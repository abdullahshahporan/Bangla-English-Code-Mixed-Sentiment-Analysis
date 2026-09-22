"""Combine four NLP models into one final sentiment prediction."""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import CLASS_NAMES, MODEL_DIR
from src.neural_models import NeuralModelConfig, create_neural_model, encode_text
from src.preprocessing import preprocess_text
from src.word_embeddings import weighted_document_vectors


CONTRAST_PATTERN = re.compile(r"\b(?:but|kintu|tobe)\b")
# Chosen in 5% steps using validation Macro F1. The test set was not used.
MODEL_WEIGHTS = (0.85, 0.05, 0.05, 0.05)


class EnsembleSentimentPredictor:
    """Load the final artifacts once and combine their probabilities.

    The word-and-character TF-IDF model receives most weight because it gave
    the best validation results. The other three models still contribute.
    """

    def __init__(self) -> None:
        self.tfidf_bundle = joblib.load(MODEL_DIR / "tfidf_logistic.pkl")
        self.word2vec_bundle = joblib.load(MODEL_DIR / "word2vec_sentiment.pkl")
        self.neural_components = [
            self._load_neural_component("bilstm"),
            self._load_neural_component("transformer"),
        ]

        self.component_weights = np.asarray(MODEL_WEIGHTS, dtype=np.float64)

    @staticmethod
    def _load_neural_component(model_type: str) -> dict[str, object]:
        """Load one neural checkpoint and reconstruct its architecture."""

        checkpoint = torch.load(
            MODEL_DIR / f"{model_type}_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        config = NeuralModelConfig(**checkpoint["config"])
        model = create_neural_model(config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return {
            "model": model,
            "config": config,
            "vocabulary": checkpoint["vocabulary"],
            "classes": checkpoint.get("class_names", CLASS_NAMES),
            "validation_macro_f1": checkpoint["validation_macro_f1"],
        }

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        source_classes: list[str],
    ) -> np.ndarray:
        """Put every model's probabilities in the common class order."""

        aligned = np.zeros((len(probabilities), len(CLASS_NAMES)), dtype=np.float64)
        for source_index, class_name in enumerate(source_classes):
            target_index = CLASS_NAMES.index(str(class_name))
            aligned[:, target_index] = probabilities[:, source_index]
        return aligned

    def _predict_tfidf(self, processed_texts: list[str]) -> np.ndarray:
        """Return aligned TF-IDF Logistic Regression probabilities."""

        features = self.tfidf_bundle["vectorizer"].transform(processed_texts)
        probabilities = self.tfidf_bundle["classifier"].predict_proba(features)
        return self._align_probabilities(
            probabilities, self.tfidf_bundle["classes"]
        )

    def _predict_word2vec(self, processed_texts: list[str]) -> np.ndarray:
        """Return aligned Word2Vec-classifier probabilities."""

        features = weighted_document_vectors(
            pd.Series(processed_texts),
            self.word2vec_bundle["tfidf_vectorizer"],
            self.word2vec_bundle["word2vec"],
        )

        probabilities = self.word2vec_bundle["classifier"].predict_proba(features)
        return self._align_probabilities(
            probabilities, self.word2vec_bundle["classes"]
        )

    def _predict_neural(
        self,
        processed_texts: list[str],
        component: dict[str, object],
        batch_size: int = 128,
    ) -> np.ndarray:
        """Return aligned probabilities from one neural model in batches."""

        config = component["config"]
        encoded_rows = [
            encode_text(text, component["vocabulary"], config.maximum_length)
            for text in processed_texts
        ]
        input_ids = torch.tensor([row[0] for row in encoded_rows], dtype=torch.long)
        lengths = torch.tensor([row[1] for row in encoded_rows], dtype=torch.long)

        probability_batches = []
        with torch.no_grad():
            for start in range(0, len(processed_texts), batch_size):
                end = start + batch_size
                logits = component["model"](input_ids[start:end], lengths[start:end])
                probability_batches.append(torch.softmax(logits, dim=1).numpy())

        probabilities = np.vstack(probability_batches)
        return self._align_probabilities(probabilities, component["classes"])

    def _focus_on_strong_negative_after_contrast(
        self, processed_texts: list[str], probabilities: np.ndarray
    ) -> np.ndarray:
        """Let a clearly stronger negative clause decide some Mixed cases.

        This is an explicit application rule. Corpus labels often call a
        positive-plus-negative sentence Mixed, even when the later negative
        clause matters more to the reader.
        """

        positive_index = CLASS_NAMES.index("Positive")
        negative_index = CLASS_NAMES.index("Negative")
        mixed_index = CLASS_NAMES.index("Mixed")

        for index, text in enumerate(processed_texts):
            if probabilities[index].argmax() != mixed_index:
                continue
            match = CONTRAST_PATTERN.search(text)
            if match is None:
                continue

            before = text[: match.start()].strip()
            after = text[match.end() :].strip()
            if len(before.split()) < 2 or len(after.split()) < 2:
                continue

            before_probabilities, after_probabilities = self._predict_tfidf(
                [before, after]
            )
            positive_before = before_probabilities[positive_index]
            negative_after = after_probabilities[negative_index]
            if (
                positive_before >= 0.50
                and negative_after >= 0.70
                and negative_after - positive_before >= 0.15
            ):
                probabilities[index] = after_probabilities

        return probabilities

    def predict_probabilities(self, texts: list[str]) -> np.ndarray:
        """Return one combined probability row for every input sentence."""

        if not texts:
            return np.empty((0, len(CLASS_NAMES)), dtype=np.float64)

        processed_texts = [preprocess_text(text) for text in texts]
        component_probabilities = [
            self._predict_tfidf(processed_texts),
            self._predict_word2vec(processed_texts),
            self._predict_neural(processed_texts, self.neural_components[0]),
            self._predict_neural(processed_texts, self.neural_components[1]),
        ]

        combined = np.average(
            np.stack(component_probabilities),
            axis=0,
            weights=self.component_weights,
        )
        return self._focus_on_strong_negative_after_contrast(
            processed_texts, combined
        )

    def predict(self, text: str) -> dict[str, object]:
        """Return the single final sentiment result shown by the interface."""

        probabilities = self.predict_probabilities([text])[0]
        predicted_index = int(probabilities.argmax())
        return {
            "model": "Validation-tuned NLP ensemble",
            "predicted_sentiment": CLASS_NAMES[predicted_index],
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(CLASS_NAMES, probabilities)
            },
        }


_CACHED_PREDICTOR: EnsembleSentimentPredictor | None = None


def get_ensemble_predictor() -> EnsembleSentimentPredictor:
    """Reuse loaded models across predictions, especially in the web interface."""

    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None:
        _CACHED_PREDICTOR = EnsembleSentimentPredictor()
    return _CACHED_PREDICTOR


def predict_sentiment(text: str) -> dict[str, object]:
    """Return one final result produced jointly by all four NLP models."""

    return get_ensemble_predictor().predict(text)
