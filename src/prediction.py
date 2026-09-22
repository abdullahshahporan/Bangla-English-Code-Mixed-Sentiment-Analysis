"""Combine four NLP models into one final sentiment prediction."""

from __future__ import annotations

import json
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

MODEL_ORDER = ("tfidf", "word2vec", "bilstm", "transformer")
ENSEMBLE_WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.json"

# Used only when Module 6 has not generated ensemble_weights.json yet.
DEFAULT_MODEL_WEIGHTS = (0.85, 0.05, 0.05, 0.05)


def load_model_weights() -> tuple[float, float, float, float]:
    """Load validation-tuned ensemble weights, or use the fallback weights."""

    if not ENSEMBLE_WEIGHTS_PATH.exists():
        return DEFAULT_MODEL_WEIGHTS

    with ENSEMBLE_WEIGHTS_PATH.open("r", encoding="utf-8") as file:
        saved_data = json.load(file)

    saved_order = tuple(saved_data.get("model_order", MODEL_ORDER))
    if saved_order != MODEL_ORDER:
        raise ValueError(
            "Saved ensemble model order does not match the prediction pipeline. "
            f"Expected {MODEL_ORDER}, found {saved_order}."
        )

    weights = tuple(float(value) for value in saved_data["weights"])

    if len(weights) != len(MODEL_ORDER):
        raise ValueError(
            f"Ensemble must contain exactly {len(MODEL_ORDER)} weights."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Ensemble weights must be finite numbers.")
    if any(weight < 0 for weight in weights):
        raise ValueError("Ensemble weights cannot be negative.")
    if not np.isclose(sum(weights), 1.0, atol=1e-8):
        raise ValueError("Ensemble weights must sum to 1.0.")

    return weights


class EnsembleSentimentPredictor:
    """Load the final artifacts once and combine their probabilities."""

    def __init__(self) -> None:
        self.tfidf_bundle = joblib.load(MODEL_DIR / "tfidf_logistic.pkl")
        self.word2vec_bundle = joblib.load(MODEL_DIR / "word2vec_sentiment.pkl")
        self.neural_components = [
            self._load_neural_component("bilstm"),
            self._load_neural_component("transformer"),
        ]

        self.component_weights = np.asarray(
            load_model_weights(),
            dtype=np.float64,
        )

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

        aligned = np.zeros(
            (len(probabilities), len(CLASS_NAMES)),
            dtype=np.float64,
        )

        for source_index, class_name in enumerate(source_classes):
            target_index = CLASS_NAMES.index(str(class_name))
            aligned[:, target_index] = probabilities[:, source_index]

        return aligned

    def _predict_tfidf(self, processed_texts: list[str]) -> np.ndarray:
        """Return aligned TF-IDF Logistic Regression probabilities."""

        features = self.tfidf_bundle["vectorizer"].transform(processed_texts)
        probabilities = self.tfidf_bundle["classifier"].predict_proba(features)

        return self._align_probabilities(
            probabilities,
            self.tfidf_bundle["classes"],
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
            probabilities,
            self.word2vec_bundle["classes"],
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
            encode_text(
                text,
                component["vocabulary"],
                config.maximum_length,
            )
            for text in processed_texts
        ]

        input_ids = torch.tensor(
            [row[0] for row in encoded_rows],
            dtype=torch.long,
        )
        lengths = torch.tensor(
            [row[1] for row in encoded_rows],
            dtype=torch.long,
        )

        probability_batches = []

        with torch.no_grad():
            for start in range(0, len(processed_texts), batch_size):
                end = start + batch_size
                logits = component["model"](
                    input_ids[start:end],
                    lengths[start:end],
                )
                probability_batches.append(
                    torch.softmax(logits, dim=1).numpy()
                )

        probabilities = np.vstack(probability_batches)

        return self._align_probabilities(
            probabilities,
            component["classes"],
        )

    def predict_component_probabilities(
        self,
        texts: list[str],
    ) -> tuple[list[str], list[np.ndarray]]:
        """Return processed texts and probabilities from all four models.

        Component order is always:
        TF-IDF, Word2Vec, BiLSTM, Transformer.
        """

        processed_texts = [preprocess_text(text) for text in texts]

        if not processed_texts:
            empty = np.empty(
                (0, len(CLASS_NAMES)),
                dtype=np.float64,
            )
            return processed_texts, [empty.copy() for _ in MODEL_ORDER]

        component_probabilities = [
            self._predict_tfidf(processed_texts),
            self._predict_word2vec(processed_texts),
            self._predict_neural(
                processed_texts,
                self.neural_components[0],
            ),
            self._predict_neural(
                processed_texts,
                self.neural_components[1],
            ),
        ]

        return processed_texts, component_probabilities

    def prepare_contrast_replacements(
        self,
        processed_texts: list[str],
    ) -> dict[int, np.ndarray]:
        """Precompute negative-clause replacements used by the contrast rule.

        The expensive TF-IDF clause predictions are computed once. The rule is
        then cheap to apply repeatedly during validation weight search.
        """

        candidate_indices: list[int] = []
        before_clauses: list[str] = []
        after_clauses: list[str] = []

        for index, text in enumerate(processed_texts):
            match = CONTRAST_PATTERN.search(text)
            if match is None:
                continue

            before = text[: match.start()].strip()
            after = text[match.end() :].strip()

            if len(before.split()) < 2 or len(after.split()) < 2:
                continue

            candidate_indices.append(index)
            before_clauses.append(before)
            after_clauses.append(after)

        if not candidate_indices:
            return {}

        before_probabilities = self._predict_tfidf(before_clauses)
        after_probabilities = self._predict_tfidf(after_clauses)

        positive_index = CLASS_NAMES.index("Positive")
        negative_index = CLASS_NAMES.index("Negative")

        replacements: dict[int, np.ndarray] = {}

        for row_index, sample_index in enumerate(candidate_indices):
            positive_before = before_probabilities[row_index, positive_index]
            negative_after = after_probabilities[row_index, negative_index]

            if (
                positive_before >= 0.50
                and negative_after >= 0.70
                and negative_after - positive_before >= 0.15
            ):
                replacements[sample_index] = after_probabilities[row_index].copy()

        return replacements

    @staticmethod
    def apply_contrast_replacements(
        probabilities: np.ndarray,
        contrast_replacements: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Apply precomputed replacements only when the ensemble predicts Mixed."""

        adjusted_probabilities = probabilities.copy()
        mixed_index = CLASS_NAMES.index("Mixed")

        for sample_index, replacement in contrast_replacements.items():
            if adjusted_probabilities[sample_index].argmax() == mixed_index:
                adjusted_probabilities[sample_index] = replacement

        return adjusted_probabilities

    def combine_component_probabilities(
        self,
        component_probabilities: list[np.ndarray],
        *,
        weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
        contrast_replacements: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Combine component probabilities with weighted soft voting."""

        if len(component_probabilities) != len(MODEL_ORDER):
            raise ValueError(
                f"Expected {len(MODEL_ORDER)} component probability arrays."
            )

        selected_weights = (
            self.component_weights
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )

        if len(selected_weights) != len(MODEL_ORDER):
            raise ValueError(
                f"Expected {len(MODEL_ORDER)} ensemble weights."
            )
        if np.any(selected_weights < 0):
            raise ValueError("Ensemble weights cannot be negative.")
        if not np.isclose(selected_weights.sum(), 1.0, atol=1e-8):
            raise ValueError("Ensemble weights must sum to 1.0.")

        combined = np.average(
            np.stack(component_probabilities, axis=0),
            axis=0,
            weights=selected_weights,
        )

        if contrast_replacements:
            combined = self.apply_contrast_replacements(
                combined,
                contrast_replacements,
            )

        return combined

    def predict_probabilities(self, texts: list[str]) -> np.ndarray:
        """Return one combined probability row for every input sentence."""

        if not texts:
            return np.empty(
                (0, len(CLASS_NAMES)),
                dtype=np.float64,
            )

        processed_texts, component_probabilities = (
            self.predict_component_probabilities(texts)
        )

        contrast_replacements = self.prepare_contrast_replacements(
            processed_texts
        )

        return self.combine_component_probabilities(
            component_probabilities,
            contrast_replacements=contrast_replacements,
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
                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities,
                )
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
