"""Module 6: evaluate the one final ensemble and save its real errors."""

from __future__ import annotations

import re

import pandas as pd

from src.config import CLASS_NAMES, RESULTS_DIR, create_project_directories
from src.dataset_utils import load_fixed_data_splits
from src.evaluation import (
    calculate_metrics,
    save_evaluation_outputs,
    update_metrics_file,
)
from src.prediction import get_ensemble_predictor


NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|na|nai|nei|nay|nahi|dont|doesnt|didnt|isnt|cant|wont)\b"
)
CONTRAST_PATTERN = re.compile(r"\b(?:but|however|kintu|tobe)\b")
SPELLING_VARIANTS = {"bhalo", "valo", "vhalo", "vaalo", "kharap", "kharappp"}


def categorize_error(row: pd.Series) -> str:
    """Assign an observable language category to one incorrect prediction."""

    text = str(row["processed_text"]).lower()
    tokens = text.split()
    if len(tokens) <= 3:
        return "Very short text"
    if NEGATION_PATTERN.search(text):
        return "Negation"
    if CONTRAST_PATTERN.search(text) or row["label"] == "Mixed":
        return "Mixed or contrastive sentiment"
    if SPELLING_VARIANTS.intersection(tokens):
        return "Romanized spelling variation"
    return "Other / manual review needed"


def build_final_analysis() -> pd.DataFrame:
    """Evaluate the combined model and update the single metrics table."""

    create_project_directories()
    _, validation_data, test_data = load_fixed_data_splits()
    predictor = get_ensemble_predictor()

    validation_probabilities = predictor.predict_probabilities(
        validation_data["original_text"].tolist()
    )
    validation_predictions = [
        CLASS_NAMES[index] for index in validation_probabilities.argmax(axis=1)
    ]
    validation_metrics = calculate_metrics(
        validation_data["label"], validation_predictions
    )

    test_probabilities = predictor.predict_probabilities(
        test_data["original_text"].tolist()
    )
    test_predictions = [
        CLASS_NAMES[index] for index in test_probabilities.argmax(axis=1)
    ]
    result = save_evaluation_outputs(
        experiment_id="M6.1",
        experiment_name="Validation-Weighted NLP Ensemble",
        family="Combined NLP System",
        test_data=test_data,
        predictions=test_predictions,
    )
    result["Representation"] = "TF-IDF + Word2Vec + BiLSTM + Transformer"
    result["Validation Macro F1"] = validation_metrics["Macro F1"]
    all_metrics = update_metrics_file(pd.DataFrame([result]))

    error_analysis = test_data[
        ["sample_id", "original_text", "processed_text", "label"]
    ].copy()
    error_analysis["predicted_label"] = test_predictions
    error_analysis["confidence"] = test_probabilities.max(axis=1)
    error_analysis = error_analysis[
        error_analysis["label"] != error_analysis["predicted_label"]
    ].copy()
    error_analysis["error_category"] = error_analysis.apply(categorize_error, axis=1)
    error_analysis.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)

    print("MODULE 6 COMPLETE")
    print(all_metrics.to_string(index=False))
    print("Final prediction method: Validation-Weighted NLP Ensemble")
    return all_metrics


if __name__ == "__main__":
    build_final_analysis()
