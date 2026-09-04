"""Shared evaluation helpers used by every classification module."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_DIR,
    RESULTS_DIR,
    create_project_directories,
)


def safe_file_name(name: str) -> str:
    """Convert an experiment name into a readable, filesystem-safe name."""

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def calculate_metrics(
    true_labels: list[str] | pd.Series,
    predicted_labels: list[str] | pd.Series,
) -> dict[str, float]:
    """Calculate the common metrics used in the final comparison."""

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
    }


def save_evaluation_outputs(
    *,
    experiment_id: str,
    experiment_name: str,
    family: str,
    test_data: pd.DataFrame,
    predictions: list[str],
) -> dict[str, object]:
    """Evaluate predictions and save one confusion matrix."""

    create_project_directories()
    true_labels = test_data["label"].tolist()
    metrics = calculate_metrics(true_labels, predictions)
    result = {
        "Experiment ID": experiment_id,
        "Model": experiment_name,
        "Family": family,
        **metrics,
    }

    file_stem = safe_file_name(experiment_id + "_" + experiment_name)
    matrix = confusion_matrix(true_labels, predictions, labels=CLASS_NAMES)
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    figure, axis = plt.subplots(figsize=(7, 6))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(experiment_name)
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_DIR / f"{file_stem}.png", dpi=160)
    plt.close(figure)
    return result


def update_metrics_file(new_results: pd.DataFrame) -> pd.DataFrame:
    """Add or replace experiment rows in the single project metrics file."""

    create_project_directories()
    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        existing_results = pd.read_csv(metrics_path)
        replaced_ids = set(new_results["Experiment ID"])
        existing_results = existing_results[
            ~existing_results["Experiment ID"].isin(replaced_ids)
        ]
        combined_results = pd.concat(
            [existing_results, new_results], ignore_index=True, sort=False
        )
    else:
        combined_results = new_results.copy()

    combined_results = combined_results.sort_values("Experiment ID").reset_index(
        drop=True
    )
    combined_results.to_csv(metrics_path, index=False)
    return combined_results
