"""Shared project settings.

Keeping paths and label names in one place prevents small inconsistencies
between notebooks and training scripts.
"""

from pathlib import Path


# Path(__file__) points to src/config.py, so parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_sentiment.csv"
SPLIT_DIR = DATA_DIR / "splits"
TRAIN_DATA_PATH = SPLIT_DIR / "train.csv"
VALIDATION_DATA_PATH = SPLIT_DIR / "validation.csv"
TEST_DATA_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"

RANDOM_SEED = 42

# These numeric labels come from the original BnSentMix-style dataset.
LABEL_ID_TO_NAME = {
    0: "Positive",
    1: "Negative",
    2: "Neutral",
    3: "Mixed",
}
LABEL_NAME_TO_ID = {name: label_id for label_id, name in LABEL_ID_TO_NAME.items()}
CLASS_NAMES = [LABEL_ID_TO_NAME[index] for index in sorted(LABEL_ID_TO_NAME)]


def create_project_directories() -> None:
    """Create every output directory used by the project."""

    for directory in (
        RAW_DATA_PATH.parent,
        PROCESSED_DATA_PATH.parent,
        SPLIT_DIR,
        MODEL_DIR,
        CONFUSION_MATRIX_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
