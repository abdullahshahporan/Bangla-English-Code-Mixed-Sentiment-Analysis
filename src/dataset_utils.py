"""Dataset loading and the single shared train/validation/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    create_project_directories,
)


REQUIRED_PROCESSED_COLUMNS = {"original_text", "processed_text", "label", "label_id"}


def validate_processed_dataframe(dataframe: pd.DataFrame) -> None:
    """Raise a helpful error when the processed dataset has the wrong shape."""

    missing_columns = REQUIRED_PROCESSED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Processed dataset is missing columns: {missing_text}")

    if dataframe.empty:
        raise ValueError("The processed dataset is empty.")

    if dataframe[list(REQUIRED_PROCESSED_COLUMNS)].isna().any().any():
        raise ValueError("The processed dataset contains missing required values.")


def create_fixed_data_splits(
    dataframe: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create and save a 70/15/15 stratified split.

    The returned dataframes have a stable ``sample_id`` column. Later modules
    use that ID to compare predictions for the exact same test examples.
    """

    validate_processed_dataframe(dataframe)
    create_project_directories()

    data = dataframe.copy().reset_index(drop=True)
    data.insert(0, "sample_id", range(len(data)))

    train_data, temporary_data = train_test_split(
        data,
        test_size=0.30,
        random_state=random_seed,
        stratify=data["label"],
    )
    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=random_seed,
        stratify=temporary_data["label"],
    )

    # Sorting makes saved files stable and easier to inspect by hand.
    train_data = train_data.sort_values("sample_id").reset_index(drop=True)
    validation_data = validation_data.sort_values("sample_id").reset_index(drop=True)
    test_data = test_data.sort_values("sample_id").reset_index(drop=True)

    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    validation_data.to_csv(VALIDATION_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    return train_data, validation_data, test_data


def load_fixed_data_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the saved data splits and explain how to create them if missing."""

    missing_paths = [
        path
        for path in (TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Fixed data splits do not exist. Run notebook "
            "`notebooks/01_preprocessing.ipynb` first."
        )

    return (
        pd.read_csv(TRAIN_DATA_PATH),
        pd.read_csv(VALIDATION_DATA_PATH),
        pd.read_csv(TEST_DATA_PATH),
    )
