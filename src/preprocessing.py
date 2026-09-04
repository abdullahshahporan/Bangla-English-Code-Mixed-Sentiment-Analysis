"""Module 1: code-mixed text cleaning and corpus preparation.

The cleaning rules intentionally preserve Bengali Unicode text and sentiment
negations such as ``not``, ``na`` and ``nai``.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import (
    LABEL_ID_TO_NAME,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    create_project_directories,
)
from src.dataset_utils import create_fixed_data_splits


HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)

# Keep English letters, digits, apostrophes, and the Bengali Unicode block.
UNNECESSARY_SYMBOL_PATTERN = re.compile(r"[^a-z0-9'\u0980-\u09FF]+")
WHITESPACE_PATTERN = re.compile(r"\s+")

PROTECTED_SENTIMENT_WORDS = {
    "not",
    "no",
    "nor",
    "never",
    "but",
    "however",
    "na",
    "nai",
    "nei",
    "nay",
    "nahi",
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "cant",
    "wont",
}
SAFE_ENGLISH_STOP_WORDS = set(ENGLISH_STOP_WORDS).difference(PROTECTED_SENTIMENT_WORDS)


def _lemmatize_english_tokens(tokens: list[str]) -> list[str]:
    """Optionally lemmatize English-looking tokens with NLTK.

    This helper is deliberately opt-in. An English lemmatizer does not
    understand Romanized Bangla, so it should not be applied automatically.
    """

    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(token) if token.isascii() else token
            for token in tokens
        ]
    except (ImportError, LookupError) as error:
        raise RuntimeError(
            "Optional lemmatization needs NLTK and its WordNet data. "
            "Install NLTK and run `nltk.download('wordnet')`, or leave "
            "lemmatization disabled."
        ) from error


def preprocess_text(
    text: object,
    *,
    remove_stopwords: bool = False,
    lemmatize_english: bool = False,
) -> str:
    """Convert one raw sentence into clean, space-separated tokens.

    Args:
        text: Any input value. Missing values become an empty string.
        remove_stopwords: Remove common English function words when ``True``.
        lemmatize_english: Apply optional WordNet lemmatization when ``True``.

    Returns:
        A cleaned string suitable for the later project modules.
    """

    if text is None or pd.isna(text):
        return ""

    cleaned_text = html.unescape(str(text)).lower()
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    cleaned_text = URL_PATTERN.sub(" ", cleaned_text)
    cleaned_text = UNNECESSARY_SYMBOL_PATTERN.sub(" ", cleaned_text)
    cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()

    tokens = cleaned_text.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in SAFE_ENGLISH_STOP_WORDS]
    if lemmatize_english:
        tokens = _lemmatize_english_tokens(tokens)

    return " ".join(tokens)


def calculate_edit_distance(first_word: str, second_word: str) -> int:
    """Return the Levenshtein edit distance between two words."""

    previous_row = list(range(len(second_word) + 1))
    for first_index, first_character in enumerate(first_word, start=1):
        current_row = [first_index]
        for second_index, second_character in enumerate(second_word, start=1):
            insertion_cost = current_row[second_index - 1] + 1
            deletion_cost = previous_row[second_index] + 1
            replacement_cost = previous_row[second_index - 1]
            if first_character != second_character:
                replacement_cost += 1
            current_row.append(min(insertion_cost, deletion_cost, replacement_cost))
        previous_row = current_row
    return previous_row[-1]


def prepare_corpus(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    *,
    remove_stopwords: bool = False,
    lemmatize_english: bool = False,
) -> pd.DataFrame:
    """Clean the raw dataset, save it, and create the shared data splits."""

    create_project_directories()
    raw_data = pd.read_csv(input_path)
    required_columns = {"Sentence", "Label"}
    missing_columns = required_columns.difference(raw_data.columns)
    if missing_columns:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing_columns)}")

    raw_sample_count = len(raw_data)
    data = raw_data[["Sentence", "Label"]].copy()
    data["Sentence"] = data["Sentence"].astype("string").str.strip()
    data = data.dropna(subset=["Sentence", "Label"])
    data = data[data["Sentence"] != ""]
    data = data.drop_duplicates(subset=["Sentence", "Label"]).reset_index(drop=True)

    data["label_id"] = pd.to_numeric(data["Label"], errors="coerce")
    data = data.dropna(subset=["label_id"])
    data["label_id"] = data["label_id"].astype(int)
    unknown_labels = set(data["label_id"]).difference(LABEL_ID_TO_NAME)
    if unknown_labels:
        raise ValueError(f"Unknown numeric labels found: {sorted(unknown_labels)}")

    data = data.rename(columns={"Sentence": "original_text"})
    data["label"] = data["label_id"].map(LABEL_ID_TO_NAME)
    data["processed_text"] = data["original_text"].map(
        lambda value: preprocess_text(
            value,
            remove_stopwords=remove_stopwords,
            lemmatize_english=lemmatize_english,
        )
    )
    data = data[data["processed_text"] != ""].reset_index(drop=True)
    data["token_count"] = data["processed_text"].str.split().str.len()

    final_data = data[
        ["original_text", "processed_text", "label", "label_id", "token_count"]
    ]
    final_data.to_csv(output_path, index=False)
    train_data, validation_data, test_data = create_fixed_data_splits(final_data)

    token_counts = Counter(
        token
        for sentence in final_data["processed_text"]
        for token in sentence.split()
    )
    print("MODULE 1 COMPLETE")
    print(f"Raw samples: {raw_sample_count:,}")
    print(f"Final samples: {len(final_data):,}")
    print(f"Classes: {', '.join(sorted(final_data['label'].unique()))}")
    print(f"Vocabulary size: {len(token_counts):,}")
    print(f"Average sentence length: {final_data['token_count'].mean():.2f}")
    print(f"95th percentile length: {final_data['token_count'].quantile(0.95):.0f}")
    print(
        "Split sizes: "
        f"train={len(train_data):,}, validation={len(validation_data):,}, "
        f"test={len(test_data):,}"
    )
    print(f"Processed dataset saved to: {output_path}")
    return final_data


if __name__ == "__main__":
    prepare_corpus()
