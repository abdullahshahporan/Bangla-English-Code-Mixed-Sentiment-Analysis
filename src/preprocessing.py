"""Reusable Unicode-safe preprocessing for code-mixed sentiment text."""

from __future__ import annotations

import html
import re

import pandas as pd


HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)

# Preserve English, Bengali, digits, and apostrophes. Removing English stop
# words or applying English lemmatization could damage Romanized Bangla and
# sentiment expressions such as "not good" or "valo na".
UNNECESSARY_SYMBOL_PATTERN = re.compile(r"[^a-z0-9'\u0980-\u09FF]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: object) -> str:
    """Return normalized text shared by notebooks and live prediction."""

    if text is None or pd.isna(text):
        return ""

    cleaned_text = html.unescape(str(text)).lower()
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    cleaned_text = URL_PATTERN.sub(" ", cleaned_text)
    cleaned_text = UNNECESSARY_SYMBOL_PATTERN.sub(" ", cleaned_text)
    return WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()
