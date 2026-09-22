"""Reusable TF-IDF-weighted Word2Vec document representation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text: str) -> list[str]:
    """Split normalized text into ordered tokens."""

    return str(text).split()


def weighted_document_vectors(
    texts: pd.Series,
    vectorizer: TfidfVectorizer,
    word2vec: Word2Vec,
) -> np.ndarray:
    """Convert documents into TF-IDF-weighted averages of Word2Vec vectors."""

    tfidf_matrix = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    document_vectors = np.zeros(
        (len(texts), word2vec.vector_size), dtype=np.float32
    )

    for row_index in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(row_index)
        weighted_sum = np.zeros(word2vec.vector_size, dtype=np.float32)
        total_weight = 0.0

        for feature_index, weight in zip(row.indices, row.data):
            token = feature_names[feature_index]
            if token in word2vec.wv:
                weighted_sum += float(weight) * word2vec.wv[token]
                total_weight += float(weight)

        if total_weight > 0:
            document_vectors[row_index] = weighted_sum / total_weight

    return document_vectors
