"""Module 3: custom Word2Vec document representations."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_DIR, RANDOM_SEED, create_project_directories
from src.dataset_utils import load_fixed_data_splits
from src.evaluation import calculate_metrics, save_evaluation_outputs, update_metrics_file


VECTOR_SIZE = 100


def tokenize(text: str) -> list[str]:
    """Split Module 1's normalized text into ordered tokens."""

    return str(text).split()


def mean_document_vector(tokens: list[str], word2vec: Word2Vec) -> np.ndarray:
    """Average all known word vectors in one document."""

    known_vectors = [word2vec.wv[token] for token in tokens if token in word2vec.wv]
    if not known_vectors:
        return np.zeros(word2vec.vector_size, dtype=np.float32)
    return np.mean(known_vectors, axis=0).astype(np.float32)


def weighted_document_vectors(
    texts: pd.Series,
    vectorizer: TfidfVectorizer,
    word2vec: Word2Vec,
) -> np.ndarray:
    """Create TF-IDF-weighted Word2Vec vectors for several documents."""

    tfidf_matrix = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    output = np.zeros((len(texts), word2vec.vector_size), dtype=np.float32)

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
            output[row_index] = weighted_sum / total_weight

    return output


def _new_classifier() -> LogisticRegression:
    """Return the common classifier for both embedding experiments."""

    return LogisticRegression(
        max_iter=1_000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def train_embedding_models() -> pd.DataFrame:
    """Train custom Skip-Gram Word2Vec and both Module 3 classifiers."""

    create_project_directories()
    train_data, validation_data, test_data = load_fixed_data_splits()
    train_tokens = train_data["processed_text"].map(tokenize).tolist()

    word2vec = Word2Vec(
        sentences=train_tokens,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        sg=1,
        workers=1,  # One worker makes seeded training reproducible.
        epochs=10,
        seed=RANDOM_SEED,
    )
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\S+",
        min_df=2,
    )
    tfidf_vectorizer.fit(train_data["processed_text"])

    mean_features = {
        "train": np.vstack(
            train_data["processed_text"].map(
                lambda text: mean_document_vector(tokenize(text), word2vec)
            )
        ),
        "validation": np.vstack(
            validation_data["processed_text"].map(
                lambda text: mean_document_vector(tokenize(text), word2vec)
            )
        ),
        "test": np.vstack(
            test_data["processed_text"].map(
                lambda text: mean_document_vector(tokenize(text), word2vec)
            )
        ),
    }
    weighted_features = {
        "train": weighted_document_vectors(
            train_data["processed_text"], tfidf_vectorizer, word2vec
        ),
        "validation": weighted_document_vectors(
            validation_data["processed_text"], tfidf_vectorizer, word2vec
        ),
        "test": weighted_document_vectors(
            test_data["processed_text"], tfidf_vectorizer, word2vec
        ),
    }

    experiments = [
        ("M3.1", "Mean Word2Vec + Logistic Regression", mean_features),
        (
            "M3.2",
            "TF-IDF Weighted Word2Vec + Logistic Regression",
            weighted_features,
        ),
    ]
    results: list[dict[str, object]] = []
    fitted_classifiers: dict[str, LogisticRegression] = {}

    for experiment_id, experiment_name, features in experiments:
        classifier = _new_classifier()
        classifier.fit(features["train"], train_data["label"])
        validation_predictions = classifier.predict(features["validation"])
        validation_metrics = calculate_metrics(
            validation_data["label"], validation_predictions
        )
        test_predictions = classifier.predict(features["test"])
        result = save_evaluation_outputs(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            family="Dense Embedding",
            test_data=test_data,
            predictions=test_predictions.tolist(),
        )
        result["Representation"] = experiment_name.split(" + ")[0]
        result["Validation Macro F1"] = validation_metrics["Macro F1"]
        results.append(result)
        fitted_classifiers[experiment_name] = classifier

    metrics_data = pd.DataFrame(results).sort_values(
        ["Validation Macro F1", "Macro F1"], ascending=False
    )
    update_metrics_file(metrics_data)

    best_name = str(metrics_data.iloc[0]["Model"])
    best_classifier = fitted_classifiers[best_name]
    joblib.dump(
        {
            "name": best_name,
            "word2vec": word2vec,
            "tfidf_vectorizer": tfidf_vectorizer,
            "classifier": best_classifier,
            "classes": best_classifier.classes_.tolist(),
            "representation": (
                "weighted" if best_name.startswith("TF-IDF") else "mean"
            ),
            "vector_size": VECTOR_SIZE,
            "validation_macro_f1": float(
                metrics_data.loc[
                    metrics_data["Model"] == best_name, "Validation Macro F1"
                ].iloc[0]
            ),
        },
        MODEL_DIR / "word2vec_sentiment.pkl",
    )

    # Token-level coverage is especially informative for spelling variation.
    all_test_tokens = [token for text in test_data["processed_text"] for token in tokenize(text)]
    known_count = sum(token in word2vec.wv for token in all_test_tokens)
    coverage_data = pd.DataFrame(
        [
            {
                "Total Tokens": len(all_test_tokens),
                "Known Tokens": known_count,
                "OOV Tokens": len(all_test_tokens) - known_count,
                "Vocabulary Coverage": known_count / max(len(all_test_tokens), 1),
            }
        ]
    )
    requested_pairs = [
        ("bhalo", "valo"),
        ("good", "bhalo"),
        ("bad", "kharap"),
    ]
    similarity_rows = []
    for first_word, second_word in requested_pairs:
        both_known = first_word in word2vec.wv and second_word in word2vec.wv
        similarity_rows.append(
            {
                "First Word": first_word,
                "Second Word": second_word,
                "Both In Vocabulary": both_known,
                "Cosine Similarity": (
                    float(word2vec.wv.similarity(first_word, second_word))
                    if both_known
                    else np.nan
                ),
            }
        )
    similarity_data = pd.DataFrame(similarity_rows)

    neighbor_rows = []
    for word in ("bhalo", "valo", "kharap", "good", "bad"):
        if word in word2vec.wv:
            for rank, (neighbor, similarity) in enumerate(
                word2vec.wv.most_similar(word, topn=5), start=1
            ):
                neighbor_rows.append(
                    {
                        "Word": word,
                        "Rank": rank,
                        "Neighbor": neighbor,
                        "Cosine Similarity": float(similarity),
                    }
                )
    nearest_words_data = pd.DataFrame(neighbor_rows)

    print("MODULE 3 COMPLETE")
    print(metrics_data.to_string(index=False))
    print(coverage_data.to_string(index=False))
    print("\nReal Word2Vec similarity examples:")
    print(similarity_data.to_string(index=False))
    print("\nNearest-word examples:")
    print(nearest_words_data.head(15).to_string(index=False))
    return metrics_data


if __name__ == "__main__":
    train_embedding_models()
