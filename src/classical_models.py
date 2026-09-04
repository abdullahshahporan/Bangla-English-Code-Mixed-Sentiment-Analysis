"""Module 2: Bag-of-Words and TF-IDF classical baselines."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.config import MODEL_DIR, RANDOM_SEED, create_project_directories
from src.dataset_utils import load_fixed_data_splits
from src.evaluation import calculate_metrics, save_evaluation_outputs, update_metrics_file


def _create_vectorizers() -> dict[str, object]:
    """Create the two text representations used in Module 2."""

    common_options = {
        "lowercase": False,  # Module 1 already lowercased the text.
        "token_pattern": r"(?u)\S+",
        "min_df": 2,
        "ngram_range": (1, 2),
    }
    return {
        "BoW": CountVectorizer(**common_options),
        "TF-IDF": TfidfVectorizer(**common_options, sublinear_tf=True),
    }


def train_classical_models() -> pd.DataFrame:
    """Train all four classical experiments and save the best validation model."""

    create_project_directories()
    train_data, validation_data, test_data = load_fixed_data_splits()

    vectorizers = _create_vectorizers()
    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }

    experiment_plan = [
        ("M2.1", "BoW", "Naive Bayes"),
        ("M2.2", "TF-IDF", "Naive Bayes"),
        ("M2.3", "BoW", "Logistic Regression"),
        ("M2.4", "TF-IDF", "Logistic Regression"),
    ]
    results: list[dict[str, object]] = []
    fitted_experiments: dict[str, tuple[object, object]] = {}

    feature_sets: dict[str, tuple[object, object, object]] = {}
    for representation_name, vectorizer in vectorizers.items():
        train_features = vectorizer.fit_transform(train_data["processed_text"])
        validation_features = vectorizer.transform(validation_data["processed_text"])
        test_features = vectorizer.transform(test_data["processed_text"])
        feature_sets[representation_name] = (
            train_features,
            validation_features,
            test_features,
        )

    for experiment_id, representation_name, classifier_name in experiment_plan:
        vectorizer = vectorizers[representation_name]
        train_features, validation_features, test_features = feature_sets[
            representation_name
        ]
        classifier_template = classifiers[classifier_name]
        # A fresh classifier avoids carrying learned parameters between runs.
        classifier = classifier_template.__class__(**classifier_template.get_params())
        classifier.fit(train_features, train_data["label"])

        experiment_name = f"{representation_name} + {classifier_name}"
        validation_predictions = classifier.predict(validation_features)
        validation_metrics = calculate_metrics(
            validation_data["label"], validation_predictions
        )

        test_predictions = classifier.predict(test_features)
        result = save_evaluation_outputs(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            family=(
                "Generative" if classifier_name == "Naive Bayes" else "Discriminative"
            ),
            test_data=test_data,
            predictions=test_predictions.tolist(),
        )
        result["Representation"] = representation_name
        result["Validation Macro F1"] = validation_metrics["Macro F1"]
        results.append(result)
        fitted_experiments[experiment_name] = (vectorizer, classifier)

    metrics_data = pd.DataFrame(results).sort_values(
        ["Validation Macro F1", "Macro F1"], ascending=False
    )
    update_metrics_file(metrics_data)

    # The final ensemble uses the strongest standard NLP baseline requested in
    # the project plan: TF-IDF with Logistic Regression.
    final_name = "TF-IDF + Logistic Regression"
    final_vectorizer, final_classifier = fitted_experiments[final_name]
    validation_f1 = float(
        metrics_data.loc[
            metrics_data["Model"] == final_name, "Validation Macro F1"
        ].iloc[0]
    )
    joblib.dump(
        {
            "name": final_name,
            "vectorizer": final_vectorizer,
            "classifier": final_classifier,
            "classes": final_classifier.classes_.tolist(),
            "validation_macro_f1": validation_f1,
        },
        MODEL_DIR / "tfidf_logistic.pkl",
    )

    print("MODULE 2 COMPLETE")
    print(metrics_data.to_string(index=False))
    print("Final ensemble artifact: TF-IDF + Logistic Regression")
    return metrics_data


if __name__ == "__main__":
    train_classical_models()
