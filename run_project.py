"""Command-line entry point for the project modules.

Examples:
    python run_project.py preprocess
    python run_project.py classical
    python run_project.py neural bilstm --epochs 12
    python run_project.py predict "camera bhalo but battery kharap"
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Run only the project stage requested by the user."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preprocess", help="Prepare data and create fixed splits")
    subparsers.add_parser("classical", help="Train Module 2 models")
    subparsers.add_parser("embeddings", help="Train Module 3 models")
    subparsers.add_parser("finalize", help="Build Module 6 comparison and errors")

    neural_parser = subparsers.add_parser("neural", help="Train one neural model")
    neural_parser.add_argument("model", choices=("rnn", "bilstm", "transformer"))
    neural_parser.add_argument("--epochs", type=int, default=12)
    neural_parser.add_argument("--batch-size", type=int, default=64)
    neural_parser.add_argument("--patience", type=int, default=3)

    prediction_parser = subparsers.add_parser("predict", help="Classify new text")
    prediction_parser.add_argument("text")
    arguments = parser.parse_args()

    if arguments.command == "preprocess":
        from src.preprocessing import prepare_corpus

        prepare_corpus()
    elif arguments.command == "classical":
        from src.classical_models import train_classical_models

        train_classical_models()
    elif arguments.command == "embeddings":
        from src.word_embeddings import train_embedding_models

        train_embedding_models()
    elif arguments.command == "neural":
        from src.train_neural_models import TrainingSettings, train_neural_model

        train_neural_model(
            arguments.model,
            TrainingSettings(
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                patience=arguments.patience,
            ),
        )
    elif arguments.command == "finalize":
        from src.final_analysis import build_final_analysis

        build_final_analysis()
    else:
        from src.prediction import predict_sentiment

        result = predict_sentiment(arguments.text)
        print(f"Model: {result['model']}")
        print(f"Predicted sentiment: {result['predicted_sentiment']}")
        print("Probabilities:")
        for class_name, probability in result["probabilities"].items():
            print(f"  {class_name:<8}: {probability:.2%}")


if __name__ == "__main__":
    main()
