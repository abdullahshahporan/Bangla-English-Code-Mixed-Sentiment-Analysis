"""Command-line prediction with the four-model sentiment ensemble.

Example:
    python run_project.py "camera bhalo but battery kharap"
"""

from __future__ import annotations

import argparse

from src.prediction import predict_sentiment


def main() -> None:
    """Predict one sentence supplied through the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Bangla-English code-mixed sentence")
    arguments = parser.parse_args()

    result = predict_sentiment(arguments.text)
    print(f"Model: {result['model']}")
    print(f"Predicted sentiment: {result['predicted_sentiment']}")
    print("Probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"  {class_name:<8}: {probability:.2%}")


if __name__ == "__main__":
    main()
