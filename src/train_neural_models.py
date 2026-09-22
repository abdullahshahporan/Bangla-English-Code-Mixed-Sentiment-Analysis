"""Shared training loop for the BiLSTM and Transformer components."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CLASS_NAMES,
    LABEL_NAME_TO_ID,
    MODEL_DIR,
    RANDOM_SEED,
    create_project_directories,
)
from src.dataset_utils import load_fixed_data_splits
from src.evaluation import calculate_metrics, save_evaluation_outputs, update_metrics_file
from src.neural_models import (
    NeuralModelConfig,
    build_vocabulary,
    choose_maximum_length,
    create_neural_model,
    encode_text,
)


class SentimentDataset(Dataset):
    """Convert saved text rows into tensors when a batch requests them."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabulary: dict[str, int],
        maximum_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.maximum_length = maximum_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        token_ids, length = encode_text(
            row["processed_text"], self.vocabulary, self.maximum_length
        )
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(LABEL_NAME_TO_ID[row["label"]], dtype=torch.long),
        }


@dataclass
class TrainingSettings:
    """Small set of training choices suitable for a course project."""

    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 3


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Make repeated experiments as consistent as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], np.ndarray]:
    """Evaluate one dataset without updating model weights."""

    model.eval()
    total_loss = 0.0
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, lengths)
            loss = loss_function(logits, labels)
            total_loss += loss.item() * len(labels)

            probabilities = torch.softmax(logits, dim=1)
            all_predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
            all_probabilities.append(probabilities.cpu().numpy())

    average_loss = total_loss / max(len(data_loader.dataset), 1)
    return average_loss, all_predictions, np.vstack(all_probabilities)


def train_neural_model(
    model_type: str,
    settings: TrainingSettings | None = None,
) -> pd.DataFrame:
    """Train one neural experiment using validation Macro F1 and early stopping."""

    if model_type not in {"bilstm", "transformer"}:
        raise ValueError("model_type must be bilstm or transformer")

    settings = settings or TrainingSettings()
    set_random_seeds()
    create_project_directories()
    train_data, validation_data, test_data = load_fixed_data_splits()

    vocabulary = build_vocabulary(train_data["processed_text"].tolist())
    maximum_length = choose_maximum_length(train_data["processed_text"].tolist())
    config = NeuralModelConfig(
        model_type=model_type,
        vocabulary_size=len(vocabulary),
        embedding_dimension=128 if model_type == "transformer" else 100,
        hidden_dimension=128,
        number_of_layers=2,
        maximum_length=maximum_length,
        attention_heads=4,
        feedforward_dimension=256,
    )
    train_loader = DataLoader(
        SentimentDataset(train_data, vocabulary, maximum_length),
        batch_size=settings.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        SentimentDataset(validation_data, vocabulary, maximum_length),
        batch_size=settings.batch_size,
    )
    test_loader = DataLoader(
        SentimentDataset(test_data, vocabulary, maximum_length),
        batch_size=settings.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_neural_model(config).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    best_validation_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_training_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item() * len(labels)

        training_loss = total_training_loss / len(train_loader.dataset)
        validation_loss, validation_ids, _ = _run_evaluation(
            model, validation_loader, loss_function, device
        )
        validation_labels = [CLASS_NAMES[index] for index in validation_ids]
        validation_metrics = calculate_metrics(
            validation_data["label"], validation_labels
        )
        validation_f1 = validation_metrics["Macro F1"]
        print(
            f"Epoch {epoch:02d} | train loss={training_loss:.4f} | "
            f"validation loss={validation_loss:.4f} | "
            f"validation Macro F1={validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                print("Early stopping: validation Macro F1 stopped improving.")
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")
    model.load_state_dict(best_state)

    experiment_details = {
        "bilstm": ("M4.2", "Bidirectional LSTM", "Sequence Neural"),
        "transformer": ("M5.1", "Transformer Encoder", "Self-Attention"),
    }
    experiment_id, experiment_name, family = experiment_details[model_type]
    _, test_ids, _ = _run_evaluation(
        model, test_loader, loss_function, device
    )
    test_predictions = [CLASS_NAMES[index] for index in test_ids]
    result = save_evaluation_outputs(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        family=family,
        test_data=test_data,
        predictions=test_predictions,
    )
    result["Representation"] = {
        "bilstm": "Trainable Embedding + BiLSTM",
        "transformer": "Embedding + Positional Encoding",
    }[model_type]
    result["Validation Macro F1"] = best_validation_f1

    result_data = pd.DataFrame([result])
    update_metrics_file(result_data)

    checkpoint_path = MODEL_DIR / f"{model_type}_best.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": config.to_dictionary(),
            "vocabulary": vocabulary,
            "class_names": CLASS_NAMES,
            "validation_macro_f1": best_validation_f1,
        },
        checkpoint_path,
    )

    print(f"{experiment_name.upper()} COMPLETE")
    print(result_data.to_string(index=False))
    print(f"Best checkpoint saved to: {checkpoint_path}")
    return result_data
