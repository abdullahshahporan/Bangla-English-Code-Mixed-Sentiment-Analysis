"""PyTorch implementations of the BiLSTM and Transformer components."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
PAD_ID = 0
UNKNOWN_ID = 1


@dataclass
class NeuralModelConfig:
    """All architecture settings saved beside a neural checkpoint."""

    model_type: str
    vocabulary_size: int
    number_of_classes: int = 4
    embedding_dimension: int = 100
    hidden_dimension: int = 128
    number_of_layers: int = 2
    dropout: float = 0.30
    maximum_length: int = 50
    attention_heads: int = 4
    feedforward_dimension: int = 256

    def to_dictionary(self) -> dict[str, object]:
        """Convert this configuration into checkpoint-friendly values."""

        return asdict(self)


def build_vocabulary(
    texts: list[str],
    minimum_frequency: int = 2,
) -> dict[str, int]:
    """Build a training-only token vocabulary with PAD and UNK entries."""

    token_counts = Counter(token for text in texts for token in str(text).split())
    vocabulary = {PAD_TOKEN: PAD_ID, UNKNOWN_TOKEN: UNKNOWN_ID}
    for token, count in sorted(token_counts.items()):
        if count >= minimum_frequency:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def choose_maximum_length(texts: list[str], percentile: float = 95.0) -> int:
    """Choose sequence length from a percentile of training sentence lengths."""

    lengths = torch.tensor(
        [max(len(str(text).split()), 1) for text in texts], dtype=torch.float32
    )
    quantile = torch.quantile(lengths, percentile / 100.0)
    return max(2, int(math.ceil(float(quantile))))


def encode_text(
    text: str,
    vocabulary: dict[str, int],
    maximum_length: int,
) -> tuple[list[int], int]:
    """Map tokens to IDs, then truncate and pad to one fixed length."""

    token_ids = [vocabulary.get(token, UNKNOWN_ID) for token in str(text).split()]
    token_ids = token_ids[:maximum_length]
    real_length = max(len(token_ids), 1)
    if not token_ids:
        token_ids = [UNKNOWN_ID]
    padding_needed = maximum_length - len(token_ids)
    token_ids.extend([PAD_ID] * padding_needed)
    return token_ids, real_length


class BiLSTMSentimentClassifier(nn.Module):
    """A stacked bidirectional LSTM for four-class sentiment prediction."""

    def __init__(self, config: NeuralModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dimension,
            padding_idx=PAD_ID,
        )
        recurrent_dropout = config.dropout if config.number_of_layers > 1 else 0.0
        self.recurrent_layer = nn.LSTM(
            input_size=config.embedding_dimension,
            hidden_size=config.hidden_dimension,
            num_layers=config.number_of_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(
            config.hidden_dimension * 2, config.number_of_classes
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return four unnormalized class scores for each sentence."""

        embedded_tokens = self.embedding(input_ids)
        packed_tokens = pack_padded_sequence(
            embedded_tokens,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden_output = self.recurrent_layer(packed_tokens)

        hidden_state, _ = hidden_output
        sentence_vector = torch.cat(
            (hidden_state[-2], hidden_state[-1]), dim=1
        )
        return self.classifier(self.dropout(sentence_vector))


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic token-position information to embeddings."""

    def __init__(self, embedding_dimension: int, maximum_length: int) -> None:
        super().__init__()
        positions = torch.arange(maximum_length).unsqueeze(1)
        frequency_terms = torch.exp(
            torch.arange(0, embedding_dimension, 2)
            * (-math.log(10_000.0) / embedding_dimension)
        )
        encoding = torch.zeros(maximum_length, embedding_dimension)
        encoding[:, 0::2] = torch.sin(positions * frequency_terms)
        encoding[:, 1::2] = torch.cos(positions * frequency_terms[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        """Add encodings for the sequence positions currently in use."""

        return embedded_tokens + self.encoding[:, : embedded_tokens.size(1)]


class TransformerSentimentClassifier(nn.Module):
    """A small encoder-only Transformer with padding-aware mean pooling."""

    def __init__(self, config: NeuralModelConfig) -> None:
        super().__init__()
        if config.embedding_dimension % config.attention_heads != 0:
            raise ValueError("embedding_dimension must be divisible by attention_heads")

        self.config = config
        self.embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dimension,
            padding_idx=PAD_ID,
        )
        self.position = SinusoidalPositionalEncoding(
            config.embedding_dimension, config.maximum_length
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.number_of_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(
            config.embedding_dimension, config.number_of_classes
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode sentences and return four class logits."""

        del lengths  # The Transformer uses its padding mask instead.
        padding_mask = input_ids.eq(PAD_ID)
        embedded_tokens = self.embedding(input_ids) * math.sqrt(
            self.config.embedding_dimension
        )
        embedded_tokens = self.position(embedded_tokens)
        encoded_tokens = self.encoder(
            embedded_tokens,
            src_key_padding_mask=padding_mask,
        )

        # PAD tokens must not contribute to the document representation.
        real_token_mask = (~padding_mask).unsqueeze(-1).to(encoded_tokens.dtype)
        token_sum = (encoded_tokens * real_token_mask).sum(dim=1)
        token_count = real_token_mask.sum(dim=1).clamp(min=1.0)
        pooled_output = token_sum / token_count
        return self.classifier(self.dropout(pooled_output))


def create_neural_model(config: NeuralModelConfig) -> nn.Module:
    """Create a model from its saved configuration."""

    if config.model_type == "bilstm":
        return BiLSTMSentimentClassifier(config)
    if config.model_type == "transformer":
        return TransformerSentimentClassifier(config)
    raise ValueError(f"Unsupported model type: {config.model_type}")
