"""Wires the shared backbone to the multi-task heads (Sections 5.3-5.4)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.backbone import SharedBackbone
from ml.models.config import ModelConfig
from ml.models.heads import MultiTaskHeads


class NeuroScanModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        disorders: list[str],
        cnn_channels: list[int],
        cnn_kernel_size: int,
        lstm_hidden: int,
        embedding_dim: int,
        head_hidden_dim: int,
    ):
        super().__init__()
        self.backbone = SharedBackbone(
            n_channels, cnn_channels, cnn_kernel_size, lstm_hidden, embedding_dim
        )
        self.heads = MultiTaskHeads(embedding_dim, disorders, head_hidden_dim)

    def forward(
        self, x: torch.Tensor, dataset_id: str | None = None
    ) -> dict[str, torch.Tensor]:
        embedding = self.backbone(x)
        return self.heads(embedding, dataset_id=dataset_id)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "NeuroScanModel":
        return cls(
            n_channels=config.n_channels,
            disorders=config.disorders,
            cnn_channels=config.backbone.cnn_channels,
            cnn_kernel_size=config.backbone.cnn_kernel_size,
            lstm_hidden=config.backbone.lstm_hidden,
            embedding_dim=config.backbone.embedding_dim,
            head_hidden_dim=config.heads.hidden_dim,
        )
