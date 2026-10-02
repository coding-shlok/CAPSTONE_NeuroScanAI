"""Shared backbone: CNN (spatial, per-window) + BiLSTM (temporal, across
windows) -> one embedding vector, architecture-identical regardless of which
of the three source datasets the input came from (Section 5.3).

This shared embedding is the technical core of the "unified architecture"
claim, so nothing dataset-specific belongs in this module — dataset
differences are resolved upstream, in preprocessing's channel mapping.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.config import BackboneConfig


class CNNEncoder(nn.Module):
    """Extracts spatial features across EEG channels within a single window.

    Input: [N, n_channels, n_timepoints] (N = batch * n_windows, windows are
    encoded independently of each other here — the BiLSTM is what relates
    windows to each other). Conv1d mixes across the channel dimension while
    sliding over time, producing a stack of learned filter responses.
    """

    def __init__(self, n_channels: int, conv_channels: list[int], kernel_size: int, dropout: float = 0.0):
        super().__init__()
        conv_layers = []
        blocks = []
        in_ch = n_channels
        for out_ch in conv_channels:
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            conv_layers.append(conv)
            blocks.append(
                nn.Sequential(conv, nn.BatchNorm1d(out_ch), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout))
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self._conv_layers = conv_layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = conv_channels[-1]

    @property
    def last_conv_block(self) -> nn.Module:
        """Grad-CAM target: the last conv block's output is the final
        spatial feature map before global pooling collapses the time axis."""
        return self.blocks[-1]

    def forward(self, x: torch.Tensor, return_feature_map: bool = False):
        fmap = x
        for block in self.blocks:
            fmap = block(fmap)
        pooled = self.pool(fmap).squeeze(-1)  # [N, out_dim]
        if return_feature_map:
            return pooled, fmap
        return pooled


class SharedBackbone(nn.Module):
    def __init__(
        self,
        n_channels: int,
        cnn_channels: list[int],
        cnn_kernel_size: int,
        lstm_hidden: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cnn = CNNEncoder(n_channels, cnn_channels, cnn_kernel_size, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(lstm_hidden * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_windows, n_channels, n_timepoints] -> [batch, embedding_dim]."""
        batch, n_windows, n_channels, n_timepoints = x.shape
        x = x.view(batch * n_windows, n_channels, n_timepoints)
        window_features = self.cnn(x)  # [batch * n_windows, cnn_out_dim]
        window_features = window_features.view(batch, n_windows, -1)
        lstm_out, _ = self.lstm(window_features)  # [batch, n_windows, 2 * lstm_hidden]
        pooled = lstm_out.mean(dim=1)  # temporal pooling across windows
        pooled = self.dropout(pooled)
        embedding = self.proj(pooled)  # [batch, embedding_dim]
        return embedding

    @classmethod
    def from_config(cls, n_channels: int, config: BackboneConfig) -> "SharedBackbone":
        return cls(
            n_channels=n_channels,
            cnn_channels=config.cnn_channels,
            cnn_kernel_size=config.cnn_kernel_size,
            lstm_hidden=config.lstm_hidden,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )
