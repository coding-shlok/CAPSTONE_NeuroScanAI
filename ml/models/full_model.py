"""Wires the shared backbone to the multi-task heads (Sections 5.3-5.4).

A single fixed n_channels (the 19-electrode common montage) works for ADHD
and MCI, whose source recordings both map cleanly onto that montage. CHB-MIT
(epilepsy) does not: it is recorded natively as 23 bipolar-derivation
channels (e.g. "FP1-F7"), not single referential electrodes, and squeezing
those into the 19-electrode montage would either collide (multiple bipolar
pairs sharing a first electrode) or silently drop channels (O1/O2 never
appear as a pair's first electrode) — plus it would fight, not match, how
CHB-MIT is used in essentially all published work on it. Rather than forcing
every dataset through one channel count, each dataset gets a thin
`InputAdapter` that projects its native channel count onto the backbone's
internal n_channels; the backbone itself (this module's whole point) stays
completely dataset-agnostic and unchanged."""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.backbone import SharedBackbone
from ml.models.config import ModelConfig
from ml.models.heads import MultiTaskHeads


class InputAdapter(nn.Module):
    """Projects a dataset's native per-window channel count onto the shared
    backbone's fixed n_channels via a 1x1 (per-timepoint) convolution.
    Identity when the dataset's native channel count already matches
    n_channels (true today for ADHD and MCI), so those datasets pick up no
    extra learned parameters or behavior change from this class existing."""

    def __init__(self, native_channels: int, n_channels: int):
        super().__init__()
        self.proj = (
            nn.Identity()
            if native_channels == n_channels
            else nn.Conv1d(native_channels, n_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_windows, native_channels, n_timepoints] -> [batch, n_windows, n_channels, n_timepoints]."""
        batch, n_windows, native_channels, n_timepoints = x.shape
        x = x.reshape(batch * n_windows, native_channels, n_timepoints)
        x = self.proj(x)
        return x.reshape(batch, n_windows, x.shape[1], n_timepoints)


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
        native_channels: dict[str, int] | None = None,
        backbone_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        native_channels = native_channels or {}
        self.backbone = SharedBackbone(
            n_channels, cnn_channels, cnn_kernel_size, lstm_hidden, embedding_dim,
            dropout=backbone_dropout,
        )
        self.heads = MultiTaskHeads(embedding_dim, disorders, head_hidden_dim, dropout=head_dropout)
        self.input_adapters = nn.ModuleDict(
            {d: InputAdapter(native_channels.get(d, n_channels), n_channels) for d in disorders}
        )

    def forward(
        self, x: torch.Tensor, dataset_id: str | None = None
    ) -> dict[str, torch.Tensor]:
        """When `dataset_id` is given (training, or single-disorder
        inference), x is expected in that dataset's own native channel
        format and is adapted before the shared backbone sees it. When
        `dataset_id` is None (multi-disorder inference over one recording,
        e.g. ml.pipeline's run_full_pipeline), x must already be in the
        shared n_channels format — i.e. only datasets whose native format
        equals n_channels (ADHD, MCI) support that all-heads-at-once path
        today; epilepsy's differently-shaped native input means a recording
        must be explicitly routed dataset_id="epilepsy" to be scored for it."""
        if dataset_id is not None:
            embedding = self.backbone(self.input_adapters[dataset_id](x))
            return self.heads(embedding, dataset_id=dataset_id)
        embedding = self.backbone(x)
        return self.heads(embedding, dataset_id=None)

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
            native_channels=config.native_channels,
            backbone_dropout=config.backbone.dropout,
            head_dropout=config.heads.dropout,
        )
