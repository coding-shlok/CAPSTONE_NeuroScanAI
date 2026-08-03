"""Multi-task heads (Section 5.4): one lightweight FC classifier per
disorder, sitting on top of the shared embedding. Each head is independent —
what makes them cooperate is the masked loss in ml/training/masked_loss.py,
not anything in this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Binary risk classifier for one disorder. Outputs a single logit —
    sigmoid(logit) is the risk probability."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)  # [batch]


class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, disorders: list[str], hidden_dim: int):
        super().__init__()
        self.disorders = list(disorders)
        self.heads = nn.ModuleDict(
            {d: TaskHead(embedding_dim, hidden_dim) for d in self.disorders}
        )

    def forward(
        self, embedding: torch.Tensor, dataset_id: str | None = None
    ) -> dict[str, torch.Tensor]:
        """If dataset_id is given, only that head is evaluated (used during
        masked multi-task training — the other heads get no gradient this
        step). Otherwise all heads run (used at inference)."""
        if dataset_id is not None:
            return {dataset_id: self.heads[dataset_id](embedding)}
        return {d: head(embedding) for d, head in self.heads.items()}
