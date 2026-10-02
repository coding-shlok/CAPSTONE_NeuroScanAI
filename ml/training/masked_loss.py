"""The masked multi-task loss (Section 5.5) — the one genuine algorithmic
idea in this spec, not boilerplate.

Because no dataset has labels for more than one disorder, a batch can only
supply a gradient signal for the one head matching its source dataset. The
masking is structural rather than a zeroed-out loss term: `model(x,
dataset_id=...)` only evaluates that one head, so the other two heads never
enter the autograd graph for this step and get no gradient at all — not
"gradient of zero," but literally none, which is what "masked" means here.
The shared backbone *is* in the graph for every batch regardless of dataset,
so it accumulates gradient from all three datasets over the course of
training; that's what makes it learn a shared EEG representation instead of
three independent ones bolted together.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_multi_task_step(
    model: nn.Module,
    tensor: torch.Tensor,
    labels: torch.Tensor,
    dataset_id: str,
    pos_weight: torch.Tensor | None = None,
    dataset_loss_weight: float | None = None,
) -> torch.Tensor:
    """One training step's loss for a single-dataset batch.

    tensor: [batch, n_windows, n_channels, n_timepoints]
    labels: [batch] binary risk labels for `dataset_id`'s disorder
    Returns a scalar loss ready for `.backward()`. Only `dataset_id`'s head
    is evaluated (model.forward masks the rest), so `.backward()` on this
    loss leaves every other head's gradient at None for this step.

    `dataset_loss_weight` scales the resulting scalar loss (not the
    per-sample BCE terms, which `pos_weight` already handles) — it's the
    training_config.yaml `inter_dataset_loss_weights` knob, applied on top
    of pos_weight rather than instead of it.
    """
    head_out = model(tensor, dataset_id=dataset_id)  # {dataset_id: logits [batch]}
    logits = head_out[dataset_id]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(logits, labels)
    if dataset_loss_weight is not None:
        loss = loss * dataset_loss_weight
    return loss
