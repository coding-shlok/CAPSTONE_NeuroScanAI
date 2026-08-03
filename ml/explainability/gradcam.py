"""Grad-CAM explainability (Section 5.7). Runs strictly after inference, on
the CNN layers of the shared backbone — never before or in parallel, per the
non-negotiable boundary in Section 3.1.

Produces two things from a single backward pass on the predicted disorder's
head:
  1. A time-region heatmap: classic Grad-CAM on the backbone's last conv
     block, giving per-timepoint importance across the full recording.
  2. Channel-importance scores: since the first Conv1d mixes all 19
     electrode channels into each learned filter, Grad-CAM's own feature map
     can't be attributed back to individual electrodes. Channel importance
     is instead read off the same backward pass via input gradient x input
     (a standard, lightweight saliency extension of Grad-CAM), which stays
     within "Grad-CAM only" — no separate attribution library (Captum is
     explicitly out of scope per Section 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ml.models.full_model import NeuroScanModel


@dataclass
class GradCAMResult:
    disorder: str
    predicted_probability: float
    temporal_heatmap: np.ndarray  # [n_windows * n_timepoints], normalized [0, 1]
    channel_importance: dict[str, float]  # normalized, sums to 1
    top_channels: list[str]  # channel_importance keys, sorted descending


class GradCAM:
    def __init__(self, model: NeuroScanModel):
        self.model = model
        self.target_layer = model.backbone.cnn.last_conv_block
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def explain(
        self, x: torch.Tensor, dataset_id: str, channel_names: list[str], top_k: int = 3
    ) -> GradCAMResult:
        """x: [1, n_windows, n_channels, n_timepoints] — one recording, already
        through preprocessing. Must be called after model.eval() inference has
        already produced the prediction being explained; this method reruns
        the forward pass (with grad enabled) purely to attribute it.
        """
        if x.shape[0] != 1:
            raise ValueError("GradCAM.explain expects a single recording (batch size 1).")

        self.model.zero_grad(set_to_none=True)
        x = x.clone().detach().requires_grad_(True)

        head_out = self.model(x, dataset_id=dataset_id)
        logit = head_out[dataset_id][0]
        probability = torch.sigmoid(logit).item()
        logit.backward()

        temporal_heatmap = self._temporal_heatmap(x.shape[1], x.shape[-1])
        channel_importance = self._channel_importance(x, channel_names)
        top_channels = sorted(channel_importance, key=channel_importance.get, reverse=True)[:top_k]

        return GradCAMResult(
            disorder=dataset_id,
            predicted_probability=probability,
            temporal_heatmap=temporal_heatmap,
            channel_importance=channel_importance,
            top_channels=top_channels,
        )

    def _temporal_heatmap(self, n_windows: int, n_timepoints: int) -> np.ndarray:
        activations = self._activations  # [n_windows, n_filters, T']
        gradients = self._gradients  # [n_windows, n_filters, T']
        weights = gradients.mean(dim=2, keepdim=True)  # global-average-pooled gradient per filter
        cam = F.relu((weights * activations).sum(dim=1))  # [n_windows, T']

        cam_min = cam.amin(dim=1, keepdim=True)
        cam_max = cam.amax(dim=1, keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam_upsampled = F.interpolate(
            cam.unsqueeze(1), size=n_timepoints, mode="linear", align_corners=False
        ).squeeze(1)  # [n_windows, n_timepoints]

        # Windows are consecutive, non-overlapping segments of one recording;
        # concatenating gives one heatmap spanning the full trace.
        return cam_upsampled.reshape(-1).numpy()

    def _channel_importance(
        self, x: torch.Tensor, channel_names: list[str]
    ) -> dict[str, float]:
        input_grad = x.grad[0]  # [n_windows, n_channels, n_timepoints]
        input_val = x.detach()[0]
        saliency = (input_grad * input_val).abs()
        per_channel = saliency.mean(dim=(0, 2))  # [n_channels]
        total = per_channel.sum().item()
        if total <= 0:
            # No signal reached this channel at all (e.g. a zero-filled
            # missing channel from preprocessing) -> zero importance, not NaN.
            return {name: 0.0 for name in channel_names}
        normalized = (per_channel / total).tolist()
        return dict(zip(channel_names, normalized))

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
