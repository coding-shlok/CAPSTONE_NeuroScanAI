"""Renders the Grad-CAM result as an overlay image on the EEG trace, saved
to disk (Section 5.7) — this is the artifact whose path gets stored on the
prediction record (Section 8's `heatmap_path`)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.explainability.gradcam import GradCAMResult


def save_heatmap_overlay(
    eeg_trace: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    result: GradCAMResult,
    out_path: str | Path,
    n_display_channels: int = 3,
) -> Path:
    """eeg_trace: [n_channels, total_timepoints] standardized signal for the
    full recording (windows concatenated in order), aligned 1:1 with
    result.temporal_heatmap."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    display_channels = result.top_channels[:n_display_channels]
    n_timepoints = eeg_trace.shape[1]
    t = np.arange(n_timepoints) / sfreq

    fig, axes = plt.subplots(
        len(display_channels) + 1,
        1,
        figsize=(10, 2 * (len(display_channels) + 1)),
        sharex=False,
        gridspec_kw={"height_ratios": [1] * len(display_channels) + [0.8]},
    )
    if len(display_channels) == 1:
        axes = [axes[0], axes[1]]

    heatmap = result.temporal_heatmap
    for ax, ch_name in zip(axes[: len(display_channels)], display_channels):
        ch_idx = channel_names.index(ch_name)
        ax.pcolormesh(
            t,
            [eeg_trace[ch_idx].min(), eeg_trace[ch_idx].max()],
            np.tile(heatmap, (2, 1)),
            cmap="Reds",
            alpha=0.4,
            shading="gouraud",
        )
        ax.plot(t, eeg_trace[ch_idx], color="black", linewidth=0.6)
        ax.set_ylabel(ch_name)
        ax.set_xlim(t[0], t[-1])

    axes[-2].set_xlabel("Time (s)")

    bar_ax = axes[-1]
    channels_sorted = sorted(
        result.channel_importance, key=result.channel_importance.get, reverse=True
    )
    values = [result.channel_importance[c] for c in channels_sorted]
    colors = ["#c0392b" if c in display_channels else "#95a5a6" for c in channels_sorted]
    bar_ax.bar(channels_sorted, values, color=colors)
    bar_ax.set_ylabel("Channel importance")
    bar_ax.tick_params(axis="x", rotation=90, labelsize=7)

    fig.suptitle(
        f"{result.disorder} risk: {result.predicted_probability:.0%} "
        f"— most influential: {', '.join(result.top_channels)}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
