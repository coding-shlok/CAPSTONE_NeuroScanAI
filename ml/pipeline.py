"""Top-level pipeline orchestrator (Section 5.1):

    Raw EDF -> [1] Preprocessing -> [2] Shared Backbone -> [3] Masked
    Multi-Task Heads -> [4] Grad-CAM -> [5] Clinical Report Builder

This is the single entry point the backend's `/api/eeg/{id}/predict` service
should call — it owns the non-negotiable boundaries from Section 3.1: the ML
module never talks to the database (it takes a path, returns a result
object), and explainability always runs strictly after inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ml.explainability.gradcam import GradCAM
from ml.explainability.render import save_heatmap_overlay
from ml.models.full_model import NeuroScanModel
from ml.preprocessing.config import PreprocessingConfig
from ml.preprocessing.pipeline import run_pipeline as run_preprocessing
from ml.reporting.report_builder import build_report, confidence_from_probability


@dataclass
class PipelineResult:
    risk_scores: dict[str, float]  # e.g. {"epilepsy": 0.12, "adhd": 0.34, "mci": 0.08}
    predicted_disorder: str
    confidence: float
    heatmap_path: str
    report_text: str
    model_version: str


def run_full_pipeline(
    edf_path: str | Path,
    model: NeuroScanModel,
    preprocessing_config: PreprocessingConfig,
    output_dir: str | Path,
    model_version: str = "dev",
    recording_id: str = "recording",
) -> PipelineResult:
    # [1] Preprocessing — never predicts, only produces a standardized tensor.
    windows = run_preprocessing(edf_path, preprocessing_config)  # [n_windows, n_channels, n_timepoints]
    x = torch.from_numpy(windows).float().unsqueeze(0)  # [1, n_windows, n_channels, n_timepoints]

    # [2] + [3] Shared backbone + all three task heads.
    model.eval()
    with torch.no_grad():
        head_out = model(x)
    risk_scores = {disorder: torch.sigmoid(logit).item() for disorder, logit in head_out.items()}
    predicted_disorder = max(risk_scores, key=risk_scores.get)

    # [4] Grad-CAM — strictly after inference, only on the predicted disorder's head.
    with GradCAM(model) as cam:
        explanation = cam.explain(
            x, dataset_id=predicted_disorder, channel_names=preprocessing_config.common_channels
        )

    eeg_trace = windows.transpose(1, 0, 2).reshape(windows.shape[1], -1)  # [n_channels, total_timepoints]
    heatmap_path = save_heatmap_overlay(
        eeg_trace,
        preprocessing_config.common_channels,
        preprocessing_config.resample.target_sfreq,
        explanation,
        Path(output_dir) / f"{recording_id}_heatmap.png",
    )

    # [5] Clinical report builder — pure formatting.
    report = build_report(predicted_disorder, risk_scores[predicted_disorder], explanation.top_channels)

    return PipelineResult(
        risk_scores=risk_scores,
        predicted_disorder=predicted_disorder,
        confidence=confidence_from_probability(risk_scores[predicted_disorder]),
        heatmap_path=str(heatmap_path),
        report_text=report.summary_text,
        model_version=model_version,
    )
