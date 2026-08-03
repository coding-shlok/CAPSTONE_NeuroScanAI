"""Shared plumbing for the per-dataset ingestion scripts: run one recording
through preprocessing, cache the resulting tensor to disk, and append a
manifest row in the format ml/training/dataset.py expects.

These ingestion scripts are structured against each dataset's publicly
documented layout, but have not been run against the real downloads (they
aren't available in this environment) — treat the exact folder/column names
as a starting point to verify against the actual data, not as ground truth.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ml.preprocessing.config import PreprocessingConfig
from ml.preprocessing.pipeline import run_pipeline
from ml.training.dataset import ManifestEntry

logger = logging.getLogger(__name__)


def preprocess_and_cache(
    edf_path: str | Path,
    label: float,
    dataset_id: str,
    output_dir: str | Path,
    preprocessing_config: PreprocessingConfig,
    recording_id: str,
) -> ManifestEntry | None:
    """Runs the shared preprocessing pipeline on one EDF file and saves the
    resulting window tensor as .npy. Returns None (and logs a warning)
    rather than raising, so one corrupt/unreadable recording doesn't abort
    ingestion of an entire dataset."""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        windows = run_pipeline(edf_path, preprocessing_config)
    except Exception as exc:
        logger.warning("Skipping %s (%s): preprocessing failed: %s", recording_id, edf_path, exc)
        return None

    tensor_path = output_dir / f"{dataset_id}_{recording_id}.npy"
    np.save(tensor_path, windows)
    return ManifestEntry(tensor_path=str(tensor_path), label=float(label), dataset_id=dataset_id)


def write_manifest(entries: list[ManifestEntry], manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")
    return manifest_path
