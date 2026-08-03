"""Generates a small synthetic multi-channel EEG recording as an .edf file.

This exists so the preprocessing pipeline (and the rest of the ml/ stack) can
be built and validated end-to-end before any of the three real datasets
(CHB-MIT, ADHD-200, OpenNeuro ds003490) are downloaded locally. It is a
smoke-test fixture, not a stand-in for real EEG signal — it lets us assert
"the code runs and produces the right shapes," not "the model learns
anything clinically meaningful."
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

# A representative subset of the common montage, deliberately not 1:1 with
# ml/configs/preprocessing_config.yaml's common_channels, so tests also
# exercise the missing-channel zero-fill path in validate_and_map_channels.
SYNTHETIC_CHANNELS = [
    "EEG Fp1-Ref",
    "EEG Fp2-Ref",
    "EEG F3-Ref",
    "EEG F4-Ref",
    "EEG C3-Ref",
    "EEG C4-Ref",
    "EEG O1-Ref",
    "EEG O2-Ref",
]


def make_synthetic_raw(
    duration_seconds: float = 30.0,
    sfreq: float = 256.0,
    n_channels: int | None = None,
    seed: int = 42,
) -> mne.io.BaseRaw:
    """Build an in-memory synthetic Raw object: mixed sine components + noise
    per channel, so ICA has non-degenerate structure to work with."""
    rng = np.random.default_rng(seed)
    channels = SYNTHETIC_CHANNELS[:n_channels] if n_channels else SYNTHETIC_CHANNELS
    n_samples = int(duration_seconds * sfreq)
    t = np.arange(n_samples) / sfreq

    data = np.zeros((len(channels), n_samples))
    for i in range(len(channels)):
        alpha = np.sin(2 * np.pi * 10 * t + rng.uniform(0, np.pi))
        beta = 0.5 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, np.pi))
        mains = 0.2 * np.sin(2 * np.pi * 60 * t)
        noise = rng.normal(0, 0.3, n_samples)
        # Scale to realistic EEG microvolt amplitudes (~1e-5 V for MNE, which
        # expects SI units internally).
        data[i] = (alpha + beta + mains + noise) * 1e-5

    info = mne.create_info(channels, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def write_synthetic_edf(
    out_path: str | Path,
    duration_seconds: float = 30.0,
    sfreq: float = 256.0,
    n_channels: int | None = None,
    seed: int = 42,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = make_synthetic_raw(duration_seconds, sfreq, n_channels, seed)
    mne.export.export_raw(str(out_path), raw, fmt="edf", overwrite=True, verbose=False)
    return out_path


if __name__ == "__main__":
    path = write_synthetic_edf("/tmp/synthetic_sample.edf")
    print(f"Wrote synthetic EDF to {path}")
