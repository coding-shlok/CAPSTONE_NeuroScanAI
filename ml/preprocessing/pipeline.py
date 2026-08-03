"""EEG standardization pipeline (MVP spec Section 5.2).

Executed in a fixed order: load -> channel mapping -> band-pass -> notch ->
ICA artifact removal -> resample -> per-channel z-score -> windowing.

Every stage is a standalone function that takes an `mne.io.Raw` (or ndarray,
for the post-MNE stages) and returns the same type, so each one can be
called and unit-tested in isolation without running the whole pipeline.
Nothing here predicts anything — the only output is a standardized tensor.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import mne
import numpy as np

from ml.preprocessing.config import PreprocessingConfig

logger = logging.getLogger(__name__)

mne.set_log_level("WARNING")


def load_edf(edf_path: str | Path) -> mne.io.BaseRaw:
    """Stage 1: load a raw EDF recording."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    return raw


def _normalize_channel_name(name: str) -> str:
    """Strip dataset-specific decoration (e.g. 'EEG Fp1-Ref', 'FP1.') down to
    the bare 10-20 label so channels from different acquisition systems can
    be matched to the common montage."""
    name = name.upper()
    name = re.sub(r"^EEG\s*", "", name)
    name = re.split(r"[-.]", name)[0]
    name = name.strip()
    return name


def validate_and_map_channels(
    raw: mne.io.BaseRaw, common_channels: list[str]
) -> mne.io.BaseRaw:
    """Stage 2: map this recording's channels onto the shared montage.

    Datasets differ in channel naming, count, and montage, which is exactly
    what makes the shared backbone claim meaningful — so this stage must
    always produce the same channel set/order, regardless of source. Channels
    present in the recording are kept and renamed to the canonical label;
    channels required by the common montage but absent from this recording
    are synthesized as flat zero signal (with a warning), since silently
    changing the tensor's channel count downstream is not an option.
    """
    normalized_lookup = {_normalize_channel_name(ch): ch for ch in raw.ch_names}
    canonical_set = {c.upper(): c for c in common_channels}

    rename_map = {}
    for norm_name, orig_name in normalized_lookup.items():
        if norm_name in canonical_set:
            rename_map[orig_name] = canonical_set[norm_name]

    found_canonical = set(rename_map.values())
    missing = [c for c in common_channels if c not in found_canonical]
    if missing:
        logger.warning(
            "Recording is missing %d/%d common-montage channels %s; "
            "filling with zero signal.",
            len(missing),
            len(common_channels),
            missing,
        )

    raw = raw.copy()
    raw.rename_channels(rename_map)
    raw.pick(list(found_canonical))

    if missing:
        zero_data = np.zeros((len(missing), raw.n_times))
        info = mne.create_info(missing, raw.info["sfreq"], ch_types="eeg")
        zero_raw = mne.io.RawArray(zero_data, info, verbose=False)
        raw.add_channels([zero_raw], force_update_info=True)

    raw.reorder_channels(common_channels)
    return raw


def bandpass_filter(raw: mne.io.BaseRaw, low_freq: float, high_freq: float) -> mne.io.BaseRaw:
    """Stage 3: band-pass filter, default 0.5-45 Hz."""
    raw = raw.copy()
    raw.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
    return raw


def notch_filter(raw: mne.io.BaseRaw, freq: float) -> mne.io.BaseRaw:
    """Stage 4: notch filter to remove mains hum (50/60 Hz, region-configurable)."""
    raw = raw.copy()
    nyquist = raw.info["sfreq"] / 2.0
    if freq >= nyquist:
        logger.warning(
            "Notch frequency %.1f Hz >= Nyquist %.1f Hz for this recording; skipping.",
            freq,
            nyquist,
        )
        return raw
    raw.notch_filter(freqs=[freq], verbose=False)
    return raw


def remove_artifacts_ica(raw: mne.io.BaseRaw, ica_config) -> mne.io.BaseRaw:
    """Stage 5: ICA-based artifact removal.

    No dedicated EOG channel is guaranteed across all three datasets, so
    blink/eye-movement components are identified by correlation with the
    frontal-most channels in the common montage rather than
    `mne.preprocessing.create_eog_epochs`.
    """
    raw = raw.copy()

    # Channels synthesized as zero signal in validate_and_map_channels (because
    # this recording's montage didn't cover them) are flat and would make the
    # ICA unmixing matrix rank-deficient if included in the fit. ICA is fit
    # and applied only to channels with real signal; flat channels are passed
    # through untouched, which is correct since there's nothing to unmix.
    data = raw.get_data()
    live_picks = [ch for ch, sig in zip(raw.ch_names, data) if sig.std() > 0]

    n_components = min(ica_config.n_components, len(live_picks) - 1)
    if n_components < 2:
        logger.warning("Too few live channels for ICA; skipping artifact removal.")
        return raw

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ica_config.method,
        random_state=ica_config.random_state,
        max_iter="auto",
    )
    ica.fit(raw, picks=live_picks, verbose=False)

    frontal_candidates = [ch for ch in ("FP1", "FP2", "F3", "F4") if ch in live_picks]
    exclude = []
    if frontal_candidates:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, ch_name=frontal_candidates, threshold=ica_config.eog_corr_threshold, verbose=False
            )
            exclude = eog_indices
        except Exception as exc:  # pragma: no cover - defensive; ICA on odd data can fail edge cases
            logger.warning("EOG-component detection failed (%s); keeping all ICA components.", exc)

    ica.exclude = exclude
    ica.apply(raw, verbose=False)
    return raw


def resample(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    """Stage 6: resample to the common sampling rate."""
    raw = raw.copy()
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    return raw


def normalize_zscore(data: np.ndarray) -> np.ndarray:
    """Stage 7: per-channel z-score normalization.

    `data` is [n_channels, n_timepoints]; each channel is normalized against
    its own mean/std so channel scale differences across acquisition
    hardware don't leak into the model.
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def segment_windows(
    data: np.ndarray, sfreq: float, window_seconds: float, overlap_seconds: float = 0.0
) -> np.ndarray:
    """Stage 8: segment into fixed-length windows.

    `data` is [n_channels, n_timepoints] -> returns [n_windows, n_channels, n_window_timepoints].
    Trailing samples that don't fill a full window are dropped.
    """
    window_size = int(round(window_seconds * sfreq))
    step = window_size - int(round(overlap_seconds * sfreq))
    if step <= 0:
        raise ValueError("overlap_seconds must be smaller than window_seconds")

    n_channels, n_timepoints = data.shape
    n_windows = max(0, (n_timepoints - window_size) // step + 1)
    if n_windows == 0:
        raise ValueError(
            f"Recording too short ({n_timepoints} samples) for a single "
            f"{window_seconds}s window at {sfreq} Hz."
        )

    windows = np.empty((n_windows, n_channels, window_size), dtype=data.dtype)
    for i in range(n_windows):
        start = i * step
        windows[i] = data[:, start : start + window_size]
    return windows


def run_pipeline(edf_path: str | Path, config: PreprocessingConfig) -> np.ndarray:
    """Run all eight stages in order and return the standardized tensor.

    Output shape: [n_windows, n_channels, n_timepoints], matching Section 5.2.
    """
    raw = load_edf(edf_path)
    raw = validate_and_map_channels(raw, config.common_channels)
    raw = bandpass_filter(raw, config.bandpass.low_freq, config.bandpass.high_freq)
    notch_freq = config.notch.freq_for(config.notch.default_region)
    raw = notch_filter(raw, notch_freq)
    raw = remove_artifacts_ica(raw, config.ica)
    raw = resample(raw, config.resample.target_sfreq)

    data = raw.get_data()  # [n_channels, n_timepoints]
    if config.normalize.method == "zscore":
        data = normalize_zscore(data)
    else:
        raise ValueError(f"Unsupported normalization method: {config.normalize.method}")

    windows = segment_windows(
        data,
        sfreq=raw.info["sfreq"],
        window_seconds=config.windowing.window_seconds,
        overlap_seconds=config.windowing.overlap_seconds,
    )
    return windows
