import numpy as np
import pytest

from ml.preprocessing.pipeline import (
    bandpass_filter,
    load_edf,
    normalize_zscore,
    notch_filter,
    remove_artifacts_ica,
    resample,
    run_pipeline,
    segment_windows,
    validate_and_map_channels,
)


def test_load_edf(synthetic_edf_path):
    raw = load_edf(synthetic_edf_path)
    assert raw.n_times > 0
    assert len(raw.ch_names) == 8  # SYNTHETIC_CHANNELS default count


def test_validate_and_map_channels_fills_missing(synthetic_edf_path, preprocessing_config):
    raw = load_edf(synthetic_edf_path)
    mapped = validate_and_map_channels(raw, preprocessing_config.common_channels)
    assert mapped.ch_names == preprocessing_config.common_channels

    data = mapped.get_data()
    live = {"FP1", "FP2", "F3", "F4", "C3", "C4", "O1", "O2"}
    for ch_name, signal in zip(mapped.ch_names, data):
        if ch_name.upper() in live:
            assert signal.std() > 0
        else:
            assert signal.std() == 0


def test_bandpass_and_notch_filters_run(synthetic_edf_path, preprocessing_config):
    raw = load_edf(synthetic_edf_path)
    raw = validate_and_map_channels(raw, preprocessing_config.common_channels)
    raw = bandpass_filter(raw, preprocessing_config.bandpass.low_freq, preprocessing_config.bandpass.high_freq)
    raw = notch_filter(raw, 60.0)
    assert raw.get_data().shape[0] == len(preprocessing_config.common_channels)


def test_ica_skips_zero_variance_channels(synthetic_edf_path, preprocessing_config):
    raw = load_edf(synthetic_edf_path)
    raw = validate_and_map_channels(raw, preprocessing_config.common_channels)
    raw = bandpass_filter(raw, preprocessing_config.bandpass.low_freq, preprocessing_config.bandpass.high_freq)
    cleaned = remove_artifacts_ica(raw, preprocessing_config.ica)
    data = cleaned.get_data()
    live = {"FP1", "FP2", "F3", "F4", "C3", "C4", "O1", "O2"}
    for ch_name, signal in zip(cleaned.ch_names, data):
        if ch_name.upper() not in live:
            assert signal.std() == 0, "flat channels must stay untouched by ICA"


def test_resample_changes_sfreq(synthetic_edf_path):
    raw = load_edf(synthetic_edf_path)
    assert raw.info["sfreq"] == 256.0
    resampled = resample(raw, 128.0)
    assert resampled.info["sfreq"] == 128.0


def test_normalize_zscore():
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 10.0, 10.0, 10.0, 10.0]])
    normalized = normalize_zscore(data)
    assert np.isclose(normalized[0].mean(), 0.0, atol=1e-8)
    assert np.isclose(normalized[0].std(), 1.0, atol=1e-8)
    # constant channel: std guarded against divide-by-zero, stays at zero
    assert np.allclose(normalized[1], 0.0)


def test_segment_windows_shape_and_drop_remainder():
    data = np.zeros((4, 1000))  # 4 channels, 1000 timepoints
    windows = segment_windows(data, sfreq=100.0, window_seconds=2.0)  # 200 samples/window
    assert windows.shape == (5, 4, 200)  # 1000 // 200 = 5, no remainder


def test_segment_windows_too_short_raises():
    data = np.zeros((4, 10))
    with pytest.raises(ValueError):
        segment_windows(data, sfreq=100.0, window_seconds=2.0)


def test_run_pipeline_end_to_end_shape(synthetic_edf_path, preprocessing_config):
    windows = run_pipeline(synthetic_edf_path, preprocessing_config)
    n_channels = len(preprocessing_config.common_channels)
    expected_window_samples = int(
        preprocessing_config.windowing.window_seconds * preprocessing_config.resample.target_sfreq
    )
    assert windows.ndim == 3
    assert windows.shape[1] == n_channels
    assert windows.shape[2] == expected_window_samples
    assert windows.shape[0] > 0
