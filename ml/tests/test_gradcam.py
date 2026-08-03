import numpy as np
import torch

from ml.explainability.gradcam import GradCAM
from ml.models.full_model import NeuroScanModel


def test_gradcam_output_shapes(model_config):
    torch.manual_seed(0)
    model = NeuroScanModel.from_config(model_config)
    model.eval()

    n_windows, n_timepoints = 5, 512
    x = torch.randn(1, n_windows, model_config.n_channels, n_timepoints)
    channel_names = [f"CH{i}" for i in range(model_config.n_channels)]

    with GradCAM(model) as cam:
        result = cam.explain(x, dataset_id="epilepsy", channel_names=channel_names)

    assert result.temporal_heatmap.shape == (n_windows * n_timepoints,)
    assert 0.0 <= result.predicted_probability <= 1.0
    assert set(result.channel_importance.keys()) == set(channel_names)
    assert np.isclose(sum(result.channel_importance.values()), 1.0, atol=1e-4)
    assert len(result.top_channels) == 3
    assert all(c in channel_names for c in result.top_channels)


def test_gradcam_rejects_batch_greater_than_one(model_config):
    model = NeuroScanModel.from_config(model_config)
    x = torch.randn(2, 5, model_config.n_channels, 512)
    channel_names = [f"CH{i}" for i in range(model_config.n_channels)]

    with GradCAM(model) as cam:
        try:
            cam.explain(x, dataset_id="epilepsy", channel_names=channel_names)
            assert False, "expected ValueError for batch size > 1"
        except ValueError:
            pass


def test_gradcam_handles_zero_signal_channel(model_config):
    """A zero-filled channel (e.g. from preprocessing's missing-channel
    fill) must get zero importance, not NaN."""
    torch.manual_seed(0)
    model = NeuroScanModel.from_config(model_config)
    model.eval()

    n_windows, n_timepoints = 5, 512
    x = torch.randn(1, n_windows, model_config.n_channels, n_timepoints)
    x[:, :, 0, :] = 0.0  # zero out first channel entirely
    channel_names = [f"CH{i}" for i in range(model_config.n_channels)]

    with GradCAM(model) as cam:
        result = cam.explain(x, dataset_id="epilepsy", channel_names=channel_names)

    assert result.channel_importance["CH0"] == 0.0
