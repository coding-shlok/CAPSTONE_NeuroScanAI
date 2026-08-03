import torch

from ml.models.full_model import NeuroScanModel


def test_forward_all_heads_shapes(model_config):
    model = NeuroScanModel.from_config(model_config)
    x = torch.randn(2, 5, model_config.n_channels, 512)
    out = model(x)
    assert set(out.keys()) == set(model_config.disorders)
    for logits in out.values():
        assert logits.shape == (2,)


def test_forward_single_head_only(model_config):
    model = NeuroScanModel.from_config(model_config)
    x = torch.randn(2, 5, model_config.n_channels, 512)
    out = model(x, dataset_id="epilepsy")
    assert list(out.keys()) == ["epilepsy"]


def test_backbone_output_independent_of_batch_size(model_config):
    """Same single recording should get the same embedding whether it's run
    alone or padded into a larger batch (no cross-sample leakage)."""
    model = NeuroScanModel.from_config(model_config)
    model.eval()
    torch.manual_seed(0)
    x_single = torch.randn(1, 5, model_config.n_channels, 512)

    with torch.no_grad():
        out_single = model(x_single)
        x_batched = torch.cat([x_single, torch.randn(3, 5, model_config.n_channels, 512)], dim=0)
        out_batched = model(x_batched)

    for disorder in model_config.disorders:
        assert torch.allclose(out_single[disorder][0], out_batched[disorder][0], atol=1e-5)
