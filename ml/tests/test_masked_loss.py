import torch

from ml.models.full_model import NeuroScanModel
from ml.training.masked_loss import masked_multi_task_step


def _has_grad(module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in module.parameters())


def test_masked_loss_only_updates_target_head(model_config):
    torch.manual_seed(0)
    model = NeuroScanModel.from_config(model_config)
    x = torch.randn(2, 5, model_config.n_channels, 512)
    labels = torch.tensor([0.0, 1.0])

    loss = masked_multi_task_step(model, x, labels, dataset_id="adhd")
    loss.backward()

    assert _has_grad(model.heads.heads["adhd"])
    for disorder in model_config.disorders:
        if disorder != "adhd":
            assert not _has_grad(model.heads.heads[disorder]), (
                f"{disorder} head must receive no gradient on an 'adhd' batch"
            )
    assert _has_grad(model.backbone), "shared backbone must get gradient on every batch"


def test_masked_loss_is_finite(model_config):
    torch.manual_seed(0)
    model = NeuroScanModel.from_config(model_config)
    x = torch.randn(2, 5, model_config.n_channels, 512)
    labels = torch.tensor([0.0, 1.0])
    loss = masked_multi_task_step(model, x, labels, dataset_id="mci")
    assert torch.isfinite(loss)
