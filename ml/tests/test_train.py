import numpy as np

from ml.training.config import TrainingConfig
from ml.training.dataset import ManifestEntry
from ml.training.train import train


def _build_synthetic_manifest(tmp_path, disorders, n_per_disorder=8, seed=0):
    rng = np.random.default_rng(seed)
    manifest = []
    for disorder in disorders:
        for i in range(n_per_disorder):
            label = i % 2
            tensor = (rng.normal(0, 1, (5, 19, 256)) + 2.0 * label).astype(np.float32)
            path = tmp_path / f"{disorder}_{i}.npy"
            np.save(path, tensor)
            manifest.append(
                ManifestEntry(tensor_path=str(path), label=float(label), dataset_id=disorder)
            )
    return manifest


def test_training_loss_decreases(tmp_path, model_config, training_config_factory):
    manifest = _build_synthetic_manifest(tmp_path, model_config.disorders)
    training_config = training_config_factory(tmp_path, epochs=5)

    result = train(model_config, training_config, manifest)
    history = result["history"]

    assert len(history) == 5
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_checkpoints_and_run_log_written(tmp_path, model_config, training_config_factory):
    manifest = _build_synthetic_manifest(tmp_path, model_config.disorders, n_per_disorder=4)
    training_config = training_config_factory(tmp_path, epochs=2)

    train(model_config, training_config, manifest)

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "checkpoint_epoch_001.pt").exists()
    assert (checkpoint_dir / "checkpoint_epoch_002.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()

    run_log_path = tmp_path / "run_log.jsonl"
    lines = run_log_path.read_text().strip().splitlines()
    assert len(lines) == 2
