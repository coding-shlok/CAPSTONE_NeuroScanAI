import pytest

from ml.models.config import ModelConfig
from ml.preprocessing.config import PreprocessingConfig
from ml.tests.synthetic_edf import write_synthetic_edf
from ml.training.config import TrainingConfig


@pytest.fixture(scope="session")
def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig.from_yaml()


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    return ModelConfig.from_yaml()


@pytest.fixture(scope="session")
def synthetic_edf_path(tmp_path_factory) -> str:
    out_dir = tmp_path_factory.mktemp("synthetic_edf")
    path = write_synthetic_edf(out_dir / "sample.edf", duration_seconds=30.0, sfreq=256.0)
    return str(path)


@pytest.fixture
def training_config_factory():
    """Builds a TrainingConfig pointed at a tmp_path, so each test's
    checkpoints/run log land in an isolated, auto-cleaned directory instead
    of the real ml/checkpoints/."""

    def _factory(tmp_path, epochs: int = 3) -> TrainingConfig:
        config = TrainingConfig.from_yaml()
        config.epochs = epochs
        config.checkpointing.checkpoint_dir = str(tmp_path / "checkpoints")
        config.run_log.path = str(tmp_path / "run_log.jsonl")
        return config

    return _factory
