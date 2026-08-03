"""Typed access to training_config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "training_config.yaml"


@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float


@dataclass
class CheckpointingConfig:
    checkpoint_dir: str
    keep_best_only_metric: str


@dataclass
class RunLogConfig:
    path: str
    dataset_version: str


@dataclass
class TrainingConfig:
    seed: int
    optimizer: OptimizerConfig
    batch_size: int
    epochs: int
    loss_weighting: dict
    checkpointing: CheckpointingConfig
    run_log: RunLogConfig

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "TrainingConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            seed=raw["seed"],
            optimizer=OptimizerConfig(**raw["optimizer"]),
            batch_size=raw["batch_size"],
            epochs=raw["epochs"],
            loss_weighting=raw["loss_weighting"],
            checkpointing=CheckpointingConfig(**raw["checkpointing"]),
            run_log=RunLogConfig(**raw["run_log"]),
        )
