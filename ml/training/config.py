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
    # Early-stopping patience, in epochs without improvement on train.py's
    # composite worst-disorder target-achievement metric. Default kept here
    # (not just in the yaml) so older configs without this key still work.
    patience: int = 20
    # Optional per-dataset scaling of each batch's loss before backward(),
    # on top of loss_weighting's per-sample pos_weight -- lets one disorder's
    # gradient contribution to the shared backbone be tuned independently of
    # its own BCE class balance. None/absent = no extra scaling (1.0 for all).
    inter_dataset_loss_weights: dict | None = None

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
            patience=raw.get("patience", 20),
            inter_dataset_loss_weights=raw.get("inter_dataset_loss_weights"),
        )
