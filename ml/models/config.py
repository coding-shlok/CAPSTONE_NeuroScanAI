"""Typed access to model_config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"


@dataclass
class BackboneConfig:
    cnn_channels: list[int]
    cnn_kernel_size: int
    lstm_hidden: int
    embedding_dim: int
    dropout: float = 0.0


@dataclass
class HeadsConfig:
    hidden_dim: int
    dropout: float = 0.0


@dataclass
class ModelConfig:
    n_channels: int
    disorders: list[str]
    backbone: BackboneConfig
    heads: HeadsConfig
    native_channels: dict[str, int]

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ModelConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            n_channels=raw["n_channels"],
            disorders=raw["disorders"],
            backbone=BackboneConfig(**raw["backbone"]),
            heads=HeadsConfig(**raw["heads"]),
            native_channels=raw.get("native_channels", {}),
        )
