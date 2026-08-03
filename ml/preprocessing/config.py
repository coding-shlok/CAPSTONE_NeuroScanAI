"""Typed access to preprocessing_config.yaml.

Keeping this as a small dataclass (rather than passing raw dicts around)
means every stage in pipeline.py gets IDE-checked field names instead of
string-keyed lookups scattered through the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml"


@dataclass
class BandpassConfig:
    low_freq: float
    high_freq: float


@dataclass
class NotchConfig:
    freq_by_region: dict
    default_region: str

    def freq_for(self, region: str | None) -> float:
        region = region or self.default_region
        return self.freq_by_region[region]


@dataclass
class IcaConfig:
    n_components: int
    method: str
    random_state: int
    eog_corr_threshold: float


@dataclass
class ResampleConfig:
    target_sfreq: float


@dataclass
class NormalizeConfig:
    method: str


@dataclass
class WindowingConfig:
    window_seconds: float
    overlap_seconds: float


@dataclass
class PreprocessingConfig:
    common_channels: list[str]
    bandpass: BandpassConfig
    notch: NotchConfig
    ica: IcaConfig
    resample: ResampleConfig
    normalize: NormalizeConfig
    windowing: WindowingConfig

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "PreprocessingConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            common_channels=raw["common_channels"],
            bandpass=BandpassConfig(**raw["bandpass"]),
            notch=NotchConfig(**raw["notch"]),
            ica=IcaConfig(**raw["ica"]),
            resample=ResampleConfig(**raw["resample"]),
            normalize=NormalizeConfig(**raw["normalize"]),
            windowing=WindowingConfig(**raw["windowing"]),
        )
