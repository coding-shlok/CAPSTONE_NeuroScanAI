"""App-wide settings, loaded from environment variables (with sane local
defaults) rather than hard-coded — Section 0.1's "config over hard-coding"
applied to the backend layer."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEUROSCAN_")

    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'neuroscan.db'}"
    storage_dir: Path = REPO_ROOT / "data" / "storage"
    heatmap_dir: Path = REPO_ROOT / "data" / "storage" / "heatmaps"
    model_checkpoint_path: Path = REPO_ROOT / "ml" / "checkpoints" / "best.pt"
    model_version: str = "dev"
    secret_key: str = "dev-only-change-me"


settings = Settings()
