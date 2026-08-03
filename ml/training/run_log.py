"""Simple JSON-lines run log (Section 5.6): dataset version, config, seed,
metrics per run. No external tracking service, per the MVP scope's explicit
exclusion of MLflow-style experiment tracking."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RunLog:
    def __init__(self, path: str | Path, run_id: str, seed: int, dataset_version: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.seed = seed
        self.dataset_version = dataset_version

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float | None, **extra) -> None:
        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "dataset_version": self.dataset_version,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **extra,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
