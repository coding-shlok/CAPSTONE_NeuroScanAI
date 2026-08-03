"""Checkpoint saving (Section 5.6): save after every epoch, keep best-by-
validation-loss. Deliberately just torch.save/load + a JSON "best" marker —
no experiment-tracking service, per the MVP scope's explicit exclusion of
MLflow-style tracking."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric_path = self.checkpoint_dir / "best_metric.json"

    def save_epoch(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float
    ) -> Path:
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

        best = self._read_best()
        if best is None or val_loss < best["val_loss"]:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            self.best_metric_path.write_text(
                json.dumps({"epoch": epoch, "val_loss": val_loss})
            )
        return path

    def _read_best(self) -> dict | None:
        if not self.best_metric_path.exists():
            return None
        return json.loads(self.best_metric_path.read_text())

    def load_best(self, model: nn.Module, map_location: str = "cpu") -> dict:
        checkpoint = torch.load(self.checkpoint_dir / "best.pt", map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
