"""Masked multi-task training loop (Section 5.5/5.6).

    for batch in dataloader:              # batch tagged with source dataset
        embedding = shared_backbone(batch.tensor)
        head_out  = heads[batch.dataset_id](embedding)
        loss      = task_loss(head_out, batch.labels)   # only this head's loss
        loss.backward()                    # shared backbone gets gradient every step
        optimizer.step()

is implemented here as: PerDatasetBatchSampler guarantees single-dataset
batches -> masked_multi_task_step evaluates only that batch's head -> the
other two heads simply aren't part of this step's autograd graph.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from ml.models.config import ModelConfig
from ml.models.full_model import NeuroScanModel
from ml.training.checkpoint import CheckpointManager
from ml.training.config import TrainingConfig
from ml.training.dataset import (
    EEGWindowDataset,
    ManifestEntry,
    PerDatasetBatchSampler,
    collate_single_dataset_batch,
    load_manifest,
)
from ml.training.masked_loss import masked_multi_task_step
from ml.training.run_log import RunLog
from ml.training.seed import set_seed


def build_dataloader(
    manifest: list[ManifestEntry], batch_size: int, seed: int, shuffle: bool
) -> tuple[DataLoader, PerDatasetBatchSampler]:
    dataset = EEGWindowDataset(manifest)
    sampler = PerDatasetBatchSampler(
        dataset.dataset_ids, batch_size=batch_size, shuffle=shuffle, seed=seed
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_single_dataset_batch
    )
    return loader, sampler


def evaluate(model: NeuroScanModel, loader: DataLoader, pos_weights: dict) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for tensor, labels, dataset_id in loader:
            loss = masked_multi_task_step(
                model, tensor, labels, dataset_id, pos_weights.get(dataset_id)
            )
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_manifest: list[ManifestEntry],
    val_manifest: list[ManifestEntry] | None = None,
    run_id: str | None = None,
) -> dict:
    set_seed(training_config.seed)

    model = NeuroScanModel.from_config(model_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.optimizer.learning_rate,
        weight_decay=training_config.optimizer.weight_decay,
    )

    train_loader, train_sampler = build_dataloader(
        train_manifest, training_config.batch_size, training_config.seed, shuffle=True
    )
    val_loader = None
    if val_manifest:
        val_loader, _ = build_dataloader(
            val_manifest, training_config.batch_size, training_config.seed, shuffle=False
        )

    pos_weights = {
        d: torch.tensor(float(w)) for d, w in training_config.loss_weighting.items()
    }

    checkpoint_mgr = CheckpointManager(training_config.checkpointing.checkpoint_dir)
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_log = RunLog(
        training_config.run_log.path,
        run_id=run_id,
        seed=training_config.seed,
        dataset_version=training_config.run_log.dataset_version,
    )

    history = []
    for epoch in range(1, training_config.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for tensor, labels, dataset_id in train_loader:
            optimizer.zero_grad()
            loss = masked_multi_task_step(
                model, tensor, labels, dataset_id, pos_weights.get(dataset_id)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        val_loss = evaluate(model, val_loader, pos_weights) if val_loader else train_loss

        checkpoint_mgr.save_epoch(model, optimizer, epoch, val_loss)
        run_log.log_epoch(epoch, train_loss, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    return {"model": model, "history": history, "run_id": run_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NeuroScan shared backbone.")
    parser.add_argument("--train-manifest", required=True, help="JSONL manifest of training windows.")
    parser.add_argument("--val-manifest", default=None, help="JSONL manifest of validation windows.")
    args = parser.parse_args()

    model_config = ModelConfig.from_yaml()
    training_config = TrainingConfig.from_yaml()
    train_manifest = load_manifest(args.train_manifest)
    val_manifest = load_manifest(args.val_manifest) if args.val_manifest else None

    result = train(model_config, training_config, train_manifest, val_manifest)
    print(f"Run {result['run_id']} finished. Final epoch: {result['history'][-1]}")


if __name__ == "__main__":
    main()
