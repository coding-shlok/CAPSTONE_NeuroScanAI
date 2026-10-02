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
from collections import defaultdict
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
from ml.training.evaluate import compute_disorder_metrics
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


def evaluate(model: NeuroScanModel, loader: DataLoader, pos_weights: dict, dataset_loss_weights: dict | None = None, device: torch.device | None = None) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for tensor, labels, dataset_id in loader:
            if device is not None:
                tensor, labels = tensor.to(device), labels.to(device)
            loss = masked_multi_task_step(
                model, tensor, labels, dataset_id,
                pos_weights.get(dataset_id),
                dataset_loss_weights.get(dataset_id) if dataset_loss_weights else None
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = NeuroScanModel.from_config(model_config).to(device)
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

    # Grouped separately from val_loader (which only feeds the aggregate BCE
    # loss checkpoint.py selects "best" on) so each epoch also reports
    # per-disorder accuracy/F1 — the blended val_loss can improve while one
    # disorder's actual metric quietly gets worse, which is invisible without this.
    val_by_dataset: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in val_manifest or []:
        val_by_dataset[entry.dataset_id].append(entry)

    pos_weights = {
        d: torch.tensor(float(w)).to(device) for d, w in training_config.loss_weighting.items()
    }

    dataset_loss_weights = getattr(training_config, 'inter_dataset_loss_weights', None)
    if dataset_loss_weights:
        dataset_loss_weights = {d: float(w) for d, w in dataset_loss_weights.items()}

    checkpoint_mgr = CheckpointManager(training_config.checkpointing.checkpoint_dir)
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_log = RunLog(
        training_config.run_log.path,
        run_id=run_id,
        seed=training_config.seed,
        dataset_version=training_config.run_log.dataset_version,
    )

    # Checkpoint "best" selection and early stopping both key off this instead
    # of raw blended val_loss. Evidence (2026-09-26 corrected run's
    # run_log.jsonl): aggregate val_loss bounced between 0.52 and 2.06 with no
    # real trend across 16 epochs, so patience-on-val_loss locked onto epoch 6
    # as "best" purely because it was a lucky low point in noise -- while
    # ADHD's and MCI's own val_f1 kept climbing well past epoch 6 (peaks
    # around epochs 9-10 and 13-15) and got discarded by early stopping before
    # ever being tried as the checkpoint. This tracks, per epoch, how close
    # the WORST-performing disorder is to ITS OWN clinical target and
    # selects/stops on that -- so one noisy disorder's loss can no longer
    # throw away a genuinely-better epoch for the other two.
    CLINICAL_TARGETS = {"adhd": ("accuracy", 0.80), "epilepsy": ("f1", 0.60), "mci": ("accuracy", 0.85)}

    def _target_achievement(per_disorder: dict) -> float | None:
        ratios = []
        for did, (metric_name, target) in CLINICAL_TARGETS.items():
            if did not in per_disorder:
                return None
            ratios.append(per_disorder[did][metric_name] / target)
        return min(ratios) if ratios else None  # 1.0 = worst disorder is exactly at its target

    history = []
    best_selection_metric = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, training_config.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for tensor, labels, dataset_id in train_loader:
            tensor, labels = tensor.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = masked_multi_task_step(
                model, tensor, labels, dataset_id,
                pos_weights.get(dataset_id),
                dataset_loss_weights.get(dataset_id) if dataset_loss_weights else None
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        val_loss = evaluate(model, val_loader, pos_weights, dataset_loss_weights, device) if val_loader else train_loss

        per_disorder = {}
        for dataset_id, entries in val_by_dataset.items():
            m = compute_disorder_metrics(model, entries, dataset_id, device)
            per_disorder[dataset_id] = m
            run_log.log_epoch(
                epoch, train_loss, val_loss,
                dataset_id=dataset_id,
                val_accuracy=m["accuracy"], val_f1=m["f1"],
                val_precision=m["precision"], val_recall=m["recall"],
            )
        if not val_by_dataset:
            run_log.log_epoch(epoch, train_loss, val_loss)

        target_achievement = _target_achievement(per_disorder)
        if target_achievement is not None:
            selection_metric, metric_name = -target_achievement, "neg_worst_disorder_target_achievement"
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}  worst-disorder target achievement={target_achievement:.2f} "
                  f"(1.0 = worst disorder exactly at its clinical target)")
        else:
            selection_metric, metric_name = val_loss, "val_loss"
        checkpoint_mgr.save_epoch(model, optimizer, epoch, selection_metric, metric_name=metric_name)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "target_achievement": target_achievement,
            "per_disorder": per_disorder,
        })

        if val_loader:
            if selection_metric < best_selection_metric - 1e-6:
                best_selection_metric = selection_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if training_config.patience and epochs_without_improvement >= training_config.patience:
                print(
                    f"Early stopping at epoch {epoch}: {metric_name} hasn't improved "
                    f"in {training_config.patience} epochs (best={best_selection_metric:.4f})."
                )
                break

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
