"""Classification metrics for a trained checkpoint against a manifest.

train.py only tracks BCE loss (Section 5.6's checkpointing metric). This
adds the accuracy/precision/recall/F1 numbers needed to actually report how
well a single-task model is doing, evaluated per-recording (one prediction
per manifest row) rather than per-window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml.models.config import ModelConfig
from ml.models.full_model import NeuroScanModel
from ml.training.dataset import ManifestEntry, load_manifest


def _predict_probs(
    model: nn.Module,
    manifest: list[ManifestEntry],
    dataset_id: str,
    device: torch.device,
) -> list[tuple[float, float]]:
    """One forward pass per manifest row. Returns (predicted_probability,
    true_label) pairs so a threshold can be applied afterward without
    re-running inference — sweeping many thresholds is then just re-scoring
    the same predictions, not re-running the model.
    """
    was_training = model.training
    model.eval()
    pairs = []
    with torch.no_grad():
        for entry in manifest:
            tensor = torch.from_numpy(np.load(entry.tensor_path)).float().unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor, dataset_id=dataset_id)[dataset_id]).item()
            pairs.append((prob, entry.label))
    if was_training:
        model.train()
    return pairs


def _metrics_at_threshold(pairs: list[tuple[float, float]], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for prob, label in pairs:
        pred = 1.0 if prob >= threshold else 0.0
        if label == 1.0 and pred == 1.0:
            tp += 1
        elif label == 0.0 and pred == 1.0:
            fp += 1
        elif label == 0.0 and pred == 0.0:
            tn += 1
        elif label == 1.0 and pred == 0.0:
            fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "n": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_disorder_metrics(
    model: nn.Module,
    manifest: list[ManifestEntry],
    dataset_id: str,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Metrics for an already-loaded, in-memory model — used both by
    evaluate_checkpoint (loads from disk first) and by train.py to log
    per-disorder val metrics every epoch without touching disk at all."""
    pairs = _predict_probs(model, manifest, dataset_id, device)
    return _metrics_at_threshold(pairs, threshold)


def sweep_thresholds(
    model: nn.Module,
    manifest: list[ManifestEntry],
    dataset_id: str,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    """Scores one set of predictions at many thresholds and returns the
    F1-best one plus the full table. Ties broken by higher recall, since a
    missed seizure/positive case is the costlier error in this project's
    stated targets (Epilepsy F1>=0.60 was failing on recall=0.26, not
    precision).
    """
    thresholds = thresholds if thresholds is not None else [i / 100 for i in range(5, 100, 5)]
    pairs = _predict_probs(model, manifest, dataset_id, device)
    table = [_metrics_at_threshold(pairs, t) for t in thresholds]
    best = max(table, key=lambda m: (m["f1"], m["recall"]))
    return best, table


def evaluate_checkpoint(
    checkpoint_path: str,
    manifest: list[ManifestEntry],
    dataset_id: str,
    model_config: ModelConfig | None = None,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> dict:
    model_config = model_config or ModelConfig.from_yaml()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroScanModel.from_config(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # strict=False for backward compatibility with checkpoints trained before
    # the per-dataset InputAdapter (ml/models/full_model.py) existed.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return compute_disorder_metrics(model, manifest, dataset_id, device, threshold=threshold)


def find_best_threshold(
    checkpoint_path: str,
    manifest: list[ManifestEntry],
    dataset_id: str,
    model_config: ModelConfig | None = None,
    device: torch.device | None = None,
    thresholds: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    """Same checkpoint, same predictions, scored at every threshold in
    `thresholds` — use this to pick the deployed decision threshold instead
    of assuming 0.5, which is only ever correct by coincidence on an
    imbalanced disorder like epilepsy (9% positive rate in this project's
    CHB-MIT validation split)."""
    model_config = model_config or ModelConfig.from_yaml()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroScanModel.from_config(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return sweep_thresholds(model, manifest, dataset_id, device, thresholds=thresholds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint's classification metrics.")
    parser.add_argument("--checkpoint", default="ml/checkpoints/best.pt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Ignore --threshold; score the same checkpoint at thresholds 0.05..0.95 "
        "and print the full table plus the F1-best one. No retraining involved — "
        "this only changes where the decision boundary is drawn on outputs the "
        "model already produces.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    if args.sweep_thresholds:
        best, table = find_best_threshold(args.checkpoint, manifest, args.dataset_id)
        print(f"{'thr':>5}  {'acc':>6}  {'prec':>5}  {'rec':>5}  {'f1':>5}   TP/FP/TN/FN")
        for m in table:
            marker = "  <-- best F1" if m is best else ""
            print(
                f"{m['threshold']:.2f}  {m['accuracy']:.1%}  {m['precision']:.2f}  "
                f"{m['recall']:.2f}  {m['f1']:.2f}   {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}{marker}"
            )
        print(
            f"\nBest threshold={best['threshold']:.2f}: accuracy={best['accuracy']:.1%} "
            f"precision={best['precision']:.2f} recall={best['recall']:.2f} f1={best['f1']:.2f}"
        )
    else:
        metrics = evaluate_checkpoint(args.checkpoint, manifest, args.dataset_id, threshold=args.threshold)
        print(f"n={metrics['n']}  accuracy={metrics['accuracy']:.1%}  precision={metrics['precision']:.2f}  "
              f"recall={metrics['recall']:.2f}  f1={metrics['f1']:.2f}")
        print(f"Confusion: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
