"""Computes the positive-class ratio in a manifest, so
training_config.yaml's loss_weighting (BCEWithLogitsLoss's pos_weight) can
be set from the real label distribution instead of guessed.

Context: epilepsy's validation confusion matrix (TP=23 FP=3 TN=887 FN=66)
has precision=0.88 but recall=0.26 — the model is heavily biased toward
predicting "no seizure," which is exactly what unweighted BCE does when
positives are rare. pos_weight = n_negative / n_positive in the *training*
manifest (not validation) directly counteracts that during training, unlike
threshold tuning (ml/training/evaluate.py --sweep-thresholds) which only
adjusts an already-trained model's decision boundary after the fact. Use
both: this for the next training run, sweep-thresholds for the checkpoint
you already have.
"""
from __future__ import annotations

from ml.training.dataset import load_manifest


def pos_weight_for_manifest(manifest_path: str) -> dict:
    manifest = load_manifest(manifest_path)
    n_pos = sum(1 for e in manifest if e.label == 1.0)
    n_neg = sum(1 for e in manifest if e.label == 0.0)
    n = len(manifest)
    pos_weight = (n_neg / n_pos) if n_pos else float("inf")
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": n_pos / n if n else 0.0,
        "recommended_pos_weight": pos_weight,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recommend a BCEWithLogitsLoss pos_weight from a manifest's real label balance."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", help="Label only, for the printed message.", default=None)
    args = parser.parse_args()

    stats = pos_weight_for_manifest(args.manifest)
    label = args.dataset_id or args.manifest
    print(f"{label}: n={stats['n']}  positives={stats['n_pos']} ({stats['pos_rate']:.1%})  negatives={stats['n_neg']}")
    print(f"Recommended loss_weighting entry: {stats['recommended_pos_weight']:.2f}")
    print(
        "(Set this in ml/configs/training_config.yaml's loss_weighting section "
        "for this dataset_id, then retrain — pos_weight only affects the loss "
        "function during training, not an already-trained checkpoint.)"
    )
