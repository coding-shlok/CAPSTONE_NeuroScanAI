"""Dataset wrapper + batch sampler enforcing the Section 5.5 constraint that
"each training batch is drawn from exactly one source dataset."

`EEGWindowDataset` is deliberately generic: it loads pre-processed window
tensors from a manifest (list of {tensor_path, label, dataset_id}) rather
than knowing anything about CHB-MIT/ADHD-200/OpenNeuro specifically — the
per-dataset ingestion scripts in ml/datasets/ are responsible for producing
that manifest from raw EDF files. This lets the training loop be exercised
today against synthetic manifests, and point at real data later without
any code change here.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class ManifestEntry:
    tensor_path: str
    label: float
    dataset_id: str


class EEGWindowDataset(Dataset):
    """Each item is one recording's preprocessed window tensor
    [n_windows, n_channels, n_timepoints], saved on disk as .npy, plus its
    binary label and source dataset id."""

    def __init__(self, manifest: list[ManifestEntry]):
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        entry = self.manifest[idx]
        tensor = np.load(entry.tensor_path)
        return (
            torch.from_numpy(tensor).float(),
            torch.tensor(entry.label, dtype=torch.float32),
            entry.dataset_id,
        )

    @property
    def dataset_ids(self) -> list[str]:
        return [e.dataset_id for e in self.manifest]


class PerDatasetBatchSampler(Sampler[list[int]]):
    """Yields batches of indices that all share the same dataset_id, and
    shuffles the order of those single-dataset batches across an epoch —
    this is what makes "each batch is drawn from exactly one source
    dataset" true by construction rather than by convention."""

    def __init__(
        self, dataset_ids: list[str], batch_size: int, shuffle: bool = True, seed: int = 42
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.groups: dict[str, list[int]] = defaultdict(list)
        for idx, did in enumerate(dataset_ids):
            self.groups[did].append(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        per_dataset_batches: dict[str, list[list[int]]] = {}
        for did, indices in self.groups.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            per_dataset_batches[did] = [
                indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)
            ]

        # Datasets differ wildly in row count AND in what a row represents
        # (ADHD/MCI: one row per whole recording; epilepsy: one row per
        # window) so raw batch counts aren't comparable across datasets —
        # slicing by batch_size alone let epilepsy outnumber ADHD/MCI by
        # 15-30x. Equalize to the largest dataset's natural batch count by
        # cycling (with reshuffling each lap) so every dataset contributes
        # the same number of gradient updates to the shared backbone.
        target = max((len(b) for b in per_dataset_batches.values()), default=0)

        all_batches: list[list[int]] = []
        for batches in per_dataset_batches.values():
            if not batches:
                continue
            cycled: list[list[int]] = []
            while len(cycled) < target:
                lap = list(batches)
                if self.shuffle:
                    rng.shuffle(lap)
                cycled.extend(lap)
            all_batches.extend(cycled[:target])

        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        if not self.groups:
            return 0
        per_dataset_counts = [
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        ]
        target = max(per_dataset_counts)
        return target * len(self.groups)


def collate_single_dataset_batch(batch):
    """All items in `batch` come from one PerDatasetBatchSampler batch, so
    they share a dataset_id by construction; this just asserts that
    invariant instead of trusting it silently."""
    tensors, labels, dataset_ids = zip(*batch)
    dataset_id = dataset_ids[0]
    if any(d != dataset_id for d in dataset_ids):
        raise ValueError(
            f"Batch mixes dataset ids {set(dataset_ids)}; the masked "
            "multi-task loss requires single-dataset batches."
        )
    return torch.stack(tensors), torch.stack(labels), dataset_id


def load_manifest(manifest_path: str | Path) -> list[ManifestEntry]:
    """Manifest is a JSONL file: one {"tensor_path", "label", "dataset_id"} per line."""
    import json

    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entries.append(ManifestEntry(**row))
    return entries
