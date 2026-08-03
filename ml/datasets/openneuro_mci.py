"""Ingestion for OpenNeuro ds003490 (Alzheimer's/FTD/healthy-control resting
EEG) — Section 4's third dataset, deliberately different montage/acquisition
protocol from CHB-MIT and the ADHD set, to stress-test the channel-mapping
standardization in ml/preprocessing/pipeline.py.

Expected raw layout (BIDS, as published on OpenNeuro):

    ds003490/
      participants.tsv       # columns include: participant_id, Group, ...
      sub-001/
        eeg/
          sub-001_task-eyesclosed_eeg.edf
      sub-002/
        ...

`Group` in participants.tsv is expected to take values "A" (Alzheimer's),
"F" (frontotemporal dementia), "C" (healthy control). Only A/C subjects are
labeled here (A=1, C=0); F subjects are excluded by default since FTD is a
distinct condition from the Alzheimer's/MCI head this dataset is meant to
train — set `include_ftd_as_negative=True` to fold them in as label 0
instead of dropping them.

Caveat: this dataset's raw EEG recordings may be published in EEGLAB (.set)
rather than .edf. If so, ml/preprocessing/pipeline.py's `load_edf` needs a
one-line extension to `mne.io.read_raw_eeglab` for this dataset's files —
everything downstream (channel mapping, filtering, windowing) is
format-agnostic since it all operates on the mne.io.Raw object either loader
returns. Confirm the actual file extension against the real download before
Week 1-2 ingestion.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ml.datasets.common import preprocess_and_cache, write_manifest
from ml.preprocessing.config import PreprocessingConfig
from ml.training.dataset import ManifestEntry

DATASET_ID = "mci"

_GROUP_LABELS = {"A": 1, "C": 0}


def discover_recordings(
    raw_data_dir: str | Path, include_ftd_as_negative: bool = False
) -> list[tuple[Path, int]]:
    raw_data_dir = Path(raw_data_dir)
    participants_tsv = raw_data_dir / "participants.tsv"
    if not participants_tsv.exists():
        raise FileNotFoundError(f"{participants_tsv} not found; expected BIDS participants.tsv.")

    group_labels = dict(_GROUP_LABELS)
    if include_ftd_as_negative:
        group_labels["F"] = 0

    recordings = []
    with open(participants_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            group = row.get("Group", "").strip()
            if group not in group_labels:
                continue
            subject_id = row["participant_id"]
            eeg_dir = raw_data_dir / subject_id / "eeg"
            edf_candidates = sorted(eeg_dir.glob("*_eeg.edf")) if eeg_dir.exists() else []
            for edf_path in edf_candidates:
                recordings.append((edf_path, group_labels[group]))
    return recordings


def ingest(
    raw_data_dir: str | Path,
    output_dir: str | Path,
    preprocessing_config: PreprocessingConfig,
    manifest_path: str | Path,
    include_ftd_as_negative: bool = False,
) -> list[ManifestEntry]:
    entries = []
    for edf_path, label in discover_recordings(raw_data_dir, include_ftd_as_negative):
        entry = preprocess_and_cache(
            edf_path,
            label=label,
            dataset_id=DATASET_ID,
            output_dir=output_dir,
            preprocessing_config=preprocessing_config,
            recording_id=edf_path.stem,
        )
        if entry:
            entries.append(entry)
    write_manifest(entries, manifest_path)
    return entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest OpenNeuro ds003490 into preprocessed window tensors.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/mci")
    parser.add_argument("--manifest", default="data/manifests/mci.jsonl")
    parser.add_argument("--include-ftd-as-negative", action="store_true")
    args = parser.parse_args()

    config = PreprocessingConfig.from_yaml()
    entries = ingest(
        args.raw_dir, args.output_dir, config, args.manifest, args.include_ftd_as_negative
    )
    print(f"Ingested {len(entries)} recordings -> {args.manifest}")
