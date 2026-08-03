"""Ingestion for the ADHD EEG dataset (Section 4).

IMPORTANT — naming caveat: the MVP scope document names this dataset
"ADHD-200". The actual ADHD-200 dataset (NITRC, fcon_1000.projects.nitrc.org)
is resting-state fMRI/sMRI, not EEG — it has no .edf recordings, so it
cannot be the literal source here. There are separate public EEG/ADHD
datasets (e.g. the IEEE Dataport "EEG data for ADHD/Control children" set),
but which one is actually intended needs to be confirmed before Week 1-2
acquisition starts; don't assume this module's folder layout matches
whatever gets downloaded without checking first.

Because the exact source is unconfirmed, this ingestion script is written
against a dataset-agnostic convention rather than a guessed real folder
layout, so it works once you point it at *some* labeled EEG/ADHD corpus:

    <raw_data_dir>/
      labels.csv          # columns: filename,label  (label: 1=ADHD, 0=control)
      <filename1>.edf
      <filename2>.edf
      ...

Update `discover_recordings` if the real dataset you settle on organizes
files differently (e.g. separate ADHD/ and Control/ subfolders).
"""
from __future__ import annotations

import csv
from pathlib import Path

from ml.datasets.common import preprocess_and_cache, write_manifest
from ml.preprocessing.config import PreprocessingConfig
from ml.training.dataset import ManifestEntry

DATASET_ID = "adhd"


def discover_recordings(raw_data_dir: str | Path) -> list[tuple[Path, int]]:
    raw_data_dir = Path(raw_data_dir)
    labels_csv = raw_data_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} not found. This ingestion script expects a "
            "labels.csv (columns: filename,label) alongside the .edf files — "
            "see the module docstring for why, and adjust to match "
            "whichever real EEG/ADHD dataset is confirmed."
        )

    recordings = []
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            edf_path = raw_data_dir / row["filename"]
            recordings.append((edf_path, int(row["label"])))
    return recordings


def ingest(
    raw_data_dir: str | Path,
    output_dir: str | Path,
    preprocessing_config: PreprocessingConfig,
    manifest_path: str | Path,
) -> list[ManifestEntry]:
    entries = []
    for edf_path, label in discover_recordings(raw_data_dir):
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

    parser = argparse.ArgumentParser(description="Ingest ADHD EEG data into preprocessed window tensors.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/adhd")
    parser.add_argument("--manifest", default="data/manifests/adhd.jsonl")
    args = parser.parse_args()

    config = PreprocessingConfig.from_yaml()
    entries = ingest(args.raw_dir, args.output_dir, config, args.manifest)
    print(f"Ingested {len(entries)} recordings -> {args.manifest}")
