"""Ingestion for CHB-MIT Scalp EEG Database (epilepsy) — Section 4, the
dataset to build and validate the pipeline against FIRST.

Expected raw layout (as published on PhysioNet):

    CHB-MIT/
      chb01/
        chb01_01.edf
        chb01_02.edf
        ...
        chb01-summary.txt
      chb02/
        ...

`chbNN-summary.txt` lists, per file, the number of seizures and their
start/end times in seconds, e.g.:

    File Name: chb01_03.edf
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

A recording is labeled positive (1) if its summary entry reports at least
one seizure, negative (0) otherwise. This is a recording-level label —
matching the pipeline's one-tensor-per-recording granularity — not a
per-window seizure-onset label; if per-window labeling turns out to matter
for training quality, that requires cutting the summary's second-resolution
intervals against the window boundaries from ml/preprocessing/pipeline.py's
segment_windows, which is a Week 1-2 follow-up, not part of this stub.
"""
from __future__ import annotations

import re
from pathlib import Path

from ml.datasets.common import preprocess_and_cache, write_manifest
from ml.preprocessing.config import PreprocessingConfig
from ml.training.dataset import ManifestEntry

DATASET_ID = "epilepsy"


def _parse_summary(summary_path: Path) -> dict[str, bool]:
    """Returns {edf_filename: has_seizure} for every file listed in a
    chbNN-summary.txt."""
    labels: dict[str, bool] = {}
    current_file = None
    text = summary_path.read_text(errors="ignore")
    for line in text.splitlines():
        file_match = re.match(r"File Name:\s*(\S+)", line)
        if file_match:
            current_file = file_match.group(1)
            labels[current_file] = False
            continue
        seizure_match = re.match(r"Number of Seizures in File:\s*(\d+)", line)
        if seizure_match and current_file is not None:
            labels[current_file] = int(seizure_match.group(1)) > 0
    return labels


def discover_recordings(raw_data_dir: str | Path) -> list[tuple[Path, int]]:
    """Returns [(edf_path, label)] across all subject folders."""
    raw_data_dir = Path(raw_data_dir)
    recordings = []
    for subject_dir in sorted(raw_data_dir.glob("chb*")):
        if not subject_dir.is_dir():
            continue
        summary_files = list(subject_dir.glob("*-summary.txt"))
        if not summary_files:
            continue
        labels = _parse_summary(summary_files[0])
        for edf_path in sorted(subject_dir.glob("*.edf")):
            label = int(labels.get(edf_path.name, False))
            recordings.append((edf_path, label))
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

    parser = argparse.ArgumentParser(description="Ingest CHB-MIT into preprocessed window tensors.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/chb_mit")
    parser.add_argument("--manifest", default="data/manifests/chb_mit.jsonl")
    args = parser.parse_args()

    config = PreprocessingConfig.from_yaml()
    entries = ingest(args.raw_dir, args.output_dir, config, args.manifest)
    print(f"Ingested {len(entries)} recordings -> {args.manifest}")
