# NeuroScan AI

EEG-based multi-disorder neurological risk classification pipeline. A single
shared deep-learning backbone ingests raw EEG recordings from three
unrelated public datasets (epilepsy, ADHD, Alzheimer's/MCI) and produces
disorder-specific, explainable risk predictions — end-to-end, from a raw
`.edf` file to a clinician-readable report.

This repository implements the MVP scope defined in
`NeuroScan_AI_MVP_Scope.docx`. The pipeline (`ml/`) is the project's
selling point and was built first, per that document's Section 0.1; the
backend/frontend/docker scaffolding around it exists to make the pipeline
demonstrable, not as ends in themselves.

## Status

- **`ml/`** — fully implemented and tested: preprocessing (MNE), shared
  CNN+BiLSTM backbone, masked multi-task heads, Grad-CAM explainability,
  report builder. Validated end-to-end on a synthetic EEG file (see
  "Datasets" below) — not yet trained on real data.
- **`backend/`** — working FastAPI app (5 endpoints, SQLite, calls into
  `ml/`), tested end-to-end. Auth is a minimal single-login gate, no RBAC.
- **`frontend/`** — not yet built (Week 6-7 per the build order); folder
  structure only.
- **`docker/`** — compose topology recorded, Dockerfiles not yet written
  (Week 7-8).

## Repository structure

```
backend/
  app/
    api/            # thin routes: auth, eeg
    services/       # upload + prediction orchestration (calls ml/)
    models/         # SQLAlchemy tables (users, recordings, predictions)
    schemas/        # Pydantic request/response models
  tests/
ml/
  configs/           # preprocessing_config.yaml, model_config.yaml, training_config.yaml
  preprocessing/      # MNE pipeline stages, config-driven
  models/             # shared backbone (CNN+BiLSTM) + multi-task heads
  training/            # masked multi-task loss, training loop, checkpointing
  explainability/       # Grad-CAM
  reporting/             # plain-language report builder
  datasets/               # per-dataset ingestion scripts (CHB-MIT, ADHD, OpenNeuro)
  checkpoints/             # trained model checkpoints (gitignored)
  pipeline.py              # top-level orchestrator: EDF -> report
  tests/                    # pytest suite + synthetic EDF generator
frontend/                    # not yet built — see frontend/README.md
docker/                       # docker-compose.yml (topology only, for now)
data/                          # local datasets + runtime storage (gitignored)
```

## Setup

```
pip install -r requirements.txt
```

## Running the tests

```
pytest
```

27 tests cover every preprocessing stage, model shapes, the masked
multi-task loss's gradient-masking behavior, Grad-CAM output, the report
builder, the training loop (loss convergence + checkpointing), and a full
`.edf` -> report end-to-end smoke test. 7 more cover the backend API
end-to-end (login, upload, predict, report) via `fastapi.testclient`.

## Running the pipeline directly

```python
from ml.models.config import ModelConfig
from ml.models.full_model import NeuroScanModel
from ml.preprocessing.config import PreprocessingConfig
from ml.pipeline import run_full_pipeline

model = NeuroScanModel.from_config(ModelConfig.from_yaml())  # untrained until Section 5.6 runs
result = run_full_pipeline(
    "path/to/recording.edf",
    model,
    PreprocessingConfig.from_yaml(),
    output_dir="data/storage/heatmaps",
)
print(result.report_text)
```

## Running the backend

```
uvicorn backend.app.main:app --reload
```

Docs at `http://localhost:8000/docs`.

## Datasets

The three real datasets (Section 4 — CHB-MIT, ADHD EEG, OpenNeuro ds003490)
are not included in this repository (`data/` is gitignored) and haven't
been used to validate this build — see `data/README.md` for the expected
layout and `ml/datasets/*.py` for the ingestion scripts. Everything in
`ml/` has instead been validated against a synthetic EEG file
(`ml/tests/synthetic_edf.py`), which exercises every code path — correct
tensor shapes, ICA on real vs. zero-filled channels, masked-loss gradient
routing, Grad-CAM — without asserting anything about clinical validity,
which requires real, labeled data.

**Before Week 1-2 dataset acquisition starts**, note the caveat documented
in `ml/datasets/adhd_eeg.py`: the MVP scope names this dataset "ADHD-200,"
but the real ADHD-200 (NITRC) is fMRI/sMRI, not EEG, so it can't be the
literal source — confirm which actual EEG/ADHD dataset is intended before
downloading anything.

## Training

```
python -m ml.training.train --train-manifest data/manifests/train.jsonl --val-manifest data/manifests/val.jsonl
```

Hyperparameters live in `ml/configs/training_config.yaml` — nothing is
hard-coded in `train.py`. Checkpoints save after every epoch to
`ml/checkpoints/`, with `best.pt` tracking the lowest validation loss; a
JSON-lines run log is written alongside them. Per Section 5.6, train each
dataset single-task first (point the manifest at one `dataset_id`) as a
baseline before combining all three via the masked multi-task loss.
