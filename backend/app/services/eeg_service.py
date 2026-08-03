"""Upload + prediction orchestration (Section 7). Routes stay thin; this is
where the actual work happens — direct SQLAlchemy queries are fine at this
scale, no repository abstraction layer.

This is also the one place the backend crosses the ML boundary: it calls
ml.pipeline.run_full_pipeline with a path and gets a result object back.
Per Section 3.1, the ML module never talks to the database itself — only
this service does, after the pipeline returns.
"""
from __future__ import annotations

import logging
import uuid

import torch
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.tables import Prediction, Recording, RecordingStatus
from ml.models.config import ModelConfig
from ml.models.full_model import NeuroScanModel
from ml.pipeline import run_full_pipeline
from ml.preprocessing.config import PreprocessingConfig

logger = logging.getLogger(__name__)

_model: NeuroScanModel | None = None


def _load_model() -> NeuroScanModel:
    global _model
    if _model is not None:
        return _model

    model = NeuroScanModel.from_config(ModelConfig.from_yaml())
    if settings.model_checkpoint_path.exists():
        checkpoint = torch.load(settings.model_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning(
            "No trained checkpoint at %s; serving predictions from an "
            "untrained model until training (Section 5.6) produces "
            "ml/checkpoints/best.pt.",
            settings.model_checkpoint_path,
        )
    model.eval()
    _model = model
    return model


def save_upload(db: Session, user_id: str, filename: str, file_bytes: bytes) -> Recording:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    recording_id = str(uuid.uuid4())
    storage_path = settings.storage_dir / f"{recording_id}.edf"
    storage_path.write_bytes(file_bytes)

    recording = Recording(
        id=recording_id,
        user_id=user_id,
        original_filename=filename,
        storage_path=str(storage_path),
        status=RecordingStatus.UPLOADED,
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return recording


def get_recording(db: Session, recording_id: str) -> Recording | None:
    return db.get(Recording, recording_id)


def get_prediction(db: Session, recording_id: str) -> Prediction | None:
    return db.query(Prediction).filter(Prediction.recording_id == recording_id).first()


def run_prediction(db: Session, recording: Recording) -> Prediction:
    """Synchronous, blocking pipeline run (Section 7: "/predict blocks until
    the report is ready" — no job queue for MVP)."""
    recording.status = RecordingStatus.PREPROCESSING
    db.commit()

    preprocessing_config = PreprocessingConfig.from_yaml()
    model = _load_model()

    recording.status = RecordingStatus.PREDICTING
    db.commit()

    try:
        result = run_full_pipeline(
            recording.storage_path,
            model,
            preprocessing_config,
            output_dir=settings.heatmap_dir,
            model_version=settings.model_version,
            recording_id=recording.id,
        )
    except Exception:
        recording.status = RecordingStatus.FAILED
        db.commit()
        raise

    prediction = Prediction(
        recording_id=recording.id,
        epilepsy_risk=result.risk_scores["epilepsy"],
        mci_risk=result.risk_scores["mci"],
        adhd_risk=result.risk_scores["adhd"],
        confidence=result.confidence,
        heatmap_path=result.heatmap_path,
        report_text=result.report_text,
        model_version=result.model_version,
    )
    db.add(prediction)
    recording.status = RecordingStatus.DONE
    db.commit()
    db.refresh(prediction)
    return prediction
