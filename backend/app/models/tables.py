"""3 tables (Section 8) — heatmap path and report text live directly on the
prediction row; no separate audit_logs/model_registry/explainability_artifacts
tables, per the MVP scope's explicit exclusion of those."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RecordingStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    PREDICTING = "predicting"
    DONE = "done"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    recordings: Mapped[list["Recording"]] = relationship(back_populates="user")


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    dataset_source: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[RecordingStatus] = mapped_column(
        Enum(RecordingStatus), default=RecordingStatus.UPLOADED
    )
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="recordings")
    prediction: Mapped["Prediction | None"] = relationship(
        back_populates="recording", uselist=False
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    recording_id: Mapped[str] = mapped_column(
        ForeignKey("recordings.id"), unique=True, nullable=False
    )
    epilepsy_risk: Mapped[float] = mapped_column(Float, nullable=False)
    mci_risk: Mapped[float] = mapped_column(Float, nullable=False)
    adhd_risk: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    heatmap_path: Mapped[str] = mapped_column(String, nullable=False)
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    recording: Mapped["Recording"] = relationship(back_populates="prediction")
