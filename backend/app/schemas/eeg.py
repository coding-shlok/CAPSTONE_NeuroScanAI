from __future__ import annotations

from pydantic import BaseModel

from backend.app.models.tables import RecordingStatus


class UploadResponse(BaseModel):
    recording_id: str
    status: RecordingStatus


class StatusResponse(BaseModel):
    recording_id: str
    status: RecordingStatus


class ReportResponse(BaseModel):
    recording_id: str
    epilepsy_risk: float
    mci_risk: float
    adhd_risk: float
    confidence: float
    heatmap_path: str
    report_text: str
    model_version: str
