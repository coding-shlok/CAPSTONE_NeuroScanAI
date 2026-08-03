"""Five thin endpoints (Section 7). Each just validates input, calls one
service function, and returns the response envelope — business logic lives
in services/eeg_service.py."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.schemas.common import ResponseEnvelope
from backend.app.schemas.eeg import ReportResponse, StatusResponse, UploadResponse
from backend.app.services import eeg_service

router = APIRouter(prefix="/api/eeg", tags=["eeg"])


@router.post("/upload", response_model=ResponseEnvelope[UploadResponse])
async def upload(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> ResponseEnvelope[UploadResponse]:
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only .edf files are accepted")
    file_bytes = await file.read()

    # TODO: replace with the authenticated user once request-level auth
    # (Section 7's single login gate) is wired through the API layer.
    recording = eeg_service.save_upload(
        db, user_id="demo-user", filename=file.filename, file_bytes=file_bytes
    )
    return ResponseEnvelope(
        success=True, data=UploadResponse(recording_id=recording.id, status=recording.status)
    )


@router.get("/{recording_id}/status", response_model=ResponseEnvelope[StatusResponse])
def get_status(recording_id: str, db: Session = Depends(get_db)) -> ResponseEnvelope[StatusResponse]:
    recording = eeg_service.get_recording(db, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return ResponseEnvelope(
        success=True, data=StatusResponse(recording_id=recording.id, status=recording.status)
    )


@router.post("/{recording_id}/predict", response_model=ResponseEnvelope[ReportResponse])
def predict(recording_id: str, db: Session = Depends(get_db)) -> ResponseEnvelope[ReportResponse]:
    recording = eeg_service.get_recording(db, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    prediction = eeg_service.run_prediction(db, recording)
    return ResponseEnvelope(success=True, data=_to_report_response(recording_id, prediction))


@router.get("/{recording_id}/report", response_model=ResponseEnvelope[ReportResponse])
def get_report(recording_id: str, db: Session = Depends(get_db)) -> ResponseEnvelope[ReportResponse]:
    prediction = eeg_service.get_prediction(db, recording_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="No report available for this recording yet")
    return ResponseEnvelope(success=True, data=_to_report_response(recording_id, prediction))


def _to_report_response(recording_id: str, prediction) -> ReportResponse:
    return ReportResponse(
        recording_id=recording_id,
        epilepsy_risk=prediction.epilepsy_risk,
        mci_risk=prediction.mci_risk,
        adhd_risk=prediction.adhd_risk,
        confidence=prediction.confidence,
        heatmap_path=prediction.heatmap_path,
        report_text=prediction.report_text,
        model_version=prediction.model_version,
    )
