from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.tables import User
from backend.app.schemas.auth import LoginRequest, LoginResponse
from backend.app.schemas.common import ResponseEnvelope
from backend.app.services.auth_service import create_access_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=ResponseEnvelope[LoginResponse])
def login(request: LoginRequest, db: Session = Depends(get_db)) -> ResponseEnvelope[LoginResponse]:
    user = db.query(User).filter(User.email == request.email).first()
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id)
    return ResponseEnvelope(success=True, data=LoginResponse(access_token=token))
