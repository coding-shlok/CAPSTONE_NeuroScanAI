from __future__ import annotations

from fastapi import FastAPI

from backend.app.api import auth, eeg
from backend.app.models.database import Base, engine

app = FastAPI(title="NeuroScan AI", version="0.1.0")

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(eeg.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
