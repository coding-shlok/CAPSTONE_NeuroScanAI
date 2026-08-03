"""Points the app at an isolated temp SQLite DB + storage dir before
`backend.app.main` (and its `settings` singleton) get imported anywhere,
so tests never touch the real data/neuroscan.db."""
import os
import tempfile
from pathlib import Path

_tmp_dir = Path(tempfile.mkdtemp(prefix="neuroscan_test_"))
os.environ["NEUROSCAN_DATABASE_URL"] = f"sqlite:///{_tmp_dir / 'test.db'}"
os.environ["NEUROSCAN_STORAGE_DIR"] = str(_tmp_dir / "storage")
os.environ["NEUROSCAN_HEATMAP_DIR"] = str(_tmp_dir / "storage" / "heatmaps")

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.database import Base, SessionLocal, engine
from backend.app.models.tables import User
from backend.app.services.auth_service import hash_password
from ml.tests.synthetic_edf import write_synthetic_edf


@pytest.fixture(scope="session", autouse=True)
def _setup_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    db.merge(
        User(id="demo-user", email="demo@example.com", password_hash=hash_password("password123"))
    )
    db.commit()
    db.close()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def synthetic_edf_bytes() -> bytes:
    path = write_synthetic_edf(_tmp_dir / "fixture_sample.edf", duration_seconds=30.0, sfreq=256.0)
    return path.read_bytes()
