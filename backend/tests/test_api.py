def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_login_success(client):
    r = client.post("/api/auth/login", json={"email": "demo@example.com", "password": "password123"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "access_token" in body["data"]


def test_login_wrong_password(client):
    r = client.post("/api/auth/login", json={"email": "demo@example.com", "password": "wrong"})
    assert r.status_code == 401


def test_upload_rejects_non_edf(client):
    r = client.post("/api/eeg/upload", files={"file": ("sample.txt", b"not an edf", "text/plain")})
    assert r.status_code == 400


def test_full_prediction_flow(client, synthetic_edf_bytes):
    r = client.post(
        "/api/eeg/upload",
        files={"file": ("sample.edf", synthetic_edf_bytes, "application/octet-stream")},
    )
    assert r.status_code == 200
    recording_id = r.json()["data"]["recording_id"]

    r = client.get(f"/api/eeg/{recording_id}/status")
    assert r.status_code == 200
    assert r.json()["data"]["status"] == "uploaded"

    r = client.post(f"/api/eeg/{recording_id}/predict")
    assert r.status_code == 200
    data = r.json()["data"]
    for key in ("epilepsy_risk", "mci_risk", "adhd_risk", "confidence"):
        assert 0.0 <= data[key] <= 1.0
    assert "Risk:" in data["report_text"]

    r = client.get(f"/api/eeg/{recording_id}/report")
    assert r.status_code == 200
    assert r.json()["data"] == data


def test_report_404_for_unknown_recording(client):
    r = client.get("/api/eeg/does-not-exist/report")
    assert r.status_code == 404


def test_predict_404_for_unknown_recording(client):
    r = client.post("/api/eeg/does-not-exist/predict")
    assert r.status_code == 404
