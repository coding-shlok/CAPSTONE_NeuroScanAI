"""End-to-end smoke test matching Section 5.9's acceptance criteria: a real
.edf file goes in, a complete report comes out, with no manual steps."""
import time

from ml.models.full_model import NeuroScanModel
from ml.pipeline import run_full_pipeline


def test_full_pipeline_edf_to_report(tmp_path, synthetic_edf_path, model_config, preprocessing_config):
    model = NeuroScanModel.from_config(model_config)

    start = time.monotonic()
    result = run_full_pipeline(
        synthetic_edf_path,
        model,
        preprocessing_config,
        output_dir=tmp_path,
        model_version="test-0.1",
        recording_id="e2e_test",
    )
    elapsed = time.monotonic() - start

    assert set(result.risk_scores.keys()) == set(model_config.disorders)
    for probability in result.risk_scores.values():
        assert 0.0 <= probability <= 1.0

    assert result.predicted_disorder in model_config.disorders
    assert 0.0 <= result.confidence <= 1.0

    heatmap_path = tmp_path / "e2e_test_heatmap.png"
    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0
    assert str(heatmap_path) == result.heatmap_path

    assert "Risk:" in result.report_text
    assert "Most influential channels:" in result.report_text
    assert "Recommendation:" in result.report_text

    # Section 5.9: inference should return in "under a few seconds on CPU."
    assert elapsed < 20.0
