from ml.reporting.report_builder import build_report, confidence_from_probability, risk_tier


def test_risk_tier_boundaries():
    assert risk_tier(0.0) == "Low"
    assert risk_tier(0.32) == "Low"
    assert risk_tier(0.33) == "Moderate"
    assert risk_tier(0.65) == "Moderate"
    assert risk_tier(0.66) == "High"
    assert risk_tier(1.0) == "High"


def test_confidence_from_probability():
    assert confidence_from_probability(0.5) == 0.0  # maximally uncertain
    assert confidence_from_probability(1.0) == 1.0  # maximally confident positive
    assert confidence_from_probability(0.0) == 1.0  # maximally confident negative


def test_build_report_high_risk_text():
    report = build_report("epilepsy", 0.84, ["F3", "F4", "Cz"])
    assert report.risk_tier == "High"
    assert "84%" in report.summary_text
    assert "F3, F4, Cz" in report.summary_text
    assert "Further neurological evaluation recommended." in report.summary_text
    assert report.summary_text.startswith("Epilepsy Risk:")


def test_build_report_unknown_disorder_falls_back_to_raw_name():
    report = build_report("unknown_disorder", 0.1, ["A", "B"])
    assert report.summary_text.startswith("unknown_disorder Risk:")
