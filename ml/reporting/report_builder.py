"""Clinical report builder (Section 5.8) — pure formatting, no ML. Turns a
risk score and the Grad-CAM's top channels into the plain-language block:

    Risk: 84% (High)
    Most influential channels: F3, F4, Cz
    Recommendation: Further neurological evaluation recommended.
"""
from __future__ import annotations

from dataclasses import dataclass

DISORDER_LABELS = {
    "epilepsy": "Epilepsy",
    "adhd": "ADHD",
    "mci": "Alzheimer's / MCI",
}

RISK_TIER_THRESHOLDS = {"low": 0.33, "moderate": 0.66}

RECOMMENDATIONS = {
    "Low": "No immediate concern indicated; continue routine monitoring.",
    "Moderate": "Findings are inconclusive; clinical correlation and follow-up recommended.",
    "High": "Further neurological evaluation recommended.",
}


@dataclass
class ClinicalReport:
    disorder: str
    risk_percent: float
    risk_tier: str
    top_channels: list[str]
    confidence: float
    recommendation: str
    summary_text: str


def risk_tier(probability: float) -> str:
    if probability < RISK_TIER_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_TIER_THRESHOLDS["moderate"]:
        return "Moderate"
    return "High"


def confidence_from_probability(probability: float) -> float:
    """Distance of the prediction from the decision boundary (0.5), mapped
    to [0, 1]. This is a simple, honest proxy, not a calibrated probability —
    the model has no separate confidence head (Section 5.4 only specifies
    per-disorder risk heads)."""
    return abs(2 * probability - 1)


def build_report(disorder: str, probability: float, top_channels: list[str]) -> ClinicalReport:
    tier = risk_tier(probability)
    recommendation = RECOMMENDATIONS[tier]
    label = DISORDER_LABELS.get(disorder, disorder)
    summary_text = (
        f"{label} Risk: {probability:.0%} ({tier})\n"
        f"Most influential channels: {', '.join(top_channels)}\n"
        f"Recommendation: {recommendation}"
    )
    return ClinicalReport(
        disorder=disorder,
        risk_percent=probability * 100,
        risk_tier=tier,
        top_channels=list(top_channels),
        confidence=confidence_from_probability(probability),
        recommendation=recommendation,
        summary_text=summary_text,
    )
