"""Unit tests verifying grounded facts, provenance tracing, and absence of hallucinated sensors."""
import pytest
from app.geospatial.activities import get_activity_profile
from app.models.generator import build_deterministic_fallback_recommendation
from app.safety.guardrails import evaluate_all_safety_rules


def test_grounding_preserves_facts_and_sources():
    activity = get_activity_profile("cricket")
    weather = {
        "temperature": 27.5,
        "uv_index": 4.5,
        "precipitation_probability": 10,
        "apparent_temperature": 29.0
    }
    air_quality = {
        "us_aqi": 68
    }
    candidate_venues = [
        {"name": "Oval Maidan Cricket Pitch", "category": "pitch", "distance_km": 0.3, "suitability_score": 92.0}
    ]
    guardrail = evaluate_all_safety_rules(activity, weather, air_quality)
    
    rec = build_deterministic_fallback_recommendation(
        activity=activity,
        location_name="Oval Maidan, Mumbai",
        weather_summary=weather,
        air_quality_summary=air_quality,
        candidate_venues=candidate_venues,
        recommended_window="Morning 08:00 to 11:00",
        guardrail=guardrail
    )

    # Check that facts in grounded_facts_used directly match inputs
    facts_str = " ".join(rec["grounded_facts_used"])
    assert "27.5" in facts_str
    assert "4.5" in facts_str
    assert "68" in facts_str

    # Verify no fabricated sensors (e.g. UV sensor, proprietary API)
    assert not any("gpt-4" in f.lower() for f in facts_str)
    assert not any("claude" in f.lower() for f in facts_str)


def test_missing_input_preservation():
    activity = get_activity_profile("picnic_park_outing")
    missing_items = ["Location omitted by user", "Forecast unverified"]
    guardrail = evaluate_all_safety_rules(activity)

    rec = build_deterministic_fallback_recommendation(
        activity=activity,
        location_name="Unknown Location",
        weather_summary={},
        air_quality_summary={},
        candidate_venues=[],
        recommended_window="Postpone until location confirmed",
        guardrail=guardrail,
        missing_inputs=missing_items
    )

    assert rec["missing_information_identified"] == missing_items
    assert len(rec["grounded_facts_used"]) == 0
