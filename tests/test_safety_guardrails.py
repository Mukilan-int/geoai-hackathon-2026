"""Unit tests for deterministic safety rules, hazard thresholds, and model override guardrails."""
import pytest
from app.geospatial.activities import get_activity_profile
from app.safety.guardrails import apply_guardrail_override, evaluate_all_safety_rules
from app.safety.rules import (
    evaluate_air_quality,
    evaluate_convective_and_wind_storm,
    evaluate_thermal_stress,
    evaluate_uv_safety,
)


def test_extreme_uv_midday_running():
    """Verify WHO Extreme UV (>=11.0) blocks strenuous outdoor running."""
    activity = get_activity_profile("running_jogging")
    result = evaluate_uv_safety(activity, uv_index=11.2, hour_of_day=13)
    assert result is not None
    assert result.triggered is True
    assert result.mandated_suitability == "not_recommended"
    assert "WHO_UV_EXTREME_STRENUOUS" in result.rule_name
    assert "World Health Organization" in result.guideline_source


def test_hazardous_aqi_hiking():
    """Verify EPA Hazardous AQI (>300) blocks outdoor hiking."""
    activity = get_activity_profile("hiking_nature_walking")
    result = evaluate_air_quality(activity, us_aqi=345, pm2_5_ugm3=295.0)
    assert result is not None
    assert result.triggered is True
    assert result.mandated_suitability == "not_recommended"
    assert "EPA_AQI_HAZARDOUS" in result.rule_name


def test_thunderstorm_golf_and_cricket():
    """Verify convective thunderstorm (WMO 95) blocks golf and cricket."""
    golf = get_activity_profile("golf")
    result = evaluate_convective_and_wind_storm(
        golf, weather_code=95, precipitation_prob=70, wind_speed_kmh=25.0, wind_gusts_kmh=40.0
    )
    assert result is not None
    assert result.triggered is True
    assert result.mandated_suitability == "not_recommended"
    assert "NWS_LIGHTNING_THUNDERSTORM_RULE" in result.rule_name


def test_gale_wind_cycling():
    """Verify gale-force wind gusts (>60 km/h) block cycling."""
    cycling = get_activity_profile("cycling")
    result = evaluate_convective_and_wind_storm(
        cycling, weather_code=2, precipitation_prob=10, wind_speed_kmh=40.0, wind_gusts_kmh=68.0
    )
    assert result is not None
    assert result.triggered is True
    assert result.mandated_suitability == "not_recommended"
    assert "BEAUFORT_GALE_WIND_RULE" in result.rule_name


def test_extreme_heat_stress_football():
    """Verify NOAA Heat Index danger (apparent temp >= 40 C) blocks outdoor football."""
    football = get_activity_profile("outdoor_football")
    result = evaluate_thermal_stress(football, apparent_temp_c=41.5, relative_humidity=70)
    assert result is not None
    assert result.triggered is True
    assert result.mandated_suitability == "not_recommended"
    assert "NOAA_HEAT_INDEX_DANGER" in result.rule_name


def test_guardrail_deterministic_override():
    """Verify that an unsafe model recommendation ('recommended' in UV 11) is overridden."""
    running = get_activity_profile("running_jogging")
    guardrail = evaluate_all_safety_rules(
        activity=running,
        weather_data={"uv_index": 11.2, "temperature": 34.0, "apparent_temperature": 38.0},
        air_quality_data={"us_aqi": 80},
        hour_of_day=13
    )
    assert guardrail.guardrail_status == "BLOCKED"
    assert guardrail.mandated_suitability == "not_recommended"

    # Simulate a model that erroneously returned 'recommended'
    flawed_model_output = {
        "activity_name": "Running or Jogging",
        "suitability": "recommended",  # Unsafe model hallucination/optimism
        "recommended_time_window": "Now at 1 PM",
        "precautions": ["Wear running shoes"],
        "reasons": ["Sunny skies are pleasant for outdoor sports."],
        "missing_information_identified": [],
        "grounded_facts_used": ["Sunny"],
        "safety_assessment": {
            "guardrail_status": "CLEAR",
            "triggered_rules": [],
            "override_applied": False,
            "source_guidelines": []
        },
        "candidate_places": [],
        "data_sources_used": [],
        "uncertainties_and_limitations": []
    }

    # Apply guardrail
    secured_output = apply_guardrail_override(flawed_model_output, guardrail)

    # Assert that the deterministic guardrail strictly overrode the model
    assert secured_output["suitability"] == "not_recommended"
    assert secured_output["safety_assessment"]["override_applied"] is True
    assert secured_output["safety_assessment"]["guardrail_status"] == "BLOCKED"
    assert any("WHO_UV_EXTREME_STRENUOUS" in r for r in secured_output["safety_assessment"]["triggered_rules"])
    assert any("MANDATORY SAFETY RULE" in r for r in secured_output["reasons"])
