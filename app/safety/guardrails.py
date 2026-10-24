"""Deterministic safety guardrails with authoritative rule evaluation and model override logic."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.geospatial.activities import ActivityProfile
from app.safety.rules import (
    RuleEvaluationResult,
    evaluate_air_quality,
    evaluate_convective_and_wind_storm,
    evaluate_personal_constraints,
    evaluate_thermal_stress,
    evaluate_uv_safety,
)
from app.utils.logger import logger


SUITABILITY_RANKS = {
    "recommended": 1,
    "caution": 2,
    "not_recommended": 3,
}


@dataclass
class GuardrailAssessment:
    guardrail_status: str  # "CLEAR", "ADVISORY", "BLOCKED"
    mandated_suitability: str  # "recommended", "caution", "not_recommended"
    triggered_rules: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    source_guidelines: List[str] = field(default_factory=list)
    evidence_snapshot: Dict[str, Any] = field(default_factory=dict)


def evaluate_all_safety_rules(
    activity: ActivityProfile,
    weather_data: Optional[Dict[str, Any]] = None,
    air_quality_data: Optional[Dict[str, Any]] = None,
    user_constraints: Optional[Dict[str, Any]] = None,
    hour_of_day: Optional[int] = None
) -> GuardrailAssessment:
    """Evaluates all deterministic safety rules against current meteorological and geospatial observations."""
    weather = weather_data or {}
    aq = air_quality_data or {}
    
    uv_index = weather.get("uv_index")
    apparent_temp = weather.get("apparent_temperature") or weather.get("temperature")
    rel_humidity = weather.get("relative_humidity")
    weather_code = weather.get("weather_code")
    rain_prob = weather.get("precipitation_probability")
    wind_speed = weather.get("wind_speed")
    wind_gusts = weather.get("wind_gusts")
    
    us_aqi = aq.get("us_aqi")
    pm2_5 = aq.get("pm2_5")

    rule_checks: List[Optional[RuleEvaluationResult]] = [
        evaluate_uv_safety(activity, uv_index, hour_of_day),
        evaluate_thermal_stress(activity, apparent_temp, rel_humidity),
        evaluate_air_quality(activity, us_aqi, pm2_5),
        evaluate_convective_and_wind_storm(activity, weather_code, rain_prob, wind_speed, wind_gusts),
        evaluate_personal_constraints(activity, user_constraints)
    ]

    highest_rank = 1  # recommended by default
    mandated_suitability = "recommended"
    triggered_rules = []
    explanations = []
    sources = set()

    for result in rule_checks:
        if result and result.triggered:
            triggered_rules.append(result.rule_name)
            explanations.append(f"[{result.hazard_title}] {result.explanation}")
            sources.add(result.guideline_source)
            
            rank = SUITABILITY_RANKS.get(result.mandated_suitability or "recommended", 1)
            if rank > highest_rank:
                highest_rank = rank
                mandated_suitability = result.mandated_suitability

    if highest_rank == 3:
        status = "BLOCKED"
    elif highest_rank == 2:
        status = "ADVISORY"
    else:
        status = "CLEAR"

    return GuardrailAssessment(
        guardrail_status=status,
        mandated_suitability=mandated_suitability,
        triggered_rules=triggered_rules,
        explanations=explanations,
        source_guidelines=list(sources),
        evidence_snapshot={
            "uv_index": uv_index,
            "apparent_temp": apparent_temp,
            "rain_prob": rain_prob,
            "wind_speed": wind_speed,
            "us_aqi": us_aqi,
            "weather_code": weather_code
        }
    )


def apply_guardrail_override(
    recommendation: Dict[str, Any],
    guardrail: GuardrailAssessment
) -> Dict[str, Any]:
    """Applies post-generation deterministic override if the SLM generated an unsafe recommendation.
    
    Ensures that an SLM cannot bypass hard scientific safety thresholds (e.g., WHO UV >= 11 or EPA AQI > 300).
    """
    model_suitability = recommendation.get("suitability", "recommended")
    model_rank = SUITABILITY_RANKS.get(model_suitability, 1)
    mandated_rank = SUITABILITY_RANKS.get(guardrail.mandated_suitability, 1)

    override_applied = False
    
    # If the deterministic safety rules mandate a strictly more conservative rating than the model produced
    if mandated_rank > model_rank:
        override_applied = True
        logger.warning(
            f"SAFETY OVERRIDE APPLIED: Overriding model suitability '{model_suitability}' -> '{guardrail.mandated_suitability}' "
            f"due to deterministic rules: {guardrail.triggered_rules}"
        )
        recommendation["suitability"] = guardrail.mandated_suitability
        
        # Invert or adjust recommended time window if fully blocked
        if guardrail.mandated_suitability == "not_recommended":
            recommendation["recommended_time_window"] = (
                "No safe outdoor time window available under current hazardous conditions. "
                "Postpone or relocate indoors."
            )
        
        # Inject deterministic explanations into reasons
        existing_reasons = recommendation.get("reasons", [])
        for exp in guardrail.explanations:
            if exp not in existing_reasons:
                existing_reasons.insert(0, f"MANDATORY SAFETY RULE: {exp}")
        recommendation["reasons"] = existing_reasons

    # Ensure safety assessment structure is populated
    recommendation["safety_assessment"] = {
        "guardrail_status": guardrail.guardrail_status,
        "triggered_rules": guardrail.triggered_rules,
        "override_applied": override_applied,
        "source_guidelines": guardrail.source_guidelines
    }
    
    # Ensure precautions include relevant hazard warnings
    existing_precautions = recommendation.get("precautions", [])
    for exp in guardrail.explanations:
        if exp not in existing_precautions and override_applied:
            existing_precautions.insert(0, exp)
    recommendation["precautions"] = existing_precautions

    return recommendation
