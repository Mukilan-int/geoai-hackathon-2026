"""SLM recommendation generator with ChatML prompting, JSON parsing, and deterministic fallback."""
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from app.geospatial.activities import ActivityProfile
from app.models.config import ModelConfig
from app.models.loader import load_model_and_tokenizer
from app.safety.guardrails import GuardrailAssessment
from app.utils.json_sanitizer import sanitize_and_parse_json, validate_schema
from app.utils.logger import logger

SYSTEM_INSTRUCTION = """You are an expert GeoAI assistant specializing in grounded outdoor activity safety for the GeoAI Hackathon 2026.
You evaluate outdoor activity requests based STRICTLY and ONLY on the provided meteorological, geospatial, and air quality evidence.

CRITICAL RULES:
1. Ground your recommendations ONLY in the facts supplied in the prompt. Do NOT invent temperatures, places, or forecast variables.
2. If any critical data is missing (e.g. location omitted, forecast unverified, air quality unknown), explicitly state it in 'missing_information_identified'.
3. Follow the safety guardrail guidance provided. If conditions are hazardous (extreme UV, thunderstorm, heavy rain, gale wind, or hazardous AQI), set suitability to 'caution' or 'not_recommended'.
4. Output MUST be ONLY a single valid JSON object strictly matching the schema with no conversational preamble or markdown commentary outside the JSON.
"""


def build_recommendation_prompt(
    activity: ActivityProfile,
    location_name: str,
    weather_summary: Dict[str, Any],
    air_quality_summary: Dict[str, Any],
    candidate_venues: List[Dict[str, Any]],
    recommended_window: str,
    guardrail: GuardrailAssessment,
    user_constraints: Optional[Dict[str, Any]] = None,
    missing_inputs: Optional[List[str]] = None
) -> str:
    """Constructs a densely grounded contextual prompt for the SLM."""
    prompt_lines = [
        f"Activity to Evaluate: {activity.name} (Intensity: {activity.intensity})",
        f"Target Location: {location_name}",
        "",
        "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---",
        f"- Current Temperature: {weather_summary.get('temperature', 'N/A')} °C (Apparent: {weather_summary.get('apparent_temperature', 'N/A')} °C)",
        f"- Relative Humidity: {weather_summary.get('relative_humidity', 'N/A')} %",
        f"- Weather Description: {weather_summary.get('weather_description', 'N/A')} (WMO Code: {weather_summary.get('weather_code', 'N/A')})",
        f"- Precipitation Probability: {weather_summary.get('precipitation_probability', 'N/A')} %",
        f"- Wind Speed: {weather_summary.get('wind_speed', 'N/A')} km/h (Gusts: {weather_summary.get('wind_gusts', 'N/A')} km/h)",
        f"- Solar UV Index: {weather_summary.get('uv_index', 'N/A')}",
        f"- Daylight Status: {weather_summary.get('daylight_status', 'N/A')}",
        "",
        "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---",
        f"- US EPA AQI: {air_quality_summary.get('us_aqi', 'N/A')} ({air_quality_summary.get('us_aqi_category', 'N/A')})",
        f"- PM2.5 Concentration: {air_quality_summary.get('pm2_5', 'N/A')} µg/m³",
        f"- PM10 Concentration: {air_quality_summary.get('pm10', 'N/A')} µg/m³",
        "",
        "--- GEOSPATIAL VENUES (OpenStreetMap) ---"
    ]
    
    if candidate_venues:
        for idx, venue in enumerate(candidate_venues[:3], 1):
            prompt_lines.append(
                f"{idx}. {venue['name']} ({venue['category']}, ~{venue.get('distance_km', 'N/A')} km away, "
                f"Suitability Score: {venue.get('suitability_score', 'N/A')})"
            )
    else:
        prompt_lines.append("No specific outdoor venues mapped within immediate search radius.")

    prompt_lines.extend([
        "",
        f"Temporal Window Analysis: {recommended_window}",
        f"Deterministic Guardrail Assessment: Status={guardrail.guardrail_status}, Mandate={guardrail.mandated_suitability}",
    ])

    if guardrail.triggered_rules:
        prompt_lines.append(f"Safety Rule Triggers: {', '.join(guardrail.triggered_rules)}")

    if user_constraints:
        prompt_lines.append(f"User Stated Constraints: {json.dumps(user_constraints)}")
        
    if missing_inputs:
        prompt_lines.append(f"Identified Missing Inputs: {', '.join(missing_inputs)}")

    prompt_lines.extend([
        "",
        "Provide a grounded outdoor recommendation in the required JSON schema format."
    ])

    return "\n".join(prompt_lines)


def build_deterministic_fallback_recommendation(
    activity: ActivityProfile,
    location_name: str,
    weather_summary: Dict[str, Any],
    air_quality_summary: Dict[str, Any],
    candidate_venues: List[Dict[str, Any]],
    recommended_window: str,
    guardrail: GuardrailAssessment,
    user_constraints: Optional[Dict[str, Any]] = None,
    missing_inputs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generates a schema-compliant recommendation deterministically when the SLM is unavailable or malformed."""
    precautions = list(activity.equipment_precautions)
    reasons = []

    # Grounded facts
    grounded_facts = []
    if weather_summary.get("temperature") is not None:
        grounded_facts.append(f"Temperature: {weather_summary['temperature']}°C")
    if weather_summary.get("uv_index") is not None:
        grounded_facts.append(f"UV Index: {weather_summary['uv_index']}")
    if weather_summary.get("precipitation_probability") is not None:
        grounded_facts.append(f"Precipitation Probability: {weather_summary['precipitation_probability']}%")
    if air_quality_summary.get("us_aqi") is not None:
        grounded_facts.append(f"US AQI: {air_quality_summary['us_aqi']}")

    # Formulate reasons
    if guardrail.mandated_suitability == "not_recommended":
        reasons.extend(guardrail.explanations)
        time_window = "No safe outdoor time window available under current hazardous conditions."
    elif guardrail.mandated_suitability == "caution":
        reasons.extend(guardrail.explanations)
        reasons.append(f"Conditions for {activity.name} at {location_name} require heightened caution.")
        time_window = recommended_window
    else:
        reasons.append(f"Weather and environmental conditions at {location_name} are favorable for {activity.name}.")
        time_window = recommended_window

    candidate_records = []
    for cand in candidate_venues[:4]:
        candidate_records.append({
            "name": cand.get("name", "Public Outdoor Area"),
            "category": cand.get("category", "park"),
            "distance_km": cand.get("distance_km"),
            "suitability_score": cand.get("suitability_score", 70.0),
            "source_evidence": cand.get("source_evidence", "OpenStreetMap features")
        })

    return {
        "activity_name": activity.name,
        "suitability": guardrail.mandated_suitability,
        "recommended_time_window": time_window,
        "precautions": precautions,
        "reasons": reasons,
        "missing_information_identified": missing_inputs or [],
        "grounded_facts_used": grounded_facts,
        "safety_assessment": {
            "guardrail_status": guardrail.guardrail_status,
            "triggered_rules": guardrail.triggered_rules,
            "override_applied": False,
            "source_guidelines": guardrail.source_guidelines
        },
        "candidate_places": candidate_records,
        "data_sources_used": [],
        "uncertainties_and_limitations": [
            "Weather forecasts represent numerical atmospheric models with inherent temporal uncertainty.",
            "OpenStreetMap features are community-mapped and may not reflect temporary facility closures.",
            "Static routing duration estimates do not model active vehicular traffic congestion."
        ]
    }


def generate_recommendation_with_model(
    activity: ActivityProfile,
    location_name: str,
    weather_summary: Dict[str, Any],
    air_quality_summary: Dict[str, Any],
    candidate_venues: List[Dict[str, Any]],
    recommended_window: str,
    guardrail: GuardrailAssessment,
    user_constraints: Optional[Dict[str, Any]] = None,
    missing_inputs: Optional[List[str]] = None,
    config: Optional[ModelConfig] = None
) -> Tuple[Dict[str, Any], bool, str]:
    """Orchestrates model inference, JSON parsing, schema validation, and fallback.
    
    Returns:
        (recommendation_dict, is_model_generated, generation_method_label)
    """
    model, tokenizer, load_err = load_model_and_tokenizer(config)
    
    # If model is unavailable or disabled, execute deterministic generation
    if model is None or tokenizer is None:
        rec = build_deterministic_fallback_recommendation(
            activity, location_name, weather_summary, air_quality_summary,
            candidate_venues, recommended_window, guardrail, user_constraints, missing_inputs
        )
        return rec, False, "Deterministic Engine (Model Unavailable or Disabled)"

    # Construct prompt
    user_prompt = build_recommendation_prompt(
        activity, location_name, weather_summary, air_quality_summary,
        candidate_venues, recommended_window, guardrail, user_constraints, missing_inputs
    )

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt}
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([formatted_prompt], return_tensors="pt")
        
        cfg = config or ModelConfig()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False  # Deterministic greedy decoding
            )
        t1 = time.perf_counter()
        gen_time = round(t1 - t0, 3)

        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Parse JSON
        parsed_dict, parse_err = sanitize_and_parse_json(response_text)
        if parsed_dict:
            # Validate schema
            is_valid, validation_errors = validate_schema(parsed_dict)
            if is_valid:
                logger.info(f"Model generated valid schema recommendation in {gen_time}s.")
                return parsed_dict, True, f"OpenVINO INT4 ({gen_time}s)"
            else:
                logger.warning(f"Model output schema validation failed: {validation_errors[:2]}")
        else:
            logger.warning(f"Model output JSON sanitization failed: {parse_err}")

    except Exception as exc:
        logger.error(f"Inference exception encountered: {exc}")

    # Fallback if model output was unparseable or failed schema
    logger.info("Executing safe deterministic fallback recommendation.")
    rec = build_deterministic_fallback_recommendation(
        activity, location_name, weather_summary, air_quality_summary,
        candidate_venues, recommended_window, guardrail, user_constraints, missing_inputs
    )
    return rec, False, "Deterministic Engine (Model Output Validation Fallback)"
