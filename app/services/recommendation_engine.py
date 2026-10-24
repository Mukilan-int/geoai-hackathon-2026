"""End-to-End Recommendation Engine coordinating live services, ranking, SLM inference, and guardrails."""
import time
from typing import Any, Dict, List, Optional, Tuple

from app.geospatial.activities import get_activity_profile
from app.geospatial.ranking import rank_candidate_places, rank_time_windows
from app.models.generator import generate_recommendation_with_model
from app.safety.guardrails import apply_guardrail_override, evaluate_all_safety_rules
from app.services.air_quality import fetch_air_quality
from app.services.geocoding import geocode_location, reverse_geocode
from app.services.overpass import find_nearby_places
from app.services.registry import DataSourceRegistry
from app.services.routing import calculate_route_distance
from app.services.sun_time import calculate_sun_times
from app.services.weather import fetch_weather_forecast
from app.utils.json_sanitizer import validate_schema
from app.utils.logger import logger


class RecommendationEngine:
    """Coordinates full recommendation pipeline adhering to GeoAI Hackathon 2026 rulebook."""

    def __init__(self):
        pass

    def run_pipeline(
        self,
        activity_name: str,
        location_query: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: float = 5.0,
        user_constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Executes full pipeline: validation -> live data -> ranking -> guardrail -> model -> response."""
        registry = DataSourceRegistry()
        missing_inputs: List[str] = []
        
        # 1. Activity validation
        activity_profile = get_activity_profile(activity_name)
        if not activity_profile:
            return {}, f"Unsupported activity '{activity_name}'. Please choose one of the 10 supported outdoor activities."

        # 2. Location resolution
        resolved_name = location_query or "Specified Coordinates"
        resolved_lat = latitude
        resolved_lon = longitude
        
        if resolved_lat is None or resolved_lon is None:
            if location_query:
                geo_info, geo_err = geocode_location(location_query, registry=registry)
                if geo_info:
                    resolved_lat = geo_info["latitude"]
                    resolved_lon = geo_info["longitude"]
                    resolved_name = geo_info["name"]
                else:
                    missing_inputs.append(f"Geocoding could not resolve coordinates for: '{location_query}'")
            else:
                missing_inputs.append("Location coordinates and place query were both omitted.")
        else:
            # Coordinates provided: get reverse geocode label
            rev_info, _ = reverse_geocode(resolved_lat, resolved_lon, registry=registry)
            if rev_info:
                resolved_name = rev_info["name"]

        # If location could not be resolved, return structured missing data response immediately
        if resolved_lat is None or resolved_lon is None:
            return {
                "activity_name": activity_profile.name,
                "suitability": "caution",
                "recommended_time_window": "Unable to evaluate weather without a confirmed location.",
                "precautions": activity_profile.equipment_precautions,
                "reasons": [
                    "A valid geographic location is required to query live meteorological forecasts and discover nearby venues.",
                    "No location could be resolved from the supplied request."
                ],
                "missing_information_identified": missing_inputs,
                "grounded_facts_used": [],
                "safety_assessment": {
                    "guardrail_status": "ADVISORY",
                    "triggered_rules": ["MISSING_LOCATION_INPUT"],
                    "override_applied": False,
                    "source_guidelines": ["GeoAI Input Verification Protocol"]
                },
                "candidate_places": [],
                "data_sources_used": registry.get_sources_used(),
                "uncertainties_and_limitations": [
                    "Cannot verify local atmospheric conditions or geographic hazards without location coordinates."
                ]
            }, None

        # 3. Retrieve Live Data
        # A. Weather
        weather_payload, weather_err = fetch_weather_forecast(resolved_lat, resolved_lon, registry=registry)
        if not weather_payload:
            missing_inputs.append(f"Live weather service unavailable: {weather_err}")
            weather_summary = {}
            upcoming_windows = []
        else:
            weather_summary = weather_payload["current"]
            upcoming_windows = weather_payload.get("upcoming_windows", [])

        # B. Air Quality
        aq_payload, aq_err = fetch_air_quality(resolved_lat, resolved_lon, registry=registry)
        if not aq_payload:
            missing_inputs.append(f"Live air quality service unavailable: {aq_err}")
            aq_summary = {}
        else:
            aq_summary = aq_payload

        # C. Solar and Daylight Information
        sun_info = calculate_sun_times(resolved_lat, resolved_lon)
        weather_summary["daylight_status"] = (
            f"Daylight ({sun_info['sunrise_utc']} to {sun_info['sunset_utc']})"
            if sun_info["is_daylight"]
            else f"Darkness/Night ({sun_info['sunset_utc']} sunset, {sun_info['sunrise_utc']} sunrise)"
        )

        # D. Nearby Outdoor Places (OSM Overpass)
        candidate_places_raw, overpass_err = find_nearby_places(
            latitude=resolved_lat,
            longitude=resolved_lon,
            radius_km=radius_km,
            activity_profile=activity_profile,
            registry=registry
        )
        if not candidate_places_raw and overpass_err:
            missing_inputs.append(f"Overpass POI discovery service notice: {overpass_err}")

        # 4. Multi-candidate Ranking & Time Window Analysis
        scored_places = rank_candidate_places(candidate_places_raw, activity_profile, resolved_lat, resolved_lon)
        time_analysis = rank_time_windows(upcoming_windows, activity_profile)
        recommended_window = time_analysis["recommended_window"]

        # 5. Deterministic Safety Checks
        guardrail = evaluate_all_safety_rules(
            activity=activity_profile,
            weather_data=weather_summary,
            air_quality_data=aq_summary,
            user_constraints=user_constraints
        )

        # 6. SLM Inference with Prompt & Schema Validation
        recommendation, is_model_gen, gen_label = generate_recommendation_with_model(
            activity=activity_profile,
            location_name=resolved_name,
            weather_summary=weather_summary,
            air_quality_summary=aq_summary,
            candidate_venues=scored_places,
            recommended_window=recommended_window,
            guardrail=guardrail,
            user_constraints=user_constraints,
            missing_inputs=missing_inputs
        )

        # 7. Apply Post-Generation Safety Guardrail Override
        recommendation = apply_guardrail_override(recommendation, guardrail)

        # 8. Enrich with Sources and Candidate Places
        recommendation["data_sources_used"] = registry.get_sources_used()
        if not recommendation.get("candidate_places"):
            recommendation["candidate_places"] = scored_places[:4]

        # 9. Final Schema Conformance Validation
        is_valid, validation_errors = validate_schema(recommendation)
        if not is_valid:
            logger.warning(f"Final output schema validation warning: {validation_errors}")

        return recommendation, None


engine = RecommendationEngine()
