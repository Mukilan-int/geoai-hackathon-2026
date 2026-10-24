"""Deterministic safety rules with recognized global standards (WHO, EPA, NOAA/NWS, ACSM)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.geospatial.activities import ActivityProfile


@dataclass
class RuleEvaluationResult:
    rule_name: str
    triggered: bool
    mandated_suitability: Optional[str]  # "caution", "not_recommended", or None
    hazard_title: str
    explanation: str
    guideline_source: str
    evidence_values: Dict[str, Any] = field(default_factory=dict)


def evaluate_uv_safety(
    activity: ActivityProfile,
    uv_index: Optional[float],
    hour_of_day: Optional[int] = None
) -> Optional[RuleEvaluationResult]:
    """WHO / EPA Global Solar UV Index Evaluation.
    
    Guideline: WHO 'Global Solar UV Index' (2002); EPA UV Index Scale.
    - UV >= 11: Extreme danger. Avoid outdoor sun exposure during midday peak (10:00-16:00).
    - UV >= 8: Very High danger. Vigorous outdoor cardio is cautioned/not recommended without continuous shade.
    """
    if uv_index is None:
        return None
    
    is_midday = hour_of_day is not None and (10 <= hour_of_day <= 16)
    
    if uv_index >= 11.0:
        if activity.intensity in ["high", "very_high"] or is_midday:
            return RuleEvaluationResult(
                rule_name="WHO_UV_EXTREME_STRENUOUS",
                triggered=True,
                mandated_suitability="not_recommended",
                hazard_title="Extreme UV Radiation Hazard",
                explanation=(
                    f"Measured UV Index of {uv_index:.1f} exceeds the WHO 'Extreme' threshold (>=11.0). "
                    f"Midday sun exposure during high-intensity {activity.name} causes rapid solar erythema (sunburn in <10 min) "
                    "and acute cellular skin/retinal damage. Strenuous midday activity is not recommended."
                ),
                guideline_source="World Health Organization (WHO) & US EPA Global Solar UV Index",
                evidence_values={"uv_index": uv_index, "hour_of_day": hour_of_day, "activity": activity.name}
            )
        else:
            return RuleEvaluationResult(
                rule_name="WHO_UV_EXTREME_MODERATE",
                triggered=True,
                mandated_suitability="caution",
                hazard_title="Extreme UV Radiation Alert",
                explanation=f"UV Index of {uv_index:.1f} requires mandatory sun-protective clothing, SPF 50+, sunglasses, and seeking shade.",
                guideline_source="World Health Organization (WHO) UV Index Guide",
                evidence_values={"uv_index": uv_index}
            )
            
    elif uv_index >= 8.0 and activity.intensity in ["high", "very_high"]:
        return RuleEvaluationResult(
            rule_name="WHO_UV_VERY_HIGH",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="Very High UV Radiation Exposure",
            explanation=f"UV Index of {uv_index:.1f} poses high sunburn risk for outdoor cardio. Shift session to early morning or late afternoon.",
            guideline_source="WHO / US EPA UV Index Guidelines",
            evidence_values={"uv_index": uv_index}
        )
        
    return None


def evaluate_thermal_stress(
    activity: ActivityProfile,
    apparent_temp_c: Optional[float],
    relative_humidity: Optional[float] = None
) -> Optional[RuleEvaluationResult]:
    """NOAA/NWS Heat Index & ACSM Guidelines for Exercise in Heat.
    
    - Apparent Temp >= 40.0°C (104°F): Danger. Heat stroke, heat cramps, or heat exhaustion highly likely with prolonged physical exertion.
    - Apparent Temp >= 34.0°C (93°F): Extreme Caution.
    - Apparent Temp <= -10.0°C: Wind chill caution.
    - Apparent Temp <= -25.0°C: Frostbite hazard.
    """
    if apparent_temp_c is None:
        return None
        
    if apparent_temp_c >= 40.0:
        return RuleEvaluationResult(
            rule_name="NOAA_HEAT_INDEX_DANGER",
            triggered=True,
            mandated_suitability="not_recommended" if activity.intensity in ["moderate", "high", "very_high"] else "caution",
            hazard_title="Dangerous Heat Stress / Hyperthermia Risk",
            explanation=(
                f"Apparent temperature (heat index) of {apparent_temp_c:.1f}°C falls into the NOAA 'Danger' zone (>=40°C). "
                f"Sustained metabolic exertion during {activity.name} under these conditions impedes evaporative sweat cooling, "
                "substantially elevating the clinical risk of exertional heat stroke."
            ),
            guideline_source="US National Oceanic and Atmospheric Administration (NOAA) & American College of Sports Medicine (ACSM)",
            evidence_values={"apparent_temp_c": apparent_temp_c, "relative_humidity": relative_humidity}
        )
    elif apparent_temp_c >= 34.0 and activity.intensity in ["high", "very_high"]:
        return RuleEvaluationResult(
            rule_name="NOAA_HEAT_INDEX_EXTREME_CAUTION",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="Elevated Heat Stress Caution",
            explanation=f"Apparent temperature is {apparent_temp_c:.1f}°C. Intense exercise risks heat exhaustion without copious hydration and regular rest in shade.",
            guideline_source="NOAA Heat Index Scale & ACSM Hydration Protocols",
            evidence_values={"apparent_temp_c": apparent_temp_c}
        )
    elif apparent_temp_c <= -25.0:
        return RuleEvaluationResult(
            rule_name="NWS_EXTREME_COLD_FROSTBITE",
            triggered=True,
            mandated_suitability="not_recommended",
            hazard_title="Severe Wind Chill and Frostbite Hazard",
            explanation=f"Apparent temperature of {apparent_temp_c:.1f}°C can cause exposed tissue frostbite in under 15 minutes.",
            guideline_source="US National Weather Service (NWS) Wind Chill Chart",
            evidence_values={"apparent_temp_c": apparent_temp_c}
        )
    elif apparent_temp_c <= -10.0:
        return RuleEvaluationResult(
            rule_name="NWS_COLD_CAUTION",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="Cold Weather Caution",
            explanation=f"Apparent temperature is {apparent_temp_c:.1f}°C. Requires multi-layered thermal insulation and wind-resistant outer shell.",
            guideline_source="US NWS Cold Weather Safety",
            evidence_values={"apparent_temp_c": apparent_temp_c}
        )
        
    return None


def evaluate_air_quality(
    activity: ActivityProfile,
    us_aqi: Optional[int],
    pm2_5_ugm3: Optional[float] = None
) -> Optional[RuleEvaluationResult]:
    """US EPA Air Quality Index (AQI) Health Thresholds.
    
    - US AQI > 300 / PM2.5 > 250 µg/m³: Hazardous. Emergency conditions; all outdoor physical activities not recommended.
    - US AQI 201-300 / PM2.5 150.5-250.4 µg/m³: Very Unhealthy. Strenuous exercise not recommended.
    - US AQI 151-200 / PM2.5 55.5-150.4 µg/m³: Unhealthy. Prolonged outdoor exertion cautioned.
    - US AQI 101-150: Unhealthy for Sensitive Groups.
    """
    if us_aqi is None and pm2_5_ugm3 is None:
        return None
        
    # Derive approximate US AQI if only PM2.5 is present
    aqi_val = us_aqi
    if aqi_val is None and pm2_5_ugm3 is not None:
        if pm2_5_ugm3 > 250:
            aqi_val = 301
        elif pm2_5_ugm3 > 150:
            aqi_val = 205
        elif pm2_5_ugm3 > 55:
            aqi_val = 155
        elif pm2_5_ugm3 > 35:
            aqi_val = 105
        else:
            aqi_val = 50

    if aqi_val >= 300:
        return RuleEvaluationResult(
            rule_name="EPA_AQI_HAZARDOUS",
            triggered=True,
            mandated_suitability="not_recommended",
            hazard_title="Hazardous Air Quality Emergency",
            explanation=(
                f"Air Quality Index of {aqi_val} (PM2.5: {pm2_5_ugm3} µg/m³) falls in the EPA 'Hazardous' category (>300). "
                f"Outdoor respiration during {activity.name} deposits severe particulate mass into deep alveolar lung tissue. "
                "All outdoor physical activities are strictly not recommended."
            ),
            guideline_source="United States Environmental Protection Agency (US EPA) Air Quality Index (40 CFR Part 58)",
            evidence_values={"us_aqi": aqi_val, "pm2_5_ugm3": pm2_5_ugm3}
        )
    elif aqi_val >= 200:
        mandate = "not_recommended" if activity.intensity in ["moderate", "high", "very_high"] else "caution"
        return RuleEvaluationResult(
            rule_name="EPA_AQI_VERY_UNHEALTHY",
            triggered=True,
            mandated_suitability=mandate,
            hazard_title="Very Unhealthy Air Quality",
            explanation=(
                f"AQI of {aqi_val} represents 'Very Unhealthy' air. Elevated minute ventilation during {activity.name} "
                "triggers pulmonary inflammation and cardiovascular stress. Indoor alternative strongly advised."
            ),
            guideline_source="US EPA Air Quality Guidelines",
            evidence_values={"us_aqi": aqi_val, "pm2_5_ugm3": pm2_5_ugm3}
        )
    elif aqi_val >= 150 and activity.intensity in ["high", "very_high"]:
        return RuleEvaluationResult(
            rule_name="EPA_AQI_UNHEALTHY_ACTIVE",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="Unhealthy Air Quality for Strenuous Exercise",
            explanation=f"AQI of {aqi_val} is 'Unhealthy'. Shorten duration and reduce exercise intensity.",
            guideline_source="US EPA AQI Standards",
            evidence_values={"us_aqi": aqi_val}
        )
        
    return None


def evaluate_convective_and_wind_storm(
    activity: ActivityProfile,
    weather_code: Optional[int],
    precipitation_prob: Optional[int],
    wind_speed_kmh: Optional[float],
    wind_gusts_kmh: Optional[float]
) -> Optional[RuleEvaluationResult]:
    """WMO Severe Weather Codes & NWS Wind and Lightning Safety.
    
    WMO codes:
    - 95, 96, 99: Thunderstorm with or without hail.
    - 65, 67, 75: Heavy rain or heavy freezing rain.
    """
    # 1. Thunderstorm / Lightning check
    is_thunderstorm = weather_code in [95, 96, 99]
    high_rain_prob = precipitation_prob is not None and precipitation_prob >= 75
    
    if is_thunderstorm:
        # Golf, cricket, tennis, cycling, hiking on exposed ridges are severely threatened by lightning
        is_high_risk_activity = activity.id in [
            "golf", "cricket", "outdoor_tennis", "cycling", "hiking_nature_walking", "outdoor_football"
        ]
        return RuleEvaluationResult(
            rule_name="NWS_LIGHTNING_THUNDERSTORM_RULE",
            triggered=True,
            mandated_suitability="not_recommended" if is_high_risk_activity else "caution",
            hazard_title="Active Thunderstorm and Lightning Threat",
            explanation=(
                f"WMO weather code {weather_code} indicates active or imminent thunderstorm conditions. "
                f"Open-field sports like {activity.name} offer zero lightning protection. "
                "Under NWS 'When Thunder Roars, Go Indoors' protocols, outdoor activity is not recommended until 30 min after last thunder."
            ),
            guideline_source="US National Weather Service (NWS) Lightning Safety Council",
            evidence_values={"weather_code": weather_code, "activity": activity.name}
        )

    # 2. Wind & Gust thresholds
    gust = wind_gusts_kmh or wind_speed_kmh
    if gust is not None:
        if gust >= 65.0:
            return RuleEvaluationResult(
                rule_name="BEAUFORT_GALE_WIND_RULE",
                triggered=True,
                mandated_suitability="not_recommended",
                hazard_title="Gale-Force Wind Hazard",
                explanation=(
                    f"Measured wind gusts of {gust:.1f} km/h (Beaufort 8) can break tree twigs, overturn bikes, "
                    f"and cause structural or flying debris injury during {activity.name}."
                ),
                guideline_source="WMO Beaufort Wind Scale & NWS High Wind Advisory",
                evidence_values={"wind_gusts_kmh": gust}
            )
        elif gust >= 45.0 and activity.id in ["cycling", "outdoor_tennis", "outdoor_photography", "golf"]:
            return RuleEvaluationResult(
                rule_name="STRONG_WIND_CAUTION",
                triggered=True,
                mandated_suitability="caution",
                hazard_title="High Crosswinds Caution",
                explanation=f"Wind gusts of {gust:.1f} km/h impair balance, ball trajectory, and gear stability.",
                guideline_source="NWS Wind Safety Guidelines",
                evidence_values={"wind_gusts_kmh": gust}
            )

    # 3. Heavy rain check
    if precipitation_prob is not None and precipitation_prob > activity.max_rain_prob_pct + 25:
        return RuleEvaluationResult(
            rule_name="HIGH_PRECIPITATION_RISK",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="High Probability of Rainfall",
            explanation=f"Precipitation probability is {precipitation_prob}%, which exceeds optimal limits for {activity.name}.",
            guideline_source="Meteorological Advisory & Sport Surface Traction Safety",
            evidence_values={"precipitation_prob": precipitation_prob}
        )

    return None


def evaluate_personal_constraints(
    activity: ActivityProfile,
    user_constraints: Optional[Dict[str, Any]]
) -> Optional[RuleEvaluationResult]:
    """Safety evaluation against user-declared physical constraints (e.g. age, joint mobility, asthma)."""
    if not user_constraints:
        return None
        
    age = user_constraints.get("age")
    mobility = str(user_constraints.get("mobility", "")).lower()
    health_notes = str(user_constraints.get("health_notes", "")).lower()
    
    # Osteoarthritis / cane / wheelchair on rough trails
    if ("cane" in mobility or "wheelchair" in mobility or "arthritis" in health_notes) and activity.id == "hiking_nature_walking":
        return RuleEvaluationResult(
            rule_name="MOBILITY_TERRAIN_MISMATCH",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="Trail Mobility Barrier",
            explanation="Unpaved hiking trails present trip hazards and uneven grades for individuals using mobility aids or managing joint arthritis. Paved park paths are recommended.",
            guideline_source="CDC Physical Activity Guidelines for Older Adults & ADA Accessibility Standards",
            evidence_values={"mobility": mobility, "health_notes": health_notes}
        )
        
    # Elderly in high thermal stress
    if age and isinstance(age, (int, float)) and age >= 70:
        return RuleEvaluationResult(
            rule_name="AGE_VULNERABILITY_ADVISORY",
            triggered=True,
            mandated_suitability="caution",
            hazard_title="Thermoregulatory Sensitivity Advisory",
            explanation="Individuals aged 70+ have reduced thirst sensation and lower heat dissipation efficiency. Regular rest intervals and shaded resting stops are advised.",
            guideline_source="World Health Organization (WHO) Age-Friendly Outdoor Spaces Guidelines",
            evidence_values={"age": age}
        )

    return None
