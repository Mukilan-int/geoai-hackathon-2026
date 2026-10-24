"""Dataset preparation script generating diverse, grounded, schema-validated training pairs."""
import json
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.json_sanitizer import validate_schema

SYSTEM_PROMPT = (
    "You are an expert GeoAI assistant specializing in grounded outdoor activity safety for the GeoAI Hackathon 2026.\n"
    "You evaluate outdoor activity requests based STRICTLY and ONLY on the provided meteorological, geospatial, and air quality evidence.\n"
    "CRITICAL RULES:\n"
    "1. Ground your recommendations ONLY in the facts supplied in the prompt. Do NOT invent temperatures, places, or forecast variables.\n"
    "2. If any critical data is missing (e.g. location omitted, forecast unverified, air quality unknown), explicitly state it in 'missing_information_identified'.\n"
    "3. Follow the safety guardrail guidance provided. If conditions are hazardous (extreme UV, thunderstorm, heavy rain, gale wind, or hazardous AQI), set suitability to 'caution' or 'not_recommended'.\n"
    "4. Output MUST be ONLY a single valid JSON object strictly matching the schema with no conversational preamble or markdown commentary outside the JSON."
)

RAW_EXAMPLES: List[Dict[str, Any]] = [
    # 1. Running in Extreme UV (Directly fixing previous evaluation failure!)
    {
        "id": "train_01_running_extreme_uv",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Running or Jogging (Intensity: very_high)\n"
            "Target Location: Bandra Bandstand, Mumbai\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 34.0 °C (Apparent: 39.5 °C)\n"
            "- Relative Humidity: 72 %\n"
            "- Weather Description: Clear sky (WMO Code: 0)\n"
            "- Precipitation Probability: 5 %\n"
            "- Wind Speed: 14.0 km/h (Gusts: 18.0 km/h)\n"
            "- Solar UV Index: 11.2\n"
            "- Daylight Status: Daylight (06:30 UTC to 18:30 UTC)\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 88 (Moderate (51-100))\n"
            "- PM2.5 Concentration: 29.5 µg/m³\n"
            "- PM10 Concentration: 55.0 µg/m³\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Bandra Bandstand Promenade (park, ~0.2 km away, Suitability Score: 85.0)\n"
            "Temporal Window Analysis: Optimal window around 06:00 UTC (Clear sky, 27°C, UV 1.5)\n"
            "Deterministic Guardrail Assessment: Status=BLOCKED, Mandate=not_recommended\n"
            "Safety Rule Triggers: WHO_UV_EXTREME_STRENUOUS, NOAA_HEAT_INDEX_EXTREME_CAUTION\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Running or Jogging",
            "suitability": "not_recommended",
            "recommended_time_window": "Early morning before 07:30 or post-sunset after 18:00 when UV index drops below 3.0.",
            "precautions": [
                "Avoid midday strenuous cardio in open direct sunlight.",
                "Apply SPF 50+ broad-spectrum sunscreen and wear UV-protective sunglasses.",
                "Maintain continuous electrolyte hydration due to 39.5°C apparent heat index."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Extreme UV Radiation Hazard] Measured UV Index of 11.2 exceeds WHO Extreme threshold (>=11.0).",
                "Midday running at 34.0°C and 72% humidity risks acute solar erythema and exertional heat stress."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Solar UV Index: 11.2",
                "Temperature: 34.0°C (Apparent: 39.5°C)",
                "Relative Humidity: 72%",
                "Bandra Bandstand Promenade (0.2 km away)"
            ],
            "safety_assessment": {
                "guardrail_status": "BLOCKED",
                "triggered_rules": ["WHO_UV_EXTREME_STRENUOUS", "NOAA_HEAT_INDEX_EXTREME_CAUTION"],
                "override_applied": False,
                "source_guidelines": [
                    "World Health Organization (WHO) & US EPA Global Solar UV Index",
                    "NOAA Heat Index Scale & ACSM Hydration Protocols"
                ]
            },
            "candidate_places": [
                {
                    "name": "Bandra Bandstand Promenade",
                    "category": "park",
                    "distance_km": 0.2,
                    "suitability_score": 85.0,
                    "source_evidence": "Mapped in OpenStreetMap (park)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "UV forecast assumes unshaded open horizon.",
                "Microclimate marine breeze may fluctuate near coastline."
            ]
        }
    },
    # 2. Cycling in Thunderstorm & Wet Road
    {
        "id": "train_02_cycling_thunderstorm",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Cycling (Intensity: high)\n"
            "Target Location: Lonavala Ghats, Maharashtra\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 22.0 °C (Apparent: 23.0 °C)\n"
            "- Relative Humidity: 88 %\n"
            "- Weather Description: Thunderstorm with slight hail (WMO Code: 96)\n"
            "- Precipitation Probability: 85 %\n"
            "- Wind Speed: 38.0 km/h (Gusts: 58.0 km/h)\n"
            "- Solar UV Index: 2.1\n"
            "- Daylight Status: Daylight\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 35 (Good (0-50))\n"
            "- PM2.5 Concentration: 8.5 µg/m³\n"
            "- PM10 Concentration: 14.0 µg/m³\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Old Mumbai-Pune Highway Ghat (highway=cycleway, ~1.2 km away, Suitability Score: 60.0)\n"
            "Temporal Window Analysis: Optimal window in 4 hours post-convective clearing.\n"
            "Deterministic Guardrail Assessment: Status=BLOCKED, Mandate=not_recommended\n"
            "Safety Rule Triggers: NWS_LIGHTNING_THUNDERSTORM_RULE, STRONG_WIND_CAUTION\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Cycling",
            "suitability": "not_recommended",
            "recommended_time_window": "No safe window currently. Postpone ride until thunderstorm cell clears and roadway dries.",
            "precautions": [
                "Seek immediate solid building shelter; avoid stopping beneath isolated trees.",
                "Do not ride on slick ghat descents during active precipitation.",
                "Ensure front and rear flashing lights are active if caught on transit."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Active Thunderstorm and Lightning Threat] WMO weather code 96 indicates active thunderstorm with hail.",
                "Wind gusts of 58.0 km/h and wet road surfaces severely compromise cycling balance and braking safety."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Weather Code: 96 (Thunderstorm with hail)",
                "Precipitation Probability: 85%",
                "Wind Gusts: 58.0 km/h",
                "Relative Humidity: 88%"
            ],
            "safety_assessment": {
                "guardrail_status": "BLOCKED",
                "triggered_rules": ["NWS_LIGHTNING_THUNDERSTORM_RULE", "STRONG_WIND_CAUTION"],
                "override_applied": False,
                "source_guidelines": [
                    "US National Weather Service (NWS) Lightning Safety Council",
                    "NWS Wind Safety Guidelines"
                ]
            },
            "candidate_places": [
                {
                    "name": "Old Mumbai-Pune Highway Ghat",
                    "category": "highway",
                    "distance_km": 1.2,
                    "suitability_score": 60.0,
                    "source_evidence": "Mapped in OpenStreetMap (highway=cycleway)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Mountain convective storms exhibit rapid local intensification not captured in regional grid scale."
            ]
        }
    },
    # 3. Picnic with Missing Location
    {
        "id": "train_03_picnic_missing_location",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Picnic or Park Outing (Intensity: low)\n"
            "Target Location: [NOT PROVIDED]\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: N/A\n"
            "- Solar UV Index: N/A\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: N/A\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "No specific outdoor venues mapped within immediate search radius.\n"
            "Identified Missing Inputs: Location coordinates and place query were both omitted.\n"
            "Deterministic Guardrail Assessment: Status=ADVISORY, Mandate=caution\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Picnic or Park Outing",
            "suitability": "caution",
            "recommended_time_window": "Unable to evaluate weather without a confirmed location.",
            "precautions": [
                "Waterproof ground mat or picnic blanket",
                "Insulated cooler box for food preservation",
                "Insect repellent and biodegradable trash bags"
            ],
            "reasons": [
                "A valid geographic location is required to query live meteorological forecasts and discover nearby venues.",
                "No location could be resolved from the supplied request."
            ],
            "missing_information_identified": [
                "Location coordinates and place query were both omitted."
            ],
            "grounded_facts_used": [],
            "safety_assessment": {
                "guardrail_status": "ADVISORY",
                "triggered_rules": ["MISSING_LOCATION_INPUT"],
                "override_applied": False,
                "source_guidelines": ["GeoAI Input Verification Protocol"]
            },
            "candidate_places": [],
            "data_sources_used": [],
            "uncertainties_and_limitations": [
                "Cannot verify local atmospheric conditions or geographic hazards without location coordinates."
            ]
        }
    },
    # 4. Hiking in Hazardous Air Quality (AQI 340)
    {
        "id": "train_04_hiking_hazardous_aqi",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Hiking or Nature Walking (Intensity: high)\n"
            "Target Location: Asola Bhatti Wildlife Sanctuary, New Delhi\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 26.0 °C (Apparent: 26.5 °C)\n"
            "- Relative Humidity: 55 %\n"
            "- Weather Description: Overcast (WMO Code: 3)\n"
            "- Precipitation Probability: 10 %\n"
            "- Wind Speed: 8.0 km/h (Gusts: 12.0 km/h)\n"
            "- Solar UV Index: 3.2\n"
            "- Daylight Status: Daylight\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 345 (Hazardous (>300))\n"
            "- PM2.5 Concentration: 295.0 µg/m³\n"
            "- PM10 Concentration: 410.0 µg/m³\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Sanctuary Nature Trail (highway=path, ~0.5 km away, Suitability Score: 80.0)\n"
            "Temporal Window Analysis: AQI projected to remain above 300 for next 24 hours.\n"
            "Deterministic Guardrail Assessment: Status=BLOCKED, Mandate=not_recommended\n"
            "Safety Rule Triggers: EPA_AQI_HAZARDOUS\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Hiking or Nature Walking",
            "suitability": "not_recommended",
            "recommended_time_window": "No safe outdoor window currently available. Indoor alternatives recommended.",
            "precautions": [
                "Remain indoors with certified HEPA filtration active.",
                "Avoid all strenuous outdoor cardiovascular exercise.",
                "Wear an N95 or equivalent respirator if essential transit is required."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Hazardous Air Quality Emergency] Measured AQI of 345 and PM2.5 of 295.0 µg/m³ falls in the EPA Hazardous zone (>300).",
                "Deep breathing during strenuous hiking causes severe particulate penetration into pulmonary alveolar tissues."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "US EPA AQI: 345 (Hazardous)",
                "PM2.5: 295.0 µg/m³",
                "PM10: 410.0 µg/m³",
                "Temperature: 26.0°C"
            ],
            "safety_assessment": {
                "guardrail_status": "BLOCKED",
                "triggered_rules": ["EPA_AQI_HAZARDOUS"],
                "override_applied": False,
                "source_guidelines": [
                    "United States Environmental Protection Agency (US EPA) Air Quality Index (40 CFR Part 58)"
                ]
            },
            "candidate_places": [
                {
                    "name": "Sanctuary Nature Trail",
                    "category": "highway",
                    "distance_km": 0.5,
                    "suitability_score": 80.0,
                    "source_evidence": "Mapped in OpenStreetMap (highway=path)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Air Quality API",
                    "service_type": "Atmospheric Composition & AQI",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Air quality data by Open-Meteo.com / CAMS / NOAA"
                }
            ],
            "uncertainties_and_limitations": [
                "Ground particulate concentrations may vary based on local biomass burning or traffic inversions."
            ]
        }
    },
    # 5. Golf with Impending Convective Cell
    {
        "id": "train_05_golf_lightning_risk",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Golf (Intensity: moderate)\n"
            "Target Location: Willingdon Sports Club, Mumbai\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 29.0 °C (Apparent: 33.0 °C)\n"
            "- Relative Humidity: 78 %\n"
            "- Weather Description: Thunderstorm: slight or moderate (WMO Code: 95)\n"
            "- Precipitation Probability: 70 %\n"
            "- Wind Speed: 28.0 km/h (Gusts: 46.0 km/h)\n"
            "- Solar UV Index: 4.5\n"
            "- Daylight Status: Daylight\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 65 (Moderate)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Willingdon Golf Course (leisure=golf_course, ~0.1 km away, Suitability Score: 95.0)\n"
            "Deterministic Guardrail Assessment: Status=BLOCKED, Mandate=not_recommended\n"
            "Safety Rule Triggers: NWS_LIGHTNING_THUNDERSTORM_RULE\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Golf",
            "suitability": "not_recommended",
            "recommended_time_window": "Wait until 30 minutes after last thunder before resuming play.",
            "precautions": [
                "Evacuate open fairways immediately; do not carry metal clubs upright.",
                "Avoid small open rain shelters or standing near tall isolated trees on the course.",
                "Seek enclosed clubhouse shelter conforming to lightning safety standards."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Active Thunderstorm and Lightning Threat] WMO weather code 95 signals active thunderstorm.",
                "Open golf fairways and metallic golf clubs pose severe lightning strike fatality hazards."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Weather Code: 95 (Thunderstorm)",
                "Precipitation Probability: 70%",
                "Willingdon Golf Course (0.1 km away)"
            ],
            "safety_assessment": {
                "guardrail_status": "BLOCKED",
                "triggered_rules": ["NWS_LIGHTNING_THUNDERSTORM_RULE"],
                "override_applied": False,
                "source_guidelines": [
                    "US National Weather Service (NWS) Lightning Safety Council"
                ]
            },
            "candidate_places": [
                {
                    "name": "Willingdon Golf Course",
                    "category": "leisure",
                    "distance_km": 0.1,
                    "suitability_score": 95.0,
                    "source_evidence": "Mapped in OpenStreetMap (leisure=golf_course)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Isolated cloud-to-ground lightning can strike up to 10 miles ahead of the main rain shaft."
            ]
        }
    },
    # 6. Cricket in Ideal Morning Conditions
    {
        "id": "train_06_cricket_ideal_morning",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Cricket (Intensity: high)\n"
            "Target Location: Shivaji Park, Dadar, Mumbai\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 24.5 °C (Apparent: 25.0 °C)\n"
            "- Relative Humidity: 60 %\n"
            "- Weather Description: Mainly clear (WMO Code: 1)\n"
            "- Precipitation Probability: 0 %\n"
            "- Wind Speed: 11.0 km/h (Gusts: 15.0 km/h)\n"
            "- Solar UV Index: 3.5\n"
            "- Daylight Status: Daylight\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 72 (Moderate)\n"
            "- PM2.5 Concentration: 22.0 µg/m³\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Shivaji Park Cricket Ground (leisure=pitch, ~0.0 km away, Suitability Score: 95.0)\n"
            "Temporal Window Analysis: Optimal window 07:00 to 10:30 before afternoon heat.\n"
            "Deterministic Guardrail Assessment: Status=CLEAR, Mandate=recommended\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Cricket",
            "suitability": "recommended",
            "recommended_time_window": "07:00 AM to 10:30 AM before midday UV and temperature peak.",
            "precautions": [
                "Protective batting pads, gloves, and helmet with visor",
                "Studded or rubber cricket shoes suitable for pitch type",
                "Sunscreen reapplied every 2 hours"
            ],
            "reasons": [
                "Weather conditions are mild at 24.5°C with zero rain probability, ideal for turf and pitch stability.",
                "Wind speeds of 11.0 km/h do not interfere with bowling trajectory or outfield fielding."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Temperature: 24.5°C",
                "Precipitation Probability: 0%",
                "Wind Speed: 11.0 km/h",
                "Shivaji Park Cricket Ground (0.0 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "CLEAR",
                "triggered_rules": [],
                "override_applied": False,
                "source_guidelines": ["Standard Outdoor Sports Climatology Protocols"]
            },
            "candidate_places": [
                {
                    "name": "Shivaji Park Cricket Ground",
                    "category": "pitch",
                    "distance_km": 0.0,
                    "suitability_score": 95.0,
                    "source_evidence": "Mapped in OpenStreetMap (leisure=pitch)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Turf pitch hardness depends on unmonitored overnight watering practices."
            ]
        }
    },
    # 7. Outdoor Football in Extreme Heat (41°C)
    {
        "id": "train_07_football_extreme_heat",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Outdoor Football (Intensity: very_high)\n"
            "Target Location: Cooperage Football Ground, Mumbai\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 38.0 °C (Apparent: 42.5 °C)\n"
            "- Relative Humidity: 65 %\n"
            "- Weather Description: Clear sky (WMO Code: 0)\n"
            "- Precipitation Probability: 0 %\n"
            "- Wind Speed: 8.0 km/h\n"
            "- Solar UV Index: 9.5\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 85 (Moderate)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Cooperage Stadium Pitch (leisure=pitch, ~0.1 km away, Suitability Score: 92.0)\n"
            "Deterministic Guardrail Assessment: Status=BLOCKED, Mandate=not_recommended\n"
            "Safety Rule Triggers: NOAA_HEAT_INDEX_DANGER, WHO_UV_VERY_HIGH\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Outdoor Football",
            "suitability": "not_recommended",
            "recommended_time_window": "Reschedule match to evening after 18:30 under artificial floodlights.",
            "precautions": [
                "Firm-ground or soft-ground studded football boots",
                "Shin guards conforming to FIFA safety standards",
                "Hydration fluids for pre-game and half-time intervals"
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Dangerous Heat Stress / Hyperthermia Risk] Apparent temperature of 42.5°C falls into NOAA Danger category (>=40°C).",
                "High-intensity football sprinting under severe heat index impedes sweat evaporation and drastically increases heat stroke incidence."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Temperature: 38.0°C (Apparent: 42.5°C)",
                "Relative Humidity: 65%",
                "UV Index: 9.5",
                "Cooperage Stadium Pitch (0.1 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "BLOCKED",
                "triggered_rules": ["NOAA_HEAT_INDEX_DANGER", "WHO_UV_VERY_HIGH"],
                "override_applied": False,
                "source_guidelines": [
                    "US National Oceanic and Atmospheric Administration (NOAA) & American College of Sports Medicine (ACSM)",
                    "WHO / US EPA UV Index Guidelines"
                ]
            },
            "candidate_places": [
                {
                    "name": "Cooperage Stadium Pitch",
                    "category": "pitch",
                    "distance_km": 0.1,
                    "suitability_score": 92.0,
                    "source_evidence": "Mapped in OpenStreetMap (leisure=pitch)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Artificial turf pitch surfaces can heat up 10-15°C higher than ambient air temperature."
            ]
        }
    },
    # 8. Outdoor Tennis in High Wind
    {
        "id": "train_08_tennis_gale_wind",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Outdoor Tennis (Intensity: high)\n"
            "Target Location: Gymkhana Courts, Pune\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 25.0 °C (Apparent: 25.0 °C)\n"
            "- Weather Description: Partly cloudy (WMO Code: 2)\n"
            "- Precipitation Probability: 15 %\n"
            "- Wind Speed: 32.0 km/h (Gusts: 52.0 km/h)\n"
            "- Solar UV Index: 5.0\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 55 (Moderate)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Gymkhana Tennis Courts (leisure=pitch, ~0.2 km away, Suitability Score: 88.0)\n"
            "Deterministic Guardrail Assessment: Status=ADVISORY, Mandate=caution\n"
            "Safety Rule Triggers: STRONG_WIND_CAUTION\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Outdoor Tennis",
            "suitability": "caution",
            "recommended_time_window": "Late afternoon when regional gusts drop below 25 km/h.",
            "precautions": [
                "Non-marking tennis court shoes with lateral support",
                "Sweatbands, cap, and electrolyte beverage",
                "Adjust shot margin of error and expect severe ball drift in gusty rallies."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [High Crosswinds Caution] Measured wind gusts of 52.0 km/h exceed the 40 km/h threshold for tennis safety.",
                "Wind gusts distort racquet ball trajectory and toss consistency on outdoor courts."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Wind Speed: 32.0 km/h (Gusts: 52.0 km/h)",
                "Temperature: 25.0°C",
                "Gymkhana Tennis Courts (0.2 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "ADVISORY",
                "triggered_rules": ["STRONG_WIND_CAUTION"],
                "override_applied": False,
                "source_guidelines": ["NWS Wind Safety Guidelines"]
            },
            "candidate_places": [
                {
                    "name": "Gymkhana Tennis Courts",
                    "category": "pitch",
                    "distance_km": 0.2,
                    "suitability_score": 88.0,
                    "source_evidence": "Mapped in OpenStreetMap (leisure=pitch)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Wind screens around perimeter fence may partially dampen court-level wind speed."
            ]
        }
    },
    # 9. Outdoor Event with Older Adults & Mobility Aids
    {
        "id": "train_09_event_older_adults",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Outdoor Event or Gathering (Intensity: low)\n"
            "Target Location: Cubbon Park, Bengaluru\n"
            "User Stated Constraints: {\"age\": 74, \"mobility\": \"walking cane\", \"health_notes\": \"knee osteoarthritis\"}\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 23.0 °C (Apparent: 23.0 °C)\n"
            "- Relative Humidity: 55 %\n"
            "- Weather Description: Partly cloudy (WMO Code: 2)\n"
            "- Precipitation Probability: 10 %\n"
            "- Wind Speed: 10.0 km/h\n"
            "- Solar UV Index: 4.0\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 58 (Moderate)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Cubbon Park Central Lawn (leisure=park, ~0.3 km away, Suitability Score: 90.0)\n"
            "Deterministic Guardrail Assessment: Status=ADVISORY, Mandate=caution\n"
            "Safety Rule Triggers: AGE_VULNERABILITY_ADVISORY\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Outdoor Event or Gathering",
            "suitability": "caution",
            "recommended_time_window": "Mid-morning (09:00 to 11:30) or late afternoon under tree canopy shade.",
            "precautions": [
                "Select venues with paved, barrier-free accessible pathways suitable for walking canes.",
                "Ensure ample seating benches are accessible every 50-100 meters for knee joint relief.",
                "Provide easily accessible hydration and shaded rest zones."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Thermoregulatory Sensitivity Advisory] Individuals aged 70+ require planned rest intervals and shaded seating.",
                "Pleasant 23.0°C temperature is comfortable, but terrain accessibility is critical for knee osteoarthritis."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "User Profile: Age 74, Walking cane, Knee osteoarthritis",
                "Temperature: 23.0°C",
                "Cubbon Park Central Lawn (0.3 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "ADVISORY",
                "triggered_rules": ["AGE_VULNERABILITY_ADVISORY"],
                "override_applied": False,
                "source_guidelines": [
                    "World Health Organization (WHO) Age-Friendly Outdoor Spaces Guidelines"
                ]
            },
            "candidate_places": [
                {
                    "name": "Cubbon Park Central Lawn",
                    "category": "park",
                    "distance_km": 0.3,
                    "suitability_score": 90.0,
                    "source_evidence": "Mapped in OpenStreetMap (leisure=park)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Park pathway gradient and cobblestone irregularities are not fully modeled in 2D OSM tags."
            ]
        }
    },
    # 10. Outdoor Photography at Golden Hour
    {
        "id": "train_10_photography_golden_hour",
        "split": "train",
        "user_prompt": (
            "Activity to Evaluate: Outdoor Photography (Intensity: low)\n"
            "Target Location: Marine Drive Promenade, Mumbai\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 28.0 °C (Apparent: 30.0 °C)\n"
            "- Relative Humidity: 68 %\n"
            "- Weather Description: Mainly clear (WMO Code: 1)\n"
            "- Precipitation Probability: 0 %\n"
            "- Wind Speed: 15.0 km/h (Gusts: 22.0 km/h)\n"
            "- Solar UV Index: 1.5\n"
            "- Daylight Status: Daylight (sunset at 18:15 UTC)\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 65 (Moderate)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Marine Drive Seawall Viewpoint (tourism=viewpoint, ~0.1 km away, Suitability Score: 95.0)\n"
            "Temporal Window Analysis: Optimal golden hour between 17:15 and 18:20.\n"
            "Deterministic Guardrail Assessment: Status=CLEAR, Mandate=recommended\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Outdoor Photography",
            "suitability": "recommended",
            "recommended_time_window": "Golden hour from 17:15 to 18:20 leading into sunset.",
            "precautions": [
                "Weather-sealed camera body or silicone rain cover",
                "Sturdy tripod with hook weight for windy vantage points",
                "Maintain situational footing awareness along the concrete tetrapod promenade."
            ],
            "reasons": [
                "Clear skies and late afternoon sun angle provide optimal illumination with minimal UV exposure.",
                "Wind speeds of 15.0 km/h are within safe tripod stability limits."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Temperature: 28.0°C",
                "UV Index: 1.5",
                "Wind Speed: 15.0 km/h",
                "Marine Drive Seawall Viewpoint (0.1 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "CLEAR",
                "triggered_rules": [],
                "override_applied": False,
                "source_guidelines": ["Optics and Environmental Photography Standards"]
            },
            "candidate_places": [
                {
                    "name": "Marine Drive Seawall Viewpoint",
                    "category": "tourism",
                    "distance_km": 0.1,
                    "suitability_score": 95.0,
                    "source_evidence": "Mapped in OpenStreetMap (tourism=viewpoint)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Marine spray may deposit airborne saline mist on glass elements during high tide."
            ]
        }
    },
    # 11. Eval Example: Running in Heavy Rain & Wind
    {
        "id": "eval_01_running_heavy_rain",
        "split": "eval",
        "user_prompt": (
            "Activity to Evaluate: Running or Jogging (Intensity: very_high)\n"
            "Target Location: Worli Sea Face, Mumbai\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 24.0 °C (Apparent: 25.0 °C)\n"
            "- Weather Description: Heavy rain showers (WMO Code: 82)\n"
            "- Precipitation Probability: 90 %\n"
            "- Wind Speed: 42.0 km/h (Gusts: 62.0 km/h)\n"
            "- Solar UV Index: 1.0\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 30 (Good)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Worli Sea Face Promenade (park, ~0.1 km away, Suitability Score: 65.0)\n"
            "Deterministic Guardrail Assessment: Status=BLOCKED, Mandate=not_recommended\n"
            "Safety Rule Triggers: BEAUFORT_GALE_WIND_RULE, HIGH_PRECIPITATION_RISK\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Running or Jogging",
            "suitability": "not_recommended",
            "recommended_time_window": "Postpone until storm band subsides and wind gusts fall below 35 km/h.",
            "precautions": [
                "Avoid exposed coastal sea walls prone to high wave overtopping.",
                "Do not run on slick polished tiles or wet manhole covers.",
                "Wear high-visibility waterproof shell if outdoor travel is unavoidable."
            ],
            "reasons": [
                "MANDATORY SAFETY RULE: [Gale-Force Wind Hazard] Measured wind gusts of 62.0 km/h present significant danger of falling branches and sea wave spray.",
                "Precipitation probability of 90% and torrential rain showers impair visibility and footing."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Weather Code: 82 (Heavy rain showers)",
                "Precipitation Probability: 90%",
                "Wind Gusts: 62.0 km/h",
                "Worli Sea Face Promenade (0.1 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "BLOCKED",
                "triggered_rules": ["BEAUFORT_GALE_WIND_RULE", "HIGH_PRECIPITATION_RISK"],
                "override_applied": False,
                "source_guidelines": [
                    "WMO Beaufort Wind Scale & NWS High Wind Advisory",
                    "Sport Surface Traction Safety"
                ]
            },
            "candidate_places": [
                {
                    "name": "Worli Sea Face Promenade",
                    "category": "park",
                    "distance_km": 0.1,
                    "suitability_score": 65.0,
                    "source_evidence": "Mapped in OpenStreetMap (park)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Coastal tide surges may breach low promenade barriers."
            ]
        }
    },
    # 12. Eval Example: Picnic in Pleasant Weather
    {
        "id": "eval_02_picnic_pleasant_afternoon",
        "split": "eval",
        "user_prompt": (
            "Activity to Evaluate: Picnic or Park Outing (Intensity: low)\n"
            "Target Location: Lodhi Gardens, New Delhi\n"
            "--- LIVE RETRIEVED METEOROLOGICAL EVIDENCE ---\n"
            "- Current Temperature: 22.0 °C (Apparent: 22.0 °C)\n"
            "- Relative Humidity: 48 %\n"
            "- Weather Description: Clear sky (WMO Code: 0)\n"
            "- Precipitation Probability: 0 %\n"
            "- Wind Speed: 9.0 km/h\n"
            "- Solar UV Index: 4.2\n"
            "--- LIVE RETRIEVED AIR QUALITY EVIDENCE ---\n"
            "- US EPA AQI: 75 (Moderate)\n"
            "--- GEOSPATIAL VENUES (OpenStreetMap) ---\n"
            "1. Lodhi Gardens Central Lawns (leisure=park, ~0.1 km away, Suitability Score: 96.0)\n"
            "Deterministic Guardrail Assessment: Status=CLEAR, Mandate=recommended\n"
            "Provide a grounded outdoor recommendation in the required JSON schema format."
        ),
        "target_output": {
            "activity_name": "Picnic or Park Outing",
            "suitability": "recommended",
            "recommended_time_window": "Late morning through early afternoon (11:00 AM to 03:30 PM).",
            "precautions": [
                "Waterproof ground mat or picnic blanket",
                "Insulated cooler box for food preservation",
                "Insect repellent and biodegradable trash bags"
            ],
            "reasons": [
                "Comfortable 22.0°C temperature, zero precipitation risk, and low wind provide ideal park picnic conditions.",
                "Lodhi Gardens offers ample grassy lawns with mature tree shade canopy."
            ],
            "missing_information_identified": [],
            "grounded_facts_used": [
                "Temperature: 22.0°C",
                "Precipitation Probability: 0%",
                "UV Index: 4.2",
                "Lodhi Gardens Central Lawns (0.1 km)"
            ],
            "safety_assessment": {
                "guardrail_status": "CLEAR",
                "triggered_rules": [],
                "override_applied": False,
                "source_guidelines": ["Recreational Park Guidelines"]
            },
            "candidate_places": [
                {
                    "name": "Lodhi Gardens Central Lawns",
                    "category": "park",
                    "distance_km": 0.1,
                    "suitability_score": 96.0,
                    "source_evidence": "Mapped in OpenStreetMap (leisure=park)"
                }
            ],
            "data_sources_used": [
                {
                    "provider": "Open-Meteo Weather Forecast API",
                    "service_type": "Numerical Weather Prediction",
                    "retrieval_timestamp": "2026-10-23T12:00:00Z",
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
                }
            ],
            "uncertainties_and_limitations": [
                "Late evening temperatures may drop rapidly during winter months."
            ]
        }
    }
]


def prepare_datasets(output_dir: str = "datasets"):
    """Generates JSONL dataset files for training and held-out evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_recommendations.jsonl")
    eval_path = os.path.join(output_dir, "eval_recommendations.jsonl")

    train_count = 0
    eval_count = 0

    with open(train_path, "w", encoding="utf-8") as f_train, open(eval_path, "w", encoding="utf-8") as f_eval:
        for ex in RAW_EXAMPLES:
            # Validate schema
            is_valid, errors = validate_schema(ex["target_output"])
            if not is_valid:
                raise ValueError(f"Schema validation failed for example {ex['id']}: {errors}")

            # Format in ChatML messages format
            chatml_record = {
                "id": ex["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["user_prompt"]},
                    {"role": "assistant", "content": json.dumps(ex["target_output"], indent=2)}
                ]
            }

            line = json.dumps(chatml_record, ensure_ascii=False) + "\n"
            if ex["split"] == "train":
                f_train.write(line)
                train_count += 1
            else:
                f_eval.write(line)
                eval_count += 1

    print(f"Dataset preparation complete:")
    print(f"  - Train records ({train_path}): {train_count}")
    print(f"  - Eval records ({eval_path}): {eval_count}")
    print(f"  - All records validated strictly against recommendation.schema.json.")


if __name__ == "__main__":
    prepare_datasets()
