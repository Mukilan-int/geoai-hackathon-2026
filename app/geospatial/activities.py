"""Activity definitions, requirements, and safety profiles for the 10 required activities."""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ActivityProfile:
    id: str
    name: str
    category: str
    intensity: str  # "low", "moderate", "high", "very_high"
    requires_daylight: bool
    max_rain_prob_pct: int
    max_wind_kmh: float
    max_wind_gust_kmh: float
    max_uv_index: float
    max_apparent_temp_c: float
    min_apparent_temp_c: float
    max_us_aqi: int
    osm_tags: List[str]
    equipment_precautions: List[str]
    safety_notes: str


ACTIVITIES: Dict[str, ActivityProfile] = {
    "running_jogging": ActivityProfile(
        id="running_jogging",
        name="Running or Jogging",
        category="cardio_endurance",
        intensity="very_high",
        requires_daylight=False,
        max_rain_prob_pct=50,
        max_wind_kmh=40.0,
        max_wind_gust_kmh=55.0,
        max_uv_index=8.0,
        max_apparent_temp_c=35.0,
        min_apparent_temp_c=0.0,
        max_us_aqi=150,
        osm_tags=["highway=footway", "highway=path", "leisure=park", "leisure=track"],
        equipment_precautions=[
            "High-traction running footwear",
            "Electrolyte hydration pack for warm conditions",
            "Reflective gear if running in low light",
            "Broad-spectrum SPF 50+ sunscreen"
        ],
        safety_notes="High metabolic heat generation; extremely vulnerable to heat stroke in high wet-bulb temperatures and UV midday peaks."
    ),
    "cycling": ActivityProfile(
        id="cycling",
        name="Cycling",
        category="road_and_trail",
        intensity="high",
        requires_daylight=False,
        max_rain_prob_pct=40,
        max_wind_kmh=35.0,
        max_wind_gust_kmh=45.0,
        max_uv_index=9.0,
        max_apparent_temp_c=36.0,
        min_apparent_temp_c=2.0,
        max_us_aqi=150,
        osm_tags=["highway=cycleway", "highway=path", "leisure=track", "route=bicycle"],
        equipment_precautions=[
            "Certified safety helmet",
            "Front and rear LED bike lights",
            "Puncture repair kit and mini-pump",
            "High-grip gloves and eye protection"
        ],
        safety_notes="Strong crosswinds (>40 km/h) destabilize bicycles. Wet asphalt reduces braking friction drastically; avoid painted road markings in rain."
    ),
    "golf": ActivityProfile(
        id="golf",
        name="Golf",
        category="sports",
        intensity="moderate",
        requires_daylight=True,
        max_rain_prob_pct=45,
        max_wind_kmh=35.0,
        max_wind_gust_kmh=48.0,
        max_uv_index=9.0,
        max_apparent_temp_c=37.0,
        min_apparent_temp_c=5.0,
        max_us_aqi=160,
        osm_tags=["leisure=golf_course", "leisure=pitch;sport=golf"],
        equipment_precautions=[
            "Wide-brimmed golf hat and UV-blocking sunglasses",
            "Spiked golf shoes for turf stability",
            "Ample drinking water for 3-4 hours of open exposure"
        ],
        safety_notes="Open fairways and metallic clubs present critical lightning hazard during convective storms. Evacuate immediately if thunder is audible."
    ),
    "cricket": ActivityProfile(
        id="cricket",
        name="Cricket",
        category="sports",
        intensity="high",
        requires_daylight=True,
        max_rain_prob_pct=30,
        max_wind_kmh=35.0,
        max_wind_gust_kmh=50.0,
        max_uv_index=9.0,
        max_apparent_temp_c=38.0,
        min_apparent_temp_c=8.0,
        max_us_aqi=160,
        osm_tags=["leisure=pitch;sport=cricket", "leisure=pitch", "leisure=sports_centre"],
        equipment_precautions=[
            "Protective batting pads, gloves, and helmet with visor",
            "Studded or rubber cricket shoes suitable for pitch type",
            "Sunscreen reapplied every 2 hours"
        ],
        safety_notes="Wet turf compromises ball grip, bowling run-up traction, and wicket bounce unpredictability. Rain halts play under standard MCC laws."
    ),
    "outdoor_football": ActivityProfile(
        id="outdoor_football",
        name="Outdoor Football",
        category="team_sports",
        intensity="very_high",
        requires_daylight=False,
        max_rain_prob_pct=50,
        max_wind_kmh=45.0,
        max_wind_gust_kmh=60.0,
        max_uv_index=8.5,
        max_apparent_temp_c=35.0,
        min_apparent_temp_c=0.0,
        max_us_aqi=150,
        osm_tags=["leisure=pitch;sport=soccer", "leisure=pitch", "leisure=stadium"],
        equipment_precautions=[
            "Firm-ground or soft-ground studded football boots",
            "Shin guards conforming to FIFA safety standards",
            "Hydration fluids for pre-game and half-time intervals"
        ],
        safety_notes="Continuous sprinting induces rapid dehydration. Avoid waterlogged pitches to prevent ACL/ankle ligament sprains."
    ),
    "outdoor_tennis": ActivityProfile(
        id="outdoor_tennis",
        name="Outdoor Tennis",
        category="racquet_sports",
        intensity="high",
        requires_daylight=True,
        max_rain_prob_pct=25,
        max_wind_kmh=28.0,
        max_wind_gust_kmh=40.0,
        max_uv_index=9.0,
        max_apparent_temp_c=36.0,
        min_apparent_temp_c=5.0,
        max_us_aqi=150,
        osm_tags=["leisure=pitch;sport=tennis", "leisure=sports_centre"],
        equipment_precautions=[
            "Non-marking tennis court shoes with lateral support",
            "Sweatbands, cap, and electrolyte beverage",
            "Spare racquet grips for humid conditions"
        ],
        safety_notes="Hardcourt surfaces become dangerously slippery at the onset of drizzle. Wind over 30 km/h severely distorts ball trajectory."
    ),
    "hiking_nature_walking": ActivityProfile(
        id="hiking_nature_walking",
        name="Hiking or Nature Walking",
        category="outdoor_trekking",
        intensity="high",
        requires_daylight=True,
        max_rain_prob_pct=40,
        max_wind_kmh=40.0,
        max_wind_gust_kmh=55.0,
        max_uv_index=8.0,
        max_apparent_temp_c=34.0,
        min_apparent_temp_c=2.0,
        max_us_aqi=150,
        osm_tags=["highway=path", "highway=trail", "route=hiking", "leisure=nature_reserve"],
        equipment_precautions=[
            "Ankle-supporting hiking boots with deep tread",
            "Offline topo map and navigation compass",
            "Emergency first-aid kit, whistle, and headlamp",
            "At least 2 liters of water per 4 hours of hiking"
        ],
        safety_notes="Trail erosion, loose scree, and reduced visibility occur in precipitation. Mountain summit trails face abrupt microclimate shifts."
    ),
    "picnic_park_outing": ActivityProfile(
        id="picnic_park_outing",
        name="Picnic or Park Outing",
        category="leisure_social",
        intensity="low",
        requires_daylight=True,
        max_rain_prob_pct=35,
        max_wind_kmh=30.0,
        max_wind_gust_kmh=45.0,
        max_uv_index=8.0,
        max_apparent_temp_c=34.0,
        min_apparent_temp_c=10.0,
        max_us_aqi=160,
        osm_tags=["leisure=park", "leisure=garden", "tourism=picnic_site", "amenity=bench"],
        equipment_precautions=[
            "Waterproof ground mat or picnic blanket",
            "Insulated cooler box for food preservation",
            "Insect repellent and biodegradable trash bags"
        ],
        safety_notes="Perishable foods spoil quickly in temperatures above 32°C. Shade availability under trees or gazebos is critical during peak sun."
    ),
    "outdoor_event_gathering": ActivityProfile(
        id="outdoor_event_gathering",
        name="Outdoor Event or Gathering",
        category="community_events",
        intensity="low",
        requires_daylight=False,
        max_rain_prob_pct=30,
        max_wind_kmh=32.0,
        max_wind_gust_kmh=45.0,
        max_uv_index=8.5,
        max_apparent_temp_c=35.0,
        min_apparent_temp_c=8.0,
        max_us_aqi=160,
        osm_tags=["amenity=place_of_worship", "leisure=park", "amenity=community_centre", "tourism=attraction"],
        equipment_precautions=[
            "Canopies or gazebos secured against unexpected gusts",
            "Portable sound equipment with weatherproofing",
            "Accessible crowd hydration points and sanitation facilities"
        ],
        safety_notes="Crowded outdoor events have heightened heat-exhaustion vulnerability. Emergency evacuation routes must be kept unobstructed."
    ),
    "outdoor_photography": ActivityProfile(
        id="outdoor_photography",
        name="Outdoor Photography",
        category="creative_leisure",
        intensity="low",
        requires_daylight=True,
        max_rain_prob_pct=40,
        max_wind_kmh=35.0,
        max_wind_gust_kmh=50.0,
        max_uv_index=9.0,
        max_apparent_temp_c=37.0,
        min_apparent_temp_c=0.0,
        max_us_aqi=180,
        osm_tags=["tourism=viewpoint", "historic=monument", "leisure=nature_reserve", "leisure=park"],
        equipment_precautions=[
            "Weather-sealed camera body or silicone rain cover",
            "Sturdy tripod with hook weight for windy vantage points",
            "Microfiber lens wipes and silica gel packs"
        ],
        safety_notes="Cliffside viewpoints or slippery shorelines require situational vigilance while looking through viewfinders. Golden hour lighting provides optimal exposure."
    )
}


def get_activity_profile(activity_key: str) -> Optional[ActivityProfile]:
    """Retrieves an activity profile by slug, exact name, or distinctive keyword."""
    if not activity_key:
        return None
    raw_clean = activity_key.strip().lower()
    normalized = raw_clean.replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # 1. Direct key match
    if normalized in ACTIVITIES:
        return ACTIVITIES[normalized]
        
    # Also test with common connectors stripped
    cleaned_key = normalized.replace("_or_", "_").replace("_and_", "_")
    if cleaned_key in ACTIVITIES:
        return ACTIVITIES[cleaned_key]

    # 2. Match exact display name
    for prof in ACTIVITIES.values():
        if raw_clean == prof.name.lower():
            return prof

    # 3. Keyword matching ignoring generic stop-words
    stop_words = {"or", "and", "of", "in", "on", "the", "a", "outdoor", "outing"}
    input_words = set(re.findall(r"\b\w+\b", raw_clean)) - stop_words

    for prof in ACTIVITIES.values():
        prof_words = set(re.findall(r"\b\w+\b", prof.name.lower())) - stop_words
        if input_words & prof_words:
            return prof
            
    return None


def list_supported_activities() -> List[Dict[str, str]]:
    """Lists all 10 supported outdoor activities with their metadata."""
    return [
        {
            "id": prof.id,
            "name": prof.name,
            "category": prof.category,
            "intensity": prof.intensity,
            "requires_daylight": prof.requires_daylight,
            "safety_summary": prof.safety_notes
        }
        for prof in ACTIVITIES.values()
    ]
