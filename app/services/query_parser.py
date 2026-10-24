"""Natural language query parser for free-form outdoor recommendation questions.

Extracts activity, location, search radius, time preference, and user constraints
from free-form user questions conforming to the GeoAI Hackathon 2026 rulebook.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from app.geospatial.activities import ACTIVITIES, get_activity_profile


# Keyword mappings to canonical activity slugs
ACTIVITY_PATTERNS = [
    # Running / Jogging
    (r"\b(running|runner|run|jogging|jog|jogger|sprint|marathon)\b", "running_jogging"),
    # Cycling
    (r"\b(cycling|cyclist|cycle|bike|biking|bicycle|commute by bike)\b", "cycling"),
    # Golf
    (r"\b(golf|golfing|golf course|driving range)\b", "golf"),
    # Cricket
    (r"\b(cricket|cricket match|cricket practice|nets|batting|bowling)\b", "cricket"),
    # Football
    (r"\b(football|soccer|five[\s-]a[\s-]side|futsal|turf football)\b", "outdoor_football"),
    # Tennis
    (r"\b(tennis|tennis match|tennis court)\b", "outdoor_tennis"),
    # Hiking / Walking
    (r"\b(hiking|hike|trek|trekking|nature walk|trail walk|morning walk|evening walk|walk outdoors|walk|walking)\b", "hiking_nature_walking"),
    # Picnic
    (r"\b(picnic|park outing|family picnic|picnicking)\b", "picnic_park_outing"),
    # Outdoor Event / Gathering
    (r"\b(birthday party|party|wedding|fair|outdoor event|gathering|celebration|concert|festival)\b", "outdoor_event_gathering"),
    # Photography
    (r"\b(sunset photos|photo session|photography|photoshoot|take photos|photographs|golden hour)\b", "outdoor_photography"),
]


def extract_activity(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts activity identifier and display name from free-form text."""
    lower_text = text.lower()
    
    # 1. First test full exact matches or specific activity phrases
    for pattern, slug in ACTIVITY_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            profile = ACTIVITIES.get(slug)
            if profile:
                return slug, profile.name

    # 2. Try generic get_activity_profile matching
    profile = get_activity_profile(text)
    if profile:
        return profile.id, profile.name

    return None, None


def extract_radius(text: str) -> Optional[float]:
    """Extracts explicit radius from phrases like 'within 15 km', 'radius of 20km', '40 km of'."""
    pattern = r"(?:within|radius(?:\s+of)?|upto|up\s+to|around)\s+(\d+(?:\.\d+)?)\s*(?:km|kilometers|kilometres|k\.m\.)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
            
    # Pattern like "40 km of Andheri"
    pattern_alt = r"(\d+(?:\.\d+)?)\s*(?:km|kilometers|kilometres|k\.m\.)\s+(?:of|from|around)"
    match_alt = re.search(pattern_alt, text, re.IGNORECASE)
    if match_alt:
        try:
            return float(match_alt.group(1))
        except ValueError:
            pass

    return None


def extract_location(text: str) -> Optional[str]:
    """Extracts candidate location/placename from free-form query."""
    clean = text.strip()
    
    # Exclude time-like phrases from candidate locations
    time_terms = {"work", "home", "the", "a", "an", "weekend", "today", "tomorrow", "yesterday", "am", "pm", "morning", "evening", "afternoon", "night", "noon"}

    location_regexes = [
        r"(?:within\s+\d+(?:\.\d+)?\s*(?:km|kilometers|kilometres)?\s+of\s+)([A-Za-z0-9\s\-]+?)(?:\?|\.|\,|(?:\s+(?:this|tomorrow|today|on|for|with|in\s+the|at\s+\d+))|$)",
        r"(?:\d+(?:\.\d+)?\s*(?:km|kilometers|kilometres)?\s+of\s+)([A-Za-z0-9\s\-]+?)(?:\?|\.|\,|(?:\s+(?:this|tomorrow|today|on|for|with|in\s+the|at\s+\d+))|$)",
        r"(?:near|around|in)\s+([A-Z][A-Za-z0-9\s\-]+?)(?:\?|\.|\,|(?:\s+(?:tomorrow|today|this|on|in\s+the|for|with|after|at\s+\d+))|$)",
        r"(?:near|around|in)\s+([a-zA-Z0-9\s\-]+?)(?:\?|\.|\,|(?:\s+(?:tomorrow|today|this|on|in\s+the|for|with|after|at\s+\d+))|$)",
        r"(?:at)\s+([A-Z][A-Za-z0-9\s\-]+?)(?:\?|\.|\,|(?:\s+(?:tomorrow|today|this|on|in\s+the|for|with|after|at\s+\d+))|$)",
    ]
    
    for pat in location_regexes:
        match = re.search(pat, clean)
        if match:
            candidate = match.group(1).strip()
            # Clean up trailing prepositions or stop words
            candidate = re.sub(r"\b(this|tomorrow|today|evening|morning|afternoon|weekend|after|work|at\s+\d+)\b.*", "", candidate, flags=re.IGNORECASE).strip()
            # Check if candidate is a time phrase like "8 AM" or "8:00 pm"
            if re.match(r"^\d{1,2}(?::\d{2})?\s*(?:am|pm)?$", candidate, re.IGNORECASE):
                continue
            if candidate and len(candidate) > 2 and candidate.lower() not in time_terms:
                return candidate

    return None


def extract_time_preference(text: str) -> Optional[str]:
    """Extracts temporal mentions such as 'tomorrow evening', '8 AM', 'this weekend', etc."""
    time_patterns = [
        r"\b(tomorrow\s+(?:morning|afternoon|evening|night))\b",
        r"\b(today\s+(?:morning|afternoon|evening|night|after\s+work))\b",
        r"\b(after\s+work\s+today)\b",
        r"\b(this\s+evening|this\s+morning|this\s+afternoon|tonight)\b",
        r"\b(this\s+weekend|next\s+weekend)\b",
        r"\b((?:on\s+)?(?:saturday|sunday|monday|tuesday|wednesday|thursday|friday)(?:\s+(?:morning|afternoon|evening|night))?)\b",
        r"\b(in\s+the\s+next\s+(?:two|2|three|3|few)\s+days)\b",
        r"\b(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b",
        r"\b(sunset|sunrise|golden\s+hour)\b",
        r"\b(tomorrow|today)\b"
    ]
    
    found = []
    for pat in time_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            found.append(match.group(1).strip())
            
    if found:
        return ", ".join(dict.fromkeys(found))
    return None


def extract_user_constraints(text: str) -> Dict[str, Any]:
    """Extracts demographic and health notes from free-form text."""
    lower = text.lower()
    constraints: Dict[str, Any] = {}
    
    # 1. Health conditions
    health_notes = []
    if "asthma" in lower:
        health_notes.append("asthma (sensitive to particulate matter and cold/dry air)")
    if "allerg" in lower:
        health_notes.append("seasonal allergies / pollen sensitivity")
    if "heart" in lower or "cardiac" in lower:
        health_notes.append("cardiac condition (requires moderate exertion limits)")
    if "knee" in lower or "joint" in lower or "arthritis" in lower:
        health_notes.append("joint / arthritis sensitivity (flat paved surfaces preferred)")
        
    # 2. Age / Demographic
    if re.search(r"\b(kid|kids|child|children)\b", lower):
        constraints["age"] = 10
        health_notes.append("young children / youth activity")
    elif re.search(r"\b(parent|parents|senior|seniors|elderly|older adult|grandmother|grandfather)\b", lower):
        constraints["age"] = 70
        health_notes.append("senior citizen / older adult")
    elif match_age := re.search(r"\b(\d{1,2})\s*[- ]*(?:year|yr)s?[- ]*old\b", lower):
        try:
            constraints["age"] = int(match_age.group(1))
        except ValueError:
            pass

    # 3. Mobility restrictions
    if "wheelchair" in lower:
        constraints["mobility"] = "wheelchair (requires paved, step-free access)"
    elif "cane" in lower or "walking stick" in lower:
        constraints["mobility"] = "walking cane (requires even paved terrain with resting benches)"
    elif "stroller" in lower or "pram" in lower:
        constraints["mobility"] = "stroller (requires paved, wide pathways)"

    if health_notes:
        constraints["health_notes"] = "; ".join(health_notes)

    return constraints


def parse_free_form_query(raw_text: str) -> Dict[str, Any]:
    """Parses a free-form question into structured recommendation parameters."""
    act_slug, act_name = extract_activity(raw_text)
    location = extract_location(raw_text)
    radius = extract_radius(raw_text) or 5.0
    time_pref = extract_time_preference(raw_text)
    constraints = extract_user_constraints(raw_text)

    return {
        "activity_slug": act_slug,
        "activity_name": act_name,
        "location": location,
        "radius_km": radius,
        "time_preference": time_pref,
        "constraints": constraints,
        "raw_query": raw_text.strip()
    }
