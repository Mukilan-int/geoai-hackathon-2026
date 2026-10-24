"""Multi-candidate geospatial venues and temporal time-windows ranking engine."""
from typing import Any, Dict, List, Optional
from app.geospatial.activities import ActivityProfile


def rank_candidate_places(
    candidates: List[Dict[str, Any]],
    activity: ActivityProfile,
    origin_lat: float,
    origin_lon: float
) -> List[Dict[str, Any]]:
    """Ranks candidate venues based on activity affinity, distance, and mapped amenities."""
    scored_candidates = []
    
    for cand in candidates:
        score = 70.0  # Base score
        dist_km = cand.get("distance_km") or 5.0
        
        # Proximity score component (shorter distance = higher score)
        # Drops by 3 points per km
        score -= min(dist_km * 3.0, 30.0)
        
        cat = cand.get("category", "").lower()
        name = cand.get("name", "").lower()
        
        # Activity tag synergy
        for tag in activity.osm_tags:
            tag_val = tag.split("=")[-1] if "=" in tag else tag
            if tag_val in cat or tag_val in name:
                score += 20.0
                break
                
        # Lighting bonus if outdoor activity is in evening
        if cand.get("lit"):
            score += 5.0
            
        final_score = round(max(min(score, 100.0), 10.0), 1)
        
        scored_candidates.append({
            "name": cand.get("name", "Unnamed Venue"),
            "category": cand.get("category", "outdoor_area"),
            "distance_km": dist_km,
            "suitability_score": final_score,
            "source_evidence": cand.get("source_evidence", "Mapped in OpenStreetMap")
        })

    # Sort descending by suitability score
    scored_candidates.sort(key=lambda x: x["suitability_score"], reverse=True)
    return scored_candidates


def rank_time_windows(
    windows: List[Dict[str, Any]],
    activity: ActivityProfile
) -> Dict[str, Any]:
    """Identifies the optimal time window among upcoming hourly forecasts.
    
    Evaluates:
    - Low precipitation probability
    - Moderate UV index (avoiding extreme midday spikes)
    - Apparent temperature closest to comfort zone (18°C - 24°C)
    - Safe wind speeds
    """
    if not windows:
        return {
            "recommended_window": "Insufficient hourly forecast data to establish specific window.",
            "best_window": None
        }

    scored_windows = []
    for win in windows:
        score = 100.0
        
        # Rain penalty
        rain_prob = win.get("precipitation_probability", 0)
        if rain_prob > activity.max_rain_prob_pct:
            score -= (rain_prob - activity.max_rain_prob_pct) * 1.5
            
        # UV penalty
        uv = win.get("uv_index", 0.0)
        if uv >= 8.0:
            score -= (uv - 7.0) * 8.0
            
        # Heat penalty (optimal 18-24 C)
        app_t = win.get("apparent_temperature") or win.get("temperature", 22.0)
        if app_t > 28.0:
            score -= (app_t - 28.0) * 3.0
        elif app_t < 10.0:
            score -= (10.0 - app_t) * 2.0
            
        # Wind penalty
        w_speed = win.get("wind_speed", 0.0)
        if w_speed > activity.max_wind_kmh:
            score -= (w_speed - activity.max_wind_kmh) * 2.0

        scored_windows.append((score, win))

    scored_windows.sort(key=lambda x: x[0], reverse=True)
    best_score, best_win = scored_windows[0]
    
    if best_score < 30.0:
        recommended_text = "No optimal window found in upcoming forecast hours due to severe weather/UV constraints."
    else:
        recommended_text = (
            f"Optimal window around {best_win.get('time')} "
            f"({best_win.get('weather_description')}, {best_win.get('temperature')}°C, "
            f"UV {best_win.get('uv_index')}, {best_win.get('precipitation_probability')}% rain probability)."
        )

    return {
        "recommended_window": recommended_text,
        "best_window": best_win,
        "all_scored": scored_windows
    }
