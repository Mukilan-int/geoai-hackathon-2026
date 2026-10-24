"""OpenStreetMap Overpass API client for finding nearby outdoor recreational venues."""
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.config import settings
from app.geospatial.activities import ActivityProfile
from app.services.registry import DataSourceRegistry
from app.utils.logger import logger

_OVERPASS_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def find_nearby_places(
    latitude: float,
    longitude: float,
    radius_km: float = 5.0,
    activity_profile: Optional[ActivityProfile] = None,
    registry: Optional[DataSourceRegistry] = None
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Queries Overpass API for outdoor venues, parks, and sports facilities around coordinates.
    
    Returns:
        (list_of_candidates, error_message)
    """
    radius_meters = int(min(max(radius_km, 1.0), 20.0) * 1000)
    cache_key = f"{round(latitude, 3)},{round(longitude, 3)}_{radius_meters}_{activity_profile.id if activity_profile else 'all'}"
    now = time.time()
    
    if cache_key in _OVERPASS_CACHE:
        cached_time, cached_list = _OVERPASS_CACHE[cache_key]
        if now - cached_time < settings.cache_ttl_seconds:
            if registry and cached_list:
                registry.record_source(
                    provider="OpenStreetMap Overpass API",
                    service_type="Geospatial Features (Cached)",
                    attribution="© OpenStreetMap contributors"
                )
            return cached_list, None

    # Construct Overpass QL query tailored to outdoor activities
    tags_filter = ""
    if activity_profile and activity_profile.osm_tags:
        # e.g. ["leisure=pitch", "leisure=park"]
        clauses = []
        for tag in activity_profile.osm_tags:
            if "=" in tag:
                k, v = tag.split("=", 1)
                clauses.append(f'node["{k}"="{v}"](around:{radius_meters},{latitude},{longitude});')
                clauses.append(f'way["{k}"="{v}"](around:{radius_meters},{latitude},{longitude});')
        # Also always include parks as universal outdoor venue
        clauses.append(f'node["leisure"="park"](around:{radius_meters},{latitude},{longitude});')
        clauses.append(f'way["leisure"="park"](around:{radius_meters},{latitude},{longitude});')
        tags_filter = "\n  ".join(clauses)
    else:
        tags_filter = f"""
  node["leisure"~"park|pitch|nature_reserve|garden|golf_course"](around:{radius_meters},{latitude},{longitude});
  way["leisure"~"park|pitch|nature_reserve|garden|golf_course"](around:{radius_meters},{latitude},{longitude});
  node["tourism"~"viewpoint|picnic_site|attraction"](around:{radius_meters},{latitude},{longitude});
  way["tourism"~"viewpoint|picnic_site|attraction"](around:{radius_meters},{latitude},{longitude});
"""

    overpass_query = f"""
[out:json][timeout:10];
(
  {tags_filter}
);
out center 15;
"""

    headers = {"User-Agent": settings.user_agent}
    try:
        response = requests.post(
            settings.overpass_url,
            data={"data": overpass_query},
            headers=headers,
            timeout=settings.http_timeout_seconds
        )
        if response.status_code != 200:
            return [], f"Overpass HTTP {response.status_code}: {response.text[:100]}"
            
        data = response.json()
        elements = data.get("elements", [])
        
        candidates = []
        seen_names = set()
        
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            category = (
                tags.get("leisure") or
                tags.get("sport") or
                tags.get("tourism") or
                tags.get("highway") or
                "outdoor_venue"
            )
            
            # Position coordinates
            if el.get("type") == "node":
                lat, lon = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                lat, lon = center.get("lat"), center.get("lon")
                
            if lat is None or lon is None:
                continue
                
            dist_km = _haversine_distance_km(latitude, longitude, float(lat), float(lon))
            
            if not name:
                name = f"Public {category.replace('_', ' ').title()}"
                
            # Deduplicate by name
            if name in seen_names:
                continue
            seen_names.add(name)
            
            candidates.append({
                "name": name,
                "category": category,
                "latitude": float(lat),
                "longitude": float(lon),
                "distance_km": dist_km,
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "surface": tags.get("surface"),
                "lit": tags.get("lit") == "yes",
                "source_evidence": f"Mapped in OpenStreetMap ({category})"
            })
            
            if len(candidates) >= 5:
                break

        # Sort by distance
        candidates.sort(key=lambda x: x["distance_km"])
        _OVERPASS_CACHE[cache_key] = (now, candidates)
        
        if registry and candidates:
            registry.record_source(
                provider="OpenStreetMap Overpass API",
                service_type="Geospatial POIs & Features",
                attribution="© OpenStreetMap contributors",
                endpoint_url=settings.overpass_url
            )
            
        return candidates, None

    except requests.exceptions.Timeout:
        logger.warning("Overpass API request timed out; degrading gracefully.")
        return [], "Overpass API request timed out."
    except Exception as exc:
        logger.error(f"Overpass API error: {exc}")
        return [], f"Overpass error: {str(exc)}"
