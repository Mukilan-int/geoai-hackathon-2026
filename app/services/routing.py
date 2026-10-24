"""Project OSRM routing service for road distances and static travel duration estimates."""
import math
import time
from typing import Any, Dict, Optional, Tuple
import requests

from app.config import settings
from app.services.registry import DataSourceRegistry
from app.utils.logger import logger

_ROUTING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def calculate_route_distance(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    registry: Optional[DataSourceRegistry] = None
) -> Dict[str, Any]:
    """Calculates road network route distance and estimated driving time via OSRM.
    
    CRITICAL NOTICE: OSRM public demo server returns static topological calculations
    based on OpenStreetMap speed limits. It does NOT provide live dynamic traffic congestion.
    """
    cache_key = f"{round(origin_lat, 4)},{round(origin_lon, 4)}->{round(dest_lat, 4)},{round(dest_lon, 4)}"
    now = time.time()
    
    if cache_key in _ROUTING_CACHE:
        cached_time, cached_val = _ROUTING_CACHE[cache_key]
        if now - cached_time < settings.cache_ttl_seconds:
            if registry:
                registry.record_source(
                    provider="Project OSRM",
                    service_type="Road Routing (Cached)",
                    attribution="Routing powered by OSRM and © OpenStreetMap contributors"
                )
            return cached_val

    straight_dist_km = _haversine_distance_km(origin_lat, origin_lon, dest_lat, dest_lon)
    url = f"{settings.osrm_url}/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=false"

    try:
        response = requests.get(url, timeout=settings.http_timeout_seconds)
        if response.status_code == 200:
            data = response.json()
            routes = data.get("routes", [])
            if routes and len(routes) > 0:
                best = routes[0]
                distance_km = round(best.get("distance", 0.0) / 1000.0, 2)
                duration_min = round(best.get("duration", 0.0) / 60.0, 1)
                
                result = {
                    "distance_km": distance_km,
                    "estimated_duration_minutes": duration_min,
                    "is_live_traffic": False,
                    "routing_method": "OSRM OpenStreetMap Road Network",
                    "note": "Estimated travel duration is static and does not reflect real-time traffic jams."
                }
                
                _ROUTING_CACHE[cache_key] = (now, result)
                if registry:
                    registry.record_source(
                        provider="Project OSRM",
                        service_type="Road Routing",
                        attribution="Routing powered by OSRM and © OpenStreetMap contributors",
                        endpoint_url=settings.osrm_url
                    )
                return result

    except Exception as exc:
        logger.warning(f"OSRM routing query failed ({exc}); falling back to straight-line distance.")

    # Graceful degradation fallback
    fallback_result = {
        "distance_km": straight_dist_km,
        "estimated_duration_minutes": round(straight_dist_km * 2.5, 1),  # rough 25 km/h urban assumption
        "is_live_traffic": False,
        "routing_method": "Haversine Straight-Line Distance (Road routing unverified)",
        "note": "OSRM server unavailable; estimated distance is straight-line Euclidean."
    }
    return fallback_result
