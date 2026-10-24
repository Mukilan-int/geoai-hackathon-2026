"""OpenStreetMap Nominatim Geocoding and Reverse Geocoding service."""
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.config import settings
from app.services.registry import DataSourceRegistry
from app.utils.logger import logger

# Simple in-memory cache for query results: query_str -> (timestamp, result_dict)
_GEOCODE_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_LAST_REQUEST_TIME: float = 0.0


def _rate_limit_wait() -> None:
    """Enforces polite rate limiting for OpenStreetMap Nominatim (minimum 1 sec between calls)."""
    global _LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - _LAST_REQUEST_TIME
    if elapsed < settings.min_nominatim_delay_sec:
        time.sleep(settings.min_nominatim_delay_sec - elapsed)
    _LAST_REQUEST_TIME = time.time()


def geocode_location(
    query: str,
    registry: Optional[DataSourceRegistry] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Geocodes a placename or address string into latitude, longitude, and metadata.
    
    Returns:
        (location_info_dict, error_message)
    """
    if not query or not query.strip():
        return None, "Empty location query."
        
    normalized_query = query.strip().lower()
    now = time.time()
    
    # Check memory cache
    if normalized_query in _GEOCODE_CACHE:
        cached_time, cached_val = _GEOCODE_CACHE[normalized_query]
        if now - cached_time < settings.cache_ttl_seconds:
            if registry and cached_val:
                registry.record_source(
                    provider="OpenStreetMap Nominatim",
                    service_type="Geocoding (Cached)",
                    attribution="© OpenStreetMap contributors"
                )
            return cached_val, None

    _rate_limit_wait()
    
    headers = {"User-Agent": settings.user_agent}
    params = {
        "q": query.strip(),
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1
    }
    
    try:
        response = requests.get(
            f"{settings.nominatim_url}/search",
            params=params,
            headers=headers,
            timeout=settings.http_timeout_seconds
        )
        if response.status_code != 200:
            return None, f"Nominatim HTTP {response.status_code}: {response.text[:100]}"
            
        data = response.json()
        if not data or not isinstance(data, list) or len(data) == 0:
            return None, f"No coordinates found for location query: '{query}'"
            
        first = data[0]
        result = {
            "name": first.get("name") or first.get("display_name", "").split(",")[0],
            "display_name": first.get("display_name"),
            "latitude": float(first["lat"]),
            "longitude": float(first["lon"]),
            "boundingbox": first.get("boundingbox", []),
            "osm_id": first.get("osm_id"),
            "osm_type": first.get("osm_type"),
            "address": first.get("address", {})
        }
        
        _GEOCODE_CACHE[normalized_query] = (now, result)
        
        if registry:
            registry.record_source(
                provider="OpenStreetMap Nominatim",
                service_type="Geocoding",
                attribution="© OpenStreetMap contributors",
                endpoint_url=settings.nominatim_url
            )
            
        return result, None

    except requests.exceptions.Timeout:
        logger.warning(f"Nominatim geocode request timed out for query '{query}'")
        return None, "Geocoding request timed out."
    except Exception as exc:
        logger.error(f"Nominatim geocoding error: {exc}")
        return None, f"Geocoding service error: {str(exc)}"


def reverse_geocode(
    latitude: float,
    longitude: float,
    registry: Optional[DataSourceRegistry] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Reverse geocodes coordinates into an address and place name."""
    cache_key = f"{round(latitude, 4)},{round(longitude, 4)}"
    now = time.time()
    
    if cache_key in _GEOCODE_CACHE:
        cached_time, cached_val = _GEOCODE_CACHE[cache_key]
        if now - cached_time < settings.cache_ttl_seconds:
            if registry and cached_val:
                registry.record_source(
                    provider="OpenStreetMap Nominatim",
                    service_type="Reverse Geocoding (Cached)",
                    attribution="© OpenStreetMap contributors"
                )
            return cached_val, None

    _rate_limit_wait()
    
    headers = {"User-Agent": settings.user_agent}
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "addressdetails": 1
    }
    
    try:
        response = requests.get(
            f"{settings.nominatim_url}/reverse",
            params=params,
            headers=headers,
            timeout=settings.http_timeout_seconds
        )
        if response.status_code != 200:
            return None, f"Nominatim reverse HTTP {response.status_code}"
            
        data = response.json()
        if not data or "lat" not in data:
            return None, "No address found for coordinates."
            
        result = {
            "name": data.get("name") or data.get("display_name", "").split(",")[0],
            "display_name": data.get("display_name"),
            "latitude": float(data["lat"]),
            "longitude": float(data["lon"]),
            "address": data.get("address", {})
        }
        
        _GEOCODE_CACHE[cache_key] = (now, result)
        
        if registry:
            registry.record_source(
                provider="OpenStreetMap Nominatim",
                service_type="Reverse Geocoding",
                attribution="© OpenStreetMap contributors"
            )
            
        return result, None
    except Exception as exc:
        return None, f"Reverse geocoding error: {str(exc)}"
