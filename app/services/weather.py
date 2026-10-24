"""Open-Meteo Numerical Weather Forecast service client."""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.config import settings
from app.services.registry import DataSourceRegistry
from app.utils.logger import logger

_WEATHER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# WMO Weather interpretation codes (WW)
WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm: slight or moderate",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


def get_wmo_description(code: Optional[int]) -> str:
    """Returns human-readable WMO weather code description."""
    if code is None:
        return "Unknown"
    return WMO_DESCRIPTIONS.get(code, f"Weather code {code}")


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    registry: Optional[DataSourceRegistry] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves current and hourly weather forecast from Open-Meteo.
    
    Returns:
        (weather_payload, error_message)
    """
    cache_key = f"{round(latitude, 3)},{round(longitude, 3)}"
    now = time.time()
    
    if cache_key in _WEATHER_CACHE:
        cached_time, cached_val = _WEATHER_CACHE[cache_key]
        if now - cached_time < settings.cache_ttl_seconds:
            if registry:
                registry.record_source(
                    provider="Open-Meteo Weather Forecast API",
                    service_type="Weather (Cached)",
                    attribution="Weather data by Open-Meteo.com (CC BY 4.0)"
                )
            return cached_val, None

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true",
        "hourly": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation_probability,precipitation,weathercode,"
            "cloudcover,windspeed_10m,windgusts_10m,winddirection_10m,uv_index"
        ),
        "timezone": "auto"
    }
    
    try:
        response = requests.get(
            settings.open_meteo_weather_url,
            params=params,
            timeout=settings.http_timeout_seconds
        )
        if response.status_code != 200:
            return None, f"Open-Meteo HTTP {response.status_code}: {response.text[:100]}"
            
        raw = response.json()
        current = raw.get("current_weather", {})
        hourly = raw.get("hourly", {})
        
        # Determine current hour index in the hourly array
        times = hourly.get("time", [])
        current_time_str = current.get("time", "")
        
        idx = 0
        if current_time_str in times:
            idx = times.index(current_time_str)
        elif len(times) > 0:
            idx = 0

        # Safe extraction helper
        def _get_val(key_name: str, fallback: Any = None) -> Any:
            arr = hourly.get(key_name, [])
            if arr and 0 <= idx < len(arr):
                val = arr[idx]
                return val if val is not None else fallback
            return fallback

        current_summary = {
            "temperature": current.get("temperature", _get_val("temperature_2m")),
            "apparent_temperature": _get_val("apparent_temperature", current.get("temperature")),
            "relative_humidity": _get_val("relative_humidity_2m"),
            "precipitation_probability": _get_val("precipitation_probability", 0),
            "precipitation": _get_val("precipitation", 0.0),
            "weather_code": current.get("weathercode", _get_val("weathercode")),
            "weather_description": get_wmo_description(current.get("weathercode", _get_val("weathercode"))),
            "wind_speed": current.get("windspeed", _get_val("windspeed_10m")),
            "wind_gusts": _get_val("windgusts_10m", current.get("windspeed")),
            "wind_direction": current.get("winddirection", _get_val("winddirection_10m")),
            "cloud_cover": _get_val("cloudcover"),
            "uv_index": _get_val("uv_index", 0.0),
            "observation_time": current_time_str or datetime.now().isoformat(),
            "timezone": raw.get("timezone", "UTC")
        }

        # Build upcoming time windows (e.g. next 6 to 12 hours)
        upcoming_windows = []
        for i in range(idx, min(idx + 12, len(times))):
            t_str = times[i]
            # hour part
            hour_label = t_str.split("T")[-1] if "T" in t_str else t_str
            w_code = hourly.get("weathercode", [None])[i] if i < len(hourly.get("weathercode", [])) else None
            upcoming_windows.append({
                "time": t_str,
                "hour_label": hour_label,
                "temperature": hourly.get("temperature_2m", [None])[i] if i < len(hourly.get("temperature_2m", [])) else None,
                "apparent_temperature": hourly.get("apparent_temperature", [None])[i] if i < len(hourly.get("apparent_temperature", [])) else None,
                "precipitation_probability": hourly.get("precipitation_probability", [0])[i] if i < len(hourly.get("precipitation_probability", [])) else 0,
                "wind_speed": hourly.get("windspeed_10m", [None])[i] if i < len(hourly.get("windspeed_10m", [])) else None,
                "uv_index": hourly.get("uv_index", [0.0])[i] if i < len(hourly.get("uv_index", [])) else 0.0,
                "weather_description": get_wmo_description(w_code)
            })

        result = {
            "current": current_summary,
            "upcoming_windows": upcoming_windows,
            "elevation": raw.get("elevation"),
            "latitude": raw.get("latitude"),
            "longitude": raw.get("longitude")
        }

        _WEATHER_CACHE[cache_key] = (now, result)

        if registry:
            registry.record_source(
                provider="Open-Meteo Weather Forecast API",
                service_type="Numerical Weather Prediction",
                attribution="Weather data by Open-Meteo.com (CC BY 4.0)",
                endpoint_url=settings.open_meteo_weather_url
            )

        return result, None

    except requests.exceptions.Timeout:
        logger.warning(f"Weather request timed out for {latitude}, {longitude}")
        return None, "Weather forecast request timed out."
    except Exception as exc:
        logger.error(f"Weather API error: {exc}")
        return None, f"Weather API error: {str(exc)}"
