"""Open-Meteo Air Quality service client with strict scale differentiation."""
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import requests

from app.config import settings
from app.services.registry import DataSourceRegistry
from app.utils.logger import logger

_AQ_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# US EPA AQI Category definitions
def describe_us_aqi(aqi: Optional[int]) -> str:
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good (0-50): Air quality is satisfactory, poses little or no risk."
    elif aqi <= 100:
        return "Moderate (51-100): Acceptable air quality; slight risk for unusually sensitive individuals."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups (101-150): General public not likely affected; active children/adults with asthma at risk."
    elif aqi <= 200:
        return "Unhealthy (151-200): Everyone may begin to experience health effects; active outdoors cautioned."
    elif aqi <= 300:
        return "Very Unhealthy (201-300): Health alert; risk of health effects increased for everyone."
    else:
        return "Hazardous (>300): Health warnings of emergency conditions; entire population is likely affected."


def fetch_air_quality(
    latitude: float,
    longitude: float,
    registry: Optional[DataSourceRegistry] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves air quality indices and pollutant concentrations from Open-Meteo.
    
    Returns:
        (air_quality_dict, error_message)
    """
    cache_key = f"{round(latitude, 3)},{round(longitude, 3)}"
    now = time.time()
    
    if cache_key in _AQ_CACHE:
        cached_time, cached_val = _AQ_CACHE[cache_key]
        if now - cached_time < settings.cache_ttl_seconds:
            if registry:
                registry.record_source(
                    provider="Open-Meteo Air Quality API",
                    service_type="Air Quality (Cached)",
                    attribution="Air quality data by Open-Meteo.com / CAMS / NOAA"
                )
            return cached_val, None

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi,european_aqi",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(
            settings.open_meteo_aq_url,
            params=params,
            timeout=settings.http_timeout_seconds
        )
        if response.status_code != 200:
            return None, f"Open-Meteo AQ HTTP {response.status_code}: {response.text[:100]}"
            
        raw = response.json()
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        
        # Take the most recent hour (index 0 or nearest)
        idx = 0
        def _get_val(col_name: str, fallback: Any = None) -> Any:
            arr = hourly.get(col_name, [])
            if arr and 0 <= idx < len(arr):
                val = arr[idx]
                return val if val is not None else fallback
            return fallback

        us_aqi_val = _get_val("us_aqi")
        eaqi_val = _get_val("european_aqi")
        pm2_5_val = _get_val("pm2_5")
        pm10_val = _get_val("pm10")
        no2_val = _get_val("nitrogen_dioxide")
        o3_val = _get_val("ozone")

        result = {
            "us_aqi": us_aqi_val,
            "us_aqi_scale": "US EPA Air Quality Index (0-500 scale)",
            "us_aqi_category": describe_us_aqi(us_aqi_val),
            "european_aqi": eaqi_val,
            "european_aqi_scale": "European Common Air Quality Index (1-5 qualitative scale)",
            "pm2_5": pm2_5_val,
            "pm2_5_unit": "µg/m³",
            "pm10": pm10_val,
            "pm10_unit": "µg/m³",
            "ozone": o3_val,
            "ozone_unit": "µg/m³",
            "nitrogen_dioxide": no2_val,
            "nitrogen_dioxide_unit": "µg/m³",
            "observation_time": times[idx] if times else datetime.now().isoformat()
        }

        _AQ_CACHE[cache_key] = (now, result)

        if registry:
            registry.record_source(
                provider="Open-Meteo Air Quality API",
                service_type="Atmospheric Composition & AQI",
                attribution="Air quality data by Open-Meteo.com / CAMS / NOAA",
                endpoint_url=settings.open_meteo_aq_url
            )

        return result, None

    except requests.exceptions.Timeout:
        logger.warning(f"Air quality request timed out for {latitude}, {longitude}")
        return None, "Air quality request timed out."
    except Exception as exc:
        logger.error(f"Air Quality API error: {exc}")
        return None, f"Air Quality API error: {str(exc)}"
