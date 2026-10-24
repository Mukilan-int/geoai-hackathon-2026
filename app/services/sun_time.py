"""Solar calculation service for daylight, sunrise, sunset, and golden hour windows."""
import math
from datetime import datetime, timezone
from typing import Dict


def calculate_sun_times(latitude: float, longitude: float, date: datetime = None) -> Dict[str, str]:
    """Computes approximate UTC sunrise, sunset, and daylight status using NOAA Solar Geometry.
    
    Returns:
        dict with sunrise_utc, sunset_utc, daylight_duration_hours, is_currently_daylight
    """
    dt = date or datetime.now(timezone.utc)
    day_of_year = dt.timetuple().tm_yday
    
    # Fractional year in radians
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (dt.hour - 12) / 24.0)
    
    # Equation of time in minutes
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    
    # Solar declination angle in radians
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    
    lat_rad = math.radians(latitude)
    zenith_rad = math.radians(90.833)  # Official zenith for sunrise/sunset
    
    # Hour angle calculation
    cos_ha = (math.cos(zenith_rad) / (math.cos(lat_rad) * math.cos(decl)) -
              math.tan(lat_rad) * math.tan(decl))
    
    if cos_ha > 1.0:
        # Polar night
        return {
            "sunrise_utc": "Polar Night (No Sunrise)",
            "sunset_utc": "Polar Night (No Sunset)",
            "daylight_duration_hours": "0.0",
            "is_daylight": False
        }
    elif cos_ha < -1.0:
        # Midnight sun
        return {
            "sunrise_utc": "Midnight Sun (No Sunset)",
            "sunset_utc": "Midnight Sun (No Sunset)",
            "daylight_duration_hours": "24.0",
            "is_daylight": True
        }
        
    ha = math.degrees(math.acos(cos_ha))
    
    # Sunrise and Sunset in minutes from UTC midnight
    solar_noon_utc_min = 720.0 - (4.0 * longitude) - eqtime
    sunrise_utc_min = solar_noon_utc_min - (ha * 4.0)
    sunset_utc_min = solar_noon_utc_min + (ha * 4.0)
    
    daylight_hours = round((ha * 8.0) / 60.0, 2)
    
    def _format_min(mins: float) -> str:
        mins = mins % 1440.0
        h = int(mins // 60)
        m = int(mins % 60)
        return f"{h:02d}:{m:02d} UTC"
        
    current_utc_min = dt.hour * 60 + dt.minute
    is_daylight = (sunrise_utc_min <= current_utc_min <= sunset_utc_min)
    
    return {
        "sunrise_utc": _format_min(sunrise_utc_min),
        "sunset_utc": _format_min(sunset_utc_min),
        "solar_noon_utc": _format_min(solar_noon_utc_min),
        "daylight_duration_hours": str(daylight_hours),
        "is_daylight": is_daylight
    }
