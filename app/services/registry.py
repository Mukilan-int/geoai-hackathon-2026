"""Provider and Data Source Registry for tracking fact provenance, licensing, and attribution."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DataSourceRegistry:
    """Tracks every external API provider that contributed live evidence to a recommendation."""
    
    def __init__(self):
        self._sources: List[Dict[str, str]] = []

    def record_source(
        self,
        provider: str,
        service_type: str,
        attribution: str,
        endpoint_url: Optional[str] = None
    ) -> None:
        """Records an external fact retrieval event with timestamp and attribution."""
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "provider": provider,
            "service_type": service_type,
            "retrieval_timestamp": timestamp_iso,
            "attribution": attribution
        }
        # Prevent exact duplicate entries
        if not any(
            s["provider"] == provider and s["service_type"] == service_type
            for s in self._sources
        ):
            self._sources.append(entry)

    def get_sources_used(self) -> List[Dict[str, str]]:
        """Returns the list of all sources recorded in this request lifecycle."""
        return list(self._sources)

    @staticmethod
    def get_configured_providers_info() -> List[Dict[str, Any]]:
        """Static directory of all supported live data providers and their licenses."""
        return [
            {
                "provider": "OpenStreetMap Nominatim",
                "service_type": "Geocoding and Reverse Geocoding",
                "license": "Open Data Commons Open Database License (ODbL) 1.0",
                "attribution": "\u00a9 OpenStreetMap contributors (https://www.openstreetmap.org/copyright)",
                "terms_of_use": "Nominatim Usage Policy: Max 1 req/sec, valid User-Agent, results cached locally.",
                "endpoint": "https://nominatim.openstreetmap.org"
            },
            {
                "provider": "OpenStreetMap Overpass API",
                "service_type": "Geospatial Features and POI Discovery",
                "license": "Open Data Commons Open Database License (ODbL) 1.0",
                "attribution": "\u00a9 OpenStreetMap contributors (https://www.openstreetmap.org/copyright)",
                "terms_of_use": "Free open query interpreter for vector features.",
                "endpoint": "https://overpass-api.de/api/interpreter"
            },
            {
                "provider": "Open-Meteo Weather Forecast API",
                "service_type": "Numerical Weather Prediction (Hourly Forecasts, UV, Wind, Precip)",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "attribution": "Weather data by Open-Meteo.com (https://open-meteo.com/)",
                "terms_of_use": "Free for non-commercial and commercial use under CC BY 4.0, no API key required.",
                "endpoint": "https://api.open-meteo.com/v1/forecast"
            },
            {
                "provider": "Open-Meteo Air Quality API",
                "service_type": "Atmospheric Composition & Air Quality (US AQI, European AQI, PM2.5, PM10)",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "attribution": "Air quality data by Open-Meteo.com / CAMS Copernicus / NOAA",
                "terms_of_use": "Free open access API under CC BY 4.0, no API key required.",
                "endpoint": "https://air-quality-api.open-meteo.com/v1/air-quality"
            },
            {
                "provider": "Project OSRM (Open Source Routing Machine)",
                "service_type": "Road Distance & Estimated Static Travel Time",
                "license": "BSD 2-Clause License & OpenStreetMap data (ODbL)",
                "attribution": "Routing powered by OSRM and \u00a9 OpenStreetMap contributors",
                "terms_of_use": "Public demo server for low-volume testing; static road network calculations without live congestion data.",
                "endpoint": "https://router.project-osrm.org"
            }
        ]
