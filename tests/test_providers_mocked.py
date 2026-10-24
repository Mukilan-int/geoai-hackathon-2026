"""Mocked unit tests for live data service parsers independent of external network availability."""
from unittest.mock import MagicMock, patch
import pytest

from app.services.air_quality import fetch_air_quality
from app.services.geocoding import geocode_location
from app.services.overpass import find_nearby_places
from app.services.registry import DataSourceRegistry
from app.services.routing import calculate_route_distance
from app.services.weather import fetch_weather_forecast


def test_mocked_nominatim_geocoding():
    registry = DataSourceRegistry()
    mock_response = [
        {
            "name": "Sanjay Gandhi National Park",
            "display_name": "Sanjay Gandhi National Park, Borivali, Mumbai, Maharashtra, India",
            "lat": "19.2215",
            "lon": "72.9125",
            "boundingbox": ["19.1", "19.3", "72.8", "73.0"],
            "osm_id": 123456,
            "osm_type": "way"
        }
    ]
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response
        result, err = geocode_location("Sanjay Gandhi National Park", registry=registry)

    assert err is None
    assert result["name"] == "Sanjay Gandhi National Park"
    assert result["latitude"] == 19.2215
    assert result["longitude"] == 72.9125
    sources = registry.get_sources_used()
    assert any("Nominatim" in s["provider"] for s in sources)


def test_mocked_open_meteo_weather():
    registry = DataSourceRegistry()
    mock_payload = {
        "latitude": 19.076,
        "longitude": 72.877,
        "elevation": 14.0,
        "timezone": "Asia/Kolkata",
        "current_weather": {
            "temperature": 29.5,
            "windspeed": 12.0,
            "winddirection": 240.0,
            "weathercode": 1,
            "time": "2026-10-23T12:00"
        },
        "hourly": {
            "time": ["2026-10-23T12:00", "2026-10-23T13:00"],
            "temperature_2m": [29.5, 30.2],
            "relative_humidity_2m": [65, 62],
            "apparent_temperature": [33.5, 34.8],
            "precipitation_probability": [5, 10],
            "precipitation": [0.0, 0.0],
            "weathercode": [1, 2],
            "cloudcover": [20, 35],
            "windspeed_10m": [12.0, 14.0],
            "windgusts_10m": [18.0, 20.0],
            "winddirection_10m": [240, 250],
            "uv_index": [7.5, 8.2]
        }
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_payload
        result, err = fetch_weather_forecast(19.076, 72.877, registry=registry)

    assert err is None
    assert result["current"]["temperature"] == 29.5
    assert result["current"]["uv_index"] == 7.5
    assert len(result["upcoming_windows"]) == 2


def test_mocked_air_quality():
    registry = DataSourceRegistry()
    mock_payload = {
        "hourly": {
            "time": ["2026-10-23T12:00"],
            "pm10": [68.0],
            "pm2_5": [32.5],
            "carbon_monoxide": [450.0],
            "nitrogen_dioxide": [28.0],
            "ozone": [62.0],
            "us_aqi": [93],
            "european_aqi": [2]
        }
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_payload
        result, err = fetch_air_quality(19.076, 72.877, registry=registry)

    assert err is None
    assert result["us_aqi"] == 93
    assert result["pm2_5"] == 32.5
    assert "Moderate" in result["us_aqi_category"]


def test_mocked_osrm_routing():
    registry = DataSourceRegistry()
    mock_payload = {
        "code": "Ok",
        "routes": [
            {
                "distance": 8420.5,
                "duration": 912.4,
                "weight_name": "routability"
            }
        ]
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_payload
        route = calculate_route_distance(19.076, 72.877, 19.120, 72.900, registry=registry)

    assert route["distance_km"] == 8.42
    assert route["estimated_duration_minutes"] == 15.2
    assert route["is_live_traffic"] is False
