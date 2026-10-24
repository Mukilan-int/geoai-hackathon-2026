"""Integration tests for FastAPI HTTP endpoints using TestClient."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["supported_activities_count"] == 10
    assert "model_status" in data


def test_get_activities():
    response = client.get("/activities")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    activity_ids = [item["id"] for item in data]
    assert "running_jogging" in activity_ids
    assert "golf" in activity_ids
    assert "outdoor_photography" in activity_ids


def test_get_sources():
    response = client.get("/sources")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 5
    provider_names = [p["provider"] for p in data]
    assert any("OpenStreetMap" in name for name in provider_names)
    assert any("Open-Meteo" in name for name in provider_names)
    assert any("OSRM" in name for name in provider_names)


def test_post_recommend_missing_location():
    """Verify that requests missing location coordinates degrade gracefully with explicit missing notice."""
    payload = {
        "activity": "Picnic or Park Outing"
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["activity_name"] == "Picnic or Park Outing"
    assert data["suitability"] == "caution"
    assert len(data["missing_information_identified"]) > 0
    assert any("omitted" in m.lower() for m in data["missing_information_identified"])


def test_post_recommend_unsupported_activity():
    payload = {
        "activity": "Scuba Diving Deep Trench",
        "location": "Mumbai"
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 400
    assert "Unsupported activity" in response.json()["detail"]


def test_post_recommend_with_coordinates_live():
    """Integration smoke test for full recommendation pipeline with coordinates."""
    payload = {
        "activity": "Cricket",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "radius_km": 5.0
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["activity_name"] == "Cricket"
    assert data["suitability"] in ["recommended", "caution", "not_recommended"]
    assert len(data["reasons"]) > 0
    assert len(data["precautions"]) > 0
    assert "safety_assessment" in data
    assert "candidate_places" in data
    assert "data_sources_used" in data
    assert "uncertainties_and_limitations" in data


def test_post_recommend_free_form_query():
    """Test rulebook free-form query handling via /recommend."""
    payload = {
        "query": "Where can I go running tomorrow evening near Koregaon Park?"
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["activity_name"] == "Running or Jogging"
    assert "suitability" in data
    assert "candidate_places" in data


def test_post_query_endpoint_alias_with_constraints():
    """Test /query endpoint alias with health constraints."""
    payload = {
        "question": "I have mild asthma, where's good for a morning walk near Powai in the next two days?"
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["activity_name"] == "Hiking or Nature Walking"
    assert "suitability" in data


def test_post_recommend_free_form_missing_location():
    """Test rulebook incomplete query handling (missing location)."""
    payload = {
        "query": "Is it okay to cycle to work at 8 AM?"
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["activity_name"] == "Cycling"
    assert data["suitability"] == "caution"
    assert len(data["missing_information_identified"]) > 0

