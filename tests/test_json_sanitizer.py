"""Unit tests for JSON extraction, markdown fence stripping, and schema validation."""
import json
import pytest
from app.utils.json_sanitizer import extract_json_candidate, sanitize_and_parse_json, validate_schema


def test_markdown_fence_stripping():
    fenced = """Here is the recommendation:
```json
{
  "activity_name": "Cycling",
  "suitability": "recommended"
}
```
Hope this helps!"""
    extracted = extract_json_candidate(fenced)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    parsed, err = sanitize_and_parse_json(fenced)
    assert err is None
    assert parsed["activity_name"] == "Cycling"


def test_raw_json_parsing():
    raw = '{"activity_name": "Golf", "suitability": "caution"}'
    parsed, err = sanitize_and_parse_json(raw)
    assert err is None
    assert parsed["activity_name"] == "Golf"


def test_trailing_comma_recovery():
    malformed = '{"activity_name": "Outdoor Tennis", "suitability": "recommended",}'
    parsed, err = sanitize_and_parse_json(malformed)
    assert err is None
    assert parsed["activity_name"] == "Outdoor Tennis"


def test_schema_validator_success():
    valid_payload = {
        "activity_name": "Cricket",
        "suitability": "recommended",
        "recommended_time_window": "08:00 to 11:00 AM",
        "precautions": ["Wear cricket helmet and sunscreen"],
        "reasons": ["Weather is clear with zero rain probability"],
        "missing_information_identified": [],
        "grounded_facts_used": ["Clear sky", "Temperature: 24C"],
        "safety_assessment": {
            "guardrail_status": "CLEAR",
            "triggered_rules": [],
            "override_applied": False,
            "source_guidelines": ["Standard Protocols"]
        },
        "candidate_places": [
            {
                "name": "Local Cricket Ground",
                "category": "pitch",
                "distance_km": 0.5,
                "suitability_score": 90.0,
                "source_evidence": "Mapped in OpenStreetMap"
            }
        ],
        "data_sources_used": [
            {
                "provider": "Open-Meteo",
                "service_type": "Weather",
                "retrieval_timestamp": "2026-10-23T12:00:00Z",
                "attribution": "CC BY 4.0"
            }
        ],
        "uncertainties_and_limitations": ["Forecast model has 3-hour resolution"]
    }
    is_valid, errors = validate_schema(valid_payload)
    assert is_valid is True
    assert len(errors) == 0


def test_schema_validator_catches_missing_required_fields():
    invalid_payload = {
        "activity_name": "Cricket",
        "suitability": "recommended"
        # missing all other required fields
    }
    is_valid, errors = validate_schema(invalid_payload)
    assert is_valid is False
    assert len(errors) > 0
