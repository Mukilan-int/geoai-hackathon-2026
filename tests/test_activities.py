"""Unit tests verifying support for all 10 required outdoor activities."""
import pytest
from app.geospatial.activities import ACTIVITIES, get_activity_profile, list_supported_activities

REQUIRED_ACTIVITIES = [
    "Running or Jogging",
    "Cycling",
    "Golf",
    "Cricket",
    "Outdoor Football",
    "Outdoor Tennis",
    "Hiking or Nature Walking",
    "Picnic or Park Outing",
    "Outdoor Event or Gathering",
    "Outdoor Photography"
]


def test_ten_activities_present():
    """Verify that exactly the 10 required outdoor activities are configured."""
    supported = list_supported_activities()
    assert len(supported) == 10
    names = [a["name"] for a in supported]
    for req in REQUIRED_ACTIVITIES:
        assert req in names, f"Missing required activity: {req}"


def test_activity_lookup_fuzzy_and_slug():
    """Verify robust lookup by slug, exact name, and lowercase keywords."""
    assert get_activity_profile("running_jogging") is not None
    assert get_activity_profile("Running or Jogging") is not None
    assert get_activity_profile("cricket") is not None
    assert get_activity_profile("Golf") is not None
    assert get_activity_profile("outdoor tennis") is not None
    assert get_activity_profile("photography") is not None


def test_activity_profiles_have_safety_thresholds():
    """Verify each activity defines explicit physical and meteorological bounds."""
    for key, prof in ACTIVITIES.items():
        assert prof.max_rain_prob_pct > 0
        assert prof.max_wind_kmh > 0
        assert prof.max_uv_index > 0
        assert prof.max_apparent_temp_c > 0
        assert len(prof.osm_tags) > 0
        assert len(prof.equipment_precautions) > 0
        assert len(prof.safety_notes) > 10
