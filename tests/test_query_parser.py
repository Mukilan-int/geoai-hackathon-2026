"""Unit tests for the natural language query parser testing all 10 Rulebook queries."""
import pytest
from app.services.query_parser import parse_free_form_query


RULEBOOK_EXAMPLE_QUERIES = [
    ("Where can I go running tomorrow evening near Koregaon Park?", "Running or Jogging", "Koregaon Park"),
    ("Best time this weekend for a kids' cricket match within 15 km of Whitefield?", "Cricket", "Whitefield"),
    ("I have mild asthma, where's good for a morning walk near Powai in the next two days?", "Hiking or Nature Walking", "Powai"),
    ("Is it okay to cycle to work at 8 AM?", "Cycling", None),
    ("Planning an outdoor birthday party on Sunday afternoon in Pune, what should I keep in mind?", "Outdoor Event or Gathering", "Pune"),
    ("Good place and time for golf on Saturday within 40 km of Andheri?", "Golf", "Andheri"),
    ("Where can I take sunset photos this evening around Bandra?", "Outdoor Photography", "Bandra"),
    ("Is Saturday or Sunday better for a hike near Lonavala?", "Hiking or Nature Walking", "Lonavala"),
    ("When is it safe for my parents to walk outdoors tomorrow in Bengaluru?", "Hiking or Nature Walking", "Bengaluru"),
    ("Outdoor tennis after work today near Hinjewadi, any good spots?", "Outdoor Tennis", "Hinjewadi"),
]


@pytest.mark.parametrize("query,expected_activity,expected_location", RULEBOOK_EXAMPLE_QUERIES)
def test_rulebook_queries_parsed(query, expected_activity, expected_location):
    parsed = parse_free_form_query(query)
    assert parsed["activity_name"] == expected_activity
    if expected_location:
        assert parsed["location"] == expected_location


def test_radius_extraction():
    parsed = parse_free_form_query("Best cricket ground within 15 km of Whitefield?")
    assert parsed["radius_km"] == 15.0

    parsed_golf = parse_free_form_query("Golf courses within 40 km of Andheri")
    assert parsed_golf["radius_km"] == 40.0


def test_constraint_extraction():
    parsed_asthma = parse_free_form_query("I have mild asthma, where to walk?")
    assert "asthma" in parsed_asthma["constraints"].get("health_notes", "").lower()

    parsed_kids = parse_free_form_query("Kids cricket match in Whitefield")
    assert parsed_kids["constraints"].get("age") == 10

    parsed_parents = parse_free_form_query("Walk for my parents in Bengaluru")
    assert parsed_parents["constraints"].get("age") == 70
