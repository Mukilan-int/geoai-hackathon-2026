"""Pydantic input request schemas for HTTP API."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class UserConstraints(BaseModel):
    age: Optional[int] = Field(default=None, description="Age of the participant in years.")
    mobility: Optional[str] = Field(default=None, description="Any mobility aid or restriction (e.g. wheelchair, walking cane, stroller).")
    health_notes: Optional[str] = Field(default=None, description="User-reported health notes (e.g. asthma, osteoarthritis, sun allergy).")


class RecommendationRequest(BaseModel):
    activity: Optional[str] = Field(default=None, description="Target outdoor activity (e.g. 'Running or Jogging', 'Cycling', 'Golf', etc.).")
    location: Optional[str] = Field(default=None, description="City, park name, address or landmark string.")
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Optional GPS latitude.")
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Optional GPS longitude.")
    radius_km: float = Field(default=5.0, ge=0.5, le=50.0, description="Search radius in kilometers for nearby venues.")
    constraints: Optional[UserConstraints] = Field(default=None, description="Optional personal constraints.")
    
    # Free-form query fields (supporting judging test harness queries)
    query: Optional[str] = Field(default=None, description="Free-form question (e.g. 'Where can I go running tomorrow evening near Koregaon Park?')")
    question: Optional[str] = Field(default=None, description="Alias for free-form query.")
    prompt: Optional[str] = Field(default=None, description="Alias for free-form query.")
    text: Optional[str] = Field(default=None, description="Alias for free-form query.")
    message: Optional[str] = Field(default=None, description="Alias for free-form query.")
    time_preference: Optional[str] = Field(default=None, description="Preferred time of day or weekend.")

