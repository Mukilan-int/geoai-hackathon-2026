"""Pydantic response models matching the official recommendation schema."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SafetyAssessmentModel(BaseModel):
    guardrail_status: str = Field(..., description="Guardrail assessment level: CLEAR, ADVISORY, or BLOCKED.")
    triggered_rules: List[str] = Field(default_factory=list, description="Identifiers of triggered safety rules.")
    override_applied: bool = Field(..., description="Whether a deterministic safety override changed model suitability.")
    source_guidelines: List[str] = Field(default_factory=list, description="Official health/sports medicine guidelines referenced.")


class CandidatePlaceModel(BaseModel):
    name: str
    category: str
    distance_km: Optional[float] = None
    suitability_score: float
    source_evidence: str


class DataSourceEntryModel(BaseModel):
    provider: str
    service_type: str
    retrieval_timestamp: str
    attribution: str


class RecommendationResponse(BaseModel):
    activity_name: str
    suitability: str = Field(..., description="suitability rating: recommended, caution, or not_recommended.")
    recommended_time_window: str
    precautions: List[str]
    reasons: List[str]
    missing_information_identified: List[str]
    grounded_facts_used: List[str]
    safety_assessment: SafetyAssessmentModel
    candidate_places: List[CandidatePlaceModel]
    data_sources_used: List[DataSourceEntryModel]
    uncertainties_and_limitations: List[str]
