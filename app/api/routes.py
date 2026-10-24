"""FastAPI HTTP API endpoints for the GeoAI recommendation service."""
import os
import psutil
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status

from app.geospatial.activities import list_supported_activities
from app.models.loader import get_model_status
from app.schemas.recommendation import RecommendationResponse
from app.schemas.requests import RecommendationRequest
from app.services.query_parser import parse_free_form_query
from app.services.recommendation_engine import engine
from app.services.registry import DataSourceRegistry
from app.utils.logger import logger

router = APIRouter()


@router.get(
    "/health",
    summary="Service and Dependency Health Check",
    tags=["System Status"]
)
def get_health() -> Dict[str, Any]:
    """Returns the operational status of the service, model loader, and host resources."""
    process = psutil.Process(os.getpid())
    rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    model_status = get_model_status()

    return {
        "status": "healthy",
        "service": "GeoAI Outdoor Activity Recommendation Service",
        "process_memory_rss_mb": rss_mb,
        "model_status": model_status,
        "framework": "FastAPI + Intel OpenVINO / PyTorch",
        "supported_activities_count": len(list_supported_activities())
    }


@router.get(
    "/activities",
    summary="List Supported Outdoor Activities",
    tags=["Activities"]
)
def get_activities() -> List[Dict[str, Any]]:
    """Returns the 10 explicitly supported outdoor activities with their metabolic and meteorological profiles."""
    return list_supported_activities()


@router.get(
    "/sources",
    summary="List External Data Providers and Attribution",
    tags=["Data Provenance"]
)
def get_sources() -> List[Dict[str, Any]]:
    """Lists all configured live data providers, licenses, attribution requirements, and endpoints."""
    return DataSourceRegistry.get_configured_providers_info()


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Generate Grounded Outdoor Recommendation",
    tags=["Recommendation Engine"],
    status_code=status.HTTP_200_OK
)
@router.post(
    "/query",
    response_model=RecommendationResponse,
    include_in_schema=False
)
@router.post(
    "/ask",
    response_model=RecommendationResponse,
    include_in_schema=False
)
def post_recommend(request: RecommendationRequest) -> Dict[str, Any]:
    """Evaluates an outdoor activity request against live weather, air quality, venue proximity, and deterministic safety rules."""
    raw_query = request.query or request.question or request.prompt or request.text or request.message
    
    activity = request.activity
    location = request.location
    radius_km = request.radius_km
    constraints_dict = request.constraints.model_dump() if request.constraints else {}

    # If free-form natural language query is provided, extract missing fields
    if raw_query:
        parsed = parse_free_form_query(raw_query)
        if not activity and parsed.get("activity_name"):
            activity = parsed["activity_name"]
        if not location and parsed.get("location"):
            location = parsed["location"]
        if radius_km == 5.0 and parsed.get("radius_km"):
            radius_km = parsed["radius_km"]
        if parsed.get("constraints"):
            # Merge parsed constraints
            for k, v in parsed["constraints"].items():
                if k not in constraints_dict or not constraints_dict[k]:
                    constraints_dict[k] = v

    # Fallback to general outdoor walking if still unspecified
    if not activity:
        activity = "Hiking or Nature Walking"
        if not constraints_dict:
            constraints_dict = {}
        constraints_dict["_notice"] = "Activity was not specified in the input query; evaluated as general outdoor walking/outing."

    recommendation, err_msg = engine.run_pipeline(
        activity_name=activity,
        location_query=location,
        latitude=request.latitude,
        longitude=request.longitude,
        radius_km=radius_km,
        user_constraints=constraints_dict if constraints_dict else None
    )

    if err_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg
        )

    return recommendation
