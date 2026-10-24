"""Application configuration with environment variable support and safe defaults."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    """Central configuration class for the GeoAI Hackathon Backend."""
    
    # Service metadata
    app_name: str = "GeoAI Outdoor Activity Recommendation Service"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Model configuration
    model_id: str = os.getenv("MODEL_ID", "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
    base_model_id: str = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path: Optional[str] = os.getenv("ADAPTER_PATH", None)
    device: str = os.getenv("OPENVINO_DEVICE", "CPU")  # "CPU" or "GPU" (Iris Xe)
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))  # Greedy decoding for schema compliance
    enable_model_inference: bool = os.getenv("ENABLE_MODEL_INFERENCE", "true").lower() == "true"
    
    # Geospatial and Live Data Endpoints
    user_agent: str = os.getenv(
        "NOMINATIM_USER_AGENT",
        "GeoAI-Hackathon-2026-CSRE-IITB/1.0 (academic.research.geospatial@iitb.ac.in)"
    )
    nominatim_url: str = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
    overpass_url: str = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
    open_meteo_weather_url: str = os.getenv("OPEN_METEO_WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
    open_meteo_aq_url: str = os.getenv("OPEN_METEO_AQ_URL", "https://air-quality-api.open-meteo.com/v1/air-quality")
    osrm_url: str = os.getenv("OSRM_URL", "https://router.project-osrm.org")
    
    # Network resilience & caching
    http_timeout_seconds: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "8.0"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 minutes
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    
    # Rate Limiting safety (politeness to free public OSM servers)
    min_nominatim_delay_sec: float = float(os.getenv("NOMINATIM_MIN_DELAY", "1.0"))


# Singleton configuration instance
settings = AppConfig()
