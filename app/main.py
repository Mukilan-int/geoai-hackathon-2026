"""Main FastAPI application entry point for GeoAI Hackathon 2026."""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.models.loader import load_model_and_tokenizer
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: pre-warm model if enabled."""
    logger.info("Initializing GeoAI Outdoor Recommendation Service...")
    if settings.enable_model_inference:
        # Pre-warm model in background or on startup
        try:
            load_model_and_tokenizer()
        except Exception as exc:
            logger.warning(f"Startup model load deferred or failed: {exc}")
    yield
    logger.info("Shutting down GeoAI Outdoor Recommendation Service.")


app = FastAPI(
    title="GeoAI Outdoor Activity Recommendation Service",
    description=(
        "Production backend for GeoAI Hackathon 2026 (CSRE, IIT Bombay). "
        "Provides grounded outdoor activity recommendations based on live meteorological forecasts, "
        "atmospheric air quality, OpenStreetMap venues, and deterministic safety guardrails. "
        "Powered by OpenVINO quantized Qwen2.5-1.5B-Instruct (<3B parameters)."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Records request latency and attaches processing headers."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions safely without leaking internal stack traces or secrets."""
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred while processing the geospatial request.",
            "path": request.url.path
        }
    )


# Register API routes
app.include_router(router, prefix="/api/v1")
app.include_router(router)  # Also expose directly at root for convenience


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
