"""Development server launcher with clean console output."""
import os
import sys
import uvicorn

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from app.config import settings

if __name__ == "__main__":
    print("=" * 70)
    print("Starting GeoAI Outdoor Recommendation API Server (Development Mode)")
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Swagger Documentation: http://localhost:{settings.port}/docs")
    print(f"OpenVINO Device: {settings.device}")
    print("=" * 70)
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
