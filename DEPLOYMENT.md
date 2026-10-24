# Production Deployment & Hosting Guide

**Project:** GeoAI Hackathon 2026 (CSRE, IIT Bombay)  
**Current Status:** Fully operational locally (`http://localhost:8000`). Ready for public cloud deployment.  

---

## 1. Local Execution (FastAPI + Uvicorn)

To launch the backend locally on Windows, macOS, or Linux:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI):** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/health`
- **Activities Directory:** `http://localhost:8000/activities`
- **External Providers Directory:** `http://localhost:8000/sources`

---

## 2. Docker Container Deployment

A multi-stage `Dockerfile` is included in the project root.

### Build and Run with Docker:
```bash
# Build the Docker image
docker build -t geoai-backend:latest .

# Run the container mapping port 8000
docker run -d -p 8000:8000 --name geoai-service geoai-backend:latest

# Check logs
docker logs -f geoai-service
```

---

## 3. Public Cloud Deployment for Judging

The GeoAI Hackathon requires an active public HTTP endpoint for judging evaluation. You can deploy this backend free of charge on any platform-neutral container host:

### Option A: Render.com (Recommended Free Cloud Hosting)
1. Push your repository to a public GitHub repo.
2. Sign in to [Render.com](https://render.com/) and click **New +** -> **Web Service**.
3. Select your GitHub repository.
4. Set the following parameters:
   - **Environment:** `Docker`
   - **Branch:** `main`
   - **Plan:** `Free`
5. Click **Create Web Service**. Render will build the Docker container and provide a live public HTTPS URL (e.g., `https://geoai-recommendation-service.onrender.com`).
6. Share this URL with the hackathon judges along with `/docs`.

### Option B: Hugging Face Spaces (Docker Space)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space SDK: Select **Docker** (Blank template).
3. Clone the repo and push this codebase to the Space git remote.
4. Hugging Face Spaces will automatically build the `Dockerfile` and expose the API publicly on port 7860/8000.

### Option C: Railway.app
1. Link your GitHub repository in Railway.
2. Railway detects the `Dockerfile` automatically and deploys the service.

---

## 4. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Operating environment (`development` or `production`). |
| `HOST` | `0.0.0.0` | Host IP binding. |
| `PORT` | `8000` | Port for HTTP requests. |
| `MODEL_ID` | `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` | Hugging Face model identifier for OpenVINO quantized inference. |
| `BASE_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model identifier for fine-tuning. |
| `OPENVINO_DEVICE` | `CPU` | Compute device (`CPU` or `GPU` for Intel Iris Xe). |
| `MAX_NEW_TOKENS` | `512` | Max token limit for response generation. |
| `ENABLE_MODEL_INFERENCE` | `true` | Set to `false` to run strictly in deterministic fallback mode. |
| `HTTP_TIMEOUT_SECONDS` | `8.0` | Network request timeout for external APIs. |
| `CACHE_TTL_SECONDS` | `900` | In-memory cache duration (15 minutes). |
| `NOMINATIM_USER_AGENT` | `GeoAI-Hackathon-2026-CSRE-IITB/1.0` | Custom User-Agent header for OpenStreetMap Nominatim. |
