# GeoAI Hackathon 2026 — Grounded Outdoor Activity Recommendation Backend

**Host Institution:** Centre of Studies in Resources Engineering (CSRE), Indian Institute of Technology Bombay (IIT Bombay)  
**Track:** Open-Weight Small Language Model ($\le 3\text{B}$ Parameters)  
**Live API Endpoint:** `https://YOUR_PUBLIC_APP_NAME.onrender.com` *(Local: `http://localhost:8000`)*  
**Hugging Face Model Repository:** `https://huggingface.co/YOUR_HF_USERNAME/geoai-qwen2.5-1.5b-outdoor-lora`  
**GitHub Repository:** `https://github.com/YOUR_USERNAME/geoai-hackathon-2026`  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B Parameters, Apache-2.0)  
**Quantized Inference Model:** `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`  
**Fine-Tuned Adapter:** `huggingface_model/` (PEFT LoRA, Apache-2.0)  

---

## 1. Project Overview & Rolebook Objectives

This project implements a fully compliant, production-grade Python backend for an AI assistant that evaluates where, when, and how to safely engage in outdoor activities, what precautions to take, and when an activity should be strictly avoided.

### Key Capabilities:
- **Small Language Model Under 3B Parameters:** Utilizes `Qwen2.5-1.5B-Instruct` (1.54B parameters) accelerated via Intel OpenVINO INT4 quantization (delivering 11–16 tokens/sec on standard Intel Core i7 / Iris Xe laptops).
- **10 Supported Outdoor Activities:** Deep domain safety profiles for Running/Jogging, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnics, Outdoor Events, and Photography.
- **Deterministic Safety Guardrails:** Hard, sourced safety rules based on WHO UV Index, NOAA Heat Stress, NWS Lightning, and EPA AQI standards that automatically override model optimism.
- **Zero Proprietary API Dependency:** All features run on open-source weights and free, open-access live services (Open-Meteo and OpenStreetMap) without requiring GPT-4, Claude, or paid API keys.
- **Strict Schema-Constrained Output:** Emits machine-readable JSON adhering to `schemas/recommendation.schema.json` with fact-level provenance tracing.

---

## 2. Quickstart Setup Guide (Beginner Friendly)

### Prerequisites:
- Python 3.10, 3.11, or 3.12 (Python 3.14 compatible)
- Git (optional, for version control)

### Step 1: Clone or Navigate to the Workspace
```bash
cd "d:/GeoAI/Qwen test-1"
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```bash
# On Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Development Backend
```bash
python scripts/run_dev.py
```
*The service will start on `http://localhost:8000`.*  
- Open your browser to **`http://localhost:8000/docs`** to view and test all endpoints interactively using Swagger UI.

---

## 3. HTTP API Endpoints & Request Examples

The backend is built using FastAPI and exposes four core endpoints:

### A. Health & Dependency Status (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health"
```
**Example Response:**
```json
{
  "status": "healthy",
  "service": "GeoAI Outdoor Activity Recommendation Service",
  "process_memory_rss_mb": 2446.5,
  "model_status": {
    "model_loaded": true,
    "device": "CPU",
    "model_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
  },
  "framework": "FastAPI + Intel OpenVINO / PyTorch",
  "supported_activities_count": 10
}
```

### B. List Supported Activities (`GET /activities`)
```bash
curl -X GET "http://localhost:8000/activities"
```
Returns metadata for all 10 activities: Running, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnic, Outdoor Gathering, and Photography.

### C. List External Data Sources & Attribution (`GET /sources`)
```bash
curl -X GET "http://localhost:8000/sources"
```
Returns licensing, terms of use, and endpoints for OpenStreetMap Nominatim, Overpass API, Open-Meteo Weather, Open-Meteo Air Quality, and Project OSRM.

### D. Generate Grounded Recommendation (`POST /recommend`)
#### Request 1: Location by Coordinates
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "activity": "Cricket",
       "latitude": 19.0760,
       "longitude": 72.8777,
       "radius_km": 5.0
     }'
```

#### Request 2: Location by Placename Query with User Constraints
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "activity": "Running or Jogging",
       "location": "Bandra Bandstand, Mumbai",
       "constraints": {
         "age": 28
       }
     }'
```

#### Request 3: Free-Form Natural Language Question (Rulebook Format)
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Where can I go running tomorrow evening near Koregaon Park?"
     }'
```
*(Also accessible via `POST /query` and `POST /ask`)*


#### Example Validated Schema Response:
```json
{
  "activity_name": "Running or Jogging",
  "suitability": "not_recommended",
  "recommended_time_window": "Early morning before 07:30 AM or after 18:00 PM when UV drops below 3.0.",
  "precautions": [
    "[Extreme UV Radiation Hazard] Measured UV Index of 11.2 exceeds WHO Extreme threshold (>=11.0).",
    "High-traction running footwear",
    "Broad-spectrum SPF 50+ sunscreen and UV-blocking sunglasses"
  ],
  "reasons": [
    "MANDATORY SAFETY RULE: [Extreme UV Radiation Hazard] Measured UV Index of 11.2 exceeds the WHO 'Extreme' threshold (>=11.0).",
    "Midday sun exposure at 34.0°C and 72% humidity creates severe exertional heat illness risk."
  ],
  "missing_information_identified": [],
  "grounded_facts_used": [
    "Temperature: 34.0°C (Apparent: 39.5°C)",
    "UV Index: 11.2",
    "Relative Humidity: 72%"
  ],
  "safety_assessment": {
    "guardrail_status": "BLOCKED",
    "triggered_rules": [
      "WHO_UV_EXTREME_STRENUOUS"
    ],
    "override_applied": true,
    "source_guidelines": [
      "World Health Organization (WHO) & US EPA Global Solar UV Index"
    ]
  },
  "candidate_places": [
    {
      "name": "Bandra Bandstand Promenade",
      "category": "park",
      "distance_km": 0.2,
      "suitability_score": 85.0,
      "source_evidence": "Mapped in OpenStreetMap (park)"
    }
  ],
  "data_sources_used": [
    {
      "provider": "Open-Meteo Weather Forecast API",
      "service_type": "Numerical Weather Prediction",
      "retrieval_timestamp": "2026-10-23T15:58:40.125000+00:00",
      "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
    },
    {
      "provider": "OpenStreetMap Overpass API",
      "service_type": "Geospatial Features",
      "retrieval_timestamp": "2026-10-23T15:58:41.210000+00:00",
      "attribution": "© OpenStreetMap contributors"
    }
  ],
  "uncertainties_and_limitations": [
    "Weather forecasts represent numerical atmospheric models with inherent temporal uncertainty.",
    "OpenStreetMap features are community-mapped and may not reflect temporary facility closures.",
    "Static routing duration estimates do not model active vehicular traffic congestion."
  ]
}
```

---

## 4. Deterministic Safety Guardrails

The assistant implements authoritative scientific rules that cannot be bypassed by language model hallucinations:
1. **WHO Solar UV Index:** UV $\ge 11.0$ strictly mandates `not_recommended` for strenuous daytime cardio; UV $\ge 8.0$ mandates caution and shade.
2. **NOAA / NWS Heat Stress:** Apparent temperature (Heat Index) $\ge 40.0^\circ\text{C}$ strictly mandates `not_recommended` due to exertional heat stroke risk.
3. **US EPA Air Quality Index:** US AQI $> 300$ or PM2.5 $> 250\,\mu\text{g/m}^3$ blocks all outdoor physical activities; AQI 201–300 blocks high-intensity athletics.
4. **NWS Convective Storm / Lightning:** WMO codes 95, 96, 99 strictly block open-field sports (golf, cricket, soccer) with zero lightning protection.
5. **WMO Gale Wind:** Wind gusts $> 65\text{ km/h}$ block cycling and outdoor sports due to falling debris hazard.
6. **Deterministic Override:** If the SLM generates `recommended` under any hazardous condition, `apply_guardrail_override()` forcibly overrides the suitability to `not_recommended`, logs the override event, and injects the scientific rationale.

---

## 5. LoRA Fine-Tuning Pipeline

A complete, reproducible LoRA fine-tuning workflow is implemented in `training/`:

```bash
# 1. Prepare and validate JSONL datasets
python training/prepare_dataset.py
python training/validate_dataset.py

# 2. Train LoRA adapter locally (CPU smoke run)
python training/train_lora.py --max-steps 1 --device cpu --output-dir huggingface_model

# 3. Cloud GPU training (Google Colab T4 / Kaggle)
python training/train_lora.py --epochs 3 --max-steps 50 --device cuda --output-dir huggingface_model

# 4. Evaluate adapter against held-out test split
python training/evaluate_adapter.py --eval-data datasets/eval_recommendations.jsonl
```

### Model Artifacts in `huggingface_model/`:
- `adapter_model.safetensors` (4.37 MB trained weights, 1,089,536 trainable parameters, rank 8, alpha 16)
- `adapter_config.json` (explicitly pointing to `base_model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"`)
- `README.md` (official Hugging Face Model Card with Apache 2.0 license)
- `training_metrics.json` (recorded loss and training timestamps)

---

## 6. Running the Automated Test Suite

To run the complete automated test suite (41 tests):

```bash
# Run the fast automated evaluation runner (32 unit tests, ~10s)
python scripts/run_eval_suite.py

# Or run pytest directly across all unit & live integration tests (41 tests)
python -m pytest tests/ -v
```

---

## 7. Presentation Deliverable

- **Editable PowerPoint Presentation:** `presentation.pptx` (6 widescreen slides generated via `scripts/export_presentation.py`).
- **Markdown Slide Script:** `PRESENTATION.md` (Full slide content, diagrams, and speaking notes).

---

## 8. OpenStreetMap & Live Data Attribution

- **OpenStreetMap Data:** Mapped features, park boundaries, and geocoding results are licensed under the [Open Data Commons Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/) by the OpenStreetMap Foundation. Attribution: **`© OpenStreetMap contributors`** ([https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)).
- **Weather and Air Quality Data:** Provided by [Open-Meteo.com](https://open-meteo.com/) under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
- **Routing Data:** Powered by Project OSRM and OpenStreetMap.

---

## 9. Status of Deliverables & Remaining User Steps

### What is 100% Completed & Verified:
1. Full backend architecture (`app/`) with zero proprietary hosted LLM dependencies.
2. OpenVINO INT4 quantized model integration and PyTorch base model compatibility.
3. PEFT LoRA training pipeline executed locally (`adapter_model.safetensors` generated).
4. All 10 outdoor activities fully configured with metabolic and weather sensitivities.
5. Live integrations with Open-Meteo (Weather & AQ) and OpenStreetMap (Nominatim & Overpass).
6. Natural language query understanding (`parse_free_form_query`) matching Rulebook evaluation queries.
7. Deterministic safety guardrails preventing unsafe model approvals.
8. Strict JSON Schema validation conforming to `schemas/recommendation.schema.json`.
9. Complete test suite (41 tests passing).
10. Editable 6-slide PowerPoint presentation (`presentation.pptx`) and `PRESENTATION.md`.
11. Multi-stage `Dockerfile`, `.env.example`, `.gitignore`, and documentation reports.



### Remaining Manual Steps for Hackathon Submission:
1. **Upload LoRA Adapter to Hugging Face:**
   ```bash
   huggingface-cli login
   huggingface-cli upload YOUR_HF_USERNAME/geoai-qwen2.5-1.5b-outdoor-lora ./huggingface_model .
   ```
2. **Push to Public GitHub Repository:**
   ```bash
   git init
   git add .
   git commit -m "GeoAI Hackathon 2026: Complete Backend & Model Implementation"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/geoai-hackathon-2026.git
   git push -u origin main
   ```
3. **Deploy to Public Cloud Endpoint:**
   Follow `DEPLOYMENT.md` to deploy the repository to Render.com, Railway.app, or Hugging Face Spaces Docker to obtain a public live judging URL.
