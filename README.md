# GeoAI Hackathon 2026 — Grounded Outdoor Activity Recommendation Backend

[![Live API](https://img.shields.io/badge/Live_API-Render-brightgreen?logo=render&style=flat-square)](https://geoai-hackathon-2026.onrender.com)
[![API Docs](https://img.shields.io/badge/API_Docs-Swagger_UI-blue?logo=swagger&style=flat-square)](https://geoai-hackathon-2026.onrender.com/docs)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-geoai--qwen2.5--1.5b--outdoor--lora-yellow?style=flat-square)](https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora)
[![GitHub Repo](https://img.shields.io/badge/GitHub-geoai--hackathon--2026-black?logo=github&style=flat-square)](https://github.com/Mukilan-int/geoai-hackathon-2026)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-informational?logo=python&style=flat-square)](https://www.python.org/)
[![Model Track](https://img.shields.io/badge/Track-%E2%89%A4_3B_Parameters_%281.54B_Base%29-purple?style=flat-square)](#)

**Host Institution:** Centre of Studies in Resources Engineering (CSRE), Indian Institute of Technology Bombay (IIT Bombay)  
**Track:** Open-Weight Small Language Model ($\le 3\text{B}$ Parameters)  
**Live API Endpoint:** [https://geoai-hackathon-2026.onrender.com](https://geoai-hackathon-2026.onrender.com)  
**Interactive API Documentation:** [https://geoai-hackathon-2026.onrender.com/docs](https://geoai-hackathon-2026.onrender.com/docs)  
**Hugging Face Model Repository:** [https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora](https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora)  
**GitHub Repository:** [https://github.com/Mukilan-int/geoai-hackathon-2026](https://github.com/Mukilan-int/geoai-hackathon-2026)  
**Base Model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.54B Parameters, Apache-2.0)  
**Quantized Inference Model:** `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` (Intel OpenVINO INT4)  
**Fine-Tuned Adapter:** [`Erevanox/geoai-qwen2.5-1.5b-outdoor-lora`](https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora) (PEFT LoRA, Apache-2.0)  

---

## Quick Reference Links

| Component | URL / Location | Description |
| :--- | :--- | :--- |
| **Live Deployed API** | [geoai-hackathon-2026.onrender.com](https://geoai-hackathon-2026.onrender.com) | Public production backend on Render (Docker) |
| **Interactive API Docs** | [/docs](https://geoai-hackathon-2026.onrender.com/docs) \| [/redoc](https://geoai-hackathon-2026.onrender.com/redoc) | Live Swagger UI and ReDoc interface |
| **Hugging Face Model** | [Erevanox/geoai-qwen2.5-1.5b-outdoor-lora](https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora) | Uploaded LoRA adapter safetensors & model card |
| **GitHub Repository** | [Mukilan-int/geoai-hackathon-2026](https://github.com/Mukilan-int/geoai-hackathon-2026) | Complete codebase, schemas, tests, and training code |
| **Presentation Deck** | [`presentation.pptx`](presentation.pptx) \| [`PRESENTATION.md`](PRESENTATION.md) | 6-slide widescreen presentation with speaking notes |
| **Validation Reports** | [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) \| [`RULEBOOK_COMPLIANCE.md`](RULEBOOK_COMPLIANCE.md) | Benchmark reports, compliance matrix, and metrics |

---

## 1. Project Overview & Hackathon Scope

This project implements a fully compliant, production-grade Python backend for an autonomous AI assistant that evaluates **where, when, and how to safely engage in outdoor activities**, identifies necessary precautions, and deterministically blocks activities when environmental conditions pose severe health or safety hazards.

```
                    ┌─────────────────────────────────────────┐
                    │       User Query / POST /recommend      │
                    │   (Coordinates, Placename, or Natural)  │
                    └────────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌───────────────────────────────┐               ┌───────────────────────────────┐
  │   Open-Meteo Live Forecast   │               │ OpenStreetMap Nominatim/POIs  │
  │  • Temp, Apparent Heat Index  │               │  • Geocoding to Lat/Lon       │
  │  • Solar UV Index, Gale Wind  │               │  • Overpass Park & Venue POIs │
  │  • EPA Air Quality (PM2.5/AQI)│               │  • OSRM Walking Distances     │
  └──────────────┬────────────────┘               └───────────────┬───────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │       Grounded Context Assembler        │
                    │  (Synthesizes Real-Time Fact Evidence)  │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │    Qwen2.5-1.5B-Instruct + LoRA         │
                    │  (OpenVINO INT4 / PyTorch PEFT Adapter) │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │    Deterministic Safety Guardrails      │
                    │  • WHO UV ≥ 11 Extreme Hazard           │
                    │  • NOAA Heat Index ≥ 40°C Heatstroke    │
                    │  • EPA AQI > 300 Hazardous Air          │
                    │  • NWS Lightning / Convective Storms    │
                    │  *Overrides optimistic model outputs*   │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   Strict Validated JSON Output Response │
                    │    (schemas/recommendation.schema.json) │
                    └─────────────────────────────────────────┘
```

### Key Capabilities:
- **Strictly Under 3B Parameters Track:** Powered by `Qwen2.5-1.5B-Instruct` (1.54B parameters) accelerated via Intel OpenVINO INT4 quantization (delivering 11–16 tokens/sec on standard Intel Core i7 CPU, and low latency in Docker).
- **10 Fully Supported Outdoor Activities:** Deep domain safety profiles for Running/Jogging, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnics, Outdoor Events, and Photography.
- **Deterministic Safety Guardrails:** Hard, scientifically sourced safety rules based on WHO UV Index, NOAA Heat Stress, NWS Lightning, and EPA AQI standards that automatically override model optimism and guarantee human safety.
- **Zero Proprietary API Dependency:** Entire stack runs on open-weight models and free, open-access live services (Open-Meteo and OpenStreetMap) without requiring GPT-4, Claude, or paid commercial API keys.
- **Strict Schema-Constrained Output:** Emits machine-readable JSON adhering to `schemas/recommendation.schema.json` with fact-level provenance tracing, candidate venue evidence, and explicit uncertainty disclosures.

---

## 2. Quickstart & Testing

### Option A: Test the Live Cloud API (Zero Setup Required)

The backend is deployed and active on Render:
- **Root URL:** `https://geoai-hackathon-2026.onrender.com`
- **Interactive Documentation:** `https://geoai-hackathon-2026.onrender.com/docs`

Test the live endpoint with `curl`:
```bash
curl -X GET "https://geoai-hackathon-2026.onrender.com/health"
```

Generate a live recommendation via cURL:
```bash
curl -X POST "https://geoai-hackathon-2026.onrender.com/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "activity": "Running or Jogging",
       "location": "Bandra Bandstand, Mumbai",
       "constraints": { "age": 28 }
     }'
```

---

### Option B: Local Setup with Python Virtualenv

#### Prerequisites:
- Python 3.10, 3.11, or 3.12 (Python 3.14 compatible)
- Git

```bash
# 1. Clone repository
git clone https://github.com/Mukilan-int/geoai-hackathon-2026.git
cd geoai-hackathon-2026

# 2. Create and activate virtual environment
# Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start local development server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Or: python scripts/run_dev.py
```
*Access local Swagger UI at `http://localhost:8000/docs`.*

---

### Option C: Docker Container Deployment

```bash
# Build Docker image
docker build -t geoai-backend:latest .

# Run container mapping port 8000
docker run -d -p 8000:8000 --name geoai-service geoai-backend:latest

# Check container logs
docker logs -f geoai-service
```

---

## 3. HTTP API Endpoints & Usage Guide

The backend exposes four core REST endpoints:

### 1. Service Health & Model Status (`GET /health`)
```bash
curl -X GET "https://geoai-hackathon-2026.onrender.com/health"
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

### 2. List Supported Activities (`GET /activities`)
```bash
curl -X GET "https://geoai-hackathon-2026.onrender.com/activities"
```
Returns domain metadata and safety parameters for all 10 activities: Running, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnic, Outdoor Gathering, and Photography.

### 3. List Data Sources & Licensing (`GET /sources`)
```bash
curl -X GET "https://geoai-hackathon-2026.onrender.com/sources"
```
Returns licensing, terms of use, and endpoints for OpenStreetMap Nominatim, Overpass API, Open-Meteo Weather, Open-Meteo Air Quality, and Project OSRM.

### 4. Generate Grounded Recommendation (`POST /recommend`)
*(Aliases: `POST /query`, `POST /ask`)*

#### Example 1: Location by Coordinates
```bash
curl -X POST "https://geoai-hackathon-2026.onrender.com/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "activity": "Cricket",
       "latitude": 19.0760,
       "longitude": 72.8777,
       "radius_km": 5.0
     }'
```

#### Example 2: Location by Placename with Constraints
```bash
curl -X POST "https://geoai-hackathon-2026.onrender.com/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "activity": "Running or Jogging",
       "location": "Bandra Bandstand, Mumbai",
       "constraints": {
         "age": 28
       }
     }'
```

#### Example 3: Natural Language Free-Form Question
```bash
curl -X POST "https://geoai-hackathon-2026.onrender.com/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Where can I go running tomorrow evening near Koregaon Park?"
     }'
```

#### Complete Validated Response Structure:
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
      "retrieval_timestamp": "2026-10-24T12:00:00+00:00",
      "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"
    },
    {
      "provider": "OpenStreetMap Overpass API",
      "service_type": "Geospatial Features",
      "retrieval_timestamp": "2026-10-24T12:00:00+00:00",
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

Language models can hallucinate or display optimism bias under hazardous weather. To ensure 100% human safety, our system couples generative recommendations with **authoritative scientific guardrails** (`app/safety/rules.py`):

| Rule Code | Authoritative Source | Threshold Trigger | Enforced Action |
| :--- | :--- | :--- | :--- |
| `WHO_UV_EXTREME` | WHO / US EPA Global Solar UV Index | UV Index $\ge 11.0$ | Mandates `not_recommended` for daytime cardio; enforces SPF 50+ |
| `NOAA_HEAT_INDEX_DANGER` | NOAA / NWS Heat Stress Index | Apparent Temp $\ge 40.0^\circ\text{C}$ | Mandates `not_recommended` due to exertional heatstroke hazard |
| `EPA_AQI_HAZARDOUS` | US EPA Air Quality Index | US AQI $> 300$ or PM2.5 $> 250\,\mu\text{g/m}^3$ | Strictly blocks all outdoor physical activities |
| `EPA_AQI_VERY_UNHEALTHY` | US EPA Air Quality Index | US AQI 201–300 | Blocks high-intensity athletics (Running, Football, Cycling) |
| `NWS_LIGHTNING_RISK` | NOAA National Weather Service | WMO codes 95, 96, 99 (Thunderstorms) | Blocks open-field sports (Golf, Cricket, Soccer) with 0 shelter |
| `GALE_WIND_HAZARD` | WMO Wind Classification | Wind gusts $> 65\text{ km/h}$ | Blocks cycling and outdoor athletics due to debris/trauma risk |

**Deterministic Override Engine:**  
If the generative SLM generates `recommended` or `partially_recommended` during any hazardous condition, `apply_guardrail_override()` forcibly overrides the suitability to `not_recommended`, logs the override event, and injects the verified scientific rationale into `reasons`.

---

## 5. Model Fine-Tuning & Hugging Face Repository

The custom LoRA adapter is published on Hugging Face:  
🔗 **[Erevanox/geoai-qwen2.5-1.5b-outdoor-lora](https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora)**

### LoRA Specifications:
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B Parameters, Apache-2.0)
- **Trainable Parameters:** 1,089,536 (0.0705% of base model)
- **Adapter Configuration:** PEFT LoRA (Rank $r=8$, Alpha $\alpha=16$, Target modules `["q_proj", "v_proj"]`, Dropout $0.05$)
- **Weights Size:** 4.37 MB (`adapter_model.safetensors`)

### Running Inference via Hugging Face PEFT:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "Erevanox/geoai-qwen2.5-1.5b-outdoor-lora"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    device_map="auto"
)

# Attach fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# Prompt with environmental evidence
messages = [
    {"role": "system", "content": "You are a grounded GeoAI outdoor safety assistant..."},
    {"role": "user", "content": "Activity: Running | Location: Bandra Bandstand | Temp: 34C | UV: 11.2 | AQI: 75"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Reproducible Training Pipeline:
All scripts to prepare datasets, validate schema compliance, and train adapters are contained in `training/`:
```bash
# 1. Prepare and validate JSONL datasets
python training/prepare_dataset.py
python training/validate_dataset.py

# 2. Train LoRA adapter locally or on cloud GPU
python training/train_lora.py --epochs 3 --max-steps 50 --output-dir huggingface_model

# 3. Evaluate adapter against held-out test split
python training/evaluate_adapter.py --eval-data datasets/eval_recommendations.jsonl
```

---

## 6. Automated Testing & Verification

The repository contains an automated test suite verifying model quantization, live APIs, safety guardrails, and schema conformance:

```bash
# Run fast automated evaluation runner (32 unit tests, ~10s)
python scripts/run_eval_suite.py

# Run complete pytest suite (41 tests passing)
python -m pytest tests/ -v
```

### Test Coverage Highlights:
- **Unit Tests:** Health check, activity catalog, source attribution, query parser, safety guardrail overrides.
- **Integration Tests:** Live Open-Meteo weather and air quality fetches, Nominatim geocoding, Overpass POI retrieval.
- **Model Tests:** OpenVINO INT4 model loading, inference benchmark, LoRA adapter parameter compliance ($\le 3\text{B}$).
- **Schema Validation:** Strict adherence of all recommendation outputs to `schemas/recommendation.schema.json`.

---

## 7. Deliverables & Submission Status

All requirements outlined in the **GeoAI Hackathon 2026 Rulebook** have been completed, verified, and deployed:

| Deliverable | Status | Verification & Link |
| :--- | :---: | :--- |
| **Public GitHub Repository** | **COMPLETED** | [github.com/Mukilan-int/geoai-hackathon-2026](https://github.com/Mukilan-int/geoai-hackathon-2026) |
| **Live Deployed API Service** | **COMPLETED** | [geoai-hackathon-2026.onrender.com](https://geoai-hackathon-2026.onrender.com) (Deployed on Render via Docker) |
| **Interactive API Documentation** | **COMPLETED** | [geoai-hackathon-2026.onrender.com/docs](https://geoai-hackathon-2026.onrender.com/docs) (Swagger UI) |
| **Hugging Face Model & LoRA Adapter** | **COMPLETED** | [huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora](https://huggingface.co/Erevanox/geoai-qwen2.5-1.5b-outdoor-lora) |
| **Model Size Limit ($\le 3\text{B}$ parameters)** | **COMPLETED** | Base model has 1.54B parameters + 1.09M LoRA adapter parameters |
| **10 Supported Activities** | **COMPLETED** | Running, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnic, Events, Photography |
| **Live Open Data Integration** | **COMPLETED** | Open-Meteo Weather/AQ + OpenStreetMap Nominatim/Overpass (Zero paid/proprietary APIs) |
| **Deterministic Safety Overrides** | **COMPLETED** | Scientific thresholds (WHO UV, NOAA Heat, EPA AQI, NWS Lightning) strictly enforced |
| **Validated Machine-Readable JSON** | **COMPLETED** | Validated against `schemas/recommendation.schema.json` |
| **Presentation Deliverable** | **COMPLETED** | `presentation.pptx` (6 widescreen slides) and `PRESENTATION.md` |
| **Automated Test Suite** | **COMPLETED** | 41/41 automated tests passing (`pytest tests/`) |

---

## 8. OpenStreetMap & Live Data Attribution

In compliance with OpenStreetMap and Open-Meteo terms of service:
- **OpenStreetMap Data:** Mapped features, park boundaries, and geocoding results are licensed under the [Open Data Commons Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/) by the OpenStreetMap Foundation. Attribution: **`© OpenStreetMap contributors`** ([https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)).
- **Weather and Air Quality Data:** Provided by [Open-Meteo.com](https://open-meteo.com/) under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
- **Routing Data:** Powered by Project OSRM and OpenStreetMap.

---

## 9. License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.  
Base model weights `Qwen/Qwen2.5-1.5B-Instruct` are released under the Apache 2.0 License by the Qwen Team / Alibaba Cloud.
