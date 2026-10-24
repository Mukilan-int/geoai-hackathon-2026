# GeoAI Hackathon 2026 — Rulebook Compliance Audit Checklist

**Host Institution:** Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
**Target Architecture:** Small Language Model (`Qwen2.5-1.5B-Instruct` / OpenVINO INT4)  
**Verification Date:** 2026-10-23  

---

## 1. Compliance Matrix

| # | Exact Rulebook Requirement | Addressing Files & Tests | Verification Method | Status | Remaining Action |
| :- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Open-Weight SLM $\le$ 3 Billion Parameters**<br>Model must be open-weight with parameter count strictly $\le$ 3B. | `app/models/config.py`<br>`MODEL_AUDIT.md`<br>`run_qwen_quantized.py` | Inspected Hugging Face model config & architecture tensors (1,543,714,816 params = 1.54B). | **PASS** | None. Model is ~1.54B parameters. |
| **2** | **Public Hugging Face Repository & Open License**<br>Must be hosted publicly on Hugging Face with open license, base pointer, and model card. | `huggingface_model/`<br>`HF_MODEL_CARD.md`<br>`huggingface_model/adapter_config.json` | Inspected `adapter_config.json` (`base_model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct`) and Apache 2.0 license. | **PARTIAL** | Upload `huggingface_model/` folder to user's personal Hugging Face account (manual token required). |
| **3** | **Fine-Tuning LoRA / QLoRA Workflow**<br>Reproducible fine-tuning pipeline on base Qwen model with versioned dataset. | `training/prepare_dataset.py`<br>`training/train_lora.py`<br>`training/validate_dataset.py`<br>`training/evaluate_adapter.py` | Executed `train_lora.py` yielding genuine `adapter_model.safetensors` (4.37 MB) with 1,089,536 trainable parameters. | **PASS** | Cloud GPU training script provided for multi-epoch scaling if desired. |
| **4** | **10 Required Outdoor Activities**<br>Must explicitly support the 10 outdoor activities with distinct profiles. | `app/geospatial/activities.py`<br>`tests/test_activities.py` | Executed `pytest tests/test_activities.py`: all 10 activities verified for metadata, constraints, and OSM tags. | **PASS** | None. |
| **5** | **Live Data Integration & Grounding**<br>Grounded in live retrieved weather, air quality, and spatial venue data. | `app/services/weather.py`<br>`app/services/air_quality.py`<br>`app/services/overpass.py`<br>`app/services/geocoding.py`<br>`tests/test_api_endpoints.py` | Executed `test_post_recommend_with_coordinates_live` making verified live calls to Open-Meteo & OpenStreetMap. | **PASS** | None. Live providers operational. |
| **6** | **Deterministic Safety Guardrails**<br>Sourced safety thresholds (WHO, NOAA, EPA) with post-generation model override. | `app/safety/rules.py`<br>`app/safety/guardrails.py`<br>`tests/test_safety_guardrails.py` | Executed `pytest tests/test_safety_guardrails.py`: verified that unsafe model output at UV 11 is strictly overridden to `not_recommended`. | **PASS** | None. |
| **7** | **Strict Structured Output (JSON Schema)**<br>Machine-readable JSON adhering to defined JSON schema with zero preamble. | `schemas/recommendation.schema.json`<br>`app/utils/json_sanitizer.py`<br>`tests/test_json_sanitizer.py` | Executed Draft-07 jsonschema validation across all pipeline outputs and evaluation splits (100% pass rate). | **PASS** | None. |
| **8** | **Working HTTP API Endpoint & Query Support**<br>FastAPI backend with `/recommend`, `/query`, `/ask`, `/health`, `/activities`, `/sources` and docs. | `app/main.py`<br>`app/api/routes.py`<br>`app/services/query_parser.py`<br>`tests/test_api_endpoints.py`<br>`tests/test_query_parser.py` | Executed 41 automated tests (100% pass). Verified free-form natural language query parsing and Swagger docs at `/docs`. | **PASS (Local)**<br>**NOT YET PUBLIC** | Deploy container / repo to public hosting (Render, Railway, or Hugging Face Spaces) for judging access. |
| **9** | **Complete Repository, README & Setup**<br>Clean source repository, beginner setup instructions, and Dockerfile. | `README.md`<br>`Dockerfile`<br>`requirements.txt`<br>`.env.example`<br>`.gitignore` | Verified files and configuration integrity in workspace. | **PASS** | Push project to user's public GitHub repository. |
| **10** | **Presentation Deliverable (4–6 Slides)**<br>Presentation covering problem, architecture, example, model, safety, deployment. | `presentation.pptx`<br>`PRESENTATION.md`<br>`scripts/export_presentation.py` | Executed `export_presentation.py` generating 6-slide PowerPoint file `presentation.pptx`. | **PASS** | None. Editable `.pptx` and `.md` available. |
| **11** | **OpenStreetMap Attribution**<br>Proper ODbL attribution displayed wherever OpenStreetMap data is used. | `app/services/registry.py`<br>`DATA_SOURCES.md`<br>`README.md` | Attribution `© OpenStreetMap contributors` included in all geocoding, venue, and routing responses. | **PASS** | None. |
| **12** | **No Prohibited Hosted LLMs**<br>Zero runtime reliance on GPT-4, Claude, Gemini Pro, or hosted proprietary APIs. | `app/models/loader.py`<br>`app/models/generator.py` | Inspected model pipeline; all inference uses local OpenVINO / Transformers on local CPU/iGPU. | **PASS** | None. |

---

## 2. Overall Status Summary

- **Local Implementation, Codebase, Live Data & Guardrails:** **100% COMPLETE & VERIFIED PASS**
- **LoRA Adapter Artifacts:** **TRAINED & EXPORTED (`huggingface_model/`)**
- **Public Hosting Action Remaining for User:**
  1. Upload `huggingface_model/` to Hugging Face Hub (via `huggingface-cli upload` or web UI).
  2. Push workspace to public GitHub repository.
  3. Deploy FastAPI service to free cloud host (Render, Railway, or Hugging Face Spaces) using provided `Dockerfile` and `DEPLOYMENT.md` so judges can query the live URL.
