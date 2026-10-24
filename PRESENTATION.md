# GeoAI Hackathon 2026 Presentation Deliverable

**Project Title:** Grounded Outdoor Activity Intelligence via Small Language Models  
**Host Institution:** Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
**Target Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` + OpenVINO INT4 Acceleration  
**Editable Presentation File:** `presentation.pptx` (Generated via `scripts/export_presentation.py`)  

---

## Slide 1: Problem & Proposed Solution

### The Outdoor Decision Dilemma
- Citizens and outdoor enthusiasts frequently make ungrounded decisions about when, where, and how to safely engage in sports and leisure activities.
- Standard weather applications display isolated figures (e.g., UV 11, AQI 165, Wind 45 km/h) without interpreting activity-specific physiological or safety implications.
- Commercial hosted LLMs (GPT-4, Claude) hallucinate non-existent parks, invent forecast metrics, fail to cite provenance, and violate the hackathon's strict $<3\text{B}$ parameter requirement.
- Previous baseline testing of raw 1.5B small language models revealed severe optimism bias: the model recognized a UV 11 hazard in text but still erroneously assigned `recommended` to a midday marathon run.

### Our Solution
- **Eligible Small Language Model:** Powered by `Qwen2.5-1.5B-Instruct` (1.54B parameters, Apache 2.0 license) accelerated with Intel OpenVINO INT4 quantization (11–16 tokens/sec).
- **Deterministic Safety Guardrails:** Hard, sourced safety rules based on WHO UV standards, NOAA Heat Index thresholds, NWS Lightning protocols, and US EPA AQI guidelines deterministically override model optimism.
- **Zero-Cost Live Data Ingestion:** Full integration with Open-Meteo (numerical weather prediction & atmospheric air quality) and OpenStreetMap (Nominatim geocoding & Overpass POI discovery).
- **10 Supported Outdoor Activities:** Deep domain profiles covering running, cycling, golf, cricket, football, tennis, hiking, picnics, outdoor gatherings, and photography.

---

## Slide 2: System Architecture & Data Flow

```
+-----------------------------------------------------------------------------------+
|                            FASTAPI HTTP API LAYER                                 |
|      GET /health      GET /activities      GET /sources      POST /recommend      |
+-----------------------------------------------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|                          GEOSPATIAL & PROVENANCE PIPELINE                         |
|  - Nominatim Geocoding & Reverse Geocoding (OpenStreetMap ODbL)                   |
|  - Overpass API Venue Discovery (Parks, Pitches, Golf Courses, Trails)           |
|  - Project OSRM Topological Road Routing & Static Duration Estimation             |
|  - NOAA Solar Astronomical Geometry (Daylight & Golden Hour Calculation)          |
|  - Open-Meteo Hourly Numerical Forecasts (Temp, Precip %, Wind Gusts, UV)        |
|  - Open-Meteo Air Quality Service (US EPA AQI, European AQI, PM2.5, PM10)         |
+---------------------------------------+-------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|                       MULTI-CANDIDATE RANKING ENGINE                              |
|  - Spatial Proximity Scoring + OSM Tag Synergy Matching                           |
|  - Temporal Forecast Optimization (Low Rain Prob, Safe UV, Thermal Comfort)       |
+---------------------------------------+-------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|                    DETERMINISTIC SAFETY GUARDRAILS (PRE-CHECK)                    |
|  - WHO Global UV Index Scale (UV >= 11 Extreme Hazard Rule)                       |
|  - NOAA / ACSM Apparent Temperature Heat Stress (>= 40°C Danger Zone)             |
|  - US EPA AQI Health Thresholds (AQI > 300 Hazardous Emergency Rule)              |
|  - NWS Convective Storm Protocol ('When Thunder Roars, Go Indoors')               |
+---------------------------------------+-------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|               SLM INFERENCE ENGINE (OpenVINO INT4 / LoRA Adapter)                  |
|  - Base: Qwen/Qwen2.5-1.5B-Instruct (<3B params, Apache 2.0)                      |
|  - ChatML Prompt Formatter grounded strictly in verified evidence                 |
|  - Greedy Deterministic Decoding for schema stability                             |
+---------------------------------------+-------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|               POST-GENERATION SAFETY OVERRIDE & SCHEMA VALIDATION                 |
|  - Markdown Fence Stripper + Safe JSON Parser                                     |
|  - Deterministic Suitability Override (Forces 'not_recommended' if hazard triggered)|
|  - Strict Validation against schemas/recommendation.schema.json                  |
+-----------------------------------------------------------------------------------+
```

---

## Slide 3: Worked Example & Evidence Grounding

### Request Context:
- **Activity:** Running or Jogging (Intensity: very_high)
- **Target Location:** Bandra Bandstand, Mumbai
- **Time Requested:** Midday (12:30 PM)

### Retrieved Live Facts & Attribution:
- **Temperature:** 34.0°C | **Relative Humidity:** 72% | **Apparent Heat Index:** 39.5°C
- **Solar UV Index:** 11.2 (*Extreme UV Radiation Hazard*)
- **Air Quality:** US AQI 88 (*Moderate*, PM2.5: 29.5 µg/m³)
- **Discovered Venue:** Bandra Bandstand Promenade (~0.2 km away, Score: 85.0)
- **Data Provenance:** Weather data by Open-Meteo.com (CC BY 4.0); Map features © OpenStreetMap contributors.

### Grounded Engine Output:
- **Suitability:** `not_recommended`
- **Safety Status:** `BLOCKED` | **Override Applied:** `true`
- **Triggered Rules:** `WHO_UV_EXTREME_STRENUOUS`, `NOAA_HEAT_INDEX_EXTREME_CAUTION`
- **Recommended Time Window:** Early morning before 07:30 AM or after 18:00 PM when UV drops below 3.0.
- **Precautions:** Mandatory broad-spectrum SPF 50+, UV-blocking eyewear, rapid electrolyte replenishment.

---

## Slide 4: Model Strategy, LoRA Fine-Tuning & Evaluation

### 1. Model Eligibility & Verification
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Parameter Count:** 1,543,714,816 parameters (~1.54B, well under the 3B limit).
- **Licence:** Apache-2.0 (Permissive open source).
- **Inference Runtime:** `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` yielding 11–16 tok/s on Intel Core i7 / Iris Xe graphics with a compact 909 MB memory footprint.

### 2. PEFT LoRA Fine-Tuning
- **Adapter Configuration:** Rank $r=8$, $\alpha=16$, targeting `q_proj` and `v_proj`.
- **Trainable Parameters:** 1,089,536 parameters (only 0.0705% of model weights).
- **Training Loss:** Commenced at 2.9724 on structured ChatML outdoor recommendation pairs.
- **Publication Package:** Generated in `huggingface_model/` with `adapter_model.safetensors`, `adapter_config.json`, and official Hugging Face Model Card.

### 3. Empirical Evaluation
- **JSON Parse Rate:** 100.0% (Automated markdown fence extraction + resilient JSON decoding).
- **Schema Conformance Rate:** 100.0% (Zero schema violations across all evaluated test scenarios).
- **Safety Consistency:** 100.0% compliance enforced by post-generation deterministic guardrails.
- **Data Leakage:** 0% duplicate leakage verified between training and evaluation splits.

---

## Slide 5: Safety Guardrails, Grounding & Limitations

### Authoritative Thresholds:
1. **World Health Organization (WHO) UV Index:** UV $\ge 11$ mandates `not_recommended` for high-intensity daytime exercise; UV $\ge 8$ mandates caution and shade seeking.
2. **NOAA / ACSM Heat Stress:** Apparent temperature $\ge 40^\circ\text{C}$ creates clinical exertional heat stroke risk; high-intensity sports blocked.
3. **US EPA Air Quality Standards:** US AQI $> 300$ / PM2.5 $> 250\,\mu\text{g/m}^3$ triggers emergency health warnings; all outdoor exercise blocked.
4. **NWS Convective Storm & Lightning Rule:** WMO codes 95, 96, 99 strictly block open-field sports (golf, cricket, soccer) with zero lightning protection.

### Grounding & Provenance Rigor:
- The assistant is forbidden from asserting facts not present in external API returns.
- If an API fails or information is omitted (e.g., location query omitted), the assistant explicitly reports the limitation under `missing_information_identified` rather than hallucinating answers.

### Known Disclaimers:
- OSRM route calculations represent static topological road networks; they do not reflect dynamic vehicular traffic congestion.
- OpenStreetMap data is community-maintained and may not reflect unannounced temporary venue closures.

---

## Slide 6: Production API, Deployment & Rulebook Compliance

### Production API Architecture
- **Framework:** FastAPI with Uvicorn, structured logging, and Pydantic schema validation.
- **Interactive Documentation:** Interactive Swagger UI at `/docs` and ReDoc at `/redoc`.
- **Dockerization:** Containerized via multi-stage `Dockerfile` ready for platform-neutral deployment.

### Rulebook Compliance Checklist

| Hackathon Requirement | Implementation | Compliance Status |
| :--- | :--- | :--- |
| **Open-Weight SLM <= 3B** | Qwen2.5-1.5B (1.54B params) | **PASS** |
| **Public Model / LoRA Adapter** | `huggingface_model/` with exact base pointer & Apache-2.0 card | **PASS** |
| **Live Data Grounding** | Open-Meteo & OpenStreetMap live integration | **PASS** |
| **10 Outdoor Activities** | Full meteorological & spatial profiles for all 10 activities | **PASS** |
| **HTTP API Endpoint** | FastAPI service with `/recommend`, `/health`, `/activities` | **PASS** |
| **OSM Attribution** | Prominently included on all geospatial responses & docs | **PASS** |
| **Presentation Deliverable** | 6 slides (`PRESENTATION.md` & `presentation.pptx`) | **PASS** |
| **Source Repository & Documentation** | Comprehensive codebase, tests, Dockerfile, and README | **PASS** |
