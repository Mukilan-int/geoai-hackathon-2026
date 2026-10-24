# Structured JSON Output & Grounding Evaluation Report
**Project:** GeoAI Hackathon 2026 (CSRE, IIT Bombay)  
**Evaluated Candidate:** `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B Parameters, Apache-2.0)  
**Execution Runtime:** Intel OpenVINO / `optimum-intel` on Intel Core i7-1265U CPU / Iris Xe iGPU  

---

## 1. Model Verification & Rulebook Compliance Audit

| Requirement | Audit Finding | Compliance Status |
| :--- | :--- | :--- |
| **Model Repository** | Public on Hugging Face: [`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov) | **Verified Compliant** |
| **Base Model & Pointer** | Explicitly points to [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with `base_model_relation: quantized` | **Verified Compliant** |
| **Parameter Count** | 1,543,714,816 parameters (~1.54B) | **Verified Compliant** (<= 3B limit) |
| **Licence** | Apache 2.0 (Permissive open source licence, commercial/research permitted) | **Verified Compliant** |
| **Loading Interface** | Supported directly via Hugging Face `optimum.intel.OVModelForCausalLM.from_pretrained(...)` | **Verified Compliant** |
| **Prohibited Dependencies** | Zero calls to external proprietary APIs (GPT-4, Claude, Gemini Pro) | **Verified Compliant** |

---

## 2. Evaluation Methodology & General Schema

To evaluate structured output reliability without hard-coding responses, a single unified JSON Schema and system prompt were applied across **five distinct, challenging outdoor scenarios**.

### Recommendation JSON Schema:
```json
{
  "type": "object",
  "required": [
    "activity_name",
    "suitability",
    "recommended_time_window",
    "precautions",
    "reasons",
    "missing_information_identified",
    "grounded_facts_used"
  ],
  "properties": {
    "activity_name": { "type": "string" },
    "suitability": { "type": "string", "enum": ["recommended", "caution", "not_recommended"] },
    "recommended_time_window": { "type": "string" },
    "precautions": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "reasons": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "missing_information_identified": { "type": "array", "items": { "type": "string" } },
    "grounded_facts_used": { "type": "array", "items": { "type": "string" } }
  },
  "additionalProperties": false
}
```

---

## 3. Empirical Results Across the 5 Test Scenarios

| Scenario ID & Condition | Input / Output Tokens | Generation Latency | Parse Method | Schema Valid | Chosen Suitability | Consistency & Safety Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. High UV Running** (34°C, UV 11 Extreme, 12–2 PM midday run) | 718 / 180 tok | 14.43 s (12.48 tok/s) | Fence Stripped | **Passed (0 errors)** | `recommended` | **Safety Inconsistency:** Model correctly identified UV 11 risk in reasons/precautions, but incorrectly assigned `recommended` instead of `caution` or `not_recommended` for midday running. |
| **2. Cycling in Rain Forecast** (25°C, 85% thunderstorm in 45m, 45 km/h gusts, ghat road) | 718 / 150 tok | 26.57 s (5.64 tok/s) | Fence Stripped | **Passed (0 errors)** | `not_recommended` | **Consistent & Safe:** Correctly warned against wet road slipping, wind gusts, and thunderstorm hazard. |
| **3. Picnic with Missing Location** (24°C, clear, Location omitted: `[NOT PROVIDED]`) | 707 / 120 tok | 23.83 s (5.04 tok/s) | Fence Stripped | **Passed (0 errors)** | `recommended` | **Partial Grounding:** Avoided fabricating specific park amenities, but left `missing_information_identified` empty rather than explicitly reporting location omission. |
| **4. Hiking in Severe Air Quality** (AQI 345 Hazardous, PM2.5 extreme, steep summit hike) | 703 / 162 tok | 28.89 s (5.61 tok/s) | Fence Stripped | **Passed (0 errors)** | `not_recommended` | **Consistent & Safe:** Correctly flagged severe respiratory risks and high physical exertion in hazardous air. |
| **5. Older Adult Stroll** (74-year-old, knee osteoarthritis, cane, Cubbon Park, 23°C) | 714 / 159 tok | 28.69 s (5.54 tok/s) | Fence Stripped | **Passed (0 errors)** | `recommended` | **Consistent & Grounded:** Recommended gentle walking with walking cane support and comfortable pace on flat paths. |

---

## 4. Summary Statistics & Key Findings

| Metric | Measured Value | Analysis & Implication for Assistant Design |
| :--- | :--- | :--- |
| **Direct JSON Rate (Raw without fences)** | **0.0% (0/5)** | Model consistently wraps its JSON in markdown ```` ```json ... ``` ```` fences despite system prompt instruction. |
| **Valid JSON Rate (After fence extraction)** | **100.0% (5/5)** | Extraction regex ```` ```(?:json)?([\s\S]*?)``` ```` yields 100% syntactically valid JSON. |
| **Schema Conformance Rate** | **100.0% (5/5)** | **Zero missing required fields** and zero type violations across all 5 test scenarios. |
| **Hallucination of Live Data Checks** | **0 occurrences** | Model did **not** claim to have contacted external APIs, fetched real-time weather, or queried OpenStreetMap. |
| **Safety Logic Weakness Identified** | **1 out of 5 scenarios** | Under extreme thermal/UV conditions (Scenario 1), the 1.5B model exhibited an alignment bias towards satisfying user requests ("recommended") despite noting hazards in the text. |

---

## 5. Architectural Recommendations for Next Phase

1. **Client-Side JSON Parser Sanitizer:**
   The assistant's response pipeline must always apply a markdown code-fence stripper before invoking `json.loads()` to ensure seamless parsing.
2. **Rule-Based Post-Validation Guardrail:**
   Because a 1.5B SLM can occasionally display optimism bias (e.g., approving a run in UV 11), the GeoAI assistant should implement a lightweight rule-based safety filter (e.g., `if uv_index >= 11 or aqi >= 300: force suitability to caution/not_recommended`).
3. **Explicit Missing-Data Prompting:**
   When integrating free APIs (Open-Meteo, OpenStreetMap), missing parameters must be pre-screened before passing context to the SLM to prevent implicit assumptions.
