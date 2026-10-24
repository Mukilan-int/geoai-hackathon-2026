# Deterministic Safety Rules & Authoritative Guardrails

**Project:** GeoAI Hackathon 2026 (CSRE, IIT Bombay)  
**Standard Authorities:** World Health Organization (WHO), US National Weather Service (NOAA/NWS), US Environmental Protection Agency (EPA), American College of Sports Medicine (ACSM).  

---

## 1. Rationale and Objective

During preliminary evaluation of candidate 1.5B small language models, an alignment inconsistency was identified:
- **Scenario:** Midday run at 34°C with UV Index 11.2 (Extreme UV Hazard).
- **Failure Mode:** The model described the severe UV risk and sunburn hazard in its text explanation, but still set `suitability: "recommended"` due to an inherent conversational bias to satisfy user requests.
- **Architectural Solution:** To guarantee user safety, the system implements **deterministic pre- and post-generation safety guardrails** (`app/safety/rules.py` and `app/safety/guardrails.py`) that strictly evaluate official scientific thresholds and automatically override model optimism.

---

## 2. Authoritative Thresholds & Guidelines

### A. Solar Ultraviolet Radiation (WHO / US EPA UV Index)
- **Guideline:** WHO *Global Solar UV Index: A Practical Guide* (2002); EPA UV Index Scale.
- **Rule `WHO_UV_EXTREME_STRENUOUS`:**
  - **Threshold:** $\text{UV Index} \ge 11.0$ (Extreme).
  - **Condition:** High or very high metabolic intensity activities (Running, Football, Cycling, Tennis) or requests during midday peak hours (10:00 AM – 04:00 PM).
  - **Mandated Suitability:** `not_recommended`.
  - **Action:** Shifts recommended window to early morning (before 07:30 AM) or post-sunset; mandates broad-spectrum SPF 50+ and UV-blocking eyewear.

### B. Thermal Stress & Heat Index (NOAA / NWS & ACSM)
- **Guideline:** US National Weather Service Heat Index Chart; American College of Sports Medicine (ACSM) *Exertional Heat Illness during Training and Competition*.
- **Rule `NOAA_HEAT_INDEX_DANGER`:**
  - **Threshold:** Apparent Temperature (Heat Index) $\ge 40.0^\circ\text{C}$ ($104^\circ\text{F}$).
  - **Condition:** Moderate, high, or very high intensity cardio.
  - **Mandated Suitability:** `not_recommended`.
  - **Physiological Basis:** High ambient humidity coupled with high temperature suppresses evaporative sweat cooling, substantially elevating clinical risk of exertional heat stroke.
- **Rule `NOAA_HEAT_INDEX_EXTREME_CAUTION`:**
  - **Threshold:** Apparent Temperature between $34.0^\circ\text{C}$ and $39.9^\circ\text{C}$.
  - **Mandated Suitability:** `caution`. Requires hydration stops every 15–20 minutes.

### C. Convective Storms & Lightning Hazard (US NWS Lightning Safety)
- **Guideline:** National Weather Service Lightning Safety Protocol: *"When Thunder Roars, Go Indoors"*.
- **Rule `NWS_LIGHTNING_THUNDERSTORM_RULE`:**
  - **Threshold:** WMO Weather Code $\in [95, 96, 99]$ (Thunderstorm with or without hail).
  - **Condition:** Open-field or metal-implement activities (Golf, Cricket, Outdoor Tennis, Football, Cycling, Hiking exposed ridges).
  - **Mandated Suitability:** `not_recommended`.
  - **Protocol:** Evacuate immediately to fully enclosed substantial buildings; wait at least 30 minutes after the last clap of thunder before resuming.

### D. Hazardous Atmospheric Air Quality (US EPA AQI Standards)
- **Guideline:** US EPA Clean Air Act Title 40 CFR Part 58.
- **Rule `EPA_AQI_HAZARDOUS`:**
  - **Threshold:** $\text{US AQI} \ge 300$ or $\text{PM2.5} \ge 250\,\mu\text{g/m}^3$.
  - **Mandated Suitability:** `not_recommended` for all activities.
  - **Physiological Basis:** Deep breathing during outdoor exercise deposits severe particulate mass into alveolar lung tissue, precipitating acute cardiovascular and respiratory emergencies.

### E. Severe Wind & Gale Conditions (WMO Beaufort Scale)
- **Rule `BEAUFORT_GALE_WIND_RULE`:**
  - **Threshold:** Measured wind gusts $\ge 65.0\text{ km/h}$ (Beaufort 8).
  - **Mandated Suitability:** `not_recommended`. Risk of falling tree branches and flying debris.
- **Rule `STRONG_WIND_CAUTION`:**
  - **Threshold:** Gusts $\ge 45.0\text{ km/h}$ for wind-sensitive sports (Cycling, Tennis, Photography, Golf).
  - **Mandated Suitability:** `caution`.

---

## 3. Post-Generation Override Mechanism

When the SLM finishes generating a response:
1. `apply_guardrail_override(...)` inspects the SLM's chosen `suitability`.
2. Suitability ranks: `recommended` (1) < `caution` (2) < `not_recommended` (3).
3. If the deterministic safety rules mandate a higher severity rank than the SLM produced (e.g., model returned `recommended` under UV 11 or AQI 345):
   - **Suitability is forcibly overridden** to the mandated rating.
   - `safety_assessment.override_applied` is set to `true`.
   - The authoritative safety rule and scientific explanation are prepended to `reasons` and `precautions`.
   - A structured warning is logged to stdout.
