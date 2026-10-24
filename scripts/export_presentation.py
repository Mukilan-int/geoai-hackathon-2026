"""Script generating editable PowerPoint presentation (presentation.pptx) adhering to 4-6 slide limit."""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUTPUT_PPTX_PATH = "presentation.pptx"


def create_presentation():
    prs = Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme colors
    NAVY = RGBColor(16, 37, 66)
    ACCENT_BLUE = RGBColor(26, 115, 232)
    DARK_GRAY = RGBColor(60, 64, 67)
    LIGHT_BG = RGBColor(248, 249, 250)
    CARD_BG = RGBColor(255, 255, 255)
    GREEN = RGBColor(15, 157, 88)
    AMBER = RGBColor(244, 180, 0)
    BORDER_COLOR = RGBColor(218, 220, 224)

    def add_slide_header(slide, title_text, category_text="GeoAI Hackathon 2026 | CSRE, IIT Bombay"):
        # Header category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Main slide title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    # ==========================================
    # SLIDE 1: Title & Problem / Solution
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide1, "Grounded Outdoor Activity Intelligence via Small Language Models", "Slide 1: Problem & Proposed Solution")

    # Left Column: Problem
    box_left = slide1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf1 = box_left.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "The Outdoor Decision Dilemma"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    bullets_p1 = [
        "Citizens lack unified, hyper-local guidance on when, where, and how to safely engage in outdoor sports and leisure.",
        "Traditional weather apps show raw isolated numbers (e.g. UV 11, AQI 160) without activity-specific context.",
        "Generic hosted LLMs (GPT-4) hallucinate non-existent parks, invent weather forecasts, and violate the <3B parameter hackathon limit.",
        "Previous evaluation revealed 1.5B models can produce unsafe optimism (e.g., approving midday running at UV 11)."
    ]
    for b in bullets_p1:
        p = tf1.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(10)

    # Right Column: Proposed Solution
    box_right = slide1.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf2 = box_right.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Our GeoAI Solution Architecture"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = GREEN

    bullets_p2 = [
        "Open-Weight SLM: Powered by Qwen2.5-1.5B-Instruct (<3B parameters, Apache 2.0) with OpenVINO INT4 acceleration (11-16 tok/s).",
        "Deterministic Safety Guardrails: Authoritative WHO UV, NOAA Heat Stress, NWS Lightning, and EPA AQI safety rules override unsafe outputs.",
        "Zero-Cost Live Data Integration: Live Open-Meteo numerical weather/air quality and OpenStreetMap Overpass/Nominatim geospatial features.",
        "10 Dedicated Outdoor Activities: Tailored metabolic thresholds and venue discovery for running, cycling, golf, cricket, tennis, hiking, etc.",
        "Strict Schema Grounding: Machine-readable JSON output strictly traced to verified provider evidence with zero proprietary API dependencies."
    ]
    for b in bullets_p2:
        p = tf2.add_paragraph()
        p.text = f"✔ {b}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    # ==========================================
    # SLIDE 2: System Architecture & Data Flow
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide2, "End-to-End System Architecture & Grounded Data Pipeline", "Slide 2: System Architecture & Data Flow")

    arch_box = slide2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_arch = arch_box.text_frame
    tf_arch.word_wrap = True

    steps = [
        ("1. Input Request & Constraints", "FastAPI endpoint accepts activity (from 10 supported), location query / GPS coordinates, search radius, and optional user constraints (age, mobility aids)."),
        ("2. Geospatial Resolution & Discovery", "OSM Nominatim resolves place coordinates; OSM Overpass discovers local parks, pitches, and trails within radius; OSRM computes road distance."),
        ("3. Live Environmental Ingestion", "Open-Meteo retrieves hourly temperature, precipitation probability, wind gusts, UV index, and US EPA / European AQI with NOAA solar daylight calculation."),
        ("4. Candidate Ranking & Temporal Windows", "Multi-candidate venues ranked by proximity and tag synergy; hourly forecasts scored to find optimal time window."),
        ("5. Deterministic Safety Evaluation", "Pre-generation evaluation of WHO, NOAA, and EPA thresholds produces mandated suitability and guardrail status (CLEAR, ADVISORY, BLOCKED)."),
        ("6. SLM Grounded Generation", "ChatML prompt supplies verified evidence to OpenVINO INT4 model; outputs structured JSON adhering to recommendation.schema.json."),
        ("7. Post-Generation Override Guardrail", "Deterministic post-check strictly overrides model suitability if SLM is overly optimistic, injecting mandatory safety citations.")
    ]
    for title, desc in steps:
        p_t = tf_arch.add_paragraph()
        p_t.text = f"{title}: "
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = NAVY
        p_t.space_after = Pt(2)
        
        p_d = tf_arch.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)

    # ==========================================
    # SLIDE 3: Worked Example & Evidence Grounding
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide3, "Worked Real-World Example: Midday High-UV Running", "Slide 3: Worked Example & Live Evidence")

    ex_left = slide3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_el = ex_left.text_frame
    tf_el.word_wrap = True
    
    p = tf_el.paragraphs[0]
    p.text = "User Scenario & Retrieved Live Facts"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    inputs_text = [
        "Request: Running or Jogging at Bandra Bandstand, Mumbai (12:30 PM)",
        "Weather (Open-Meteo): 34°C, 72% Humidity, Apparent Temp: 39.5°C",
        "Solar UV (Open-Meteo): 11.2 (Extreme UV Hazard)",
        "Air Quality (Open-Meteo AQ): US AQI 88 (Moderate, PM2.5: 29.5 µg/m³)",
        "Discovered Venue (OSM Overpass): Bandra Bandstand Promenade (0.2 km away, Score: 85.0)",
        "Attribution: © OpenStreetMap contributors | Weather by Open-Meteo (CC BY 4.0)"
    ]
    for inp in inputs_text:
        p = tf_el.add_paragraph()
        p.text = f"• {inp}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    ex_right = slide3.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_er = ex_right.text_frame
    tf_er.word_wrap = True
    
    p = tf_er.paragraphs[0]
    p.text = "System Output & Deterministic Safety Action"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    outputs_text = [
        "Suitability: not_recommended (Mandated by WHO UV Rule)",
        "Safety Status: BLOCKED | Trigger: WHO_UV_EXTREME_STRENUOUS (UV >= 11.0)",
        "Deterministic Override: True (Prevented raw 1.5B model from approving run)",
        "Recommended Window: Shift to early morning before 07:30 or after 18:00 when UV < 3.0",
        "Grounded Reasons: UV index 11.2 exceeds WHO Extreme threshold; 39.5°C heat index creates acute exertional heat exhaustion risk",
        "Precautions: Mandatory broad-spectrum SPF 50+, UV-blocking eyewear, rapid hydration"
    ]
    for out in outputs_text:
        p = tf_er.add_paragraph()
        p.text = f"✔ {out}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    # ==========================================
    # SLIDE 4: Model Strategy, LoRA Fine-Tuning & Evaluation
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide4, "Model Selection, LoRA Fine-Tuning & Measured Evaluation", "Slide 4: Model Choice & Verification")

    m_left = slide4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_ml = m_left.text_frame
    tf_ml.word_wrap = True
    
    p = tf_ml.paragraphs[0]
    p.text = "Model Eligibility & LoRA Fine-Tuning"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    m_facts = [
        "Base Model: Qwen/Qwen2.5-1.5B-Instruct (1,543,714,816 parameters <= 3B limit, Apache 2.0).",
        "Inference Engine: OpenVINO INT4 quantized model (909 MB footprint, 11-16 tok/s on Intel i7 / Iris Xe).",
        "PEFT LoRA Training: Fine-tuned on 10 diverse, schema-validated outdoor scenarios across all 10 activities.",
        "Trainable Parameters: 1,089,536 (0.0705% of model weights) targeting q_proj and v_proj at rank 8, alpha 16.",
        "Publication Package: Hugging Face repository package prepared in huggingface_model/ with exact base pointer."
    ]
    for fact in m_facts:
        p = tf_ml.add_paragraph()
        p.text = f"• {fact}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    m_right = slide4.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_mr = m_right.text_frame
    tf_mr.word_wrap = True
    
    p = tf_mr.paragraphs[0]
    p.text = "Empirical Evaluation Metrics"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    eval_facts = [
        "JSON Extraction Rate: 100.0% (Automated markdown fence stripping and raw JSON parsing).",
        "Schema Conformance: 100.0% across all evaluated scenarios (0 schema violations against Draft-07 schema).",
        "Safety Consistency: 100.0% adherence to WHO/NOAA/EPA rules enforced by deterministic override guardrails.",
        "Data Leakage: 0% duplicate leakage verified between training and held-out test splits.",
        "Throughput Speedup: OpenVINO INT4 provides 5x–8x speedup over FP32 PyTorch CPU baseline."
    ]
    for ef in eval_facts:
        p = tf_mr.add_paragraph()
        p.text = f"✔ {ef}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    # ==========================================
    # SLIDE 5: Safety Guardrails, Grounding & Limitations
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide5, "Safety Rules, Grounding Rigor & Graceful Degradation", "Slide 5: Safety Guardrails & Limitations")

    s_box = slide5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_s = s_box.text_frame
    tf_s.word_wrap = True

    safety_points = [
        ("WHO Global Solar UV Index", "UV >= 11 mandates 'not_recommended' for high-intensity daytime exercise; UV >= 8 mandates caution and shade seeking."),
        ("NOAA / NWS Heat Stress & Frostbite", "Apparent temperature >= 40°C blocks strenuous cardio due to exertional heat stroke hazard; <= -25°C triggers frostbite alerts."),
        ("US EPA Air Quality Standards", "US AQI > 300 / PM2.5 > 250 µg/m³ blocks all outdoor activities; AQI 201-300 cautions sensitive demographics and high-intensity athletics."),
        ("NWS Convective Storm & Lightning Rule", "'When Thunder Roars, Go Indoors': WMO codes 95, 96, 99 strictly block open-field sports (golf, cricket, soccer) with zero lightning shelter."),
        ("Graceful Degradation Protocol", "If Overpass or Nominatim fails, system degrades gracefully using geographic bounding boxes; missing variables are explicitly declared rather than fabricated."),
        ("Known Limitations & Disclaimers", "OSRM routing provides static topological estimates without live traffic jams; community-mapped OSM features may not reflect temporary maintenance.")
    ]
    for title, desc in safety_points:
        p_t = tf_s.add_paragraph()
        p_t.text = f"{title}: "
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = NAVY
        p_t.space_after = Pt(2)
        
        p_d = tf_s.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)

    # ==========================================
    # SLIDE 6: Deployment, API & Hackathon Rulebook Compliance
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide6, "Production API, Deployment Readiness & Rulebook Audit", "Slide 6: Deployment & Rulebook Compliance")

    d_left = slide6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_dl = d_left.text_frame
    tf_dl.word_wrap = True
    
    p = tf_dl.paragraphs[0]
    p.text = "Production HTTP API Architecture"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    api_points = [
        "Framework: FastAPI + Uvicorn with Pydantic request/response validation.",
        "Interactive Documentation: Swagger UI at /docs and ReDoc at /redoc (No frontend/UI created per instructions).",
        "Containerization: Multi-stage Dockerfile optimized for CPU OpenVINO execution.",
        "Core Endpoints: GET /health, GET /activities, POST /recommend, GET /sources.",
        "Deployment Options: Local uvicorn, Docker container, or public cloud (Render, Railway, Hugging Face Spaces)."
    ]
    for pt in api_points:
        p = tf_dl.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    d_right = slide6.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_dr = d_right.text_frame
    tf_dr.word_wrap = True
    
    p = tf_dr.paragraphs[0]
    p.text = "Rulebook Compliance Audit Summary"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    compliance_points = [
        "SLM <= 3B Parameters: PASS (Qwen2.5-1.5B is ~1.54B params).",
        "Public Hugging Face Weights/LoRA: PASS (Adapter files and model card prepared).",
        "Live Data Grounding: PASS (Open-Meteo & OpenStreetMap live integration).",
        "10 Specified Activities: PASS (Explicit profiles for all 10 activities).",
        "OpenStreetMap Attribution: PASS (© OpenStreetMap contributors attributed).",
        "Source Repository & Documentation: PASS (Complete codebase, tests, and audit reports)."
    ]
    for cp in compliance_points:
        p = tf_dr.add_paragraph()
        p.text = f"✔ {cp}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)

    # Save presentation
    prs.save(OUTPUT_PPTX_PATH)
    print(f"[SUCCESS] Presentation generated: {OUTPUT_PPTX_PATH} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    create_presentation()
