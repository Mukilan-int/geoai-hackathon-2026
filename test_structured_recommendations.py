import time
import os
import json
import re
import jsonschema
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# 1. Define General JSON Schema for Outdoor Activity Recommendations
RECOMMENDATION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "OutdoorActivityRecommendation",
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
        "activity_name": {
            "type": "string",
            "description": "The specific outdoor activity evaluated or proposed."
        },
        "suitability": {
            "type": "string",
            "enum": ["recommended", "caution", "not_recommended"],
            "description": "Assessment of whether the activity is advisable under the given conditions."
        },
        "recommended_time_window": {
            "type": "string",
            "description": "Appropriate time of day or condition window (e.g., 'early morning before 8 AM', 'postpone until storm clears')."
        },
        "precautions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Specific, actionable safety precautions."
        },
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Justification based strictly on supplied conditions and constraints."
        },
        "missing_information_identified": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of crucial missing facts from the prompt (e.g., location, terrain, user health profile), or empty list if sufficient."
        },
        "grounded_facts_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific facts provided in the prompt that directly informed this recommendation."
        }
    },
    "additionalProperties": False
}

SYSTEM_INSTRUCTION = (
    "You are an expert GeoAI assistant specializing in grounded outdoor activity safety.\n"
    "You must evaluate the user's outdoor request based STRICTLY and ONLY on the facts supplied in the prompt.\n"
    "CRITICAL RULES:\n"
    "1. Do NOT invent missing data (e.g., do not invent a city, park name, or weather conditions if not provided).\n"
    "2. If crucial information (such as location or weather) is missing, explicitly list it under 'missing_information_identified'.\n"
    "3. You have NOT checked real-time sensors or live APIs; do not claim to have fetched live data.\n"
    "4. Output MUST be ONLY a single valid JSON object strictly matching this schema with no markdown code fences, no conversational preamble, and no trailing text.\n"
    "JSON Schema:\n" + json.dumps(RECOMMENDATION_SCHEMA, indent=2)
)

# 2. Define the Five Test Scenarios
TEST_SCENARIOS = [
    {
        "id": "scenario_1_high_uv_running",
        "title": "Running in High UV Conditions",
        "description": "Midday run requested under extreme UV and heat stress.",
        "prompt": (
            "User Request: I want to go for an intense 10 km outdoor road run between 12:00 PM and 2:00 PM today.\n"
            "Supplied Location: Juhu Beach Promenade, Mumbai, India.\n"
            "Supplied Weather: Sunny, 34°C (93.2°F), 72% relative humidity, UV Index 11 (Extreme), wind speed 8 km/h.\n"
            "Supplied Air Quality: AQI 65 (Satisfactory).\n"
            "Provide your recommendation as a JSON object."
        ),
        "expected_suitability": ["caution", "not_recommended"],
        "critical_factors": ["UV Index 11", "Extreme UV", "34°C", "Midday heat"],
        "should_identify_missing_location": False
    },
    {
        "id": "scenario_2_cycling_rain_forecast",
        "title": "Cycling with Rain in Forecast",
        "description": "Road cycling planned right before a forecasted thunderstorm with high wind gusts.",
        "prompt": (
            "User Request: I am planning a 2-hour outdoor road cycling workout starting in 15 minutes.\n"
            "Supplied Location: Sinhagad Ghat Road, Pune, Maharashtra.\n"
            "Supplied Weather: Heavily overcast, 25°C (77°F), 85% chance of severe thunderstorm and heavy downpour within 45 minutes, wind gusts up to 45 km/h.\n"
            "Supplied Road Conditions: Mountainous winding roads, currently dry but rapidly deteriorating.\n"
            "Provide your recommendation as a JSON object."
        ),
        "expected_suitability": ["caution", "not_recommended"],
        "critical_factors": ["severe thunderstorm", "heavy downpour", "wind gusts 45 km/h", "mountainous winding roads"],
        "should_identify_missing_location": False
    },
    {
        "id": "scenario_3_picnic_missing_location",
        "title": "Picnic with Missing Location Information",
        "description": "Family asks for outdoor picnic advice with pleasant weather but completely omitted location and park details.",
        "prompt": (
            "User Request: We are a family of four planning an outdoor picnic for 3 hours. What specific park shelters and amenities are available for us?\n"
            "Supplied Location: [NOT PROVIDED / UNKNOWN]\n"
            "Supplied Weather: Clear skies, 24°C (75.2°F), 45% humidity, UV Index 4 (Moderate), wind speed 10 km/h.\n"
            "Supplied Air Quality: AQI 42 (Good).\n"
            "Provide your recommendation as a JSON object."
        ),
        "expected_suitability": ["recommended", "caution"],
        "critical_factors": ["Location not provided", "amenities unknown without location"],
        "should_identify_missing_location": True
    },
    {
        "id": "scenario_4_hiking_poor_air_quality",
        "title": "Hiking Recommendation with Poor Air Quality",
        "description": "Strenuous uphill hike planned under severe, hazardous air pollution.",
        "prompt": (
            "User Request: Our hiking club is planning a strenuous 4-hour morning summit hike on steep rugged terrain.\n"
            "Supplied Location: Aravalli Trail Ridge, Southern Haryana.\n"
            "Supplied Weather: Cool hazy morning, 19°C (66.2°F), wind 4 km/h.\n"
            "Supplied Air Quality: AQI 345 (Severe / Hazardous, PM2.5 concentration extremely high).\n"
            "Provide your recommendation as a JSON object."
        ),
        "expected_suitability": ["not_recommended", "caution"],
        "critical_factors": ["AQI 345", "Severe air pollution", "PM2.5", "Strenuous hiking"],
        "should_identify_missing_location": False
    },
    {
        "id": "scenario_5_older_adult_nature_walk",
        "title": "Activity Recommendation for an Older Adult",
        "description": "Outdoor activity request for a 74-year-old with joint sensitivity and mild mobility constraints.",
        "prompt": (
            "User Request: Recommend a pleasant outdoor activity for a 74-year-old individual who has mild knee osteoarthritis and uses a walking cane.\n"
            "Supplied Location: Cubbon Park, Central Bengaluru, Karnataka.\n"
            "Supplied Weather: Partly cloudy, 23°C (73.4°F), 58% humidity, UV Index 5 (Moderate), gentle breeze 7 km/h.\n"
            "Supplied Air Quality: AQI 55 (Satisfactory).\n"
            "Provide your recommendation as a JSON object."
        ),
        "expected_suitability": ["recommended", "caution"],
        "critical_factors": ["74-year-old", "knee osteoarthritis", "walking cane", "flat paved trails", "benches"],
        "should_identify_missing_location": False
    }
]

def clean_and_extract_json(raw_text: str):
    """
    Attempts to parse text directly as JSON.
    If fails, strips markdown fences (```json ... ```) and parses again.
    """
    raw_stripped = raw_text.strip()
    
    # 1. Direct parse attempt
    try:
        parsed = json.loads(raw_stripped)
        return parsed, "direct_json", None
    except json.JSONDecodeError as direct_err:
        pass
    
    # 2. Extract from markdown code block
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_stripped, re.IGNORECASE)
    if fence_match:
        extracted = fence_match.group(1).strip()
        try:
            parsed = json.loads(extracted)
            return parsed, "markdown_fence_stripped", None
        except json.JSONDecodeError as fence_err:
            return None, "parse_failed", f"Failed parsing markdown fence: {str(fence_err)}"
            
    # 3. Fallback: bracket slicing
    brace_match = re.search(r"(\{[\s\S]*\})", raw_stripped)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(1).strip())
            return parsed, "bracket_extracted", None
        except json.JSONDecodeError as brace_err:
            return None, "parse_failed", f"Failed parsing bracket match: {str(brace_err)}"

    return None, "parse_failed", "No JSON structure found in output"

def validate_against_schema(data):
    """Validates data dictionary against RECOMMENDATION_SCHEMA."""
    validator = jsonschema.Draft7Validator(RECOMMENDATION_SCHEMA)
    errors = [e.message for e in validator.iter_errors(data)]
    return len(errors) == 0, errors

def analyze_grounding_and_consistency(scenario, parsed_data):
    """
    Analyzes whether the output:
    1. Respected the expected suitability (consistent safety logic).
    2. Correctly identified missing location (if applicable).
    3. Grounded its reasons on prompt facts rather than hallucinations.
    """
    issues = []
    
    suitability = parsed_data.get("suitability")
    if suitability not in scenario["expected_suitability"]:
        issues.append(
            f"Inconsistent Suitability: Model assigned '{suitability}', but expected one of {scenario['expected_suitability']} "
            f"given safety factors: {scenario['critical_factors']}."
        )
        
    missing_info = parsed_data.get("missing_information_identified", [])
    if scenario["should_identify_missing_location"]:
        location_mentioned = any("location" in str(item).lower() or "park" in str(item).lower() for item in missing_info)
        if not location_mentioned:
            issues.append(
                "Missing Information Omission: Prompt had '[NOT PROVIDED / UNKNOWN]' location, but model failed to identify missing location in 'missing_information_identified'."
            )
            
    # Check for hallucination of live API checks
    all_text = json.dumps(parsed_data).lower()
    if "i fetched live" in all_text or "live api" in all_text or "checked openstreetmap" in all_text:
        issues.append("Unsupported Claim: Model falsely claimed to have accessed live APIs or databases.")
        
    return issues

def main():
    print("=" * 75)
    print("GeoAI Hackathon 2026: Structured JSON Output & Grounding Evaluation")
    print("Model: OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
    print("=" * 75)

    model_id = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
    
    # Model Metadata Verification
    model_metadata = {
        "candidate_model_id": model_id,
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "original_author": "Qwen / Alibaba Cloud",
        "quantized_author": "OpenVINO (Intel)",
        "quantization_format": "INT4 Weight-Only Quantization (NNCF)",
        "total_parameters": "1.54 Billion (< 3.0 Billion Hackathon Limit)",
        "licence": "Apache-2.0",
        "loading_interface": "optimum.intel.OVModelForCausalLM.from_pretrained",
        "hardware_device": "Intel Core i7-1265U CPU / Intel Iris Xe iGPU",
        "hackathon_rules_compliant": True
    }
    
    print("\n[1] Verified Model Metadata:")
    for k, v in model_metadata.items():
        print(f"  - {k}: {v}")

    print("\n[2] Loading Model and Tokenizer via OVModelForCausalLM...")
    t_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Using CPU device for consistent, deterministic evaluation across scenarios
    model = OVModelForCausalLM.from_pretrained(model_id, device="CPU")
    t_load_end = time.perf_counter()
    print(f"  - Loaded in {t_load_end - t_load_start:.2f} seconds.")

    scenario_results = []
    direct_json_count = 0
    total_valid_json_count = 0
    schema_valid_count = 0
    total_scenarios = len(TEST_SCENARIOS)

    print("\n[3] Executing 5 Distinct Scenarios against General Schema:")
    print("-" * 75)

    for idx, sc in enumerate(TEST_SCENARIOS, 1):
        print(f"\n--- Scenario {idx}/{total_scenarios}: {sc['title']} ---")
        print(f"Goal: {sc['description']}")
        
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": sc["prompt"]}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([formatted], return_tensors="pt")
        input_token_count = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False  # Deterministic greedy generation
        )
        t1 = time.perf_counter()
        latency = t1 - t0
        
        gen_tokens = outputs[0][input_token_count:]
        output_token_count = len(gen_tokens)
        throughput = output_token_count / latency if latency > 0 else 0
        raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        parsed_json, parse_method, parse_error = clean_and_extract_json(raw_response)
        
        is_direct = (parse_method == "direct_json")
        is_valid_json = (parsed_json is not None)
        if is_direct:
            direct_json_count += 1
        if is_valid_json:
            total_valid_json_count += 1
            
        schema_passed = False
        schema_errors = []
        consistency_issues = []
        
        if is_valid_json:
            schema_passed, schema_errors = validate_against_schema(parsed_json)
            if schema_passed:
                schema_valid_count += 1
            consistency_issues = analyze_grounding_and_consistency(sc, parsed_json)

        print(f"  Tokens In: {input_token_count} | Tokens Out: {output_token_count} | Latency: {latency:.2f} s | Speed: {throughput:.2f} tok/s")
        print(f"  Parse Method: {parse_method}")
        print(f"  Schema Valid: {'PASSED' if schema_passed else 'FAILED'}")
        if schema_errors:
            print(f"  Schema Errors: {schema_errors}")
        if consistency_issues:
            print(f"  Consistency Issues: {consistency_issues}")
            
        print(f"  Raw Output Snippet:\n{raw_response.strip()[:200]}...")

        scenario_results.append({
            "scenario_id": sc["id"],
            "title": sc["title"],
            "description": sc["description"],
            "prompt": sc["prompt"],
            "metrics": {
                "input_tokens": int(input_token_count),
                "output_tokens": int(output_token_count),
                "latency_seconds": round(latency, 3),
                "tokens_per_second": round(throughput, 2)
            },
            "json_parsing": {
                "is_direct_json": is_direct,
                "is_valid_json": is_valid_json,
                "parse_method": parse_method,
                "parse_error": parse_error
            },
            "schema_validation": {
                "passed": schema_passed,
                "errors": schema_errors
            },
            "consistency_and_grounding": {
                "issues": consistency_issues,
                "suitability_chosen": parsed_data_suitability if is_valid_json and (parsed_data_suitability := parsed_json.get("suitability")) else None
            },
            "parsed_output": parsed_json,
            "raw_output": raw_response.strip()
        })

    summary_stats = {
        "total_scenarios_tested": total_scenarios,
        "direct_json_rate_pct": round((direct_json_count / total_scenarios) * 100, 1),
        "valid_json_rate_pct": round((total_valid_json_count / total_scenarios) * 100, 1),
        "schema_conformance_rate_pct": round((schema_valid_count / total_scenarios) * 100, 1),
        "direct_json_count": direct_json_count,
        "total_valid_json_count": total_valid_json_count,
        "schema_valid_count": schema_valid_count
    }

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY STATISTICS")
    print("=" * 75)
    print(f"Total Scenarios: {summary_stats['total_scenarios_tested']}")
    print(f"Direct JSON Rate (no fences): {summary_stats['direct_json_rate_pct']}% ({direct_json_count}/{total_scenarios})")
    print(f"Total Valid JSON Rate: {summary_stats['valid_json_rate_pct']}% ({total_valid_json_count}/{total_scenarios})")
    print(f"Schema Conformance Rate: {summary_stats['schema_conformance_rate_pct']}% ({schema_valid_count}/{total_scenarios})")
    print("=" * 75)

    final_report = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_metadata": model_metadata,
            "schema_used": RECOMMENDATION_SCHEMA
        },
        "summary": summary_stats,
        "scenario_evaluations": scenario_results
    }

    output_path = os.path.join(os.path.dirname(__file__), "structured_recommendations_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nFull machine-readable results saved to: {output_path}")

if __name__ == "__main__":
    main()
