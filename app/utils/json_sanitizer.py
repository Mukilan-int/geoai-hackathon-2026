"""Safe JSON parsing, code fence sanitizer, and schema validation."""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import jsonschema

from app.utils.logger import logger

# Regex to strip markdown code fences like ```json ... ``` or ``` ... ```
JSON_FENCE_REGEX = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# Default path to the official recommendation schema
DEFAULT_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas",
    "recommendation.schema.json"
)


def extract_json_candidate(raw_text: str) -> str:
    """Extracts raw JSON content even if wrapped in markdown code fences or conversational preamble."""
    if not raw_text or not isinstance(raw_text, str):
        return ""
    
    text = raw_text.strip()
    
    # 1. Try finding markdown fence match
    fence_match = JSON_FENCE_REGEX.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    
    # 2. Try finding outermost matching braces { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    
    return text


def sanitize_and_parse_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Sanitizes text and parses into a Python dictionary.
    
    Returns:
        (parsed_dict, error_message)
    """
    candidate = extract_json_candidate(raw_text)
    if not candidate:
        return None, "Empty or non-string response received."
    
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed, None
        return None, f"Expected JSON object, got {type(parsed).__name__}."
    except json.JSONDecodeError as err:
        # Secondary attempt: fix common trailing commas before closing braces/brackets
        try:
            fixed = re.sub(r",\s*([\}\]])", r"\1", candidate)
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass
        return None, f"JSONDecodeError: {str(err)}"


def load_recommendation_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads and caches the JSON schema definition."""
    path = schema_path or DEFAULT_SCHEMA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recommendation schema not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: Dict[str, Any], schema_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Validates a dictionary against the official recommendation schema.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    try:
        schema = load_recommendation_schema(schema_path)
        validator = jsonschema.Draft7Validator(schema)
        errors = [err.message for err in validator.iter_errors(data)]
        if errors:
            return False, errors
        return True, []
    except Exception as exc:
        logger.error(f"Schema loading or validation exception: {exc}")
        return False, [f"Validation exception: {str(exc)}"]
