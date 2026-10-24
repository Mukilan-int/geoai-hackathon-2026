"""Model loading interface for OpenVINO quantized model and Transformers/PEFT."""
import os
import time
from typing import Any, Dict, Optional, Tuple

from app.models.config import ModelConfig
from app.utils.logger import logger

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_MODEL_INIT_ERROR = None
_MODEL_LOAD_METRICS = {}


def get_model_status() -> Dict[str, Any]:
    """Returns runtime status of the small language model engine."""
    global _CACHED_MODEL, _CACHED_TOKENIZER, _MODEL_INIT_ERROR, _MODEL_LOAD_METRICS
    return {
        "model_loaded": _CACHED_MODEL is not None,
        "tokenizer_loaded": _CACHED_TOKENIZER is not None,
        "device": ModelConfig.device,
        "model_id": ModelConfig.model_id,
        "base_model_id": ModelConfig.base_model_id,
        "adapter_path": ModelConfig.adapter_path,
        "load_metrics": _MODEL_LOAD_METRICS,
        "error": _MODEL_INIT_ERROR
    }


def load_model_and_tokenizer(
    config: Optional[ModelConfig] = None
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Loads tokenizer and OpenVINO quantized model with error handling and caching.
    
    Returns:
        (model, tokenizer, error_message)
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _MODEL_INIT_ERROR, _MODEL_LOAD_METRICS
    
    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER, None

    cfg = config or ModelConfig()
    
    if not cfg.enable_inference:
        logger.info("Model inference is disabled by configuration (ENABLE_MODEL_INFERENCE=false).")
        return None, None, "Model inference disabled by config."

    logger.info(f"Loading OpenVINO model '{cfg.model_id}' on device '{cfg.device}'...")
    t0 = time.perf_counter()

    try:
        from transformers import AutoTokenizer
        from optimum.intel import OVModelForCausalLM

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        
        # 2. Load OpenVINO quantized model
        model = OVModelForCausalLM.from_pretrained(
            cfg.model_id,
            device=cfg.device
        )
        t1 = time.perf_counter()
        load_duration = round(t1 - t0, 3)
        
        _CACHED_MODEL = model
        _CACHED_TOKENIZER = tokenizer
        _MODEL_INIT_ERROR = None
        _MODEL_LOAD_METRICS = {
            "load_time_seconds": load_duration,
            "device": cfg.device,
            "model_id": cfg.model_id
        }
        logger.info(f"Successfully loaded {cfg.model_id} on {cfg.device} in {load_duration}s.")
        return model, tokenizer, None

    except Exception as exc:
        err_msg = f"Failed to load OpenVINO model '{cfg.model_id}' on '{cfg.device}': {str(exc)}"
        logger.error(err_msg)
        _MODEL_INIT_ERROR = err_msg
        return None, None, err_msg
