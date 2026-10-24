"""Model configuration and inference hyperparameters."""
from dataclasses import dataclass
from typing import Optional
from app.config import settings


@dataclass
class ModelConfig:
    model_id: str = settings.model_id
    base_model_id: str = settings.base_model_id
    adapter_path: Optional[str] = settings.adapter_path
    device: str = settings.device
    max_new_tokens: int = settings.max_new_tokens
    temperature: float = settings.temperature  # 0.0 greedy
    enable_inference: bool = settings.enable_model_inference
