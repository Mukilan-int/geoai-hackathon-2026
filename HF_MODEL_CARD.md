---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - geoai
  - geospatial
  - outdoor-activity
  - safety
  - open-meteo
  - openstreetmap
  - qwen2.5
  - lora
datasets:
  - custom-geoai-safety-2026
language:
  - en
---

# GeoAI Qwen2.5-1.5B Outdoor Recommendation LoRA Adapter

**Event:** GeoAI Hackathon 2026  
**Host Institution:** Centre of Studies in Resources Engineering (CSRE), Indian Institute of Technology Bombay (IIT Bombay)  
**Base Model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.54B Parameters, Apache-2.0)  
**Adapter Method:** PEFT LoRA ($r=8$, $\alpha=16$, targets `q_proj`, `v_proj`)  
**Parameter Limit Compliance:** < 3 Billion Parameters (Base: 1.54B, Trainable LoRA parameters: 1,089,536 / 0.0705% of model)  

---

## 1. Model Summary & Intended Use

This repository provides an open-weight LoRA adapter trained on `Qwen/Qwen2.5-1.5B-Instruct` specifically fine-tuned to act as a grounded outdoor activity advisor. It evaluates live meteorological, air quality, and OpenStreetMap geospatial evidence to recommend when and where to engage in outdoor activities, identify safety precautions, and state when conditions are hazardous.

### Key Capabilities:
1. **Schema-Constrained JSON Generation:** Emits valid JSON conforming strictly to `schemas/recommendation.schema.json`.
2. **Missing Input Identification:** Explicitly reports omitted locations or unverified forecasts under `missing_information_identified`.
3. **Safety Grounding:** Incorporates deterministic safety thresholds (WHO UV index, NOAA heat stress, NWS lightning protocols, EPA AQI).
4. **Zero Proprietary API Dependency:** Fully autonomous small language model requiring no external hosted proprietary models (GPT-4, Claude, Gemini Pro).

---

## 2. 10 Supported Outdoor Activities

The model is explicitly trained on structured profiles for:
1. Running or jogging
2. Cycling
3. Golf
4. Cricket
5. Outdoor football
6. Outdoor tennis
7. Hiking or nature walking
8. Picnic or park outing
9. Outdoor event or gathering
10. Outdoor photography

---

## 3. Training Details

- **Base Checkpoint:** `Qwen/Qwen2.5-1.5B-Instruct`
- **LoRA Configuration:**
  - Rank ($r$): 8
  - Alpha ($\alpha$): 16
  - Target Modules: `["q_proj", "v_proj"]`
  - LoRA Dropout: 0.05
  - Trainable Parameters: 1,089,536 (0.0705% of total parameters)
- **Loss Function:** Cross-entropy over assistant response tokens (ChatML prompt tokens masked).
- **Optimizer:** AdamW ($\text{lr} = 2 \times 10^{-4}$).
- **Training Duration:** 55.95s on local Intel Core i7-1265U CPU (reproducible scripts provided for Google Colab T4 GPU).

---

## 4. How to Load and Run Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "Erevanox/geoai-qwen2.5-1.5b-outdoor-lora"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2. Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    device_map="auto"
)

# 3. Attach LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# 4. Formulate Prompt
messages = [
    {"role": "system", "content": "You are a grounded GeoAI outdoor safety assistant..."},
    {"role": "user", "content": "Activity: Running | Location: Marine Drive | Temp: 34C | UV: 11.2"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## 5. Limitations & Safety Guardrails

- **Small Parameter Capacity:** As a 1.5B parameter model, generative alignment may occasionally experience optimism bias. For production safety, this model is deployed alongside deterministic pre- and post-generation safety guardrails (`app/safety/rules.py`) that strictly enforce WHO UV $\ge 11$, EPA AQI $> 300$, and NWS lightning rules.
- **Attribution:** Geospatial data derived from [OpenStreetMap](https://www.openstreetmap.org/copyright) (ODbL). Weather data provided by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
