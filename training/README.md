# LoRA Fine-Tuning Pipeline for GeoAI Hackathon 2026

**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B Parameters, Apache 2.0)  
**Host Target:** Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
**Task:** Grounded, Structured GeoAI Outdoor Activity Recommendation and Safety Reasoning

---

## 1. Overview and Objective

The goal of this fine-tuning pipeline is to adapt the small language model (`Qwen2.5-1.5B-Instruct`) to:
1. Produce strictly valid JSON matching `schemas/recommendation.schema.json` without markdown wrapping fences.
2. Accurately identify missing user inputs (e.g., omitted location coordinates, unverified forecast).
3. Enforce conservative safety decisions aligned with WHO UV index, NOAA heat stress, and NWS lightning guidelines.
4. Eliminate hallucinations by grounding every claim in supplied meteorological and geospatial observations.

---

## 2. Hardware Environment and Strategy

- **Development Hardware:** Intel Core i7-1265U (10 Cores, 12 Threads), 32 GB RAM, Intel Iris Xe graphics (No CUDA GPU).
- **Inference Candidate:** `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` (Optimized via Intel NNCF INT4 weight quantization).
- **Fine-Tuning Base:** `Qwen/Qwen2.5-1.5B-Instruct` (Unquantized PyTorch base model).
- **Strategy:**
  - For local execution on CPU, we provide a lightweight smoke test (`--max-steps 2`) that verifies the full PyTorch + PEFT LoRA training loop and exports valid adapter artifacts.
  - For full-scale multi-epoch training, we provide cloud-ready reproducible configurations for Google Colab (T4 GPU) and Kaggle.

---

## 3. Dataset Description

The dataset covers all 10 hackathon outdoor activities:
- Running/Jogging, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnic, Outdoor Gathering, and Photography.
- Scenarios include extreme UV midday exposure, convective thunderstorms with hail, hazardous AQI (>300), high wind gusts, and elderly/mobility constraints.
- Splits:
  - `datasets/train_recommendations.jsonl`: 10 diverse training pairs.
  - `datasets/eval_recommendations.jsonl`: 2 strictly held-out evaluation pairs (zero ID/prompt leakage).

---

## 4. Execution Commands

### Step 1: Prepare and Validate Datasets
```bash
python training/prepare_dataset.py
python training/validate_dataset.py
```

### Step 2: Run LoRA Fine-Tuning
Local CPU smoke run:
```bash
python training/train_lora.py --max-steps 2 --device cpu --output-dir huggingface_model
```

Cloud GPU full run (Google Colab / Kaggle T4):
```bash
python training/train_lora.py --epochs 3 --max-steps 50 --device cuda --output-dir huggingface_model
```

### Step 3: Evaluate Base Model vs. Fine-Tuned Adapter
```bash
python training/evaluate_adapter.py --eval-data datasets/eval_recommendations.jsonl
```

### Step 4: Optional Merge and Export
```bash
python training/merge_or_export_adapter.py --adapter-path huggingface_model --output-dir merged_model
```
