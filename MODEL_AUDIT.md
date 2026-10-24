# Small Language Model Audit & Technical Specification

**Project:** GeoAI Hackathon 2026 (CSRE, IIT Bombay)  
**Audit Date:** 2026-10-23  
**Auditor:** Senior AI & Geospatial Software Engineer  

---

## 1. Candidate Models Evaluated

| Specification | Primary Inference Candidate | Fine-Tuning Base Model | Fine-Tuned Adapter |
| :--- | :--- | :--- | :--- |
| **Model Identifier** | `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` | `Qwen/Qwen2.5-1.5B-Instruct` | `huggingface_model/` (Local LoRA) |
| **Repository Link** | [Hugging Face Repository](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov) | [Hugging Face Repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Publication Package Prepared |
| **Exact Base Model** | Points to `Qwen/Qwen2.5-1.5B-Instruct` | Original Base Architecture | `base_model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct` |
| **Architecture** | `Qwen2ForCausalLM` (INT4 weights) | `Qwen2ForCausalLM` (FP32/BF16) | PEFT LoRA on `q_proj`, `v_proj` |
| **Exact Parameters** | 1,543,714,816 parameters (~1.54B) | 1,543,714,816 parameters (~1.54B) | 1,089,536 trainable params (0.0705%) |
| **Parameter Limit Compliance** | **PASS** ($\le 3$ Billion limit) | **PASS** ($\le 3$ Billion limit) | **PASS** ($\le 3$ Billion limit) |
| **Licence** | Apache 2.0 (Open-Source Permissive) | Apache 2.0 (Open-Source Permissive) | Apache 2.0 (Open-Source Permissive) |
| **Loading Interface** | `OVModelForCausalLM.from_pretrained(...)` | `AutoModelForCausalLM.from_pretrained(...)` | `PeftModel.from_pretrained(...)` |
| **Quantization** | INT4 Weight-Only (NNCF sym/asym) | None (Unquantized) | LoRA Adapter ($r=8, \alpha=16$) |
| **Storage Footprint** | ~909 MB | ~3.1 GB | ~4.37 MB (`adapter_model.safetensors`) |

---

## 2. Hardware Environment & Throughput Benchmark

- **Processor:** Intel Core i7-1265U (10 Physical Cores, 12 Logical Threads, 1.80 GHz to 4.80 GHz Turbo)
- **RAM:** 31.69 GB DDR4
- **Graphics:** Intel Iris Xe Graphics (Unified Memory Architecture, 96 EUs)
- **CUDA Availability:** False (Zero NVIDIA hardware present)

### Measured Generation Benchmarks (Greedy Deterministic Decoding):
- **PyTorch FP32 CPU Baseline (`Qwen/Qwen2.5-1.5B-Instruct`):**
  - Warm load time: 15.65 s
  - Throughput: 1.28–2.31 tokens/second
  - Peak RAM footprint: 6,340 MB (~6.2 GB)
- **OpenVINO INT4 CPU (`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`):**
  - Warm load time: 5.64 s (2.8x faster)
  - Throughput: **9.86–11.07 tokens/second** (~5x speedup)
  - Peak RAM footprint: **2,446 MB** (-61.4% memory reduction)
- **OpenVINO INT4 iGPU (Intel Iris Xe Graphics):**
  - Throughput: **12.55–16.46 tokens/second** (~7x–12x speedup)
  - Peak RAM footprint: 2,979 MB

---

## 3. Strict Compliance Regarding Hosted / Prohibited Models

The GeoAI Hackathon 2026 rulebook strictly prohibits reliance on commercial hosted models.

- **Prohibited Models:** GPT-4, GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, or any external hosted API.
- **Verification:**
  1. The codebase contains zero API keys or HTTP calls to OpenAI, Anthropic, or Google AI endpoints for model inference.
  2. All inference runs locally inside the process memory via Intel OpenVINO runtime or PyTorch CPU kernels.
  3. The system operates fully even when disconnected from proprietary LLM cloud services.

---

## 4. Verification Date & Integrity Check

- **Audit Date:** 2026-10-23
- **Base Model Checksum Verified:** Yes (`models--Qwen--Qwen2.5-1.5B-Instruct` in local cache)
- **OpenVINO INT4 Checksum Verified:** Yes (`models--OpenVINO--Qwen2.5-1.5B-Instruct-int4-ov` in local cache)
- **LoRA Adapter Weights Verified:** Yes (`huggingface_model/adapter_model.safetensors`, 4,372,840 bytes)
