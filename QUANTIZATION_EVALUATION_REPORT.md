# Quantized SLM Evaluation Report for GeoAI Hackathon 2026
**Host Institution:** Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
**Candidate Architecture:** Qwen2.5-1.5B-Instruct  
**Target Hardware:** Intel Core i7-1265U (10 Cores, 12 Threads) | Intel Iris Xe Graphics | 31.7 GB RAM | Windows 11  

---

## 1. Executive Summary

To optimize local inference for the GeoAI Hackathon 2026 without purchasing dedicated hardware or paid APIs, we benchmarked the **INT4 quantized variant** (`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`) against the baseline **PyTorch FP32 CPU model** (`Qwen/Qwen2.5-1.5B-Instruct`).

### Key Takeaways:
1. **Speedup:** OpenVINO INT4 accelerated generation from **1.28–2.31 tok/s** (PyTorch FP32 CPU) to **9.86–11.07 tok/s on CPU** (~5x–8x speedup) and **12.55–16.46 tok/s on the Intel Iris Xe integrated GPU** (~7x–12x speedup).
2. **Memory Reduction:** Process peak memory footprint dropped by **over 53%**, from **6,340 MB (~6.2 GB)** down to **2,512 MB (~2.45 GB)** on CPU and **2,979 MB** on iGPU.
3. **Download Footprint:** Compressed model storage dropped from **~3.1 GB** down to **~909 MB** (a 70% storage saving).
4. **Hackathon Compliance:** The OpenVINO INT4 approach complies fully with the rules: it uses an open-weight base model under 3B parameters (<1.54B parameters), is hosted on a public Hugging Face repository, uses an Apache 2.0 open licence, and loads natively via Hugging Face's official `from_pretrained(...)` API through the `optimum.intel` integration.

---

## 2. Model Eligibility & Rulebook Verification

| Hackathon Requirement | Candidate 1: `Qwen/Qwen2.5-1.5B-Instruct` (Baseline) | Candidate 2: `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` (Evaluated) | Candidate 3: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (Alternative) |
| :--- | :--- | :--- | :--- |
| **Hugging Face Repository** | Public: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Public: [`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov) | Public: [`Qwen/Qwen2.5-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |
| **Exact Parameters & Architecture** | 1.54B parameters (`Qwen2ForCausalLM`) | 1.54B parameters (`Qwen2ForCausalLM`, INT4 quantized weights) | 1.54B parameters (`qwen2` GGUF architecture) |
| **Parameter Limit Compliance** | Pass (<= 3 Billion) | Pass (<= 3 Billion) | Pass (<= 3 Billion) |
| **Licence** | Apache 2.0 (Permissive) | Apache 2.0 (Permissive) | Apache 2.0 (Permissive) |
| **`from_pretrained(...)` Compatibility** | Native: `AutoModelForCausalLM.from_pretrained(...)` | Native: `OVModelForCausalLM.from_pretrained(...)` (Optimum-Intel HF API) | Conditional: Requires `gguf_file="..."` or external `llama.cpp` runtime |
| **Hardware Fit (Intel 12th Gen + Iris Xe)** | Poor: Unquantized FP32 CPU execution bound by memory bandwidth | Outstanding: Hardware-tailored AVX2/VNNI for Core i7 and OpenCL for Iris Xe | Moderate: Requires compiled `llama-cpp-python` or external binary |
| **Hackathon Submission Compliance** | 100% compliant | 100% compliant | Requires disclosure of custom C++ runtime if not using `from_pretrained` |

> [!NOTE]
> **Distinction on GGUF vs `from_pretrained`:**
> Standard `llama.cpp` uses standalone executables (`llama-cli`) or Python bindings (`llama-cpp-python`) that do not use Hugging Face's `from_pretrained(...)` standard model class by default. While Hugging Face `transformers` has recently added `gguf_file` support into `AutoModelForCausalLM.from_pretrained(...)`, on CPU it currently dequantizes weights to float tensors in memory, failing to leverage hardware INT4 vector compute. In contrast, `optimum-intel`'s `OVModelForCausalLM.from_pretrained(...)` satisfies the exact Hugging Face abstraction while delivering native INT4 kernel execution on Intel CPUs and GPUs.

---

## 3. Side-by-Side Benchmark Results

All tests were executed on the user's Intel Core i7-1265U system under Windows 11 using deterministic greedy decoding for fair repeatability.

### Benchmark Prompt 1: Baseline Comparison
*"Suggest three outdoor activities for a sunny day."*

| Metric | Original: PyTorch FP32 (CPU) | Quantized: OpenVINO INT4 (CPU) | Quantized: OpenVINO INT4 (Iris Xe GPU) |
| :--- | :--- | :--- | :--- |
| **Execution Device** | CPU (`torch.float32`) | CPU (`Intel Core i7-1265U`) | iGPU (`Intel Iris Xe Graphics`) |
| **Warm Load Time** | 15.651 s | **5.641 s** (2.8x faster) | 15.350 s (includes OpenCL compile) |
| **Throughput (tokens/s)** | 1.28 – 2.31 tok/s | **9.86 tok/s** (~5x faster) | **16.46 tok/s** (~8x–12x faster) |
| **Generation Latency** | 35.94 – 124.56 s | **4.157 s** | **2.248 s** |
| **Output Tokens** | 41–159 tokens | 41 tokens | 37 tokens |
| **Peak Process RAM** | 6,339.93 MB (~6.2 GB) | **2,446.66 MB** (~2.4 GB) | **2,979.62 MB** (~2.9 GB) |
| **RAM Reduction** | Baseline | **-61.4% RAM** | **-53.0% RAM** |

---

### Benchmark Prompt 2: Contextual GeoAI Recommendation
*Context: Sanjay Gandhi National Park, 32°C, 65% humidity, UV index 8 (Very High), wind 12 km/h, AQI 85.*  
*Prompt: "Given the supplied weather and location information, recommend an outdoor activity, a suitable time, relevant precautions, and the reasons for your recommendation. Do not invent missing data."*

| Metric | OpenVINO INT4 (CPU) | OpenVINO INT4 (Iris Xe GPU) |
| :--- | :--- | :--- |
| **Input Tokens** | 123 tokens | 123 tokens |
| **Output Tokens** | 256 tokens | 238 tokens |
| **Generation Time** | 23.131 s | **18.969 s** |
| **Throughput** | 11.07 tokens/s | **12.55 tokens/s** |
| **Peak Process RAM** | 2,512.76 MB | 2,950.96 MB |
| **Quality & Hallucination Check** | Adhered to strict instructions; recommended hiking; cited morning/late afternoon timing; addressed UV index 8, humidity, and wind; did not fabricate unprovided sensors. | Recommended hiking in morning/late afternoon; emphasized sun protection, hydration for high humidity, and wind protection; accurately parsed all input parameters. |

---

## 4. Hardware Limitations & Observations

1. **Integrated GPU vs CPU:**
   - The Intel Iris Xe iGPU provided the highest peak generation speed (**16.46 tokens/second**).
   - The iGPU shares memory with system RAM (unified memory). It consumes slightly more RAM initialization (~2.9 GB vs ~2.5 GB on CPU) due to OpenCL runtime buffers.
2. **CPU Execution Stability:**
   - The CPU target achieved steady **9.86–11.07 tokens/second** with extremely fast warm load time (**5.64 seconds**) and a smaller memory footprint (**2.51 GB**).
3. **No NVIDIA CUDA Hardware:**
   - PyTorch's native CUDA-only quantization libraries (such as `bitsandbytes` or `AutoAWQ`) cannot run here. OpenVINO is the optimal vendor-supported engine for Intel hardware.

---

## 5. Next Steps Recommendation for Hackathon Development

1. **Adopt OpenVINO INT4 as the Primary Inference Engine:**
   - Use `OVModelForCausalLM.from_pretrained("OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", device="GPU")` for maximum generation speed (or `device="CPU"` if lower initial compile latency is preferred).
2. **Deterministic & Tool-Calling Evaluation:**
   - Benchmark structured JSON / function calling with Qwen2.5's ChatML `<tool_call>` syntax before integrating external weather or geospatial APIs.
3. **Maintain Compliance Documentation:**
   - Keep `run_qwen_quantized.py` and `quantized_benchmark_results.json` as reproducible proof in your hackathon submission repository.
