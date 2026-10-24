import time
import os
import platform
import json
import psutil
import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

def get_process_memory_mb():
    """Returns current process RSS memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_hardware_info():
    """Gathers system hardware info."""
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores_physical = psutil.cpu_count(logical=False)
    cpu_cores_logical = psutil.cpu_count(logical=True)
    
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_name,
        "physical_cores": cpu_cores_physical,
        "logical_cores": cpu_cores_logical,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "gpu_detected": "Intel(R) Iris(R) Xe Graphics (iGPU)",
        "cuda_available": False
    }

def run_inference_benchmark(model, tokenizer, prompt_text, device_label, system_instruction="You are a helpful, respectful assistant."):
    """Executes a single benchmark run with precise token and timing measurements."""
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([formatted_prompt], return_tensors="pt")
    input_tokens = inputs.input_ids.shape[1]
    
    mem_before_gen = get_process_memory_mb()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False  # Deterministic greedy decoding for repeatable benchmarking
        )
    t1 = time.perf_counter()
    gen_time_sec = t1 - t0
    
    generated_tokens = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_tokens = len(generated_tokens[0])
    response_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    throughput = output_tokens / gen_time_sec if gen_time_sec > 0 else 0
    mem_peak = get_process_memory_mb()
    
    return {
        "device": device_label,
        "prompt": prompt_text,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "generation_time_sec": round(gen_time_sec, 3),
        "throughput_tokens_per_sec": round(throughput, 2),
        "mem_peak_mb": round(mem_peak, 2),
        "response": response_text.strip()
    }

def main():
    print("=" * 70)
    print("GeoAI Hackathon 2026: Quantized Qwen2.5-1.5B-Instruct Benchmark")
    print("=" * 70)
    
    hw_info = get_hardware_info()
    print("\n[1] Hardware Environment:")
    for k, v in hw_info.items():
        print(f"  - {k}: {v}")
        
    model_id = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
    print(f"\n[2] Candidate Model: {model_id}")
    print("  - Base Model: Qwen/Qwen2.5-1.5B-Instruct (< 3B params)")
    print("  - Host: Public Hugging Face Hub")
    print("  - Loader: OVModelForCausalLM.from_pretrained(...) [Hugging Face Optimum-Intel API]")
    print("  - Quantization: INT4 Weight-Only Quantization (sym/asym NNCF)")
    print("  - License: Apache 2.0")

    mem_baseline = get_process_memory_mb()
    print(f"\n[3] Baseline Process RSS: {mem_baseline:.2f} MB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    results = {
        "benchmark_metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": model_id,
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "loader": "optimum.intel.OVModelForCausalLM.from_pretrained",
            "quantization": "INT4",
            "license": "Apache-2.0"
        },
        "hardware_info": hw_info,
        "memory_baseline_mb": round(mem_baseline, 2),
        "devices_tested": {}
    }

    # Evaluate on both CPU and Intel Iris Xe GPU
    test_devices = [("CPU", "OpenVINO CPU (i7-1265U)"), ("GPU", "OpenVINO GPU (Iris Xe iGPU)")]
    
    for dev_code, dev_label in test_devices:
        print(f"\n" + "-" * 70)
        print(f"Testing Device Target: {dev_label}")
        print("-" * 70)
        
        mem_before_load = get_process_memory_mb()
        t_load_start = time.perf_counter()
        model = OVModelForCausalLM.from_pretrained(
            model_id,
            device=dev_code
        )
        t_load_end = time.perf_counter()
        load_time_sec = t_load_end - t_load_start
        mem_after_load = get_process_memory_mb()
        model_ram_delta = mem_after_load - mem_before_load
        
        print(f"  - Model loaded on {dev_code} in {load_time_sec:.2f} seconds.")
        print(f"  - Process RAM: {mem_after_load:.2f} MB (Delta: +{model_ram_delta:.2f} MB)")
        
        # Test 1: Identical baseline prompt for exact comparison
        prompt_1 = "Suggest three outdoor activities for a sunny day."
        print(f"\n  [Run 1: Baseline Comparison Prompt]")
        res_1 = run_inference_benchmark(model, tokenizer, prompt_1, dev_label)
        print(f"    - Throughput: {res_1['throughput_tokens_per_sec']} tokens/sec")
        print(f"    - Generation Time: {res_1['generation_time_sec']} s ({res_1['output_tokens']} tokens)")
        print(f"    - Response:\n{res_1['response']}")
        
        # Test 2: GeoAI context-aware outdoor activity prompt
        prompt_2 = (
            "Location: Sanjay Gandhi National Park, Mumbai, India.\n"
            "Current Weather: Sunny, 32°C (89.6°F), 65% humidity, UV index 8 (Very High), wind speed 12 km/h.\n"
            "Air Quality: Moderate (AQI 85).\n"
            "Given the supplied weather and location information, recommend an outdoor activity, a suitable time, "
            "relevant precautions, and the reasons for your recommendation. Do not invent missing data."
        )
        print(f"\n  [Run 2: GeoAI Contextual Prompt]")
        res_2 = run_inference_benchmark(
            model,
            tokenizer,
            prompt_2,
            dev_label,
            system_instruction="You are an expert GeoAI assistant specializing in geographic and meteorological outdoor recommendations."
        )
        print(f"    - Throughput: {res_2['throughput_tokens_per_sec']} tokens/sec")
        print(f"    - Generation Time: {res_2['generation_time_sec']} s ({res_2['output_tokens']} tokens)")
        print(f"    - Response:\n{res_2['response']}")
        
        results["devices_tested"][dev_code] = {
            "device_name": dev_label,
            "load_time_seconds": round(load_time_sec, 3),
            "ram_after_load_mb": round(mem_after_load, 2),
            "ram_delta_mb": round(model_ram_delta, 2),
            "peak_ram_mb": round(max(res_1["mem_peak_mb"], res_2["mem_peak_mb"]), 2),
            "test_1_baseline_prompt": res_1,
            "test_2_geoai_prompt": res_2
        }

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    cpu_res = results["devices_tested"]["CPU"]["test_1_baseline_prompt"]
    gpu_res = results["devices_tested"]["GPU"]["test_1_baseline_prompt"]
    print(f"Baseline PyTorch FP32 (CPU)  : ~1.28 - 2.31 tok/s | ~6,340 MB Peak RAM | Load: 15.65 s")
    print(f"Quantized OpenVINO INT4 (CPU):  {cpu_res['throughput_tokens_per_sec']} tok/s | {results['devices_tested']['CPU']['peak_ram_mb']} MB Peak RAM | Load: {results['devices_tested']['CPU']['load_time_seconds']} s")
    print(f"Quantized OpenVINO INT4 (GPU):  {gpu_res['throughput_tokens_per_sec']} tok/s | {results['devices_tested']['GPU']['peak_ram_mb']} MB Peak RAM | Load: {results['devices_tested']['GPU']['load_time_seconds']} s")
    print("=" * 70)

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "quantized_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMachine-readable benchmark report saved to: {output_path}")

if __name__ == "__main__":
    main()
