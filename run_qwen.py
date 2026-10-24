import time
import os
import platform
import json
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_process_memory_mb():
    """Returns the current process RSS memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_hardware_info():
    """Gathers CPU, RAM, and GPU hardware information."""
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores_physical = psutil.cpu_count(logical=False)
    cpu_cores_logical = psutil.cpu_count(logical=True)
    
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "None"
    
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_name,
        "physical_cores": cpu_cores_physical,
        "logical_cores": cpu_cores_logical,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
    }

def main():
    print("=" * 60)
    print("Qwen2.5-1.5B-Instruct Minimal Benchmark & Inference")
    print("=" * 60)

    # 1. Hardware Inspection
    hw_info = get_hardware_info()
    print("\n[1] Hardware Environment:")
    for k, v in hw_info.items():
        print(f"  - {k}: {v}")

    # Determine execution device and dtype
    if hw_info["cuda_available"]:
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        limitations = "Dedicated NVIDIA GPU detected. CUDA acceleration enabled."
    else:
        device = "cpu"
        dtype = torch.float32  # Standard, high-precision and compatible for CPU execution
        limitations = (
            "No CUDA-capable dedicated GPU available. "
            "Running on CPU (Intel Core i7-1265U with integrated Intel Iris Xe). "
            "Generation speed is bound by CPU memory bandwidth and AVX/vector instruction throughput."
        )
    print(f"\nExecution Target: {device.upper()} (dtype: {dtype})")

    # 2. Memory Baseline
    mem_baseline = get_process_memory_mb()
    print(f"\n[2] Initial Process Memory: {mem_baseline:.2f} MB")

    # 3. Model Loading
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\n[3] Loading Model & Tokenizer: '{model_name}'...")
    
    t_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    )
    t_load_end = time.perf_counter()
    load_time_sec = t_load_end - t_load_start

    mem_after_load = get_process_memory_mb()
    model_mem_delta = mem_after_load - mem_baseline
    print(f"  - Model loaded successfully in {load_time_sec:.2f} seconds.")
    print(f"  - Process Memory after load: {mem_after_load:.2f} MB (Delta: +{model_mem_delta:.2f} MB)")

    # 4. Prompt Preparation
    prompt_text = "Suggest three outdoor activities for a sunny day."
    messages = [
        {"role": "system", "content": "You are a helpful, respectful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
    input_token_count = model_inputs.input_ids.shape[1]

    print(f"\n[4] Running Inference...")
    print(f"  - Prompt: \"{prompt_text}\"")
    print(f"  - Input Tokens: {input_token_count}")

    # 5. Response Generation & Profiling
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    t_gen_end = time.perf_counter()
    gen_time_sec = t_gen_end - t_gen_start

    # Isolate generated tokens from input tokens
    generated_tokens = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    output_token_count = len(generated_tokens[0])
    tokens_per_sec = output_token_count / gen_time_sec if gen_time_sec > 0 else 0

    mem_peak = get_process_memory_mb()

    # 6. Display Model Output
    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print("=" * 60)
    print(response_text.strip())
    print("=" * 60)

    # 7. Summary & Metrics Record
    summary = {
        "model": model_name,
        "prompt": prompt_text,
        "device": device,
        "dtype": str(dtype),
        "loading_time_seconds": round(load_time_sec, 3),
        "response_time_seconds": round(gen_time_sec, 3),
        "input_tokens": int(input_token_count),
        "output_tokens": int(output_token_count),
        "total_tokens": int(input_token_count + output_token_count),
        "tokens_per_second": round(tokens_per_sec, 2),
        "memory_usage": {
            "initial_process_ram_mb": round(mem_baseline, 2),
            "ram_after_model_load_mb": round(mem_after_load, 2),
            "ram_delta_model_mb": round(model_mem_delta, 2),
            "peak_ram_during_generation_mb": round(mem_peak, 2),
            "system_total_ram_gb": hw_info["total_ram_gb"],
            "system_available_ram_gb": hw_info["available_ram_gb"]
        },
        "hardware_info": hw_info,
        "hardware_limitations": limitations
    }

    print("\n[5] Recorded Performance Metrics:")
    print(f"  - Model Loading Time: {summary['loading_time_seconds']} s")
    print(f"  - Response Generation Time: {summary['response_time_seconds']} s")
    print(f"  - Output Tokens: {summary['output_tokens']} tokens")
    print(f"  - Throughput: {summary['tokens_per_second']} tokens/s")
    print(f"  - Process RAM Delta for Model: {summary['memory_usage']['ram_delta_model_mb']} MB")
    print(f"  - Peak Process RAM: {summary['memory_usage']['peak_ram_during_generation_mb']} MB")
    print(f"  - Hardware Limitations: {summary['hardware_limitations']}")

    # Save to JSON
    output_json_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBenchmark metrics successfully saved to: {output_json_path}")

if __name__ == "__main__":
    main()
