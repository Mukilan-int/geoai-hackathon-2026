"""Evaluation script benchmarking base model vs fine-tuned LoRA adapter on held-out dataset."""
import argparse
import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.json_sanitizer import sanitize_and_parse_json, validate_schema


def evaluate_model_on_split(
    model,
    tokenizer,
    eval_jsonl_path: str,
    device_str: str = "cpu",
    label: str = "Model"
) -> dict:
    """Evaluates a causal LM on the held-out evaluation dataset."""
    print(f"\n--- Evaluating {label} on {eval_jsonl_path} ---")
    results = []
    valid_json_count = 0
    schema_valid_count = 0
    total_time = 0.0

    with open(eval_jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for idx, rec in enumerate(records, 1):
        rec_id = rec.get("id", f"eval_{idx}")
        # Build prompt excluding the assistant target
        prompt_messages = rec["messages"][:2]
        formatted = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device_str)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False  # Deterministic greedy decoding
            )
        gen_time = time.perf_counter() - t0
        total_time += gen_time

        gen_tokens = outputs[0][inputs.input_ids.shape[1] :]
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        parsed, err = sanitize_and_parse_json(output_text)
        is_json = parsed is not None
        if is_json:
            valid_json_count += 1
            is_schema, schema_errs = validate_schema(parsed)
            if is_schema:
                schema_valid_count += 1
        else:
            is_schema = False
            schema_errs = [err]

        print(f"  [{idx}/{len(records)}] {rec_id} | Time: {gen_time:.2f}s | Valid JSON: {is_json} | Schema Valid: {is_schema}")
        results.append({
            "id": rec_id,
            "gen_time_sec": round(gen_time, 2),
            "is_valid_json": is_json,
            "is_schema_valid": is_schema,
            "parsed_suitability": parsed.get("suitability") if parsed else None,
            "errors": schema_errs if not is_schema else []
        })

    total_records = len(records)
    json_rate = round((valid_json_count / total_records) * 100.0, 1)
    schema_rate = round((schema_valid_count / total_records) * 100.0, 1)
    avg_latency = round(total_time / total_records, 2)

    summary = {
        "model_label": label,
        "total_records": total_records,
        "valid_json_count": valid_json_count,
        "json_parse_rate_pct": json_rate,
        "schema_valid_count": schema_valid_count,
        "schema_conformance_rate_pct": schema_rate,
        "avg_latency_sec": avg_latency,
        "eval_details": results
    }
    print(f"\nSummary for {label}:")
    print(f"  - Valid JSON Rate: {json_rate}% ({valid_json_count}/{total_records})")
    print(f"  - Schema Conformance: {schema_rate}% ({schema_valid_count}/{total_records})")
    print(f"  - Average Generation Time: {avg_latency}s")
    return summary


def run_evaluation(
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str = "huggingface_model",
    eval_data: str = "datasets/eval_recommendations.jsonl",
    device: str = "cpu"
):
    print("=" * 70)
    print("Held-Out Evaluation: Base Model vs. Fine-Tuned LoRA Adapter")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Base Model Evaluation
    print("\n[Stage 1] Loading Base Model for Evaluation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    base_model.eval()

    base_summary = evaluate_model_on_split(
        base_model, tokenizer, eval_data, device_str=device, label="Base Model (Qwen2.5-1.5B)"
    )

    # 2. Check if adapter exists
    adapter_summary = None
    adapter_weights_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(adapter_weights_file):
        print("\n[Stage 2] Loading LoRA Adapter onto Base Model...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        peft_model.eval()
        adapter_summary = evaluate_model_on_split(
            peft_model, tokenizer, eval_data, device_str=device, label="Fine-Tuned LoRA Adapter"
        )
    else:
        print(f"\n[Stage 2] No adapter weights found at '{adapter_weights_file}'. Skipping adapter evaluation.")

    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model_summary": base_summary,
        "adapter_summary": adapter_summary
    }

    out_file = os.path.join(os.path.dirname(eval_data), "evaluation_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[SAVED] Evaluation summary written to: {out_file}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="huggingface_model")
    parser.add_argument("--eval-data", default="datasets/eval_recommendations.jsonl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_evaluation(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        eval_data=args.eval_data,
        device=args.device
    )
