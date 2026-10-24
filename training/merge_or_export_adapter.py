"""Script to merge LoRA adapter into base model and export unified weights."""
import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.logger import logger


def merge_and_export(
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str = "huggingface_model",
    output_dir: str = "merged_model"
):
    """Merges trained LoRA weights into base model and saves unified standalone checkpoint."""
    print("=" * 70)
    print("Export / Merge LoRA Adapter into Standalone Checkpoint")
    print(f"Base Model: {base_model_name}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    adapter_weights = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights):
        raise FileNotFoundError(
            f"Adapter weights not found at {adapter_weights}. "
            "Please run 'training/train_lora.py' first."
        )

    print("\n[1] Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    print("\n[2] Applying LoRA Adapter...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("\n[3] Merging Adapter Weights into Base Model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    print(f"\n[4] Saving Merged Model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Standalone merged model exported to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="huggingface_model")
    parser.add_argument("--output-dir", default="merged_model")
    args = parser.parse_args()

    merge_and_export(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir
    )
