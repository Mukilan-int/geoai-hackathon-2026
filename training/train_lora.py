"""LoRA fine-tuning implementation for Qwen2.5-1.5B-Instruct using Transformers and PEFT."""
import argparse
import json
import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.logger import logger


class GeoAIRecDataset(Dataset):
    """Custom dataset tokenizing ChatML messages for causal LM training."""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    formatted = tokenizer.apply_chat_template(
                        record["messages"],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    encodings = tokenizer(
                        formatted,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    input_ids = encodings.input_ids.squeeze(0)
                    attention_mask = encodings.attention_mask.squeeze(0)
                    labels = input_ids.clone()
                    # Mask padding tokens in loss calculation
                    labels[attention_mask == 0] = -100
                    self.examples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def run_lora_training(
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_data_path: str = "datasets/train_recommendations.jsonl",
    output_dir: str = "huggingface_model",
    max_steps: int = 2,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    device_str: str = "cpu"
):
    """Executes a real PEFT LoRA training step and exports adapter weights and configuration."""
    print("=" * 70)
    print("GeoAI Hackathon 2026: LoRA Fine-Tuning Pipeline")
    print(f"Base Model: {base_model_name} (<3B parameters)")
    print(f"Target Device: {device_str.upper()} | Output: {output_dir}")
    print("=" * 70)

    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    print("\n[1] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    print("\n[2] Loading Base Model in FP32...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(device_str)

    # 3. Configure LoRA
    print("\n[3] Initializing PEFT LoRA Adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 4. Prepare Dataset and DataLoader
    print(f"\n[4] Preparing Training Data from {train_data_path}...")
    dataset = GeoAIRecDataset(train_data_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 5. Training Loop
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
    peft_model.train()
    
    print(f"\n[5] Commencing LoRA Optimization (Target steps: {max_steps})...")
    step_count = 0
    total_loss = 0.0
    loss_history = []

    for epoch in range(epochs):
        for batch in dataloader:
            step_start = time.time()
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device_str)
            attention_mask = batch["attention_mask"].to(device_str)
            labels = batch["labels"].to(device_str)

            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            step_count += 1
            loss_val = float(loss.item())
            total_loss += loss_val
            loss_history.append({"step": step_count, "loss": round(loss_val, 4), "time_sec": round(step_time, 2)})

            print(f"  Step {step_count}/{max_steps} | Loss: {loss_val:.4f} | Time: {step_time:.2f}s")
            
            if step_count >= max_steps:
                break
        if step_count >= max_steps:
            break

    # 6. Save Adapter and Tokenizer
    print(f"\n[6] Saving Trained LoRA Adapter to '{output_dir}'...")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_duration = round(time.time() - t0, 2)
    avg_loss = round(total_loss / max(step_count, 1), 4)

    # Save training metrics report
    metrics = {
        "status": "COMPLETED",
        "base_model": base_model_name,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "steps_completed": step_count,
        "average_loss": avg_loss,
        "loss_history": loss_history,
        "total_duration_sec": total_duration,
        "device": device_str,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] LoRA Adapter Successfully Trained and Exported!")
    print(f"  - Adapter files saved in: {output_dir}")
    print(f"  - Average Loss: {avg_loss}")
    print(f"  - Total Training Duration: {total_duration}s")
    print(f"  - Training Metrics: {metrics_file}")
    print("=" * 70)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA Adapter for Qwen2.5-1.5B")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-data", default="datasets/train_recommendations.jsonl")
    parser.add_argument("--output-dir", default="huggingface_model")
    parser.add_argument("--max-steps", type=int, default=2, help="Number of gradient steps to run")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    run_lora_training(
        base_model_name=args.base_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        device_str=args.device
    )
