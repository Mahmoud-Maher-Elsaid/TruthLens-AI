"""LoRA/QLoRA verifier training for an open, Apache-2.0 Qwen2.5 base model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prompt(row: dict[str, str]) -> str:
    return f"Classify the claim using only the evidence. Return one label: SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\nClaim: {row['claim']}\nEvidence: {row['evidence']}\nLabel: {row['label']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/training/training_config.json"))
    parser.add_argument("--data", type=Path, default=Path("ml/datasets/processed/verifier.jsonl"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data without loading model weights")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    if args.smoke: rows = rows[:4]
    required = {"claim", "evidence", "label"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Training rows must contain claim, evidence, and label")
    if args.dry_run:
        print(json.dumps({"status": "validated", "rows": len(rows), "base_model": cfg["base_model"], "quantization": "disabled" if args.no_4bit else "4-bit NF4"}, indent=2))
        return
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
    set_seed(cfg["seed"])
    quant = None if args.no_4bit else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant, device_map="auto" if quant else None, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if quant: model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    dataset = Dataset.from_list([{"text": prompt(row)} for row in rows]).map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"]), batched=True, remove_columns=["text"])
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(output_dir=str(output), num_train_epochs=cfg["epochs"], max_steps=1 if args.smoke else -1, per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=1 if args.smoke else cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"], logging_steps=1, save_strategy="no" if args.smoke else "epoch", report_to="none", seed=cfg["seed"], bf16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    metrics = trainer.train().metrics
    model.save_pretrained(output); tokenizer.save_pretrained(output)
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
