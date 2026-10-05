"""Measured QLoRA run for the three-label TruthLens verifier task."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}


def instruction(row: dict[str, str]) -> str:
    return (
        "Classify the claim using only the evidence. Return exactly one label: "
        "SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\n"
        f"Claim: {row['claim']}\nEvidence: {row['evidence']}\nLabel:"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    required = {"claim", "evidence", "label"}
    if not rows or any(not required.issubset(row) or row["label"] not in VALID_LABELS for row in rows):
        raise ValueError("Rows must contain claim, evidence, and a valid TruthLens label.")
    return rows


class TokenizedVerifierDataset:
    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            answer = tokenizer(row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            maximum_prompt = max_length - len(answer)
            prompt_ids = tokenizer(
                instruction(row),
                add_special_tokens=True,
                truncation=True,
                max_length=maximum_prompt,
            )["input_ids"]
            input_ids = prompt_ids + answer
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + answer,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


@dataclass
class VerifierCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        inputs, masks, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            inputs.append(item["input_ids"] + [self.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/training/training_config.json"))
    parser.add_argument("--train", type=Path, default=Path("ml/datasets/processed/train.jsonl"))
    parser.add_argument(
        "--validation", type=Path, default=Path("ml/datasets/processed/validation.jsonl")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(cfg["output_dir"])
    training_rows = read_rows(args.train)
    validation_rows = read_rows(args.validation)
    if args.smoke:
        training_rows = training_rows[:6]
        validation_rows = validation_rows[:3]
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; refusing to report a CPU run as QLoRA.")
    set_seed(cfg["seed"])
    use_4bit = not args.no_4bit
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quantization,
        device_map="auto" if use_4bit else None,
        torch_dtype=torch.bfloat16,
    )
    load_seconds = time.perf_counter() - load_started
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    train_dataset = TokenizedVerifierDataset(training_rows, tokenizer, cfg["max_length"])
    validation_dataset = TokenizedVerifierDataset(validation_rows, tokenizer, cfg["max_length"])
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=cfg["epochs"],
        max_steps=1 if args.smoke else -1,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=1,
        eval_strategy="steps" if args.smoke else "epoch",
        eval_steps=1 if args.smoke else None,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=1,
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=VerifierCollator(tokenizer.pad_token_id),
    )
    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    evaluation = trainer.evaluate()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    recorded_arguments = {
        key: value
        for key, value in training_args.to_dict().items()
        if "token" not in key.lower()
    }
    run = {
        "status": "completed",
        "run_type": "tiny_gpu_smoke" if args.smoke else "practical_qlora",
        "base_model": cfg["base_model"],
        "quantization": "4-bit NF4 double quantization" if use_4bit else "disabled",
        "seed": cfg["seed"],
        "data": {
            "train_path": str(args.train),
            "validation_path": str(args.validation),
            "train_samples": len(training_rows),
            "validation_samples": len(validation_rows),
        },
        "lora": {
            "r": cfg["lora_r"],
            "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"],
            "target_modules": cfg["target_modules"],
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": 100 * trainable / total,
        },
        "training_arguments": recorded_arguments,
        "model_load_seconds": load_seconds,
        "training_wall_seconds": elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
        "adapter_path": str(output),
    }
    (output / "training_run.json").write_text(
        json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, indent=2, default=str))


if __name__ == "__main__":
    main()
