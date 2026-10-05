"""QLoRA Experiment 2 with cleaned neutral data and validation checkpoint selection."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment2_format import prompt_ids

VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    required = {"claim", "evidence", "label", "source_id", "claim_group_id"}
    if not rows or any(not required.issubset(row) or row["label"] not in VALID_LABELS for row in rows):
        raise ValueError(f"Invalid Experiment 2 rows in {path}")
    return rows


class TokenizedVerifierDataset:
    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            answer = tokenizer(row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            prefix = prompt_ids(tokenizer, row, max_length - len(answer))
            input_ids = prefix + answer
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prefix) + answer,
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
        return {
            "input_ids": torch.tensor(
                [item["input_ids"] + [self.pad_token_id] * (width - len(item["input_ids"])) for item in features],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [item["labels"] + [-100] * (width - len(item["labels"])) for item in features],
                dtype=torch.long,
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("ml/training/experiment2_config.json")
    )
    parser.add_argument(
        "--train", type=Path, default=Path("ml/datasets/processed/experiment2/train.jsonl")
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("ml/datasets/processed/experiment2/validation.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    training_rows = read_rows(args.train)
    validation_rows = read_rows(args.validation)
    train_groups = {row["claim_group_id"] for row in training_rows}
    validation_groups = {row["claim_group_id"] for row in validation_rows}
    if train_groups & validation_groups:
        raise RuntimeError("Claim-group leakage detected between training and validation")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "experiment": cfg["experiment"],
                    "base_model": cfg["base_model"],
                    "output_dir": cfg["output_dir"],
                    "train_rows": len(training_rows),
                    "validation_rows": len(validation_rows),
                    "train_validation_group_overlap": 0,
                    "max_length": cfg["max_length"],
                    "learning_rate": cfg["learning_rate"],
                    "max_epochs": cfg["max_epochs"],
                    "early_stopping_patience": cfg["early_stopping_patience"],
                    "quantization": "4-bit NF4 double quantization",
                },
                indent=2,
            )
        )
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 2 requires CUDA")
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    load_seconds = time.perf_counter() - load_started
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
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=cfg["max_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=VerifierCollator(tokenizer.pad_token_id),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"],
                early_stopping_threshold=cfg["early_stopping_threshold"],
            )
        ],
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
        "experiment": cfg["experiment"],
        "base_model": cfg["base_model"],
        "quantization": "4-bit NF4 double quantization",
        "prompt_template": "Qwen chat template with explicit three-way definitions",
        "authoritative_label": "source NLI label after semantic-risk filtering",
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
        "checkpoint_selection": {
            "metric": "eval_loss on masked label tokens",
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "early_stopping_patience": cfg["early_stopping_patience"],
            "early_stopping_threshold": cfg["early_stopping_threshold"],
        },
        "deterministic_generation_settings_for_evaluation": cfg["generation"],
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
