"""Train the Experiment 3 multi-evidence verifier with standard BF16 PEFT LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .format import prompt_ids, validate_row
    from .metrics import callback_classification_metrics, classification_report, generate_labels
except ImportError:  # Supports direct PowerShell execution.
    from format import prompt_ids, validate_row
    from metrics import callback_classification_metrics, classification_report, generate_labels

ROOT = Path(__file__).resolve().parents[3]


def configure_project_cache() -> None:
    service_path = ROOT / "services" / "ai-api"
    if str(service_path) not in sys.path:
        sys.path.insert(0, str(service_path))
    from app.core.model_cache import configure_model_cache

    configure_model_cache()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    for row in rows:
        validate_row(row)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LabelDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            answer = tokenizer(row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            prefix = prompt_ids(tokenizer, row, max_length - len(answer))
            self.items.append(
                {
                    "input_ids": prefix + answer,
                    "attention_mask": [1] * (len(prefix) + len(answer)),
                    "labels": [-100] * len(prefix) + answer,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


@dataclass
class LabelCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        return {
            "input_ids": torch.tensor([item["input_ids"] + [self.pad_token_id] * (width - len(item["input_ids"])) for item in features], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in features], dtype=torch.long),
            "labels": torch.tensor([item["labels"] + [-100] * (width - len(item["labels"])) for item in features], dtype=torch.long),
        }


def verify_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    audit = manifest.get("audit", {})
    expected = {"SUPPORTED": 400, "CONTRADICTED": 400, "INSUFFICIENT_EVIDENCE": 400}
    if audit.get("splits", {}).get("train", {}).get("class_counts") != expected:
        raise RuntimeError("Experiment 3 train class balance is not the frozen 400-per-class design.")
    if audit.get("splits", {}).get("validation", {}).get("class_counts") != {key: 60 for key in expected}:
        raise RuntimeError("Experiment 3 validation class balance is not the frozen 60-per-class design.")
    if any(audit.get("exact_duplicate_counts", {}).values()) or any(audit.get("near_duplicate_pairs", {}).values()):
        raise RuntimeError("Experiment 3 duplicate audit is not clean.")
    if any(audit.get("leakage_checks", {}).values()):
        raise RuntimeError("Experiment 3 cross-split leakage audit is not clean.")
    if manifest.get("seed") != config.get("seed"):
        raise RuntimeError("Manifest seed does not match training configuration.")


def validate_only(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    manifest = load_json(ROOT / data["manifest"])
    verify_manifest(config, manifest)
    train = load_rows(ROOT / data["output_dir"] / "train.jsonl")
    validation = load_rows(ROOT / data["output_dir"] / "validation.jsonl")
    return {
        "status": "validated_no_model_loaded",
        "experiment": config["experiment"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "strategy": config["training"]["strategy"],
        "quantization": config["training"]["quantization"],
        "checkpoint_metric": "validation_macro_f1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/training/experiment3/config.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    if args.validate_only:
        print(json.dumps(validate_only(config), indent=2))
        return

    configure_project_cache()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 3 requires CUDA; standard BF16 LoRA will not run on CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Experiment 3 requires BF16 support on the selected CUDA device.")

    data = config["data"]
    training = config["training"]
    manifest = load_json(ROOT / data["manifest"])
    verify_manifest(config, manifest)
    train_rows = load_rows(ROOT / data["output_dir"] / "train.jsonl")
    validation_rows = load_rows(ROOT / data["output_dir"] / "validation.jsonl")
    smoke = bool(args.smoke)
    if smoke:
        smoke_config = config["smoke"]
        train_rows = train_rows[: int(smoke_config["train_examples"])]
        validation_rows = validation_rows[: int(smoke_config["validation_examples"])]
        output = ROOT / smoke_config["output_dir"]
        max_steps = int(smoke_config["max_steps"])
    else:
        output = ROOT / training["output_dir"]
        max_steps = -1
    existing_output = [] if not output.exists() else list(output.iterdir())
    allowed_smoke_directory = ROOT / training["output_dir"] / "smoke"
    if (not smoke and existing_output and any(path != allowed_smoke_directory for path in existing_output) and not args.overwrite_output) or (smoke and existing_output and not args.overwrite_output):
        raise RuntimeError(f"Refusing to overwrite existing output: {output}. Use --overwrite-output only for an intentional rerun.")
    output.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(training["tokenizer"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    torch.cuda.reset_peak_memory_stats()
    model_load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(training["base_model"], dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = training["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], bias=lora["bias"], task_type="CAUSAL_LM"))
    model_load_seconds = time.perf_counter() - model_load_started
    trainable, total = model.get_nb_trainable_parameters()

    class MacroF1Trainer(Trainer):
        def evaluation_loop(
            self,
            dataloader: Any,
            description: str,
            prediction_loss_only: bool | None = None,
            ignore_keys: list[str] | None = None,
            metric_key_prefix: str = "eval",
        ) -> Any:
            # Trainer.dispatches callbacks only after evaluation_loop returns. Add
            # generated-label metrics here so early stopping and checkpoint
            # selection receive eval_macro_f1 in the same evaluation lifecycle.
            loop_output = super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )
            predicted = generate_labels(self.model, tokenizer, validation_rows, int(training["max_length"]))
            report = classification_report([row["label"] for row in validation_rows], predicted)
            loop_output.metrics.update(callback_classification_metrics(report, metric_key_prefix))
            record = {"global_step": self.state.global_step, "metrics": report}
            with (output / "validation_history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            return loop_output

    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=1 if smoke else training["max_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        max_grad_norm=training["max_grad_norm"],
        lr_scheduler_type=training["scheduler"],
        logging_steps=1 if smoke else 10,
        eval_strategy="steps",
        eval_steps=1 if smoke else training["eval_steps"],
        save_strategy="steps",
        save_steps=1 if smoke else training["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    trainer = MacroF1Trainer(
        model=model,
        args=arguments,
        train_dataset=LabelDataset(train_rows, tokenizer, int(training["max_length"])),
        eval_dataset=LabelDataset(validation_rows, tokenizer, int(training["max_length"])),
        data_collator=LabelCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training["early_stopping_patience"], early_stopping_threshold=training["early_stopping_threshold"])],
    )
    started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    predicted = generate_labels(trainer.model, tokenizer, validation_rows, int(training["max_length"]))
    validation_report = classification_report([row["label"] for row in validation_rows], predicted)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    run = {
        "status": "smoke_completed_not_scientific" if smoke else "completed",
        "experiment": config["experiment"],
        "config_sha256": sha256(ROOT / args.config),
        "manifest": manifest,
        "base_model": training["base_model"],
        "tokenizer": training["tokenizer"],
        "strategy": training["strategy"],
        "lora": lora,
        "seed": config["seed"],
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "checkpoint_selection": {"metric": "validation macro F1", "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric},
        "validation_report": validation_report,
        "runtime": {"model_load_seconds": model_load_seconds, "training_wall_seconds": wall_seconds, "peak_vram_bytes": torch.cuda.max_memory_allocated()},
        "execution": {
            "training_executed": True,
            "optimizer_steps": int(trainer.state.global_step),
            "configured_max_optimizer_steps": int(trainer.state.max_steps),
            "completed_epochs": float(trainer.state.epoch or 0),
            "note": "Execution fields are recorded from Trainer state. training_arguments.do_train is serialized configuration, not execution evidence.",
        },
        "train_metrics": result.metrics,
        "trainable_parameters": {"trainable": trainable, "total": total, "percent": 100 * trainable / total},
        "training_arguments": {key: value for key, value in arguments.to_dict().items() if "token" not in key.lower()},
    }
    (output / "training_run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2, default=str))


if __name__ == "__main__":
    main()
