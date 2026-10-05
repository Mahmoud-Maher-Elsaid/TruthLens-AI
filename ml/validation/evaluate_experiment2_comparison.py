"""Evaluate the baseline, Experiment 1, and Experiment 2 on one held-out split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).resolve().parent
TRAINING_DIR = VALIDATION_DIR.parent / "training"
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(TRAINING_DIR))
common = importlib.import_module("common")
experiment2_format = importlib.import_module("experiment2_format")

LABELS = common.LABELS
NLI_TO_TRUTHLENS = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT_EVIDENCE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment1_prompt(row: dict[str, Any], _tokenizer: Any) -> str:
    return (
        "Classify the claim using only the evidence. Return exactly one label: "
        "SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\n"
        f"Claim: {row['claim']}\nEvidence: {row['evidence']}\nLabel:"
    )


def experiment2_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        experiment2_format.messages(row),
        tokenize=False,
        add_generation_prompt=True,
    )


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required_final = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "training_run.json",
        "training.log",
    }
    required_checkpoint = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    missing_final = sorted(name for name in required_final if not (args.experiment2 / name).is_file())
    run_path = args.experiment2 / "training_run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing Experiment 2 run metadata: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    best_checkpoint = Path(run["checkpoint_selection"]["best_checkpoint"])
    missing_checkpoint = sorted(
        name for name in required_checkpoint if not (best_checkpoint / name).is_file()
    )
    if missing_final or missing_checkpoint:
        raise RuntimeError(
            f"Missing Experiment 2 artifacts: final={missing_final}, checkpoint={missing_checkpoint}"
        )
    if run.get("status") != "completed":
        raise RuntimeError(f"Experiment 2 training is not completed: {run.get('status')}")
    if best_checkpoint.name != "checkpoint-50":
        raise RuntimeError(f"Unexpected best checkpoint: {best_checkpoint}")
    final_hash = sha256(args.experiment2 / "adapter_model.safetensors")
    checkpoint_hash = sha256(best_checkpoint / "adapter_model.safetensors")
    if final_hash != checkpoint_hash:
        raise RuntimeError("Exported Experiment 2 adapter does not match the selected checkpoint")

    experiment1_required = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"}
    missing_experiment1 = sorted(
        name for name in experiment1_required if not (args.experiment1 / name).is_file()
    )
    if missing_experiment1:
        raise RuntimeError(f"Missing Experiment 1 artifacts: {missing_experiment1}")

    rows = common.read_jsonl(args.data)
    training = common.read_jsonl(args.train)
    validation = common.read_jsonl(args.validation)
    counts = Counter(row["label"] for row in rows)
    if len(rows) != 180 or counts != Counter({label: 60 for label in LABELS}):
        raise RuntimeError(f"Unexpected Experiment 2 test composition: rows={len(rows)}, {counts}")
    if {row.get("source_split") for row in rows} != {"dev"}:
        raise RuntimeError("Experiment 2 test contains rows outside the held-out dev source split")
    if any(row.get("source_split") == "controlled_contrast" for row in rows):
        raise RuntimeError("Controlled training examples leaked into Experiment 2 test")

    sets: dict[str, dict[str, set[str]]] = {}
    for name, split_rows in (("train", training), ("validation", validation), ("test", rows)):
        sets[name] = {
            "source_ids": {str(row["source_id"]) for row in split_rows},
            "group_ids": {str(row["claim_group_id"]) for row in split_rows},
        }
    leakage = {
        "train_test_source_ids": len(sets["train"]["source_ids"] & sets["test"]["source_ids"]),
        "train_test_group_ids": len(sets["train"]["group_ids"] & sets["test"]["group_ids"]),
        "validation_test_source_ids": len(
            sets["validation"]["source_ids"] & sets["test"]["source_ids"]
        ),
        "validation_test_group_ids": len(
            sets["validation"]["group_ids"] & sets["test"]["group_ids"]
        ),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Experiment 2 held-out split leakage: {leakage}")

    adapter1 = json.loads((args.experiment1 / "adapter_config.json").read_text(encoding="utf-8"))
    adapter2 = json.loads((args.experiment2 / "adapter_config.json").read_text(encoding="utf-8"))
    base1 = adapter1["base_model_name_or_path"]
    base2 = adapter2["base_model_name_or_path"]
    if base1 != base2 or base2 != run["base_model"]:
        raise RuntimeError(f"Adapter base-model mismatch: experiment1={base1}, experiment2={base2}")

    return {
        "status": "validated",
        "test_path": str(args.data),
        "test_sha256": sha256(args.data),
        "test_rows": len(rows),
        "test_labels": dict(sorted(counts.items())),
        "source_splits": ["dev"],
        "controlled_examples_in_test": 0,
        "leakage": leakage,
        "experiment1_adapter": str(args.experiment1),
        "experiment2_adapter": str(args.experiment2),
        "experiment2_best_checkpoint": str(best_checkpoint),
        "experiment2_export_matches_best_checkpoint": True,
        "experiment2_adapter_sha256": final_hash,
        "missing_artifacts": [],
        "base_model": base2,
        "baseline_model": args.baseline_model,
        "output": str(args.output),
    }


def output_validity(invalid: int, count: int) -> dict[str, Any]:
    return {
        "valid_exact_label_rate": (count - invalid) / count,
        "invalid_output_rate": invalid / count,
        "invalid_outputs": invalid,
        "invalid_fallback": "INSUFFICIENT_EVIDENCE",
    }


def evaluate_baseline(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    model = CrossEncoder(model_id)
    load_seconds = time.perf_counter() - started
    id2label = {int(key): value.lower() for key, value in model.model.config.id2label.items()}
    if set(id2label.values()) != set(NLI_TO_TRUTHLENS):
        raise RuntimeError(f"Unsafe baseline label mapping: {id2label}")
    model.predict([(rows[0]["evidence"], rows[0]["claim"])], batch_size=1, show_progress_bar=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    predictions: list[str] = []
    latencies: list[float] = []
    started_all = time.perf_counter()
    for row in rows:
        started = time.perf_counter()
        scores = np.asarray(
            model.predict(
                [(row["evidence"], row["claim"])],
                batch_size=1,
                show_progress_bar=False,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(NLI_TO_TRUTHLENS[id2label[int(scores[0].argmax())]])
    runtime = time.perf_counter() - started_all
    payload = {
        "model": model_id,
        "load_seconds": load_seconds,
        "runtime_seconds": runtime,
        "latency": common.latency_summary(latencies),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "output_validity": output_validity(0, len(rows)),
        "metrics": common.classification_metrics([row["label"] for row in rows], predictions),
        "predictions": [
            {"source_id": row["source_id"], "gold": row["label"], "prediction": prediction}
            for row, prediction in zip(rows, predictions, strict=True)
        ],
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def evaluate_adapter(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    adapter_name: str,
    adapter_path: Path,
    prompt_builder: Callable[[dict[str, Any], Any], str],
    max_length: int,
) -> dict[str, Any]:
    import torch

    model.set_adapter(adapter_name)

    def infer(row: dict[str, Any]) -> tuple[str, str, bool, float]:
        prompt = prompt_builder(row, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000
        raw = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        normalized = raw.upper()
        valid = normalized in LABELS
        return (normalized if valid else "INSUFFICIENT_EVIDENCE", raw, valid, latency)

    infer(rows[0])
    torch.cuda.reset_peak_memory_stats()
    predicted: list[str] = []
    raw_outputs: list[str] = []
    latencies: list[float] = []
    invalid = 0
    started_all = time.perf_counter()
    for row in rows:
        prediction, raw, valid, latency = infer(row)
        predicted.append(prediction)
        raw_outputs.append(raw)
        latencies.append(latency)
        invalid += int(not valid)
    runtime = time.perf_counter() - started_all
    return {
        "adapter": str(adapter_path),
        "runtime_seconds": runtime,
        "latency": common.latency_summary(latencies),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 12},
        "output_validity": output_validity(invalid, len(rows)),
        "metrics": common.classification_metrics([row["label"] for row in rows], predicted),
        "predictions": [
            {
                "source_id": row["source_id"],
                "gold": row["label"],
                "prediction": prediction,
                "raw_output": raw,
            }
            for row, prediction, raw in zip(rows, predicted, raw_outputs, strict=True)
        ],
    }


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        name: right["metrics"][name] - left["metrics"][name]
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument(
        "--experiment1",
        type=Path,
        default=Path("ml/training/outputs/truthlens-qwen2.5-1.5b-lora"),
    )
    parser.add_argument(
        "--experiment2",
        type=Path,
        default=Path("ml/training/outputs/experiment2-qwen2.5-1.5b-qlora"),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("ml/datasets/processed/experiment2/train.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("ml/datasets/processed/experiment2/validation.jsonl"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/datasets/processed/experiment2/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/benchmarks/experiment2_verifier_comparison.json"),
    )
    args = parser.parse_args()

    validation = validate_inputs(args)
    if args.dry_run:
        print(json.dumps(validation, indent=2))
        return

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("The three-model evaluation requires CUDA")
    rows = common.read_jsonl(args.data)
    baseline = evaluate_baseline(rows, args.baseline_model)

    experiment1_config = PeftConfig.from_pretrained(args.experiment1)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        experiment1_config.base_model_name_or_path,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.experiment1, adapter_name="experiment1")
    model.load_adapter(args.experiment2, adapter_name="experiment2")
    model.eval()
    shared_load_seconds = time.perf_counter() - started
    tokenizer1 = AutoTokenizer.from_pretrained(args.experiment1)
    tokenizer2 = AutoTokenizer.from_pretrained(args.experiment2)
    experiment1 = evaluate_adapter(
        rows,
        model,
        tokenizer1,
        "experiment1",
        args.experiment1,
        experiment1_prompt,
        512,
    )
    experiment2 = evaluate_adapter(
        rows,
        model,
        tokenizer2,
        "experiment2",
        args.experiment2,
        experiment2_prompt,
        768,
    )
    experiment1["shared_base_and_adapters_load_seconds"] = shared_load_seconds
    experiment2["shared_base_and_adapters_load_seconds"] = shared_load_seconds
    payload = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": common.utc_now(),
        "environment": common.environment(),
        "dataset": validation,
        "same_test_samples": True,
        "models": {
            "baseline": baseline,
            "experiment1": experiment1,
            "experiment2": experiment2,
        },
        "deltas": {
            "experiment1_minus_baseline": metric_delta(baseline, experiment1),
            "experiment2_minus_baseline": metric_delta(baseline, experiment2),
            "experiment2_minus_experiment1": metric_delta(experiment1, experiment2),
        },
    }
    common.write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
