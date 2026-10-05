"""Measure FP32 versus NF4 4-bit inference for the production NLI verifier."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))
common = importlib.import_module("common")

LABELS = common.LABELS
NLI_TO_TRUTHLENS = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT_EVIDENCE",
}


def validate_data(path: Path, expected_model: str) -> dict[str, Any]:
    rows = common.read_jsonl(path)
    counts = Counter(row["label"] for row in rows)
    if len(rows) != 180 or counts != Counter({label: 60 for label in LABELS}):
        raise RuntimeError(f"Unexpected quantization benchmark split: rows={len(rows)}, {counts}")
    if {row.get("source_split") for row in rows} != {"dev"}:
        raise RuntimeError("Quantization benchmark data is not the held-out Experiment 2 dev split")
    production_model = "cross-encoder/nli-deberta-v3-small"
    if expected_model != production_model:
        raise RuntimeError(
            f"Refusing to benchmark non-production verifier {expected_model!r}; expected {production_model!r}"
        )
    return {
        "status": "validated",
        "model": expected_model,
        "data": str(path),
        "samples": len(rows),
        "labels": dict(sorted(counts.items())),
        "source_split": "dev",
        "input_order": "evidence as premise; claim as hypothesis",
        "modes": ["float32", "4bit_nf4_double_quant"],
    }


def memory_snapshot(torch: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def benchmark_mode(
    model_id: str,
    rows: list[dict[str, Any]],
    four_bit: bool,
    repetitions: int,
    warmup_runs: int,
) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_load = memory_snapshot(torch)
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if four_bit
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16 if four_bit else torch.float32,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    after_load = memory_snapshot(torch)
    id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
    if set(id2label.values()) != set(NLI_TO_TRUTHLENS):
        raise RuntimeError(f"Unsafe NLI label mapping: {id2label}")

    def infer(row: dict[str, Any]) -> tuple[str, list[float], float]:
        inputs = tokenizer(
            row["evidence"],
            row["claim"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)
        started_inference = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started_inference) * 1000
        probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        index = int(logits.argmax(dim=-1).item())
        return NLI_TO_TRUTHLENS[id2label[index]], probabilities, latency_ms

    for index in range(warmup_runs):
        infer(rows[index % len(rows)])
    torch.cuda.reset_peak_memory_stats()
    latency_samples: list[float] = []
    prediction_runs: list[list[str]] = []
    first_probabilities: list[list[float]] = []
    started_all = time.perf_counter()
    for repetition in range(repetitions):
        predictions: list[str] = []
        probabilities: list[list[float]] = []
        for row in rows:
            prediction, scores, latency_ms = infer(row)
            predictions.append(prediction)
            probabilities.append(scores)
            latency_samples.append(latency_ms)
        prediction_runs.append(predictions)
        if repetition == 0:
            first_probabilities = probabilities
    runtime_seconds = time.perf_counter() - started_all
    predictions = prediction_runs[0]
    repeat_consistency = sum(
        prediction == prediction_runs[0][index]
        for run in prediction_runs
        for index, prediction in enumerate(run)
    ) / (len(rows) * repetitions)
    after_inference = memory_snapshot(torch)
    record = {
        "mode": "4bit_nf4_double_quant" if four_bit else "float32",
        "model": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "load_seconds": load_seconds,
        "runtime_seconds": runtime_seconds,
        "repetitions": repetitions,
        "warmup_runs": warmup_runs,
        "latency": common.latency_summary(latency_samples),
        "memory": {
            "before_load": before_load,
            "after_load": after_load,
            "after_inference": after_inference,
            "model_allocated_delta_bytes": (
                after_load["allocated_bytes"] - before_load["allocated_bytes"]
            ),
        },
        "quality": common.classification_metrics(
            [row["label"] for row in rows], predictions
        ),
        "output_validity": {
            "invalid_output_rate": 0.0,
            "invalid_outputs": 0,
            "note": "Sequence-classification logits always map through the verified id2label table.",
        },
        "repeat_label_consistency": repeat_consistency,
        "predictions": predictions,
        "probabilities": first_probabilities,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/datasets/processed/experiment2/test.jsonl"),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/benchmarks/quantization_results.json"),
    )
    args = parser.parse_args()
    validation = validate_data(args.data, args.model)
    if args.repetitions < 2:
        raise ValueError("At least two repetitions are required for output-consistency measurement")
    if args.warmup_runs < 1:
        raise ValueError("At least one warm-up run is required")
    if args.dry_run:
        validation.update(
            {
                "repetitions": args.repetitions,
                "timed_inferences_per_mode": 180 * args.repetitions,
                "warmup_runs": args.warmup_runs,
                "output": str(args.output),
            }
        )
        print(json.dumps(validation, indent=2))
        return

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for measured VRAM and NF4 inference")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark configuration requires BF16-capable CUDA hardware")
    rows = common.read_jsonl(args.data)
    higher = benchmark_mode(args.model, rows, False, args.repetitions, args.warmup_runs)
    quantized = benchmark_mode(args.model, rows, True, args.repetitions, args.warmup_runs)
    agreement = sum(
        first == second
        for first, second in zip(
            higher["predictions"], quantized["predictions"], strict=True
        )
    ) / len(rows)
    probability_delta = np.abs(
        np.asarray(higher["probabilities"]) - np.asarray(quantized["probabilities"])
    )
    higher_allocated = higher["memory"]["model_allocated_delta_bytes"]
    quantized_allocated = quantized["memory"]["model_allocated_delta_bytes"]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": common.utc_now(),
        "environment": common.environment(),
        "configuration": validation
        | {
            "repetitions": args.repetitions,
            "warmup_runs": args.warmup_runs,
            "timed_inferences_per_mode": len(rows) * args.repetitions,
            "higher_precision": "FP32 model weights and computation",
            "quantized": "NF4 4-bit weights, double quantization, BF16 computation",
        },
        "higher_precision": higher,
        "quantized": quantized,
        "comparison": {
            "model_allocated_vram_savings_bytes": higher_allocated - quantized_allocated,
            "model_allocated_vram_reduction_fraction": (
                1 - quantized_allocated / higher_allocated if higher_allocated > 0 else None
            ),
            "peak_inference_vram_savings_bytes": (
                higher["memory"]["after_inference"]["peak_allocated_bytes"]
                - quantized["memory"]["after_inference"]["peak_allocated_bytes"]
            ),
            "load_time_delta_seconds": quantized["load_seconds"] - higher["load_seconds"],
            "mean_latency_delta_ms": (
                quantized["latency"]["mean_ms"] - higher["latency"]["mean_ms"]
            ),
            "p50_latency_delta_ms": (
                quantized["latency"]["p50_ms"] - higher["latency"]["p50_ms"]
            ),
            "p95_latency_delta_ms": (
                quantized["latency"]["p95_ms"] - higher["latency"]["p95_ms"]
            ),
            "accuracy_delta": (
                quantized["quality"]["accuracy"] - higher["quality"]["accuracy"]
            ),
            "macro_f1_delta": (
                quantized["quality"]["macro_f1"] - higher["quality"]["macro_f1"]
            ),
            "label_output_consistency": agreement,
            "mean_absolute_probability_delta": float(probability_delta.mean()),
            "max_absolute_probability_delta": float(probability_delta.max()),
        },
    }
    common.write_json(args.output, payload)
    console_summary = {
        "status": payload["status"],
        "output": str(args.output),
        "samples": len(rows),
        "timed_inferences_per_mode": len(rows) * args.repetitions,
        "higher_precision": {
            "load_seconds": higher["load_seconds"],
            "model_allocated_delta_bytes": higher_allocated,
            "latency": higher["latency"],
            "accuracy": higher["quality"]["accuracy"],
            "macro_f1": higher["quality"]["macro_f1"],
        },
        "quantized": {
            "load_seconds": quantized["load_seconds"],
            "model_allocated_delta_bytes": quantized_allocated,
            "latency": quantized["latency"],
            "accuracy": quantized["quality"]["accuracy"],
            "macro_f1": quantized["quality"]["macro_f1"],
        },
        "comparison": payload["comparison"],
    }
    print(json.dumps(console_summary, indent=2))


if __name__ == "__main__":
    main()
