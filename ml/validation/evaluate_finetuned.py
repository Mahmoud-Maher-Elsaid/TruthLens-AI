from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    classification_metrics,
    environment,
    latency_summary,
    parse_generated_label,
    read_jsonl,
    utc_now,
    write_json,
)


def prompt(row: dict[str, str]) -> str:
    return (
        "Classify the claim using only the evidence. Return exactly one label: "
        "SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\n"
        f"Claim: {row['claim']}\nEvidence: {row['evidence']}\nLabel:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", type=Path, default=Path("ml/training/outputs/truthlens-qwen2.5-1.5b-lora")
    )
    parser.add_argument("--data", type=Path, default=Path("ml/datasets/processed/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("ml/benchmarks/finetuned_verifier.json"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("ml/benchmarks/verifier_comparison.json")
    )
    parser.add_argument(
        "--baseline", type=Path, default=Path("ml/benchmarks/baseline_verifier.json")
    )
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_jsonl(args.data)
    config = PeftConfig.from_pretrained(args.adapter)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    predicted, raw_outputs, latencies = [], [], []
    invalid_outputs = 0
    for row in rows:
        inputs = tokenizer(prompt(row), return_tensors="pt", truncation=True, max_length=768).to(
            model.device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        parsed = parse_generated_label(generated)
        if parsed is None:
            invalid_outputs += 1
            parsed = "INSUFFICIENT_EVIDENCE"  # matches the deployed provider's safe fallback
        predicted.append(parsed)
        raw_outputs.append(generated)
    gold = [row["label"] for row in rows]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": utc_now(),
        "environment": environment(),
        "model": {
            "base": config.base_model_name_or_path,
            "adapter": str(args.adapter),
            "inference_quantization": "4-bit NF4 double quantization",
        },
        "dataset": {
            "id": "pietrolesci/nli_fever",
            "split": "held-out subset of dev",
            "path": str(args.data),
        },
        "load_seconds": load_seconds,
        "latency": latency_summary(latencies),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "output_validity": {
            "valid_exact_label_rate": (len(rows) - invalid_outputs) / len(rows),
            "invalid_outputs": invalid_outputs,
            "invalid_fallback": "INSUFFICIENT_EVIDENCE (same as deployed provider)",
        },
        "metrics": classification_metrics(gold, predicted),
        "predictions": [
            {
                "source_id": row.get("source_id"),
                "gold": row["label"],
                "prediction": prediction,
                "raw_output": raw,
            }
            for row, prediction, raw in zip(rows, predicted, raw_outputs, strict=True)
        ],
    }
    write_json(args.output, payload)
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": utc_now(),
        "dataset": payload["dataset"],
        "same_test_samples": True,
        "finetuned": payload["metrics"],
    }
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        comparison["baseline"] = baseline["metrics"]
        comparison["delta"] = {
            "accuracy": payload["metrics"]["accuracy"] - baseline["metrics"]["accuracy"],
            "macro_f1": payload["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
        }
        comparison["conclusion"] = (
            "The fine-tuned verifier improved both measured accuracy and macro F1."
            if comparison["delta"]["accuracy"] > 0 and comparison["delta"]["macro_f1"] > 0
            else "No across-metric improvement is claimed from this run."
        )
    else:
        comparison["status"] = "partial"
        comparison["baseline"] = None
        comparison["conclusion"] = "Baseline artifact unavailable; no comparison is claimed."
    write_json(args.comparison, comparison)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
