"""Evaluate an Experiment 3 adapter on the frozen 180-row held-out protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .format import validate_row
    from .metrics import classification_report, decode_label, deterministic_generation_config, generation_inputs
except ImportError:  # Supports direct PowerShell execution.
    from format import validate_row
    from metrics import classification_report, decode_label, deterministic_generation_config, generation_inputs

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    formatted = [
        {"claim": row["claim"], "label": row["label"], "source_id": row["source_id"], "claim_group_id": row["claim_group_id"], "evidence_set": [row["evidence"]]}
        for row in rows
    ]
    for row in formatted:
        validate_row(row)
    return formatted


def configure_project_cache() -> None:
    service_path = ROOT / "services" / "ai-api"
    if str(service_path) not in sys.path:
        sys.path.insert(0, str(service_path))
    from app.core.model_cache import configure_model_cache

    configure_model_cache()


def resolve_adapter_base_model(adapter: Path) -> str:
    """Read the immutable PEFT metadata without loading a model or tokenizer."""
    from peft import PeftConfig

    metadata = PeftConfig.from_pretrained(str(adapter))
    base_model_name_or_path = metadata.base_model_name_or_path
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path.strip():
        raise RuntimeError("PEFT adapter metadata does not declare base_model_name_or_path.")
    return base_model_name_or_path


def evaluation_sources(adapter: Path) -> dict[str, str]:
    """Resolve source identities before any GPU model load."""
    base_model_name_or_path = resolve_adapter_base_model(adapter)
    return {
        "base_model_name_or_path": base_model_name_or_path,
        "tokenizer_source": base_model_name_or_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/training/experiment3/config.json"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    manifest = load_json(ROOT / config["data"]["manifest"])
    test_path = ROOT / config["data"]["held_out_test"]
    if sha256(test_path) != manifest["source_hashes"]["held_out_test"]:
        raise RuntimeError("Frozen held-out test hash does not match the Experiment 3 manifest.")
    rows = frozen_rows(test_path)
    if len(rows) != 180:
        raise RuntimeError("Frozen held-out protocol must contain exactly 180 rows.")

    adapter = args.adapter.resolve()
    if not adapter.is_dir():
        raise RuntimeError(f"Adapter directory does not exist: {adapter}")
    sources = evaluation_sources(adapter)

    configure_project_cache()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Experiment 3 held-out evaluation requires a BF16-capable CUDA device.")
    tokenizer = AutoTokenizer.from_pretrained(sources["tokenizer_source"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats()
    base_model = AutoModelForCausalLM.from_pretrained(
        sources["base_model_name_or_path"], dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    base_model.to("cuda")
    model = PeftModel.from_pretrained(base_model, str(adapter))
    model.eval()
    predictions: list[str] = []
    timings: list[float] = []
    with torch.inference_mode():
        for row in rows:
            encoded = generation_inputs(tokenizer, row, config["training"]["max_length"], "cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(
                **encoded,
                generation_config=deterministic_generation_config(model, tokenizer),
            )
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
            predictions.append(decode_label(tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)))
    report = classification_report([row["label"] for row in rows], predictions)
    adapter_file = adapter / "adapter_model.safetensors"
    result = {
        "experiment": config["experiment"],
        "protocol": config["promotion"]["evaluation_protocol"],
        "held_out_test": {"path": str(test_path), "sha256": sha256(test_path), "rows": len(rows)},
        "checkpoint_identity": {
            "adapter_path": str(adapter),
            "adapter_sha256": sha256(adapter_file) if adapter_file.is_file() else None,
            **sources,
        },
        "metrics": report,
        "latency_ms": {"mean": statistics.mean(timings), "p50": statistics.median(timings), "p95": sorted(timings)[max(0, int(len(timings) * 0.95) - 1)]},
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "predictions": [{"source_id": row["source_id"], "expected": row["label"], "predicted": predicted} for row, predicted in zip(rows, predictions, strict=True)],
    }
    output = args.output or adapter / "held_out_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
