from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classification_metrics, environment, read_jsonl, utc_now, write_json  # noqa: E402

NLI_TO_TRUTHLENS = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT_EVIDENCE",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--data", type=Path, default=Path("ml/datasets/processed/test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("ml/benchmarks/baseline_verifier.json")
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder

    rows = read_jsonl(args.data)
    started = time.perf_counter()
    model = CrossEncoder(args.model)
    load_seconds = time.perf_counter() - started
    id2label = {int(key): value.lower() for key, value in model.model.config.id2label.items()}
    if set(id2label.values()) != set(NLI_TO_TRUTHLENS):
        raise RuntimeError(f"Unsafe NLI label mapping: {id2label}")

    pairs = [(row["evidence"], row["claim"]) for row in rows]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    raw = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    predicted = [NLI_TO_TRUTHLENS[id2label[index]] for index in raw.argmax(axis=1).tolist()]
    gold = [row["label"] for row in rows]
    metrics = classification_metrics(gold, predicted)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": utc_now(),
        "environment": environment(),
        "model": args.model,
        "model_revision": getattr(model.model.config, "_commit_hash", None),
        "dataset": {
            "id": "pietrolesci/nli_fever",
            "split": "held-out subset of dev",
            "path": str(args.data),
            "sampling": "deterministic balanced sample",
        },
        "label_mapping": {
            "entailment": "SUPPORTED",
            "contradiction": "CONTRADICTED",
            "neutral": "INSUFFICIENT_EVIDENCE",
            "input_order": "evidence as premise; claim as hypothesis",
        },
        "load_seconds": load_seconds,
        "runtime_seconds": inference_seconds,
        "throughput_samples_per_second": len(rows) / inference_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "metrics": metrics,
        "predictions": [
            {
                "source_id": row.get("source_id"),
                "gold": row["label"],
                "prediction": prediction,
            }
            for row, prediction in zip(rows, predicted, strict=True)
        ],
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
