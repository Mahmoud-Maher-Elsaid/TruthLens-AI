"""Prepare deterministic, balanced FEVER-NLI splits for verifier experiments."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

LABELS = {"SUPPORTS": "SUPPORTED", "REFUTES": "CONTRADICTED", "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE"}
SMOKE = [
    {"claim": "The Eiffel Tower is in Paris.", "evidence": "The Eiffel Tower is located in Paris, France.", "label": "SUPPORTED"},
    {"claim": "The Eiffel Tower was completed in 1920.", "evidence": "The tower was completed in 1889.", "label": "CONTRADICTED"},
    {"claim": "The tower was painted blue in 1901.", "evidence": "The source only describes its construction and location.", "label": "INSUFFICIENT_EVIDENCE"},
    {"claim": "Water freezes at 0°C at standard pressure.", "evidence": "At standard atmospheric pressure, pure water freezes at 0°C.", "label": "SUPPORTED"},
    {"claim": "Tokyo is the capital of France.", "evidence": "Paris is the capital of France.", "label": "CONTRADICTED"},
    {"claim": "Ada Lovelace visited the Moon.", "evidence": "Ada Lovelace was a nineteenth-century mathematician.", "label": "INSUFFICIENT_EVIDENCE"}
]


def normalize(row: dict[str, Any]) -> dict[str, str] | None:
    claim = str(row.get("claim") or row.get("sentence1") or "").strip()
    evidence = str(row.get("evidence") or row.get("sentence2") or row.get("premise") or "").strip()
    raw = str(row.get("label") or row.get("gold_label") or "").upper()
    label = LABELS.get(raw, raw)
    if claim and evidence and label in set(LABELS.values()):
        return {"claim": claim, "evidence": evidence, "label": label}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="fever/fever")
    parser.add_argument("--config", default="v1.0")
    parser.add_argument("--output", type=Path, default=Path("ml/datasets/processed/verifier.jsonl"))
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = list(SMOKE)
    if not args.smoke:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset_id, args.config, split="train")
        rows = [item for raw in dataset for item in [normalize(dict(raw))] if item][: args.limit]
        if not rows:
            raise RuntimeError("Dataset schema produced no claim/evidence rows; select a compatible NLI projection.")
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output), "seed": args.seed}))


if __name__ == "__main__": main()
