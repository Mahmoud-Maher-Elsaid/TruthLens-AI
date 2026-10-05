from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(rows: list[dict[str, object]], k: int) -> dict[str, float | int]:
    recalls, reciprocal = [], []
    for row in rows:
        relevant = set(row["relevant_ids"]); retrieved = list(row["retrieved_ids"])[:k]
        recalls.append(len(relevant & set(retrieved)) / max(len(relevant), 1))
        ranks = [retrieved.index(item) + 1 for item in relevant if item in retrieved]
        reciprocal.append(1 / min(ranks) if ranks else 0)
    return {f"recall@{k}": sum(recalls) / max(len(recalls), 1), "mrr": sum(reciprocal) / max(len(reciprocal), 1), "queries": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("input", type=Path); parser.add_argument("--k", type=int, default=5); parser.add_argument("--output", type=Path, default=Path("ml/benchmarks/retrieval.json")); args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8")); result = evaluate(rows, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
