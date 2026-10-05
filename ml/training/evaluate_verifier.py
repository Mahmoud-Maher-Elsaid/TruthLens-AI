from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


def metrics(gold: list[str], predicted: list[str]) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(gold, predicted, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(gold, predicted), "macro_precision": precision, "macro_recall": recall, "macro_f1": f1, "labels": LABELS, "confusion_matrix": confusion_matrix(gold, predicted, labels=LABELS).tolist(), "samples": len(gold)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL rows containing label and prediction")
    parser.add_argument("--output", type=Path, default=Path("ml/benchmarks/verifier.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    result = metrics([row["label"] for row in rows], [row["prediction"] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
