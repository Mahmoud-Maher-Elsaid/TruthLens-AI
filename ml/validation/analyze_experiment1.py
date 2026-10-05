"""Dependency-free diagnosis of Experiment 1 data and held-out errors."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "source_id_duplicates": len(rows) - len({row["source_id"] for row in rows}),
        "claim_duplicates": len(rows) - len({row["claim"] for row in rows}),
        "claim_chars": {
            "min": min(len(row["claim"]) for row in rows),
            "mean": statistics.fmean(len(row["claim"]) for row in rows),
            "max": max(len(row["claim"]) for row in rows),
        },
        "evidence_chars": {
            "min": min(len(row["evidence"]) for row in rows),
            "mean": statistics.fmean(len(row["evidence"]) for row in rows),
            "max": max(len(row["evidence"]) for row in rows),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    splits = {
        name: jsonl(root / "ml" / "datasets" / "processed" / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    fine = json.loads(
        (root / "ml" / "benchmarks" / "finetuned_verifier.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (root / "ml" / "benchmarks" / "baseline_verifier.json").read_text(encoding="utf-8")
    )
    fine_predictions = {row["source_id"]: row["prediction"] for row in fine["predictions"]}
    baseline_predictions = {
        row["source_id"]: row["prediction"] for row in baseline["predictions"]
    }
    test_by_id = {row["source_id"]: row for row in splits["test"]}
    ie_as_contradicted = [
        {
            "source_id": source_id,
            "claim": test_by_id[source_id]["claim"],
            "evidence": test_by_id[source_id]["evidence"],
            "baseline_prediction": baseline_predictions[source_id],
        }
        for source_id, prediction in fine_predictions.items()
        if test_by_id[source_id]["label"] == "INSUFFICIENT_EVIDENCE"
        and prediction == "CONTRADICTED"
    ]
    supported_errors = [
        {
            "source_id": source_id,
            "claim": test_by_id[source_id]["claim"],
            "evidence": test_by_id[source_id]["evidence"],
            "finetuned_prediction": prediction,
            "baseline_prediction": baseline_predictions[source_id],
        }
        for source_id, prediction in fine_predictions.items()
        if test_by_id[source_id]["label"] == "SUPPORTED" and prediction != "SUPPORTED"
    ]
    report = {
        "splits": {name: split_stats(rows) for name, rows in splits.items()},
        "cross_split_source_overlap": {
            "train_validation": len(
                {row["source_id"] for row in splits["train"]}
                & {row["source_id"] for row in splits["validation"]}
            ),
            "train_test": len(
                {row["source_id"] for row in splits["train"]}
                & {row["source_id"] for row in splits["test"]}
            ),
            "validation_test": len(
                {row["source_id"] for row in splits["validation"]}
                & {row["source_id"] for row in splits["test"]}
            ),
        },
        "ie_predicted_contradicted": ie_as_contradicted,
        "supported_errors": supported_errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
