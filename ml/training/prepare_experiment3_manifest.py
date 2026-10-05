"""Audit local Experiment 3 splits without downloading data or model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    from .experiment3_format import evidence_passages, validate_row
except ImportError:  # Allows direct execution: python ml/training/prepare_experiment3_manifest.py
    from experiment3_format import evidence_passages, validate_row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(row: dict[str, Any]) -> set[str]:
    text = f"{row.get('claim', '')} {' '.join(evidence_passages(row))}"
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def near_duplicate_pairs(rows: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Return high-overlap pairs within a claim group for pre-training review."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["claim_group_id"]), []).append(row)
    flagged: list[dict[str, Any]] = []
    for group_rows in by_group.values():
        for left, right in combinations(group_rows, 2):
            left_tokens, right_tokens = tokens(left), tokens(right)
            union = left_tokens | right_tokens
            score = len(left_tokens & right_tokens) / len(union) if union else 0.0
            if score >= threshold:
                flagged.append(
                    {
                        "claim_group_id": str(left["claim_group_id"]),
                        "left_source_id": str(left["source_id"]),
                        "right_source_id": str(right["source_id"]),
                        "left_label": left["label"],
                        "right_label": right["label"],
                        "token_jaccard": round(score, 6),
                    }
                )
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("ml/datasets/processed/experiment2/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("ml/datasets/processed/experiment2/validation.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("ml/datasets/processed/experiment2/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("ml/training/experiment3_manifest.json"))
    args = parser.parse_args()
    splits = {name: read_jsonl(path) for name, path in (("train", args.train), ("validation", args.validation), ("test", args.test))}
    for rows in splits.values():
        for row in rows:
            validate_row(row)
    groups = {name: {str(row["claim_group_id"]) for row in rows} for name, rows in splits.items()}
    source_ids = {name: {str(row["source_id"]) for row in rows} for name, rows in splits.items()}
    leakage = {
        f"{left}_{right}_group_overlap": len(groups[left] & groups[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    } | {
        f"{left}_{right}_source_overlap": len(source_ids[left] & source_ids[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(leakage.values()):
        raise RuntimeError(f"Experiment 3 leakage detected: {leakage}")
    exact_duplicates = {
        name: len(rows) - len({(row["claim"].lower(), tuple(evidence_passages(row)), row["label"]) for row in rows})
        for name, rows in splits.items()
    }
    manifest = {
        "experiment": "experiment3_multi_evidence_contradiction_boundary",
        "status": "prepared",
        "splits": {
            name: {"rows": len(rows), "class_counts": dict(Counter(row["label"] for row in rows)), "sha256": sha256(path)}
            for name, rows, path in (("train", splits["train"], args.train), ("validation", splits["validation"], args.validation), ("test", splits["test"], args.test))
        },
        "leakage_checks": leakage,
        "exact_duplicate_counts": exact_duplicates,
        "near_duplicate_pairs": {name: near_duplicate_pairs(rows) for name, rows in splits.items()},
        "near_duplicate_threshold": "token Jaccard >= 0.85 within claim groups",
        "test_policy": "Test rows are audit-only and must not influence training, checkpoint selection, thresholds, or hyperparameters.",
        "label_counts_are": "measured by this preparation run, not assumed from configuration",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
