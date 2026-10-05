"""Prepare deterministic, balanced FEVER-NLI splits for verifier experiments."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE",
    "ENTAILMENT": "SUPPORTED",
    "CONTRADICTION": "CONTRADICTED",
    "NEUTRAL": "INSUFFICIENT_EVIDENCE",
}


def normalize(row: dict[str, Any], label_names: list[str] | None = None) -> dict[str, str] | None:
    # pietrolesci/nli_fever names the FEVER claim `premise` and its assembled
    # evidence passage `hypothesis`; preserve their semantic roles explicitly.
    claim = str(row.get("claim") or row.get("premise") or row.get("sentence1") or "").strip()
    evidence = str(
        row.get("evidence") or row.get("hypothesis") or row.get("sentence2") or ""
    ).strip()
    raw = str(row.get("fever_gold_label") or row.get("gold_label") or "").upper()
    if not raw and row.get("label") is not None:
        value = row["label"]
        raw = label_names[value].upper() if label_names and isinstance(value, int) else str(value).upper()
    label = LABEL_MAP.get(raw, raw)
    if not claim or not evidence or label not in LABELS:
        return None
    source_id = f"{row.get('cid', '')}:{row.get('fid', '')}".strip(":")
    if not source_id:
        source_id = str(row.get("id") or f"row-{abs(hash((claim, evidence)))}")
    return {"claim": claim, "evidence": evidence, "label": label, "source_id": source_id}


def balanced_sample(rows: list[dict[str, str]], per_label: int, seed: int) -> list[dict[str, str]]:
    randomizer = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in LABELS:
        candidates = [row for row in rows if row["label"] == label]
        randomizer.shuffle(candidates)
        if len(candidates) < per_label:
            raise RuntimeError(f"Only {len(candidates)} rows are available for {label}; need {per_label}.")
        selected.extend(candidates[:per_label])
    randomizer.shuffle(selected)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="pietrolesci/nli_fever")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/datasets/processed"))
    parser.add_argument("--train-per-label", type=int, default=100)
    parser.add_argument("--validation-per-label", type=int, default=20)
    parser.add_argument("--test-per-label", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    source_train = load_dataset(args.dataset_id, split="train")
    source_dev = load_dataset(args.dataset_id, split="dev")
    train_names = getattr(source_train.features.get("label"), "names", None)
    dev_names = getattr(source_dev.features.get("label"), "names", None)
    train_pool = [item for row in source_train for item in [normalize(dict(row), train_names)] if item]
    dev_pool = [item for row in source_dev for item in [normalize(dict(row), dev_names)] if item]
    training = balanced_sample(train_pool, args.train_per_label, args.seed)
    held_out = balanced_sample(
        dev_pool,
        args.validation_per_label + args.test_per_label,
        args.seed + 1,
    )
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for label in LABELS:
        label_rows = [row for row in held_out if row["label"] == label]
        validation.extend(label_rows[: args.validation_per_label])
        test.extend(label_rows[args.validation_per_label :])
    random.Random(args.seed + 2).shuffle(validation)
    random.Random(args.seed + 3).shuffle(test)

    train_ids = {row["source_id"] for row in training}
    validation_ids = {row["source_id"] for row in validation}
    test_ids = {row["source_id"] for row in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Source row leakage detected between train, validation, and test splits.")

    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
        "test": args.output_dir / "test.jsonl",
    }
    for name, rows in (("train", training), ("validation", validation), ("test", test)):
        write_jsonl(paths[name], rows)
    manifest = {
        "dataset_id": args.dataset_id,
        "dataset_revision": str(source_train.info.version),
        "source_splits": {"train": "train", "validation_and_test": "dev"},
        "seed": args.seed,
        "semantic_mapping": {
            "FEVER SUPPORTS / NLI entailment": "SUPPORTED",
            "FEVER REFUTES / NLI contradiction": "CONTRADICTED",
            "FEVER NOT ENOUGH INFO / NLI neutral": "INSUFFICIENT_EVIDENCE",
        },
        "splits": {
            name: {"rows": len(rows), "labels": dict(Counter(row["label"] for row in rows))}
            for name, rows in (("train", training), ("validation", validation), ("test", test))
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "leakage_check": "passed (source_id sets are disjoint)",
        "licensing_note": (
            "The Hugging Face derivative dataset card does not declare a license. "
            "FEVER source data is distributed under CC BY-SA 3.0; this local experiment "
            "does not redistribute the processed rows."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
