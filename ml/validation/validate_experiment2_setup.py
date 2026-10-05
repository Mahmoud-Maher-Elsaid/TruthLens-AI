"""Quick, model-free validation of the Experiment 2 dataset and prompt setup."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "training"))
experiment2_format = importlib.import_module("experiment2_format")
LABELS = experiment2_format.LABELS
messages = experiment2_format.messages


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def input_ids(value: Any) -> list[int]:
    """Normalize Transformers 4.x list and 5.x BatchEncoding return shapes."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ml" / "training" / "experiment2_config.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "ml" / "datasets" / "processed" / "experiment2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional local tokenizer path; defaults to the configured base model.",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    splits = {
        name: load_jsonl(args.data_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }

    expected_output = REPO_ROOT / config["output_dir"]
    experiment1_output = (
        REPO_ROOT / "ml" / "training" / "outputs" / "truthlens-qwen2.5-1.5b-lora"
    )
    if expected_output.resolve() == experiment1_output.resolve():
        raise ValueError("Experiment 2 output path aliases the Experiment 1 output path")

    source_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[str]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    for split_name, rows in splits.items():
        counts = Counter(row["label"] for row in rows)
        if set(counts) != set(LABELS):
            raise ValueError(f"{split_name} has invalid label set: {sorted(counts)}")
        source_sets[split_name] = {str(row["source_id"]) for row in rows}
        group_sets[split_name] = {str(row["claim_group_id"]) for row in rows}
        label_counts[split_name] = dict(sorted(counts.items()))

    overlap: dict[str, dict[str, int]] = {}
    split_names = list(splits)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            overlap[key] = {
                "source_ids": len(source_sets[left] & source_sets[right]),
                "group_ids": len(group_sets[left] & group_sets[right]),
            }
            if overlap[key]["source_ids"] or overlap[key]["group_ids"]:
                raise ValueError(f"Split leakage detected: {key}={overlap[key]}")

    tokenizer_source = args.tokenizer or config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
        use_fast=True,
    )
    token_lengths: dict[str, dict[str, int]] = {}
    truncation_counts: dict[str, int] = {}
    max_length = int(config["max_length"])
    for split_name, rows in splits.items():
        lengths: list[int] = []
        for row in rows:
            token_ids = input_ids(
                tokenizer.apply_chat_template(
                    messages(row),
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            target_ids = tokenizer(row["label"], add_special_tokens=False)["input_ids"]
            lengths.append(len(token_ids) + len(target_ids) + 1)
        token_lengths[split_name] = {
            "min": min(lengths),
            "median": int(median(lengths)),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        }
        truncation_counts[split_name] = sum(length > max_length for length in lengths)

    target_tokens = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"] for label in LABELS
    }
    result = {
        "status": "passed",
        "base_model": config["base_model"],
        "tokenizer_source": str(tokenizer_source),
        "max_length": max_length,
        "output_dir": str(expected_output.relative_to(REPO_ROOT)),
        "experiment1_output_preserved": str(experiment1_output.relative_to(REPO_ROOT)),
        "label_counts": label_counts,
        "split_overlap": overlap,
        "token_lengths": token_lengths,
        "examples_over_max_length": truncation_counts,
        "target_token_ids": target_tokens,
        "deterministic_generation": config["generation"],
    }
    output_path = args.output or (args.data_dir / "setup_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
