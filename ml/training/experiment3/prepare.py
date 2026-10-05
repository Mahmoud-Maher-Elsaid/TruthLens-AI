"""Build deterministic source-audited multi-evidence data for Experiment 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def claim_group(row: dict[str, Any]) -> str:
    return str(row.get("claim_group_id") or str(row["source_id"]).split(":", 1)[0])


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256(f"{seed}:{'|'.join(values)}".encode("utf-8")).hexdigest()


def enrich(row: dict[str, Any], source_family: str) -> dict[str, Any]:
    required = {"claim", "evidence", "label", "source_id"}
    if not required.issubset(row) or row["label"] not in LABELS:
        raise ValueError(f"Invalid audited source row: {row.get('source_id', '<unknown>')}")
    return {
        **row,
        "claim_group_id": claim_group(row),
        "source_family": source_family,
    }


def choose_replacement(
    reserve_rows: list[dict[str, Any]], prohibited_sources: set[str], prohibited_groups: set[str], seed: int
) -> dict[str, Any]:
    candidates = [
        row
        for row in reserve_rows
        if row["label"] == "CONTRADICTED"
        and row["source_id"] not in prohibited_sources
        and row["claim_group_id"] not in prohibited_groups
        and len(str(row["evidence"])) >= 40
    ]
    if not candidates:
        raise RuntimeError("No distinct audited natural CONTRADICTED replacement candidate is available.")
    return min(candidates, key=lambda row: stable_key(seed, "replacement", str(row["source_id"])))


def choose_distractors(primary: dict[str, Any], pool: list[dict[str, Any]], count: int, seed: int, maximum_overlap: float) -> list[dict[str, Any]]:
    claim_tokens = token_set(str(primary["claim"]))
    candidates = []
    for row in pool:
        if row["source_id"] == primary["source_id"] or row["claim_group_id"] == primary["claim_group_id"]:
            continue
        overlap = len(claim_tokens & token_set(str(row["evidence"]))) / max(len(claim_tokens), 1)
        if overlap <= maximum_overlap:
            candidates.append(row)
    if len(candidates) < count:
        raise RuntimeError(f"Insufficient unrelated distractors for {primary['source_id']}")
    return sorted(candidates, key=lambda row: stable_key(seed, str(primary["source_id"]), str(row["source_id"])))[:count]


def choose_agreement(primary: dict[str, Any], pool: list[dict[str, Any]], seed: int) -> dict[str, Any] | None:
    candidates = [
        row
        for row in pool
        if row["source_id"] != primary["source_id"]
        and row["claim_group_id"] == primary["claim_group_id"]
        and row["label"] == primary["label"]
    ]
    return min(candidates, key=lambda row: stable_key(seed, "agreement", str(primary["source_id"]), str(row["source_id"]))) if candidates else None


def evidence_entry(row: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "source_id": row["source_id"],
        "claim_group_id": row["claim_group_id"],
        "source_family": row["source_family"],
        "source_label": row["label"],
        "role": role,
        "text": row["evidence"],
    }


def build_examples(primary_rows: list[dict[str, Any]], pool: list[dict[str, Any]], config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    data = config["data"]
    seed = int(config["seed"])
    examples: list[dict[str, Any]] = []
    for primary in sorted(primary_rows, key=lambda row: str(row["source_id"])):
        agreement = choose_agreement(primary, pool, seed)
        distractors = choose_distractors(
            primary,
            pool,
            int(data["evidence_passages"]) - 1 - int(agreement is not None),
            seed,
            float(data["distractor_max_claim_overlap"]),
        )
        evidence = [evidence_entry(primary, "decisive")] + [evidence_entry(row, "distractor") for row in distractors]
        if agreement is not None:
            evidence.append(evidence_entry(agreement, "agreeing"))
        evidence.sort(key=lambda item: str(item["source_id"]))
        tags = ["clean_" + primary["label"].lower(), "one_decisive_passage", "distracting_evidence"]
        if agreement is not None:
            tags.append("multiple_passages_agree")
        lexical_overlap = float(primary.get("lexical_overlap", 0.0))
        if primary["label"] in {"CONTRADICTED", "INSUFFICIENT_EVIDENCE"} and lexical_overlap >= 0.4:
            tags.append("hard_contradiction_neutral_boundary")
        if primary["label"] == "INSUFFICIENT_EVIDENCE" and lexical_overlap >= 0.35:
            tags.append("related_but_not_contradictory")
        examples.append(
            {
                "id": f"{split}:{primary['source_id']}",
                "claim": primary["claim"],
                "label": primary["label"],
                "label_basis": {
                    "source_id": primary["source_id"],
                    "claim_group_id": primary["claim_group_id"],
                    "source_label": primary["label"],
                    "source_family": primary["source_family"],
                },
                "source_id": primary["source_id"],
                "claim_group_id": primary["claim_group_id"],
                "evidence_set": [item["text"] for item in evidence],
                "evidence_metadata": evidence,
                "scenario_tags": tags,
            }
        )
    return examples


def build_conflict_challenge(pool: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_group[row["claim_group_id"]].append(row)
    challenge: list[dict[str, Any]] = []
    for group, rows in by_group.items():
        support = next((row for row in rows if row["label"] == "SUPPORTED"), None)
        contradiction = next((row for row in rows if row["label"] == "CONTRADICTED"), None)
        if support is None or contradiction is None or support["claim"] != contradiction["claim"]:
            continue
        evidence = sorted([evidence_entry(support, "supports"), evidence_entry(contradiction, "contradicts")], key=lambda item: str(item["source_id"]))
        challenge.append(
            {
                "id": f"conflict:{group}",
                "claim": support["claim"],
                "evidence_set": [item["text"] for item in evidence],
                "evidence_metadata": evidence,
                "relationship_labels": [item["source_label"] for item in evidence],
                "usage": "secondary_frozen_challenge_only",
            }
        )
    return sorted(challenge, key=lambda row: stable_key(seed, row["id"]))


def audit_examples(train: list[dict[str, Any]], validation: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {"train": train, "validation": validation, "test": test}
    groups = {name: {row["claim_group_id"] for row in rows} for name, rows in splits.items()}
    sources = {name: {row["source_id"] for row in rows} for name, rows in splits.items()}
    duplicate_counts = {
        name: len(rows) - len({(row["claim"].lower(), tuple(row["evidence_set"]), row["label"]) for row in rows})
        for name, rows in splits.items()
    }
    near_duplicates: dict[str, list[dict[str, Any]]] = {}
    for name, rows in splits.items():
        flagged: list[dict[str, Any]] = []
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_group[row["claim_group_id"]].append(row)
        for group_rows in by_group.values():
            for index, left in enumerate(group_rows):
                for right in group_rows[index + 1 :]:
                    left_tokens = token_set(left["claim"] + " " + " ".join(left["evidence_set"]))
                    right_tokens = token_set(right["claim"] + " " + " ".join(right["evidence_set"]))
                    union = left_tokens | right_tokens
                    score = len(left_tokens & right_tokens) / len(union) if union else 0.0
                    if score >= 0.85:
                        flagged.append({"left": left["source_id"], "right": right["source_id"], "token_jaccard": round(score, 6)})
        near_duplicates[name] = flagged
    leakage = {
        f"{left}_{right}_source_overlap": len(sources[left] & sources[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    } | {
        f"{left}_{right}_claim_group_overlap": len(groups[left] & groups[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    return {
        "splits": {name: {"rows": len(rows), "class_counts": dict(Counter(row["label"] for row in rows))} for name, rows in splits.items()},
        "exact_duplicate_counts": duplicate_counts,
        "near_duplicate_pairs": near_duplicates,
        "leakage_checks": leakage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/training/experiment3/config.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_json(config_path)
    paths = {name: ROOT / value for name, value in config["data"].items() if name in {"primary_train", "primary_validation", "held_out_test", "replacement_pool"}}
    train_source = [enrich(row, "experiment2_train") for row in load_jsonl(paths["primary_train"])]
    validation_source = [enrich(row, "experiment2_validation") for row in load_jsonl(paths["primary_validation"])]
    test_source = [enrich(row, "experiment2_held_out_test") for row in load_jsonl(paths["held_out_test"])]
    reserve_source = [enrich(row, "audited_reserve_train") for row in load_jsonl(paths["replacement_pool"])]
    excluded = config["data"]["excluded_source_id"]
    train_without_duplicate = [row for row in train_source if row["source_id"] != excluded]
    if len(train_source) - len(train_without_duplicate) != 1:
        raise RuntimeError("Configured duplicate exclusion did not remove exactly one train row.")
    prohibited_sources = {row["source_id"] for row in train_source + validation_source + test_source}
    prohibited_groups = {row["claim_group_id"] for row in train_source + validation_source + test_source}
    replacement = choose_replacement(reserve_source, prohibited_sources, prohibited_groups, int(config["seed"]))
    train_primary = train_without_duplicate + [replacement]
    train_examples = build_examples(train_primary, train_primary, config, "train")
    validation_examples = build_examples(validation_source, validation_source, config, "validation")
    test_examples = [
        {
            "id": f"test:{row['source_id']}", "claim": row["claim"], "label": row["label"], "source_id": row["source_id"],
            "claim_group_id": row["claim_group_id"], "evidence_set": [row["evidence"]], "evidence_metadata": [evidence_entry(row, "decisive")],
        }
        for row in test_source
    ]
    audit = audit_examples(train_examples, validation_examples, test_examples)
    if any(audit["exact_duplicate_counts"].values()) or any(audit["leakage_checks"].values()) or any(audit["near_duplicate_pairs"].values()):
        raise RuntimeError(f"Experiment 3 audit failed: {audit}")
    conflict_challenge = build_conflict_challenge(train_primary + validation_source, int(config["seed"]))
    if not conflict_challenge:
        raise RuntimeError("No source-audited conflicting-evidence challenge records were available.")
    manifest = {
        "experiment": config["experiment"], "status": "prepared", "seed": config["seed"],
        "source_hashes": {name: sha256(path) for name, path in paths.items()},
        "replacement": {"excluded_source_id": excluded, "replacement_source_id": replacement["source_id"], "replacement_claim_group_id": replacement["claim_group_id"]},
        "audit": audit,
        "conflict_challenge_records": len(conflict_challenge),
        "held_out_test_policy": "Frozen source rows are formatted with exactly one evidence passage; they are not used during preparation, training, validation, checkpoint selection, or early stopping.",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    output_dir = ROOT / config["data"]["output_dir"]
    write_jsonl(output_dir / "train.jsonl", train_examples)
    write_jsonl(output_dir / "validation.jsonl", validation_examples)
    write_jsonl(output_dir / "conflict_challenge.jsonl", conflict_challenge)
    manifest_path = ROOT / config["data"]["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
