"""Prepare cleaned pair-level FEVER-NLI data for QLoRA Experiment 2."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
PAIR_LABEL_MAP = {
    "entailment": "SUPPORTED",
    "contradiction": "CONTRADICTED",
    "neutral": "INSUFFICIENT_EVIDENCE",
}
FEVER_LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE",
}
NEGATIONS = {"not", "no", "never", "neither", "without", "zero"}
CATEGORICAL_GROUPS = (
    {
        "american", "british", "canadian", "chinese", "english", "french", "german",
        "indian", "irish", "italian", "japanese", "mexican", "russian", "spanish",
    },
    {
        "actor", "actress", "athlete", "author", "director", "doctor", "engineer",
        "lawyer", "musician", "nurse", "painter", "politician", "singer", "teacher",
        "writer",
    },
)
ANTONYM_PAIRS = (
    ("canceled", "released"),
    ("cancelled", "released"),
    ("alive", "dead"),
    ("married", "unmarried"),
    ("professional", "amateur"),
    ("fictional", "real"),
)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def overlap(row: dict[str, Any]) -> float:
    claim, evidence = tokens(row["claim"]), tokens(row["evidence"])
    return len(claim & evidence) / max(len(claim), 1)


def semantic_risk_flags(claim: str, evidence: str) -> list[str]:
    claim_tokens, evidence_tokens = tokens(claim), tokens(evidence)
    flags: list[str] = []
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence))
    if claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers):
        flags.append("numeric_mismatch")
    if bool(claim_tokens & NEGATIONS) != bool(evidence_tokens & NEGATIONS):
        flags.append("negation_mismatch")
    for category in CATEGORICAL_GROUPS:
        claim_values = claim_tokens & category
        evidence_values = evidence_tokens & category
        if claim_values and evidence_values and claim_values.isdisjoint(evidence_values):
            flags.append("categorical_mismatch")
            break
    for left, right in ANTONYM_PAIRS:
        if (left in claim_tokens and right in evidence_tokens) or (
            right in claim_tokens and left in evidence_tokens
        ):
            flags.append("antonym_conflict")
            break
    return flags


def normalize_pair(row: dict[str, Any], label_names: list[str], source_split: str) -> dict[str, Any] | None:
    claim = clean_text(row.get("premise"))
    evidence = clean_text(row.get("hypothesis"))
    value = row.get("label")
    if not isinstance(value, int) or value < 0 or value >= len(label_names):
        return None
    pair_name = label_names[value].lower()
    label = PAIR_LABEL_MAP.get(pair_name)
    if not label or len(claim) < 10 or len(evidence) < 40:
        return None
    cid = clean_text(row.get("cid"))
    fid = clean_text(row.get("fid"))
    source_id = f"{cid}:{fid}" if cid or fid else hashlib.sha256(
        f"{claim}\0{evidence}".encode()
    ).hexdigest()[:20]
    group_id = cid or hashlib.sha256(claim.lower().encode()).hexdigest()[:20]
    fever_raw = clean_text(row.get("fever_gold_label")).upper()
    fever_label = FEVER_LABEL_MAP.get(fever_raw)
    risk_flags = semantic_risk_flags(claim, evidence)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": label,
        "source_id": source_id,
        "claim_group_id": group_id,
        "source_split": source_split,
        "pair_nli_label": pair_name,
        "fever_claim_label": fever_label,
        "pair_fever_disagree": fever_label is not None and fever_label != label,
        "lexical_overlap": round(overlap({"claim": claim, "evidence": evidence}), 6),
        "semantic_risk_flags": risk_flags,
    }


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row["claim"].lower(), row["evidence"].lower(), row["label"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def controlled_contrast_triples(count: int = 50) -> list[dict[str, Any]]:
    subjects = [
        "Aster Archive", "Boreal Institute", "Cinder Observatory", "Dawn Laboratory",
        "Elm Research Center", "Fjord Museum", "Garnet Library", "Harbor Academy",
        "Iris Workshop", "Juniper Station",
    ]
    cities = ["Lunaris", "Norvale", "Pinecross", "Redhaven", "Stonebridge"]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        subject = f"{subjects[index % len(subjects)]} {index + 1}"
        year = 1950 + index
        other_year = year + 17
        city = cities[index % len(cities)]
        other_city = cities[(index + 2) % len(cities)]
        pattern = index % 5
        if pattern == 0:
            claim = f"{subject} opened in {year}."
            support = f"The {subject} officially opened in {year} and is located in {city}."
            conflict = f"The {subject} officially opened in {other_year}, not in {year}."
            neutral = f"The {subject} is located in {city} and contains a public reading room."
        elif pattern == 1:
            claim = f"{subject} is located in {city}."
            support = f"Visitors can find the {subject} in {city}; it opened in {year}."
            conflict = f"The {subject} is located in {other_city}, not in {city}."
            neutral = f"The {subject} opened in {year} and hosts an annual science exhibition."
        elif pattern == 2:
            staff = 40 + index
            claim = f"{subject} employs {staff} people."
            support = f"The {subject} has a staff of exactly {staff} people and operates in {city}."
            conflict = f"The {subject} employs exactly {staff + 25} people, not {staff}."
            neutral = f"The {subject} operates in {city} and maintains three public galleries."
        elif pattern == 3:
            claim = f"{subject} has a blue roof."
            support = f"Architectural records describe the roof of the {subject} as blue."
            conflict = f"The roof of the {subject} is red rather than blue."
            neutral = f"The {subject} was renovated in {year} and includes a large auditorium."
        else:
            awards = 2 + index % 7
            claim = f"{subject} received {awards} regional awards."
            support = f"Records confirm that the {subject} received {awards} regional awards."
            conflict = f"The {subject} received {awards + 3} regional awards, not {awards}."
            neutral = f"The {subject} runs educational programs for residents of {city}."
        group_id = f"controlled-contrast-{index:03d}"
        for label, evidence in (
            ("SUPPORTED", support),
            ("CONTRADICTED", conflict),
            ("INSUFFICIENT_EVIDENCE", neutral),
        ):
            rows.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "label": label,
                    "source_id": f"{group_id}:{label}",
                    "claim_group_id": group_id,
                    "source_split": "controlled_contrast",
                    "pair_nli_label": None,
                    "fever_claim_label": None,
                    "pair_fever_disagree": False,
                    "lexical_overlap": round(overlap({"claim": claim, "evidence": evidence}), 6),
                    "semantic_risk_flags": [],
                }
            )
    return rows


def clean_neutral(row: dict[str, Any]) -> bool:
    return (
        row["label"] == "INSUFFICIENT_EVIDENCE"
        and 0.12 <= row["lexical_overlap"] <= 0.7
        and not row["semantic_risk_flags"]
    )


def contrast_aware_training_sample(
    rows: list[dict[str, Any]], per_label: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    randomizer = random.Random(seed)
    controlled = controlled_contrast_triples()
    controlled_per_label = len(controlled) // len(LABELS)
    natural_per_label = per_label - controlled_per_label
    selected: list[dict[str, Any]] = []
    composition: dict[str, Any] = {}
    for label in LABELS:
        candidates = [row for row in rows if row["label"] == label]
        rejected_risky = 0
        if label == "INSUFFICIENT_EVIDENCE":
            rejected_risky = len(candidates) - sum(clean_neutral(row) for row in candidates)
            candidates = [row for row in candidates if clean_neutral(row)]
        randomizer.shuffle(candidates)
        hard_target = round(natural_per_label * 0.4)
        hard = sorted(candidates, key=lambda row: row["lexical_overlap"], reverse=True)[:hard_target]
        used = {row["source_id"] for row in hard}
        remaining = [row for row in candidates if row["source_id"] not in used]
        used.update(row["source_id"] for row in hard)
        randomizer.shuffle(remaining)
        random_rows = remaining[: natural_per_label - len(hard)]
        controlled_rows = [row for row in controlled if row["label"] == label]
        label_rows = controlled_rows + hard + random_rows
        if len(label_rows) != per_label:
            raise RuntimeError(f"Could not sample {per_label} rows for {label}; got {len(label_rows)}")
        selected.extend(label_rows)
        composition[label] = {
            "controlled_contrast_rows": len(controlled_rows),
            "natural_high_overlap_rows": len(hard),
            "natural_random_rows": len(random_rows),
            "risky_neutral_candidates_rejected": rejected_risky,
        }
    randomizer.shuffle(selected)
    return selected, composition


def stable_partition(group_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return digest[0] % 2


def balanced_group_sample(
    rows: list[dict[str, Any]], per_label: int, seed: int
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for label in LABELS:
        candidates = [row for row in rows if row["label"] == label]
        randomizer.shuffle(candidates)
        label_rows: list[dict[str, Any]] = []
        for row in candidates:
            if row["claim_group_id"] in used_groups:
                continue
            used_groups.add(row["claim_group_id"])
            label_rows.append(row)
            if len(label_rows) == per_label:
                break
        if len(label_rows) != per_label:
            raise RuntimeError(f"Could not group-sample {per_label} rows for {label}")
        selected.extend(label_rows)
    randomizer.shuffle(selected)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="pietrolesci/nli_fever")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ml/datasets/processed/experiment2")
    )
    parser.add_argument("--experiment1-test", type=Path, default=Path("ml/datasets/processed/test.jsonl"))
    parser.add_argument("--train-per-label", type=int, default=400)
    parser.add_argument("--validation-per-label", type=int, default=60)
    parser.add_argument("--test-per-label", type=int, default=60)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    from datasets import load_dataset

    raw_train = load_dataset(args.dataset_id, split="train")
    raw_dev = load_dataset(args.dataset_id, split="dev")
    train_names = raw_train.features["label"].names
    dev_names = raw_dev.features["label"].names
    train_pool = deduplicate(
        [
            normalized
            for row in raw_train
            if (normalized := normalize_pair(dict(row), train_names, "train"))
        ]
    )
    normalized_dev = [
        normalized
        for row in raw_dev
        if (normalized := normalize_pair(dict(row), dev_names, "dev"))
    ]
    dev_pool = deduplicate(normalized_dev)
    experiment1_test = [
        json.loads(line)
        for line in args.experiment1_test.read_text(encoding="utf-8").splitlines()
        if line
    ]
    legacy_ids = {row["source_id"] for row in experiment1_test}
    # Recover the exact Experiment 1 source IDs before content deduplication. Two
    # held-out rows are duplicate claim/evidence/label triples with distinct FEVER
    # row IDs, so looking only in the deduplicated pool would incorrectly lose them.
    by_source = {row["source_id"]: row for row in normalized_dev}
    missing_legacy = sorted(legacy_ids - set(by_source))
    if missing_legacy:
        raise RuntimeError(f"Could not recover {len(missing_legacy)} Experiment 1 test rows")
    corrected_legacy_test = [by_source[row["source_id"]] for row in experiment1_test]
    protected_groups = {row["claim_group_id"] for row in corrected_legacy_test}

    training, composition = contrast_aware_training_sample(
        train_pool, args.train_per_label, args.seed
    )
    eligible_dev = [
        row
        for row in dev_pool
        if row["claim_group_id"] not in protected_groups
        and (row["label"] != "INSUFFICIENT_EVIDENCE" or clean_neutral(row))
    ]
    validation_pool = [
        row for row in eligible_dev if stable_partition(row["claim_group_id"], args.seed) == 0
    ]
    test_pool = [
        row for row in eligible_dev if stable_partition(row["claim_group_id"], args.seed) == 1
    ]
    validation = balanced_group_sample(validation_pool, args.validation_per_label, args.seed + 1)
    test = balanced_group_sample(test_pool, args.test_per_label, args.seed + 2)

    split_rows = {
        "train": training,
        "validation": validation,
        "test": test,
        "experiment1_test_pair_labels": corrected_legacy_test,
    }
    for name, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    groups = {
        name: {row["claim_group_id"] for row in rows}
        for name, rows in split_rows.items()
    }
    ids = {name: {row["source_id"] for row in rows} for name, rows in split_rows.items()}
    checks = {
        "train_validation_source_overlap": len(ids["train"] & ids["validation"]),
        "train_test_source_overlap": len(ids["train"] & ids["test"]),
        "validation_test_source_overlap": len(ids["validation"] & ids["test"]),
        "validation_test_claim_group_overlap": len(groups["validation"] & groups["test"]),
        "legacy_group_in_train": len(groups["experiment1_test_pair_labels"] & groups["train"]),
        "legacy_group_in_validation": len(
            groups["experiment1_test_pair_labels"] & groups["validation"]
        ),
    }
    if any(checks.values()):
        raise RuntimeError(f"Leakage check failed: {checks}")

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "labels": dict(Counter(row["label"] for row in rows)),
            "pair_fever_disagreements": sum(row["pair_fever_disagree"] for row in rows),
            "mean_lexical_overlap": round(
                sum(row["lexical_overlap"] for row in rows) / len(rows), 6
            ),
        }

    manifest = {
        "experiment": "experiment2_clean_neutral_contrast",
        "dataset_id": args.dataset_id,
        "dataset_revision": str(raw_train.info.version),
        "seed": args.seed,
        "authoritative_label": "source NLI ClassLabel after semantic-risk filtering",
        "pair_label_mapping": PAIR_LABEL_MAP,
        "fever_claim_label_role": (
            "audit-only; it is aligned with the derivative NLI label and is not independent pair annotation"
        ),
        "cleaning": [
            "removed empty/invalid labels",
            "required claim >= 10 characters and evidence >= 40 characters",
            "normalized whitespace",
            "deduplicated exact claim/evidence/label triples",
            "excluded neutral candidates with numeric, negation, categorical, or antonym conflicts",
            "restricted natural neutral rows to moderate lexical overlap (0.12 through 0.70)",
            "added 50 controlled claim triplets with support, active conflict, and missing-evidence variants",
        ],
        "training_sampling": composition,
        "splits": {name: summary(rows) for name, rows in split_rows.items()},
        "leakage_checks": checks,
        "experiment1_preservation": (
            "Experiment 1 files are unchanged. Its 90 test source IDs are exported for later "
            "diagnostic comparison and excluded by claim group from "
            "Experiment 2 training and validation."
        ),
        "known_label_limitation": (
            "The derivative FEVER NLI labels are not independently re-annotated for every assembled "
            "passage. Heuristic filtering reduces obvious semantic conflicts but cannot remove all noise."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
