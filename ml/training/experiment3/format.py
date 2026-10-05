"""Frozen prompt and row contract for Experiment 3."""

from __future__ import annotations

import re
from typing import Any

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
SYSTEM_PROMPT = """You are a strict evidence verifier. Classify the claim using all supplied evidence.

SUPPORTED requires evidence that establishes every material part of the claim.
CONTRADICTED requires evidence that explicitly states an incompatible fact.
INSUFFICIENT_EVIDENCE applies when evidence is silent, incomplete, ambiguous, merely related, or conflicting.
Absence of support is not contradiction. Return exactly one label and no explanation."""


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def evidence_passages(row: dict[str, Any]) -> list[str]:
    passages = row.get("evidence_set")
    if passages is None:
        passages = [row.get("evidence", "")]
    if not isinstance(passages, list):
        raise ValueError("evidence_set must be a list")
    values = [clean(value) for value in passages if clean(value)]
    if not 1 <= len(values) <= 3:
        raise ValueError("Experiment 3 requires one to three evidence passages")
    return values


def validate_row(row: dict[str, Any], require_label: bool = True) -> None:
    if require_label and row.get("label") not in LABELS:
        raise ValueError(f"Unknown label: {row.get('label')}")
    if len(clean(row.get("claim"))) < 10:
        raise ValueError("Claim is too short")
    passages = evidence_passages(row)
    metadata = row.get("evidence_metadata")
    if metadata is not None and (not isinstance(metadata, list) or len(metadata) != len(passages)):
        raise ValueError("evidence_metadata must align with evidence_set")
    for key in ("source_id", "claim_group_id"):
        if not clean(row.get(key)):
            raise ValueError(f"Missing {key}")


def user_content(row: dict[str, Any]) -> str:
    validate_row(row, require_label="label" in row)
    blocks = "\n\n".join(
        f"EVIDENCE {index}:\n{passage}" for index, passage in enumerate(evidence_passages(row), 1)
    )
    return f"CLAIM:\n{clean(row['claim'])}\n\n{blocks}"


def messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content(row)}]


def prompt_ids(tokenizer: Any, row: dict[str, Any], max_prompt_tokens: int) -> list[int]:
    return tokenizer.apply_chat_template(
        messages(row), tokenize=True, add_generation_prompt=True, truncation=True, max_length=max_prompt_tokens
    )
