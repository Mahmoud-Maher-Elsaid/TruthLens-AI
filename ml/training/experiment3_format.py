"""Deterministic formatting and validation contract for Experiment 3.

The module is intentionally dependency-free so a preparation audit can run
before any model or dataset download. Training code must use the same contract.
"""

from __future__ import annotations

import re
from typing import Any

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def evidence_passages(row: dict[str, Any]) -> list[str]:
    passages = row.get("evidence_set")
    if passages is None:
        passages = [row.get("evidence", "")]
    if not isinstance(passages, list):
        raise ValueError("evidence_set must be a list")
    cleaned = [_clean(item) for item in passages if _clean(item)]
    if not 1 <= len(cleaned) <= 3:
        raise ValueError("Experiment 3 requires one to three evidence passages")
    return cleaned


def validate_row(row: dict[str, Any]) -> None:
    if row.get("label") not in LABELS:
        raise ValueError(f"Unknown label: {row.get('label')}")
    if len(_clean(row.get("claim"))) < 10:
        raise ValueError("Claim is too short")
    evidence_passages(row)
    for key in ("source_id", "claim_group_id"):
        if not _clean(row.get(key)):
            raise ValueError(f"Missing {key}")


def prompt(row: dict[str, Any]) -> str:
    validate_row(row)
    passages = evidence_passages(row)
    evidence = "\n".join(f"Evidence {index}: {text}" for index, text in enumerate(passages, 1))
    return (
        "Classify the claim using the complete evidence set. Return exactly one label: "
        "SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\n"
        f"Claim: {_clean(row['claim'])}\n{evidence}\nLabel:"
    )
