from __future__ import annotations

from typing import Any

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")

SYSTEM_PROMPT = """You are a strict evidence verifier. Classify the claim using only the supplied evidence.

Definitions:
- SUPPORTED: the evidence explicitly establishes every material part of the claim.
- CONTRADICTED: the evidence explicitly states a fact that is incompatible with the claim.
- INSUFFICIENT_EVIDENCE: the evidence is silent, incomplete, ambiguous, or merely related; it neither proves nor disproves every material part of the claim.

Important decision rule: absence of support is not contradiction. Use CONTRADICTED only for an active conflict. Return exactly one label and no explanation."""


def messages(row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<claim>\n{row['claim']}\n</claim>\n<evidence>\n{row['evidence']}\n</evidence>",
        },
    ]


def prompt_ids(tokenizer: Any, row: dict[str, str], max_prompt_tokens: int) -> list[int]:
    return tokenizer.apply_chat_template(
        messages(row),
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )


def prompt_text(tokenizer: Any, row: dict[str, str]) -> str:
    return tokenizer.apply_chat_template(
        messages(row), tokenize=False, add_generation_prompt=True
    )
