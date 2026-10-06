import re
from dataclasses import dataclass

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

_SUBJECT_PRONOUN = re.compile(r"^(it|they|he|she)\b", re.IGNORECASE)
_PREDICATE_AUXILIARY = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"can|could|may|might|must|shall|should|will|would)\b",
    re.IGNORECASE,
)
_IMPLICIT_PREDICATE_STARTS = {
    "appears", "became", "began", "contains", "covers", "created", "died",
    "features", "includes", "increased", "lies", "located", "measures", "opened",
    "produced", "ranks", "reaches", "released", "remains", "runs", "stands",
    "weighs",
}
_CONJUNCTION = re.compile(r"\s+and\s+", re.IGNORECASE)


def _predicate_start(value: str) -> bool:
    first = re.match(r"([A-Za-z]+)\b", value.strip())
    if first is None:
        return False
    word = first.group(1).lower()
    return bool(_PREDICATE_AUXILIARY.match(value.strip())) or word in _IMPLICIT_PREDICATE_STARTS


def _subject_before_predicate(value: str) -> tuple[str, str] | None:
    match = _PREDICATE_AUXILIARY.search(value)
    if match is None:
        return None
    subject = value[: match.start()].strip()
    predicate = value[match.start() :].strip()
    if not subject or not predicate:
        return None
    return subject, predicate


def _split_coordinated_predicates(value: str) -> list[str]:
    """Split only when ``and`` introduces a recognizable independent predicate.

    This deliberately uses a bounded auxiliary/verb-start list. It leaves noun
    coordination, compound objects, and list phrases together rather than
    guessing from the conjunction alone.
    """
    subject_parts = _subject_before_predicate(value)
    if subject_parts is None:
        return [value]
    subject, _ = subject_parts
    for match in _CONJUNCTION.finditer(value):
        left = value[: match.start()].strip()
        right = value[match.end() :].strip()
        if not left or not right:
            continue
        if _predicate_start(right):
            return [left, *[f"{subject} {part}" for part in _split_coordinated_predicates(right)]]
        # Preserve a repeated subject already present in the second clause.
        repeated_subject = re.match(
            rf"^{re.escape(subject)}\s+(.+)$", right, re.IGNORECASE
        )
        if repeated_subject and _predicate_start(repeated_subject.group(1)):
            return [left, *_split_coordinated_predicates(right)]
    return [value]


class ExtractedClaims(BaseModel):
    """Structured contract used by an LLM claim-extraction chain in real modes."""

    claims: list[str] = Field(description="Atomic, independently verifiable factual claims")


CLAIM_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=ExtractedClaims)


@dataclass(slots=True)
class ClaimExtractor:
    max_claims: int = 20

    def extract(self, text: str) -> list[str]:
        """Deterministic fallback extraction; real providers may populate the same schema."""
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        claims: list[str] = []
        antecedent: str | None = None
        for part in parts:
            candidate = part.strip(" \t-*•")
            if len(candidate) < 8:
                continue
            lower = candidate.lower()
            if lower.startswith(("hello", "hi ", "in my opinion", "i recommend")):
                continue
            if candidate.endswith("?"):
                continue
            pronoun = _SUBJECT_PRONOUN.match(candidate)
            if pronoun and antecedent:
                candidate = antecedent + candidate[pronoun.end() :]
            clauses = _split_coordinated_predicates(candidate)
            terminal = candidate[-1] if candidate[-1] in ".!?" else "."
            for clause in clauses:
                clause = clause.strip()
                if clause and clause[-1] not in ".!?":
                    clause += terminal
                if len(clause) >= 8:
                    claims.append(clause)
            subject_parts = _subject_before_predicate(clauses[0]) if clauses else None
            if subject_parts:
                antecedent = subject_parts[0]
            if len(claims) >= self.max_claims:
                return claims[: self.max_claims]
        return claims

    @staticmethod
    def normalize(claim: str) -> str:
        value = re.sub(r"\s+", " ", claim).strip()
        return value[:-1] if value.endswith(".") else value
