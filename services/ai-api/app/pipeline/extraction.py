import re
from dataclasses import dataclass

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


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
        for part in parts:
            candidate = part.strip(" \t-*•")
            if len(candidate) < 8:
                continue
            lower = candidate.lower()
            if lower.startswith(("hello", "hi ", "in my opinion", "i recommend")):
                continue
            if candidate.endswith("?"):
                continue
            claims.append(candidate)
            if len(claims) >= self.max_claims:
                break
        return claims

    @staticmethod
    def normalize(claim: str) -> str:
        value = re.sub(r"\s+", " ", claim).strip()
        return value[:-1] if value.endswith(".") else value
