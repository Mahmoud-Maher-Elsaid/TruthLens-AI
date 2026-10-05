from typing import Protocol

from app.models.schemas import ClaimResult
from app.retrieval.base import EvidenceChunk


class VerificationProvider(Protocol):
    name: str

    def verify(
        self,
        claim_id: str,
        claim: str,
        normalized: str,
        ranked: list[tuple[EvidenceChunk, float]],
    ) -> ClaimResult: ...
