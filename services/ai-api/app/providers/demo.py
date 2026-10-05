from app.models.schemas import ClaimResult, EvidenceItem, EvidenceStrength, SourceMetadata, Verdict
from app.retrieval.base import EvidenceChunk

REFERENCE = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed "
    "from 1887 to 1889 for the 1889 World's Fair. The tower is 330 metres tall after the "
    "installation of its current antenna. It is named after engineer Gustave Eiffel."
)


class DemoVerificationProvider:
    name = "deterministic-demo-fixture-v1"

    @staticmethod
    def verify(claim_id: str, claim: str, normalized: str, ranked: list[tuple[EvidenceChunk, float]]) -> ClaimResult:
        lower = normalized.lower()
        verdict = Verdict.INSUFFICIENT_EVIDENCE
        strength = EvidenceStrength.LOW
        explanation = "The bundled evidence does not establish or refute this claim."
        correction = None
        relationship = "related"
        if "eiffel" in lower and ("paris" in lower or "330" in lower):
            verdict = Verdict.SUPPORTED
            strength = EvidenceStrength.HIGH
            relationship = "supports"
            explanation = "The bundled reference directly states this location or height."
        elif "1920" in lower or ("eiffel" in lower and "rome" in lower):
            verdict = Verdict.CONTRADICTED
            strength = EvidenceStrength.HIGH
            relationship = "contradicts"
            explanation = "The reference dates construction to 1887–1889, not 1920, and locates the tower in Paris."
            correction = "The Eiffel Tower was completed in 1889 in Paris, France."
        evidence = [
            EvidenceItem(
                id=f"evidence-{chunk.id}", text=chunk.text,
                metadata=SourceMetadata(document_name=str(chunk.metadata.get("document_name", "Eiffel Tower reference")), document_id=chunk.metadata.get("document_id"), source_url=chunk.metadata.get("source_url"), section=chunk.metadata.get("section"), page=chunk.metadata.get("page"), chunk_id=chunk.id),
                retrieval_score=max(0.0, min(1.0, chunk.score)), rerank_score=max(-20.0, min(20.0, rerank_score)), relationship=relationship,
            )
            for chunk, rerank_score in ranked[:2]
        ]
        return ClaimResult(id=claim_id, claim=claim, normalized_claim=normalized, verdict=verdict, evidence_strength=strength, evidence=evidence, explanation=explanation, corrected_statement=correction)
