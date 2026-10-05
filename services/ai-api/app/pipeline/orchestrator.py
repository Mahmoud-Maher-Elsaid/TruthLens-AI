import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.models.schemas import (
    AnalyzeRequest,
    AuditReport,
    AuditSummary,
    ClaimResult,
    ProgressEvent,
    Verdict,
)
from app.pipeline.extraction import ClaimExtractor
from app.pipeline.reranking import EvidenceReranker
from app.providers.base import VerificationProvider
from app.providers.demo import DemoVerificationProvider
from app.providers.huggingface import BaselineNLIProvider, FinetunedGenerativeProvider
from app.retrieval.base import EvidenceChunk, VectorStore


class AuditState(TypedDict, total=False):
    text: str
    claims: list[str]
    results: list[ClaimResult]


def build_graph() -> object:
    graph = StateGraph(AuditState)
    for name in ("validate", "extract", "retrieve_rerank_verify", "aggregate_cite_report"):
        graph.add_node(name, lambda state: state)
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "extract")
    graph.add_edge("extract", "retrieve_rerank_verify")
    graph.add_edge("retrieve_rerank_verify", "aggregate_cite_report")
    graph.add_edge("aggregate_cite_report", END)
    return graph.compile()


class AuditOrchestrator:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings, self.store = settings, store
        self.extractor = ClaimExtractor()
        self.reranker = EvidenceReranker(settings.reranker_model, enabled=settings.mode != "DEMO")
        self.provider: VerificationProvider
        if settings.mode == "DEMO" or settings.verifier_provider == "demo":
            self.provider = DemoVerificationProvider()
        elif settings.verifier_provider == "finetuned":
            self.provider = FinetunedGenerativeProvider(settings.finetuned_adapter or "")
        else:
            self.provider = BaselineNLIProvider(settings.verifier_model)
        self.graph = build_graph()
        self._semaphore = asyncio.Semaphore(4)

    async def initialize(self) -> None:
        if "empty" in self.store.readiness or "demo" in self.store.readiness:
            from app.providers.demo import REFERENCE
            await self.store.add([REFERENCE], [{"document_name": "Eiffel Tower reference", "section": "Overview"}])

    @staticmethod
    def _merge_ranked(
        first: list[tuple[EvidenceChunk, float]], second: list[tuple[EvidenceChunk, float]]
    ) -> list[tuple[EvidenceChunk, float]]:
        merged: dict[str, tuple[EvidenceChunk, float]] = {}
        for chunk, score in [*first, *second]:
            key = f"{chunk.id}:{chunk.text}"
            existing = merged.get(key)
            if existing is None or float(score) > existing[1]:
                merged[key] = (chunk, float(score))
        return sorted(merged.values(), key=lambda item: item[1], reverse=True)

    async def _verify_claim(self, index: int, claim: str) -> ClaimResult:
        async with self._semaphore:
            normalized = self.extractor.normalize(claim)
            ranked = self.reranker.rerank(normalized, await self.store.search(normalized, self.settings.top_k), self.settings.rerank_k)
            result = self.provider.verify(f"claim-{index + 1}", claim, normalized, ranked)
            if result.verdict == Verdict.INSUFFICIENT_EVIDENCE and self.settings.mode != "DEMO":
                second = await self.store.search(f"Evidence about: {normalized}", self.settings.top_k)
                retry_ranked = self.reranker.rerank(normalized, second, self.settings.rerank_k)
                merged = self._merge_ranked(ranked, retry_ranked)
                result = self.provider.verify(
                    f"claim-{index + 1}", claim, normalized, merged[: self.settings.rerank_k]
                )
            return result

    async def analyze(self, request: AnalyzeRequest) -> AuditReport:
        claims = self.extractor.extract(request.text)
        results = await asyncio.gather(*(self._verify_claim(i, claim) for i, claim in enumerate(claims)))
        supported = sum(item.verdict == Verdict.SUPPORTED for item in results)
        contradicted = sum(item.verdict == Verdict.CONTRADICTED for item in results)
        insufficient = sum(item.verdict == Verdict.INSUFFICIENT_EVIDENCE for item in results)
        coverage = (supported + contradicted) / len(results) if results else 0.0
        return AuditReport(
            analysis_id=hashlib.sha256(request.text.encode()).hexdigest()[:16], mode=self.settings.mode,
            provider=self.provider.name, is_precomputed_demo=self.settings.mode == "DEMO", original_text=request.text,
            summary=AuditSummary(claims_analyzed=len(results), supported=supported, contradicted=contradicted, insufficient_evidence=insufficient, evidence_coverage_score=coverage, score_definition="Share of extracted claims with evidence sufficient to support or contradict them."),
            claims=results,
            limitations=(
                [
                    "Demo mode uses deterministic rules and bundled evidence; it is not fresh model inference.",
                    "Evidence strength is categorical and is not a calibrated probability.",
                ]
                if self.settings.mode == "DEMO"
                else [
                    "Results are bounded by the indexed evidence and pretrained verifier behavior.",
                    "Evidence strength is categorical and is not a calibrated probability.",
                ]
            ),
        )

    async def stream(self, request: AnalyzeRequest) -> AsyncIterator[ProgressEvent]:
        yield ProgressEvent(event="validation_started", message="Validating input…", progress=5)
        claims = self.extractor.extract(request.text)
        yield ProgressEvent(event="claims_extracted", message=f"Extracted {len(claims)} candidate claims", progress=15, total_claims=len(claims))
        yield ProgressEvent(event="retrieval_started", message="Starting retrieval, reranking, and verification…", progress=25)
        report = await self.analyze(request)
        for index, claim in enumerate(report.claims):
            yield ProgressEvent(event="claim_verified", message=f"Verified claim {index + 1} of {len(report.claims)}: {claim.verdict}", progress=60 + round(25 * (index + 1) / max(len(report.claims), 1)), claim_index=index + 1, total_claims=len(report.claims), data={"claim_id": claim.id, "verdict": claim.verdict})
        yield ProgressEvent(event="analysis_complete", message="Audit complete", progress=100, data=report)
