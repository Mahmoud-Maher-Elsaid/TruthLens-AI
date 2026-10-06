from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuditMode(StrEnum):
    STANDARD = "standard"
    DEEP = "deep"
    DEMO = "demo"


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceStrength(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AnalyzeRequest(StrictModel):
    text: str = Field(min_length=3, max_length=50_000)
    mode: AuditMode = AuditMode.STANDARD
    document_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def non_blank(self) -> "AnalyzeRequest":
        if not self.text.strip():
            raise ValueError("text must contain non-whitespace characters")
        return self


class SourceMetadata(StrictModel):
    document_name: str
    document_id: str | None = None
    source_url: HttpUrl | None = None
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    chunk_id: str


class EvidenceItem(StrictModel):
    id: str
    text: str
    metadata: SourceMetadata
    retrieval_score: float | None = Field(default=None, ge=-1, le=1)
    rerank_score: float | None = Field(default=None, ge=-20, le=20)
    relationship: Literal["supports", "contradicts", "related"]


class ClaimResult(StrictModel):
    id: str
    claim: str
    normalized_claim: str
    verdict: Verdict
    evidence_strength: EvidenceStrength
    evidence: list[EvidenceItem]
    explanation: str
    corrected_statement: str | None = None
    conflicting_evidence: bool = False


class AuditSummary(StrictModel):
    claims_analyzed: int = Field(ge=0)
    supported: int = Field(ge=0)
    contradicted: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    evidence_coverage_score: float = Field(ge=0, le=1)
    score_definition: str


class AuditReport(StrictModel):
    analysis_id: str
    mode: Literal["DEMO", "LOCAL", "PRODUCTION"]
    provider: str
    is_precomputed_demo: bool
    original_text: str
    summary: AuditSummary
    claims: list[ClaimResult]
    limitations: list[str]


class ProgressEvent(StrictModel):
    event: Literal[
        "validation_started",
        "claims_extracted",
        "retrieval_started",
        "evidence_found",
        "reranking_complete",
        "claim_verified",
        "report_started",
        "analysis_complete",
        "analysis_failed",
    ]
    message: str
    progress: int = Field(ge=0, le=100)
    claim_index: int | None = None
    total_claims: int | None = None
    data: dict[str, Any] | AuditReport | None = None


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    version: str
    backend_mode: str
    provider: str
    model_readiness: str
    vector_store_readiness: str


class DocumentResponse(StrictModel):
    document_id: str
    filename: str
    pages: int | None
    characters_indexed: int
    chunks_indexed: int
    vector_backend: str


class CompareVariant(StrictModel):
    name: Literal["Base LLM", "RAG-enhanced", "TruthLens"]
    description: str
    claims: list[ClaimResult]
    evidence_enabled: bool
    orchestration_enabled: bool


class CompareResponse(StrictModel):
    mode: str
    is_demo: bool
    variants: list[CompareVariant]
    disclaimer: str


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail
