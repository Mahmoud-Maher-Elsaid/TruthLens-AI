import asyncio
import io
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models.schemas import (
    AnalyzeRequest,
    AuditReport,
    CompareResponse,
    CompareVariant,
    DocumentResponse,
    HealthResponse,
)

router = APIRouter()


def _safe_name(name: str | None) -> str:
    return (re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "document").name)[:120] or "document")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    orchestrator = request.app.state.orchestrator
    vector_readiness = store.readiness
    if settings.mode != "DEMO" and settings.vector_backend == "qdrant":
        try:
            exists = await asyncio.wait_for(
                store.client.collection_exists(settings.qdrant_collection), timeout=3
            )
            vector_readiness = "ready" if exists else "degraded (collection absent)"
        except Exception:
            vector_readiness = "degraded (Qdrant unavailable)"
    model_readiness = "ready (deterministic demo)" if settings.mode == "DEMO" else "ready (loaded models)"
    if settings.mode != "DEMO" and (
        "fallback" in orchestrator.reranker.mode or "demo" in orchestrator.provider.name
    ):
        model_readiness = "degraded (model fallback active)"
    degraded = "degraded" in model_readiness or "degraded" in vector_readiness or "empty" in vector_readiness
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=request.app.version,
        backend_mode=settings.mode,
        provider=orchestrator.provider.name,
        model_readiness=model_readiness,
        vector_store_readiness=vector_readiness,
    )


@router.post("/analyze", response_model=AuditReport)
async def analyze(payload: AnalyzeRequest, request: Request) -> AuditReport:
    try:
        return await asyncio.wait_for(
            request.app.state.orchestrator.analyze(payload),
            timeout=request.app.state.settings.request_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Analysis timed out.") from exc


@router.post("/analyze/stream")
async def analyze_stream(payload: AnalyzeRequest, request: Request) -> StreamingResponse:
    async def events():
        try:
            async with asyncio.timeout(request.app.state.settings.request_timeout_seconds):
                async for event in request.app.state.orchestrator.stream(payload):
                    yield event.model_dump_json() + "\n"
        except TimeoutError:
            yield json.dumps({"event": "analysis_failed", "message": "Analysis timed out.", "progress": 100, "data": {"code": "REQUEST_TIMEOUT", "request_id": request.state.request_id}}) + "\n"
        except Exception:
            yield json.dumps({"event": "analysis_failed", "message": "Analysis failed safely.", "progress": 100, "data": {"code": "INTERNAL_ERROR", "request_id": request.state.request_id}}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentResponse:
    settings = request.app.state.settings
    filename = _safe_name(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".markdown"}:
        raise HTTPException(status_code=415, detail="Supported file types are PDF, TXT, and Markdown.")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Document exceeds the configured size limit.")
    if not content:
        raise HTTPException(status_code=422, detail="Document is empty.")
    document_id = str(uuid.uuid4())
    pages, chunks, metadata = None, [], []
    try:
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            pages = len(reader.pages)
            for page_number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    chunks.append(text)
                    metadata.append({"document_id": document_id, "document_name": filename, "page": page_number})
        else:
            text = content.decode("utf-8").strip()
            step = max(1, settings.chunk_size - settings.chunk_overlap)
            for start in range(0, len(text), step):
                chunk = text[start : start + settings.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                    metadata.append(
                        {"document_id": document_id, "document_name": filename, "section": f"offset-{start}"}
                    )
    except (ValueError, UnicodeDecodeError, OSError, PdfReadError) as exc:
        raise HTTPException(status_code=422, detail="Document is malformed or unreadable.") from exc
    if not chunks:
        raise HTTPException(status_code=422, detail="Document contains no extractable text.")
    indexed = await request.app.state.store.add(chunks, metadata)
    backend = "lexical-demo" if settings.mode == "DEMO" else settings.vector_backend
    return DocumentResponse(document_id=document_id, filename=filename, pages=pages, characters_indexed=sum(map(len, chunks)), chunks_indexed=indexed, vector_backend=backend)


@router.post("/compare", response_model=CompareResponse)
async def compare(payload: AnalyzeRequest, request: Request) -> CompareResponse:
    report = await analyze(payload, request)
    return CompareResponse(mode=request.app.state.settings.mode, is_demo=request.app.state.settings.mode == "DEMO", variants=[
        CompareVariant(name="Base LLM", description="Input-only baseline; no evidence retrieval.", claims=[], evidence_enabled=False, orchestration_enabled=False),
        CompareVariant(name="RAG-enhanced", description="Illustrative view reusing TruthLens claims; no independent RAG pipeline was run.", claims=report.claims, evidence_enabled=True, orchestration_enabled=False),
        CompareVariant(name="TruthLens", description="Atomic claims, evidence, reranking, bounded verification, and citations.", claims=report.claims, evidence_enabled=True, orchestration_enabled=True),
    ], disclaimer="Illustrative comparison, not a three-system benchmark. Base LLM was not run; RAG reuses TruthLens results. Demo mode is deterministic.")


@router.get("/benchmarks")
async def benchmarks(request: Request) -> dict[str, object]:
    root = Path(request.app.state.repo_root) / "ml" / "benchmarks"
    artifacts = []
    for path in sorted(root.glob("*.json")):
        try:
            artifacts.append(
                {"name": path.name, "data": json.loads(path.read_text(encoding="utf-8"))}
            )
        except (OSError, json.JSONDecodeError):
            continue
    return {"status": "available" if artifacts else "not_measured", "artifacts": artifacts, "message": None if artifacts else "Benchmarks have not been generated yet."}
