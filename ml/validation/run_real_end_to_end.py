"""Measured LOCAL-mode end-to-end validation for the real TruthLens pipeline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "services" / "ai-api"))

from app.api.routes import upload_document
from app.core.config import Settings
from app.models.schemas import AnalyzeRequest
from app.pipeline.orchestrator import AuditOrchestrator
from app.retrieval.store import FaissStore
from common import (
    environment,
    latency_summary,
    load_demo_kb_chunks,
    utc_now,
    write_json,
)
from fastapi import UploadFile

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VERIFIER_MODEL = "cross-encoder/nli-deberta-v3-small"

CASES = [
    {
        "name": "supported",
        "text": "The Eiffel Tower is located in Paris, France.",
        "expected": ["SUPPORTED"],
    },
    {
        "name": "contradicted",
        "text": "The Eiffel Tower was built in 1920.",
        "expected": ["CONTRADICTED"],
    },
    {
        "name": "insufficient_evidence",
        "text": "The Eiffel Tower was painted blue in 1901.",
        "expected": ["INSUFFICIENT_EVIDENCE"],
    },
    {
        "name": "multiple_claims",
        "text": (
            "The Eiffel Tower is in Paris. It was completed in 1920. "
            "It is 330 metres tall with its current antenna. "
            "The tower was painted blue in 1901."
        ),
        "expected": [
            "SUPPORTED",
            "CONTRADICTED",
            "SUPPORTED",
            "INSUFFICIENT_EVIDENCE",
        ],
    },
    {
        "name": "document_ingestion",
        "text": "The Lantern Museum opened in 2005.",
        "expected": ["SUPPORTED"],
        "ingest": {
            "filename": "lantern_museum.txt",
            "text": (
                "The Lantern Museum officially opened in 2005. It is located beside the "
                "River Luma and contains a public collection of historical lamps."
            ),
        },
    },
    {
        "name": "conflicting_evidence",
        "text": "The Eiffel Tower was built in 1920.",
        "expected": ["CONTRADICTED"],
        "ingest": {
            "filename": "unreliable_1920_gazette.txt",
            "text": (
                "The fictional 1920 Centennial Gazette states that the Eiffel Tower was built "
                "in 1920 and completed that same year."
            ),
        },
        "expected_retrieved_documents": [
            "eiffel_tower.md",
            "unreliable_1920_gazette.txt",
        ],
    },
]


class StageRecorder:
    def __init__(self) -> None:
        self.durations: dict[str, list[float]] = defaultdict(list)
        self.failures: list[dict[str, str]] = []
        self.retrieval_traces: list[dict[str, Any]] = []
        self.rerank_traces: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.durations.clear()
        self.failures.clear()
        self.retrieval_traces.clear()
        self.rerank_traces.clear()

    def record(self, stage: str, started: float) -> None:
        self.durations[stage].append((time.perf_counter() - started) * 1000)

    def failure(self, stage: str, exc: Exception) -> None:
        self.failures.append(
            {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        )

    def summary(self) -> dict[str, Any]:
        return {
            stage: {
                "calls": len(values),
                "total_ms": sum(values),
                "mean_ms": statistics.fmean(values),
                "max_ms": max(values),
                "measurements_ms": values,
            }
            for stage, values in self.durations.items()
        }


class TimedExtractor:
    def __init__(self, delegate: Any, recorder: StageRecorder) -> None:
        self.delegate, self.recorder = delegate, recorder

    def extract(self, text: str) -> list[str]:
        started = time.perf_counter()
        try:
            return self.delegate.extract(text)
        except Exception as exc:
            self.recorder.failure("claim_extraction", exc)
            raise
        finally:
            self.recorder.record("claim_extraction", started)

    def normalize(self, claim: str) -> str:
        started = time.perf_counter()
        try:
            return self.delegate.normalize(claim)
        except Exception as exc:
            self.recorder.failure("claim_normalization", exc)
            raise
        finally:
            self.recorder.record("claim_normalization", started)


class TimedStore:
    def __init__(self, delegate: Any, recorder: StageRecorder) -> None:
        self.delegate, self.recorder = delegate, recorder

    @property
    def readiness(self) -> str:
        return self.delegate.readiness

    async def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> int:
        started = time.perf_counter()
        try:
            return await self.delegate.add(texts, metadata)
        except Exception as exc:
            self.recorder.failure("document_embedding_and_indexing", exc)
            raise
        finally:
            self.recorder.record("document_embedding_and_indexing", started)

    async def search(self, query: str, top_k: int) -> list[Any]:
        started = time.perf_counter()
        try:
            results = await self.delegate.search(query, top_k)
            self.recorder.retrieval_traces.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "results": [
                        {
                            "chunk_id": item.id,
                            "text": item.text,
                            "metadata": item.metadata,
                            "retrieval_score": item.score,
                        }
                        for item in results
                    ],
                }
            )
            return results
        except Exception as exc:
            self.recorder.failure("faiss_retrieval", exc)
            raise
        finally:
            self.recorder.record("faiss_retrieval", started)


class TimedReranker:
    def __init__(self, delegate: Any, recorder: StageRecorder) -> None:
        self.delegate, self.recorder = delegate, recorder

    @property
    def mode(self) -> str:
        return self.delegate.mode

    def rerank(self, claim: str, chunks: list[Any], limit: int) -> list[tuple[Any, float]]:
        started = time.perf_counter()
        try:
            ranked = self.delegate.rerank(claim, chunks, limit)
            self.recorder.rerank_traces.append(
                {
                    "claim": claim,
                    "limit": limit,
                    "results": [
                        {
                            "chunk_id": chunk.id,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                            "retrieval_score": chunk.score,
                            "rerank_score": score,
                        }
                        for chunk, score in ranked
                    ],
                }
            )
            return ranked
        except Exception as exc:
            self.recorder.failure("cross_encoder_reranking", exc)
            raise
        finally:
            self.recorder.record("cross_encoder_reranking", started)


class TimedProvider:
    def __init__(self, delegate: Any, recorder: StageRecorder) -> None:
        self.delegate, self.recorder = delegate, recorder

    @property
    def name(self) -> str:
        return self.delegate.name

    def verify(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.delegate.verify(*args, **kwargs)
        except Exception as exc:
            self.recorder.failure("baseline_verification", exc)
            raise
        finally:
            self.recorder.record("baseline_verification", started)


async def ingest_document(
    settings: Settings,
    store: TimedStore,
    filename: str,
    text: str,
) -> tuple[dict[str, Any], float]:
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=settings, store=store))
    )
    upload = UploadFile(file=io.BytesIO(text.encode("utf-8")), filename=filename)
    started = time.perf_counter()
    response = await upload_document(request, upload)
    return response.model_dump(mode="json"), (time.perf_counter() - started) * 1000


def citations(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim["id"],
            "evidence_id": evidence["id"],
            "document_name": evidence["metadata"]["document_name"],
            "section": evidence["metadata"].get("section"),
            "page": evidence["metadata"].get("page"),
            "chunk_id": evidence["metadata"]["chunk_id"],
            "retrieval_score": evidence.get("retrieval_score"),
            "rerank_score": evidence.get("rerank_score"),
        }
        for claim in report["claims"]
        for evidence in claim["evidence"]
    ]


def dependency_check() -> dict[str, bool]:
    packages = (
        "faiss",
        "sentence_transformers",
        "transformers",
        "torch",
        "sklearn",
        "fastapi",
        "pypdf",
    )
    return {package: importlib.util.find_spec(package) is not None for package in packages}


def validate_configuration(args: argparse.Namespace) -> dict[str, Any]:
    dependencies = dependency_check()
    if not all(dependencies.values()):
        raise RuntimeError(f"Missing LOCAL-mode dependencies: {dependencies}")
    settings = Settings(
        mode="LOCAL",
        vector_backend="faiss",
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        verifier_model=args.verifier_model,
        verifier_provider="baseline",
        top_k=4,
        rerank_k=4,
    )
    if settings.mode != "LOCAL" or settings.verifier_provider != "baseline":
        raise RuntimeError(f"Unsafe LOCAL configuration: {settings.model_dump()}")
    model_artifact = json.loads(args.model_validation.read_text(encoding="utf-8"))
    expected = {
        "embedding": args.embedding_model,
        "reranker": args.reranker_model,
        "baseline_verifier": args.verifier_model,
    }
    model_checks: dict[str, Any] = {}
    for role, model_id in expected.items():
        record = model_artifact["models"][role]
        passed = (
            record["status"] == "passed"
            and record["requested_id"] == model_id
            and record["tokenizer_loaded"]
            and record["model_loaded"]
        )
        model_checks[role] = {
            "requested_id": model_id,
            "validated_revision": record.get("revision"),
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(f"Model validation is missing or stale for {role}: {record}")
    texts, metadata = load_demo_kb_chunks(args.kb)
    return {
        "status": "validated",
        "mode": settings.mode,
        "vector_backend": settings.vector_backend,
        "verifier_provider": settings.verifier_provider,
        "dependencies": dependencies,
        "models": model_checks,
        "knowledge_base": {
            "path": str(args.kb),
            "chunks": len(texts),
            "documents": sorted({item["document_name"] for item in metadata}),
        },
        "cases": [case["name"] for case in CASES],
        "outputs": {
            "summary": str(args.summary_output),
            "latency": str(args.latency_output),
            "fixtures": str(args.fixtures),
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings(
        mode="LOCAL",
        vector_backend="faiss",
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        verifier_model=args.verifier_model,
        verifier_provider="baseline",
        top_k=4,
        rerank_k=4,
    )
    texts, metadata = load_demo_kb_chunks(args.kb)
    setup_timings: dict[str, float] = {}
    started = time.perf_counter()
    store = FaissStore(settings.embedding_model)
    setup_timings["embedding_model_load_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    indexed = await store.add(texts, metadata)
    setup_timings["initial_document_embedding_and_faiss_indexing_ms"] = (
        time.perf_counter() - started
    ) * 1000
    started = time.perf_counter()
    orchestrator = AuditOrchestrator(settings, store)
    setup_timings["reranker_and_verifier_load_ms"] = (time.perf_counter() - started) * 1000

    fallbacks: list[dict[str, str]] = []
    if settings.mode != "LOCAL":
        fallbacks.append({"component": "mode", "observed": settings.mode})
    if type(store).__name__ != "FaissStore":
        fallbacks.append({"component": "retrieval", "observed": type(store).__name__})
    if not orchestrator.reranker.mode.startswith("cross-encoder:"):
        fallbacks.append({"component": "reranker", "observed": orchestrator.reranker.mode})
    if not orchestrator.provider.name.startswith("huggingface-nli:"):
        fallbacks.append({"component": "verifier", "observed": orchestrator.provider.name})
    if settings.verifier_provider != "baseline":
        fallbacks.append(
            {"component": "verifier_provider", "observed": settings.verifier_provider}
        )
    if fallbacks:
        raise RuntimeError(f"LOCAL validation refused because fallbacks were detected: {fallbacks}")

    recorder = StageRecorder()
    orchestrator.extractor = TimedExtractor(orchestrator.extractor, recorder)
    timed_store = TimedStore(store, recorder)
    orchestrator.store = timed_store
    orchestrator.reranker = TimedReranker(orchestrator.reranker, recorder)
    orchestrator.provider = TimedProvider(orchestrator.provider, recorder)

    args.fixtures.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in CASES:
        recorder.reset()
        ingestion: dict[str, Any] | None = None
        ingestion_latency_ms: float | None = None
        started = time.perf_counter()
        try:
            if "ingest" in case:
                ingestion, ingestion_latency_ms = await ingest_document(
                    settings,
                    timed_store,
                    case["ingest"]["filename"],
                    case["ingest"]["text"],
                )
            analysis_started = time.perf_counter()
            report = await orchestrator.analyze(AnalyzeRequest(text=case["text"]))
            analysis_latency_ms = (time.perf_counter() - analysis_started) * 1000
            serialization_started = time.perf_counter()
            payload = report.model_dump(mode="json")
            recorder.record("structured_report_serialization", serialization_started)
            actual = [item["verdict"] for item in payload["claims"]]
        except Exception as exc:  # noqa: BLE001 - persist a failed case and continue
            recorder.failure("case_execution", exc)
            record = {
                "schema_version": 1,
                "status": "failed",
                "fixture_kind": "real_non_demo_inference",
                "generated_at": utc_now(),
                "case": case["name"],
                "input": case["text"],
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "stage_failures": recorder.failures,
                "stage_timings": recorder.summary(),
                "overall_latency_ms": (time.perf_counter() - started) * 1000,
            }
            write_json(args.fixtures / f"{case['name']}.json", record)
            results.append(record)
            continue

        retrieved_documents = sorted(
            {
                str(item["metadata"].get("document_name"))
                for trace in recorder.retrieval_traces
                for item in trace["results"]
            }
        )
        cited = citations(payload)
        expected_documents = set(case.get("expected_retrieved_documents", []))
        expected_documents_present = expected_documents.issubset(retrieved_documents)
        retrieval_calls = len(recorder.retrieval_traces)
        record = {
            "schema_version": 1,
            "status": "completed",
            "fixture_kind": "real_non_demo_inference",
            "generated_at": utc_now(),
            "case": case["name"],
            "input": case["text"],
            "expected_verdicts": case["expected"],
            "actual_verdicts": actual,
            "expected_match": actual == case["expected"],
            "overall_latency_ms": (time.perf_counter() - started) * 1000,
            "analysis_latency_ms": analysis_latency_ms,
            "document_ingestion": ingestion,
            "document_ingestion_latency_ms": ingestion_latency_ms,
            "stage_timings": recorder.summary(),
            "stage_failures": recorder.failures,
            "runtime_branches": {
                "retrieval_expansion_triggered": retrieval_calls > len(payload["claims"]),
                "retrieval_calls": retrieval_calls,
            },
            "fallbacks": {
                "demo_mode": payload["mode"] == "DEMO",
                "precomputed_demo": payload["is_precomputed_demo"],
                "lexical_store": False,
                "vector_similarity_reranker": False,
                "finetuned_verifier": False,
            },
            "retrieval_traces": recorder.retrieval_traces,
            "reranking_traces": recorder.rerank_traces,
            "retrieved_documents": retrieved_documents,
            "expected_retrieved_documents": sorted(expected_documents),
            "expected_retrieved_documents_present": expected_documents_present,
            "citations": cited,
            "pipeline_checks": {
                "claim_extraction": len(payload["claims"]) > 0,
                "normalization": all(item["normalized_claim"] for item in payload["claims"]),
                "faiss_retrieval": bool(recorder.retrieval_traces),
                "cross_encoder_reranking": bool(recorder.rerank_traces),
                "baseline_verification": all(item["verdict"] in {
                    "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
                } for item in payload["claims"]),
                "citations": bool(cited)
                and all(item["chunk_id"] and item["document_name"] for item in cited),
                "structured_result": True,
                "non_demo": payload["mode"] == "LOCAL" and not payload["is_precomputed_demo"],
                "document_ingestion": ingestion is None or ingestion["chunks_indexed"] > 0,
                "conflicting_sources_retrieved": expected_documents_present,
            },
            "report": payload,
        }
        write_json(args.fixtures / f"{case['name']}.json", record)
        results.append(record)

    recorder.reset()
    latency_samples: list[float] = []
    request = AnalyzeRequest(text=CASES[0]["text"])
    for _ in range(args.latency_runs):
        started = time.perf_counter()
        await orchestrator.analyze(request)
        latency_samples.append((time.perf_counter() - started) * 1000)
    latency = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": utc_now(),
        "environment": environment(),
        "pipeline": {
            "mode": "LOCAL",
            "embedding_model": args.embedding_model,
            "vector_index": "FAISS IndexFlatIP",
            "reranker_model": args.reranker_model,
            "verifier_model": args.verifier_model,
            "claim_extraction": "deterministic application extractor",
            "verifier_provider": "baseline",
            "fallbacks_detected": fallbacks,
        },
        "request": CASES[0]["text"],
        "latency": latency_summary(latency_samples),
        "stage_timings": recorder.summary(),
        "timing_semantics": (
            "Per-component timings are measured at each real call. Totals may overlap because "
            "multiple claims are verified concurrently."
        ),
    }
    write_json(args.latency_output, latency)
    completed = [item for item in results if item["status"] == "completed"]
    failed = [item for item in results if item["status"] == "failed"]
    return {
        "schema_version": 1,
        "status": "completed" if not failed else "partial",
        "completed_at": utc_now(),
        "environment": environment(),
        "configuration": {
            "mode": settings.mode,
            "vector_backend": settings.vector_backend,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "verifier_model": settings.verifier_model,
            "verifier_provider": settings.verifier_provider,
            "claim_extractor": "app.pipeline.extraction.ClaimExtractor",
            "top_k": settings.top_k,
            "rerank_k": settings.rerank_k,
        },
        "runtime_identity": {
            "store": type(store).__name__,
            "store_readiness": store.readiness,
            "reranker": orchestrator.reranker.mode,
            "provider": orchestrator.provider.name,
            "fallbacks_detected": fallbacks,
        },
        "setup": {
            "initial_documents": sorted({item["document_name"] for item in metadata}),
            "initial_chunks_indexed": indexed,
            "timings": setup_timings,
        },
        "cases": len(results),
        "cases_completed": len(completed),
        "cases_failed": len(failed),
        "cases_matching_expected": sum(item["expected_match"] for item in completed),
        "all_pipeline_checks_passed": all(
            all(item["pipeline_checks"].values()) for item in completed
        ),
        "mean_case_latency_ms": (
            statistics.fmean(item["overall_latency_ms"] for item in completed)
            if completed
            else None
        ),
        "failures": [item["failure"] | {"case": item["case"]} for item in failed],
        "latency_artifact": str(args.latency_output),
        "fixtures": [str(args.fixtures / f"{case['name']}.json") for case in CASES],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kb", type=Path, default=Path("data/demo_kb"))
    parser.add_argument("--fixtures", type=Path, default=Path("data/fixtures/real_examples"))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--verifier-model", default=VERIFIER_MODEL)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument(
        "--model-validation",
        type=Path,
        default=Path("ml/benchmarks/model_validation.json"),
    )
    parser.add_argument(
        "--latency-output", type=Path, default=Path("ml/benchmarks/system_latency.json")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("ml/benchmarks/end_to_end_validation.json"),
    )
    args = parser.parse_args()
    configuration = validate_configuration(args)
    if args.dry_run:
        print(json.dumps(configuration, indent=2))
        return
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "completed_at": utc_now(),
            "configuration": configuration,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_json(args.summary_output, failure)
        print(json.dumps(failure, indent=2))
        raise
    write_json(args.summary_output, result)
    console_summary = {
        "status": result["status"],
        "summary_output": str(args.summary_output),
        "latency_output": result["latency_artifact"],
        "cases": result["cases"],
        "cases_completed": result["cases_completed"],
        "cases_failed": result["cases_failed"],
        "cases_matching_expected": result["cases_matching_expected"],
        "all_pipeline_checks_passed": result["all_pipeline_checks_passed"],
        "runtime_identity": result["runtime_identity"],
        "fixtures": result["fixtures"],
    }
    print(json.dumps(console_summary, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
