from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "services" / "ai-api"))

from common import environment, load_demo_kb_chunks, utc_now, write_json  # noqa: E402
from app.pipeline.reranking import EvidenceReranker  # noqa: E402
from app.retrieval.store import FaissStore  # noqa: E402

CASES = [
    {
        "query": "Where is the Eiffel Tower located?",
        "expected_text": "Champ de Mars in Paris, France",
    },
    {
        "query": "When was the Eiffel Tower constructed?",
        "expected_text": "constructed from 1887 to 1889",
    },
    {
        "query": "How tall is the Eiffel Tower with its current antenna?",
        "expected_text": "330 metres tall",
    },
    {
        "query": "Was the Eiffel Tower built in 1920?",
        "expected_text": "not built in 1920",
    },
    {
        "query": "Is the Eiffel Tower located in Rome?",
        "expected_text": "not located in Rome",
    },
]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    texts, metadata = load_demo_kb_chunks(args.kb)
    store_started = time.perf_counter()
    store = FaissStore(args.embedding_model)
    model_load_seconds = time.perf_counter() - store_started
    started = time.perf_counter()
    indexed = await store.add(texts, metadata)
    indexing_ms = (time.perf_counter() - started) * 1000
    reranker_started = time.perf_counter()
    reranker = EvidenceReranker(args.reranker_model, enabled=True)
    reranker_load_seconds = time.perf_counter() - reranker_started
    if not reranker.mode.startswith("cross-encoder:"):
        raise RuntimeError("The real cross-encoder reranker failed to load; refusing a fallback result.")

    rows: list[dict[str, Any]] = []
    reciprocal: list[float] = []
    recalls: list[float] = []
    for case in CASES:
        expected_ids = {
            item["expected_chunk_id"]
            for text, item in zip(texts, metadata, strict=True)
            if case["expected_text"].lower() in text.lower()
        }
        retrieval_started = time.perf_counter()
        retrieved = await store.search(case["query"], args.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        rerank_started = time.perf_counter()
        reranked = reranker.rerank(case["query"], retrieved, args.rerank_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        retrieved_ids = [chunk.id for chunk in retrieved]
        ranks = [retrieved_ids.index(item) + 1 for item in expected_ids if item in retrieved_ids]
        recalls.append(len(expected_ids & set(retrieved_ids)) / max(len(expected_ids), 1))
        reciprocal.append(1 / min(ranks) if ranks else 0.0)
        rows.append(
            {
                **case,
                "expected_chunk_ids": sorted(expected_ids),
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "expected_evidence_retrieved": bool(ranks),
                "citation_metadata_preserved": all(
                    chunk.id == chunk.metadata.get("expected_chunk_id")
                    and bool(chunk.metadata.get("document_name"))
                    and bool(chunk.metadata.get("section"))
                    for chunk in retrieved
                ),
                "top_k": [
                    {
                        "chunk_id": chunk.id,
                        "retrieval_score": chunk.score,
                        "metadata": chunk.metadata,
                        "text": chunk.text,
                    }
                    for chunk in retrieved
                ],
                "reranked": [
                    {
                        "chunk_id": chunk.id,
                        "rerank_score": score,
                        "retrieval_score": chunk.score,
                        "metadata": chunk.metadata,
                    }
                    for chunk, score in reranked
                ],
            }
        )

    return {
        "schema_version": 1,
        "status": "completed",
        "completed_at": utc_now(),
        "environment": environment(),
        "configuration": {
            "knowledge_base": str(args.kb),
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "top_k": args.top_k,
            "rerank_k": args.rerank_k,
            "vector_index": "faiss.IndexFlatIP with normalized embeddings",
        },
        "document_loading": {
            "files": sorted({item["document_name"] for item in metadata}),
            "chunks": len(texts),
            "metadata_fields": sorted(metadata[0]),
            "indexed_chunks": indexed,
        },
        "timing": {
            "embedding_model_load_seconds": model_load_seconds,
            "reranker_model_load_seconds": reranker_load_seconds,
            "indexing_ms": indexing_ms,
            "mean_retrieval_ms": statistics.fmean(row["retrieval_ms"] for row in rows),
            "mean_rerank_ms": statistics.fmean(row["rerank_ms"] for row in rows),
        },
        "metrics": {
            f"recall@{args.top_k}": statistics.fmean(recalls),
            "mrr": statistics.fmean(reciprocal),
            "expected_evidence_rate": sum(row["expected_evidence_retrieved"] for row in rows)
            / len(rows),
            "citation_preservation_rate": sum(row["citation_metadata_preserved"] for row in rows)
            / len(rows),
            "queries": len(rows),
        },
        "queries": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    timing = payload["timing"]
    recall_key = f"recall@{payload['configuration']['top_k']}"
    lines = [
        "# Real local RAG validation",
        "",
        f"Validated at `{payload['completed_at']}` using the bundled controlled knowledge base.",
        "The run used the real Sentence Transformer embedding, CPU FAISS index, and real cross-encoder reranker; no demo lexical fallback was accepted.",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Documents | {len(payload['document_loading']['files'])} |",
        f"| Chunks indexed | {payload['document_loading']['indexed_chunks']} |",
        f"| Queries | {metrics['queries']} |",
        f"| Recall@{payload['configuration']['top_k']} | {metrics[recall_key]:.4f} |",
        f"| MRR | {metrics['mrr']:.4f} |",
        f"| Expected-evidence rate | {metrics['expected_evidence_rate']:.4f} |",
        f"| Citation-metadata preservation | {metrics['citation_preservation_rate']:.4f} |",
        f"| Mean retrieval time | {timing['mean_retrieval_ms']:.2f} ms |",
        f"| Mean reranking time | {timing['mean_rerank_ms']:.2f} ms |",
        "",
        "## Query results",
        "",
        "| Query | Expected evidence found | Retrieval (ms) | Reranking (ms) |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["queries"]:
        lines.append(
            f"| {row['query']} | {'yes' if row['expected_evidence_retrieved'] else 'no'} | "
            f"{row['retrieval_ms']:.2f} | {row['rerank_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a controlled smoke validation over the small `data/demo_kb` corpus, not a broad-domain retrieval benchmark. Timings include query embedding and model scoring on the recorded machine and should not be generalized to production hardware.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, default=Path("data/demo_kb"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rerank-k", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("ml/benchmarks/rag_validation.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/RAG_VALIDATION.md"))
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    write_json(args.output, payload)
    write_markdown(args.markdown, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
