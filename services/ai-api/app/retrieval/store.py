import hashlib
import re
from typing import Any

from app.core.model_cache import configure_model_cache
from app.retrieval.base import EvidenceChunk, VectorStore


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class LexicalDemoStore(VectorStore):
    """Dependency-free demo store. LOCAL mode can switch to the FAISS adapter."""

    def __init__(self) -> None:
        self._chunks: list[EvidenceChunk] = []

    async def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> int:
        for text, item_metadata in zip(texts, metadata, strict=True):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            self._chunks.append(EvidenceChunk(digest, text, item_metadata, 0.0))
        return len(texts)

    async def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        query_tokens = _tokens(query)
        ranked: list[EvidenceChunk] = []
        for chunk in self._chunks:
            chunk_tokens = _tokens(chunk.text)
            union = query_tokens | chunk_tokens
            score = len(query_tokens & chunk_tokens) / max(len(union), 1)
            ranked.append(EvidenceChunk(chunk.id, chunk.text, chunk.metadata, score))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]

    @property
    def readiness(self) -> str:
        return "ready (deterministic lexical demo index)"


class FaissStore(VectorStore):
    """Lazy FAISS adapter so CPU demo installations do not download model weights."""

    def __init__(self, model_name: str) -> None:
        cache = configure_model_cache()
        try:
            import faiss  # type: ignore
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to enable FAISS retrieval") from exc
        self._faiss = faiss
        self._np = np
        self._model = SentenceTransformer(model_name, cache_folder=str(cache.sentence_transformers))
        self._index: Any | None = None
        self._items: list[tuple[str, dict[str, Any], str]] = []

    async def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> int:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(self._np.asarray(vectors, dtype="float32"))
        for text, item_metadata in zip(texts, metadata, strict=True):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            self._items.append((text, item_metadata, digest))
        return len(texts)

    async def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if self._index is None or not self._items:
            return []
        vector = self._model.encode([query], normalize_embeddings=True)
        scores, indexes = self._index.search(self._np.asarray(vector, dtype="float32"), top_k)
        results: list[EvidenceChunk] = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if index < 0:
                continue
            text, metadata, digest = self._items[int(index)]
            results.append(EvidenceChunk(digest, text, metadata, float(score)))
        return results

    @property
    def readiness(self) -> str:
        return "ready" if self._index is not None else "ready (empty index)"


class QdrantStore(VectorStore):
    """Production Qdrant adapter using the same Sentence Transformer embeddings as FAISS."""

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        collection: str,
        model_name: str,
    ) -> None:
        if not url:
            raise RuntimeError("TRUTHLENS_QDRANT_URL is required for Qdrant")
        cache = configure_model_cache()
        try:
            from qdrant_client import AsyncQdrantClient
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to enable Qdrant") from exc
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self._model = SentenceTransformer(model_name, cache_folder=str(cache.sentence_transformers))

    async def _ensure_collection(self, size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )

    async def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> int:
        from qdrant_client.models import PointStruct

        vectors = self._model.encode(texts, normalize_embeddings=True).tolist()
        await self._ensure_collection(len(vectors[0]))
        points = []
        for text, item_metadata, vector in zip(texts, metadata, vectors, strict=True):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            points.append(
                PointStruct(
                    id=int(digest, 16),
                    vector=vector,
                    payload={"text": text, "chunk_id": digest, **item_metadata},
                )
            )
        await self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    async def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if not await self.client.collection_exists(self.collection):
            return []
        vector = self._model.encode(query, normalize_embeddings=True).tolist()
        result = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            EvidenceChunk(
                id=str(point.payload.get("chunk_id", point.id)),
                text=str(point.payload.get("text", "")),
                metadata={
                    key: value
                    for key, value in point.payload.items()
                    if key not in {"text", "chunk_id"}
                },
                score=float(point.score),
            )
            for point in result.points
            if point.payload
        ]

    @property
    def readiness(self) -> str:
        return "configured (lazy connection)"
