import re

from app.core.model_cache import configure_model_cache
from app.retrieval.base import EvidenceChunk


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def deduplicate_chunks(chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    """Remove exact and near-duplicate passages within the same source.

    Source identity is retained: identical text from different documents remains
    available so the verifier can detect cross-source disagreement.
    """
    unique: list[EvidenceChunk] = []
    seen_exact: set[tuple[str, str]] = set()
    seen_tokens: list[tuple[str, set[str]]] = []
    for chunk in chunks:
        source = str(chunk.metadata.get("document_name", ""))
        exact_key = (source, _normalized_text(chunk.text))
        if exact_key in seen_exact:
            continue
        tokens = _token_set(chunk.text)
        near_duplicate = any(
            previous_source == source
            and len(tokens & previous_tokens) / max(len(tokens | previous_tokens), 1) >= 0.85
            for previous_source, previous_tokens in seen_tokens
        )
        if near_duplicate:
            continue
        seen_exact.add(exact_key)
        seen_tokens.append((source, tokens))
        unique.append(chunk)
    return unique


class EvidenceReranker:
    def __init__(self, model_name: str, enabled: bool = False) -> None:
        self.model_name = model_name
        self.mode = "vector-similarity fallback"
        self._model = None
        if enabled:
            cache = configure_model_cache()
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name, cache_folder=str(cache.sentence_transformers))
                self.mode = f"cross-encoder:{model_name}"
            except (ImportError, OSError):
                self._model = None

    def rerank(self, claim: str, chunks: list[EvidenceChunk], limit: int) -> list[tuple[EvidenceChunk, float]]:
        chunks = deduplicate_chunks(chunks)
        if not chunks:
            return []
        if self._model is None:
            return [(chunk, chunk.score) for chunk in chunks[:limit]]
        scores = self._model.predict([(claim, chunk.text) for chunk in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:limit]]
