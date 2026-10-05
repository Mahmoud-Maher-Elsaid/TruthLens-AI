import asyncio
import os

import pytest

from app.core.model_cache import configure_model_cache, repository_root, resolve_model_cache
from app.models.schemas import ClaimResult, EvidenceStrength, Verdict
from app.pipeline.extraction import CLAIM_OUTPUT_PARSER, ClaimExtractor
from app.pipeline.orchestrator import AuditOrchestrator
from app.pipeline.reranking import deduplicate_chunks
from app.providers.huggingface import BaselineNLIProvider
from app.retrieval.base import EvidenceChunk
from app.retrieval.store import LexicalDemoStore


def test_model_cache_overrides_inherited_locations_and_stays_in_repository(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "foreign-cache"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "deprecated-cache"))
    paths = configure_model_cache()
    assert paths.root == repository_root() / ".cache" / "huggingface"
    assert str(paths.hub).startswith(str(paths.root))
    assert str(paths.sentence_transformers).startswith(str(paths.root))
    assert os.environ["HF_HOME"] == str(paths.root)
    assert os.environ["HF_HUB_CACHE"] == str(paths.hub)
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(paths.hub)
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(paths.sentence_transformers)


def test_model_cache_rejects_locations_outside_the_repository(tmp_path):
    with pytest.raises(ValueError, match="inside the TruthLens repository"):
        resolve_model_cache(tmp_path)


def test_claim_extractor_ignores_question_and_greeting():
    claims = ClaimExtractor().extract("Hello there. Paris is in France. Is Rome in Italy?")
    assert claims == ["Paris is in France."]
    assert "claims" in CLAIM_OUTPUT_PARSER.get_format_instructions()


async def test_lexical_retrieval_returns_best_match():
    store = LexicalDemoStore()
    await store.add(["Paris is in France", "Tokyo is in Japan"], [{}, {}])
    assert (await store.search("Paris France", 1))[0].text == "Paris is in France"


def _provider_with_scores(scores_by_text: dict[str, dict[str, float]]) -> BaselineNLIProvider:
    provider = object.__new__(BaselineNLIProvider)
    provider.name = "stub-nli"

    def classify(payload):
        scores = scores_by_text[payload["text"]]
        return [{"label": label, "score": score} for label, score in scores.items()]

    provider._classifier = classify
    return provider


def _chunk(text: str, score: float, document: str = "reference.md") -> EvidenceChunk:
    return EvidenceChunk(text[:12], text, {"document_name": document}, score)


def test_multi_passage_relationships_and_conflict_are_preserved():
    provider = _provider_with_scores(
        {
            "The tower was completed in 1889.": {"LABEL_2": 0.91, "LABEL_0": 0.04, "LABEL_1": 0.05},
            "The tower was not built in 1920.": {"LABEL_2": 0.12, "LABEL_0": 0.10, "LABEL_1": 0.78},
        }
    )
    result = provider.verify(
        "claim-1",
        "The Eiffel Tower was built in 1920.",
        "The Eiffel Tower was built in 1920",
        [
            (_chunk("The tower was completed in 1889.", 0.70), 8.0),
            (_chunk("The tower was not built in 1920.", 0.58), 7.0),
        ],
    )
    assert result.verdict == Verdict.CONTRADICTED
    assert result.conflicting_evidence is True
    assert [item.relationship for item in result.evidence] == ["supports", "contradicts"]


def test_explicit_negation_can_resolve_neutral_nli_without_false_promotion():
    provider = _provider_with_scores(
        {"The tower was not built in 1920.": {"LABEL_2": 0.10, "LABEL_0": 0.12, "LABEL_1": 0.78}}
    )
    result = provider.verify(
        "claim-1", "The Eiffel Tower was built in 1920.", "The Eiffel Tower was built in 1920",
        [(_chunk("The tower was not built in 1920.", 0.58), 7.0)],
    )
    assert result.verdict == Verdict.CONTRADICTED
    assert result.evidence[0].relationship == "contradicts"


def test_negation_of_a_different_value_does_not_overturn_support():
    provider = _provider_with_scores(
        {
            "The harbor is located in Delta.": {"LABEL_2": 0.91, "LABEL_0": 0.04, "LABEL_1": 0.05},
            "The harbor is not located in Riverton.": {"LABEL_2": 0.03, "LABEL_0": 0.94, "LABEL_1": 0.03},
        }
    )
    result = provider.verify(
        "claim-1",
        "The harbor is located in Delta.",
        "The harbor is located in Delta",
        [
            (_chunk("The harbor is located in Delta.", 0.80), 9.0),
            (_chunk("The harbor is not located in Riverton.", 0.55), 4.0),
        ],
    )
    assert result.verdict == Verdict.SUPPORTED
    assert result.conflicting_evidence is False
    assert [item.relationship for item in result.evidence] == ["supports", "related"]


def test_later_unrelated_value_does_not_make_negation_targeted():
    provider = _provider_with_scores(
        {
            "The archive is located in Delta Province.": {"LABEL_2": 0.91, "LABEL_0": 0.04, "LABEL_1": 0.05},
            "The archive is not located in Riverton. It is a landmark in Delta Province.": {
                "LABEL_2": 0.03,
                "LABEL_0": 0.94,
                "LABEL_1": 0.03,
            },
        }
    )
    result = provider.verify(
        "claim-1",
        "The archive is located in Delta Province.",
        "The archive is located in Delta Province",
        [
            (_chunk("The archive is located in Delta Province.", 0.80), 9.0),
            (_chunk("The archive is not located in Riverton. It is a landmark in Delta Province.", 0.55), 4.0),
        ],
    )
    assert result.verdict == Verdict.SUPPORTED
    assert result.evidence[1].relationship == "related"


def test_related_neutral_passages_remain_insufficient_evidence():
    provider = _provider_with_scores(
        {"The tower is a cultural icon in France.": {"LABEL_2": 0.10, "LABEL_0": 0.08, "LABEL_1": 0.82}}
    )
    result = provider.verify(
        "claim-1", "The tower was painted blue in 1901.", "The tower was painted blue in 1901",
        [(_chunk("The tower is a cultural icon in France.", 0.55), 5.0)],
    )
    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert result.evidence_strength in {EvidenceStrength.MEDIUM, EvidenceStrength.LOW}
    assert result.evidence[0].relationship == "related"


def test_reranker_deduplicates_same_source_but_keeps_other_sources():
    first = _chunk("Paris is in France and the tower is famous.", 0.9, "reference.md")
    duplicate = _chunk("Paris is in France and the tower is famous.", 0.8, "reference.md")
    near = _chunk("Paris is in France and the tower is famous today.", 0.7, "reference.md")
    other = _chunk("Paris is in France and the tower is famous.", 0.6, "other.md")
    result = deduplicate_chunks([first, duplicate, near, other])
    assert [chunk.metadata["document_name"] for chunk in result] == ["reference.md", "other.md"]


def test_retry_merges_new_evidence_instead_of_replacing_first_passage():
    class Store:
        readiness = "ready"

        def __init__(self):
            self.calls = 0

        async def search(self, query, top_k):
            self.calls += 1
            return [_chunk("The archive mentions the tower.", 0.2)] if self.calls == 1 else [_chunk("The tower was not built in 1920.", 0.7)]

    class Reranker:
        mode = "stub"

        def rerank(self, claim, chunks, limit):
            return [(chunk, chunk.score) for chunk in chunks]

    class Provider:
        name = "stub"

        def __init__(self):
            self.calls = []

        def verify(self, claim_id, claim, normalized, ranked):
            self.calls.append(ranked)
            verdict = Verdict.INSUFFICIENT_EVIDENCE if len(self.calls) == 1 else Verdict.CONTRADICTED
            return ClaimResult(
                id=claim_id, claim=claim, normalized_claim=normalized, verdict=verdict,
                evidence_strength=EvidenceStrength.MEDIUM, evidence=[], explanation="stub",
            )

    orchestrator = object.__new__(AuditOrchestrator)
    orchestrator.settings = type("Settings", (), {"mode": "LOCAL", "top_k": 2, "rerank_k": 2})()
    orchestrator.store = Store()
    orchestrator.reranker = Reranker()
    orchestrator.provider = Provider()
    orchestrator.extractor = ClaimExtractor()
    orchestrator._semaphore = asyncio.Semaphore(1)
    result = asyncio.run(orchestrator._verify_claim(0, "The Eiffel Tower was built in 1920."))
    assert result.verdict == Verdict.CONTRADICTED
    assert len(orchestrator.provider.calls[1]) == 2
