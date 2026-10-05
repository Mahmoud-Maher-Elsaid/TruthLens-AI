import math
import re
from dataclasses import dataclass

from app.core.model_cache import configure_model_cache
from app.models.schemas import (
    ClaimResult,
    EvidenceItem,
    EvidenceStrength,
    SourceMetadata,
    Verdict,
)
from app.retrieval.base import EvidenceChunk


@dataclass(frozen=True, slots=True)
class EvidenceAggregationPolicy:
    """Explicit, testable aggregation thresholds kept outside the verdict loop."""

    explicit_contradiction_weight: float = 0.35
    contradiction_support_ratio: float = 0.75
    high_aggregate_score: float = 0.45
    high_probability: float = 0.8
    high_relevance: float = 0.25
    medium_probability: float = 0.55
    medium_relevance: float = 0.15


DEFAULT_AGGREGATION_POLICY = EvidenceAggregationPolicy()

_NEGATION_PATTERN = re.compile(r"\b(?:not|never|incorrectly|rather than)\b")
_PREDICATE_TOKENS = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had"}
_RELATION_TOKENS = {
    "a", "an", "and", "as", "at", "by", "completed", "constructed", "from", "in", "is",
    "located", "of", "on", "opened", "painted", "the", "to", "was", "were", "with",
    "built", "tall",
}


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    chunk: EvidenceChunk
    rerank_score: float
    verdict: Verdict
    probability: float
    relevance: float
    strength: EvidenceStrength
    explicit_contradiction: bool = False

    def weighted_score(self, policy: EvidenceAggregationPolicy = DEFAULT_AGGREGATION_POLICY) -> float:
        # This is an internal ordering score, not a user-facing confidence.
        return self.probability * self.relevance + (
            policy.explicit_contradiction_weight if self.explicit_contradiction else 0.0
        )


def _evidence_items(assessments: list[EvidenceAssessment]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"evidence-{assessment.chunk.id}",
            text=assessment.chunk.text,
            metadata=SourceMetadata(
                document_name=str(assessment.chunk.metadata.get("document_name", "Indexed evidence")),
                document_id=assessment.chunk.metadata.get("document_id"),
                source_url=assessment.chunk.metadata.get("source_url"),
                section=assessment.chunk.metadata.get("section"),
                page=assessment.chunk.metadata.get("page"),
                chunk_id=assessment.chunk.id,
            ),
            retrieval_score=max(-1.0, min(1.0, assessment.chunk.score)),
            rerank_score=max(-20.0, min(20.0, assessment.rerank_score)),
            relationship={
                Verdict.SUPPORTED: "supports",
                Verdict.CONTRADICTED: "contradicts",
                Verdict.INSUFFICIENT_EVIDENCE: "related",
            }[assessment.verdict],
        )
        for assessment in assessments
    ]


def _claim_target_tokens(claim: str) -> set[str]:
    """Return the asserted value, excluding the subject and relation wording."""
    tokens = re.findall(r"[a-z0-9]+", claim.lower())
    try:
        predicate_index = next(index for index, token in enumerate(tokens) if token in _PREDICATE_TOKENS)
    except StopIteration:
        predicate_index = -1
    values = tokens[predicate_index + 1 :] if predicate_index >= 0 else tokens
    return set(values) - _RELATION_TOKENS


def _has_explicit_negation(evidence: str) -> bool:
    return bool(_NEGATION_PATTERN.search(evidence.lower()))


def _explicit_contradiction(claim: str, evidence: str) -> bool:
    """Recognize negation only when it names a value asserted by the claim.

    Shared subjects and relations alone are insufficient: “not located in Rome”
    does not contradict a claim that something is located in Paris.
    """
    targets = _claim_target_tokens(claim)
    if not targets:
        return False
    for match in _NEGATION_PATTERN.finditer(evidence.lower()):
        # Limit matching to the negated clause. A later, unrelated mention of
        # a claim value must not turn “not located in Rome” into a denial of
        # “located in Paris, France.”
        local_tokens = re.findall(r"[a-z0-9]+", evidence[match.end() :].lower())[:8]
        if targets & set(local_tokens):
            return True
    return False


def _relevance(chunk: EvidenceChunk, rerank_score: float) -> float:
    retrieval = max(0.0, min(1.0, chunk.score))
    rerank = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, rerank_score))))
    return max(0.05, 0.5 * retrieval + 0.5 * rerank)


def _strength(
    probability: float,
    relevance: float,
    explicit: bool,
    policy: EvidenceAggregationPolicy = DEFAULT_AGGREGATION_POLICY,
) -> EvidenceStrength:
    if explicit and relevance >= policy.high_relevance:
        return EvidenceStrength.HIGH
    if probability >= policy.high_probability and relevance >= policy.high_relevance:
        return EvidenceStrength.HIGH
    if probability >= policy.medium_probability and relevance >= policy.medium_relevance:
        return EvidenceStrength.MEDIUM
    return EvidenceStrength.LOW


class BaselineNLIProvider:
    """Pretrained NLI verifier with conservative multi-passage aggregation."""

    def __init__(
        self,
        model_name: str,
        aggregation_policy: EvidenceAggregationPolicy = DEFAULT_AGGREGATION_POLICY,
    ) -> None:
        configure_model_cache()
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to enable baseline verification") from exc
        self.name = f"huggingface-nli:{model_name}"
        self.aggregation_policy = aggregation_policy
        self._classifier = pipeline("text-classification", model=model_name, top_k=None)

    def verify(
        self,
        claim_id: str,
        claim: str,
        normalized: str,
        ranked: list[tuple[EvidenceChunk, float]],
    ) -> ClaimResult:
        if not ranked:
            return ClaimResult(
                id=claim_id,
                claim=claim,
                normalized_claim=normalized,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                evidence_strength=EvidenceStrength.LOW,
                evidence=[],
                explanation="No evidence passage was retrieved for this claim.",
            )
        policy = getattr(self, "aggregation_policy", DEFAULT_AGGREGATION_POLICY)
        assessments: list[EvidenceAssessment] = []
        for passage, rerank_score in ranked:
            raw_scores = self._classifier({"text": passage.text, "text_pair": normalized})
            scores = raw_scores[0] if raw_scores and isinstance(raw_scores[0], list) else raw_scores
            mapped: dict[str, float] = {}
            for item in scores:
                label = str(item["label"]).lower()
                if "entail" in label or label == "label_2":
                    mapped["entailment"] = float(item["score"])
                elif "contrad" in label or label == "label_0":
                    mapped["contradiction"] = float(item["score"])
                else:
                    mapped["neutral"] = float(item["score"])
            winning = max(mapped, key=mapped.get)
            probability = mapped[winning]
            explicit = _explicit_contradiction(normalized, passage.text)
            verdict = {
                "entailment": Verdict.SUPPORTED,
                "contradiction": Verdict.CONTRADICTED,
                "neutral": Verdict.INSUFFICIENT_EVIDENCE,
            }[winning]
            # A negated statement about a different asserted value is not a
            # contradiction. This prevents an NLI false positive such as
            # “not located in Rome” from overturning “located in Paris.”
            if verdict == Verdict.CONTRADICTED and _has_explicit_negation(passage.text) and not explicit:
                verdict = Verdict.INSUFFICIENT_EVIDENCE
            # A neutral NLI result is promoted only for a directly matched,
            # explicit negation. This preserves insufficient evidence for
            # merely related passages and fixes the observed 1920 failure.
            if verdict == Verdict.INSUFFICIENT_EVIDENCE and explicit:
                verdict = Verdict.CONTRADICTED
            relevance = _relevance(passage, float(rerank_score))
            assessments.append(
                EvidenceAssessment(
                    chunk=passage,
                    rerank_score=float(rerank_score),
                    verdict=verdict,
                    probability=probability,
                    relevance=relevance,
                    strength=_strength(probability, relevance, explicit, policy),
                    explicit_contradiction=explicit,
                )
            )
        support = [item for item in assessments if item.verdict == Verdict.SUPPORTED and item.strength != EvidenceStrength.LOW]
        contradiction = [item for item in assessments if item.verdict == Verdict.CONTRADICTED and item.strength != EvidenceStrength.LOW]
        support_score = sum(item.weighted_score(policy) for item in support)
        contradiction_score = sum(item.weighted_score(policy) for item in contradiction)
        conflict = bool(support and contradiction)
        if contradiction and (
            not support
            or any(item.explicit_contradiction for item in contradiction)
            or contradiction_score >= support_score * policy.contradiction_support_ratio
        ):
            verdict = Verdict.CONTRADICTED
        elif support and not contradiction:
            verdict = Verdict.SUPPORTED
        elif support and contradiction:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
        else:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
        final_strength = (
            EvidenceStrength.HIGH
            if verdict in {Verdict.SUPPORTED, Verdict.CONTRADICTED}
            and max(support_score, contradiction_score) >= policy.high_aggregate_score
            else EvidenceStrength.MEDIUM if assessments else EvidenceStrength.LOW
        )
        relationships = ", ".join(
            f"{item.chunk.metadata.get('document_name', 'passage')}: {item.verdict.value}"
            for item in assessments
        )
        return ClaimResult(
            id=claim_id,
            claim=claim,
            normalized_claim=normalized,
            verdict=verdict,
            evidence_strength=final_strength,
            evidence=_evidence_items(assessments),
            conflicting_evidence=conflict,
            explanation=(
                f"Multi-passage NLI assessment: {relationships}. "
                + ("Supporting and contradicting evidence coexist; the final verdict is conservative. " if conflict else "")
                + "Evidence strength combines verifier, retrieval, and reranking signals."
            ),
        )


class FinetunedGenerativeProvider:
    """Loads the causal-LM PEFT adapter produced by ml/training/train_qlora.py."""

    def __init__(self, adapter_path: str) -> None:
        if not adapter_path:
            raise RuntimeError("TRUTHLENS_FINETUNED_ADAPTER is required for finetuned mode")
        configure_model_cache()
        try:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-gpu.txt to enable the fine-tuned verifier") from exc
        self.name = f"peft-adapter:{adapter_path}"
        self._model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto")
        self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    def verify(
        self,
        claim_id: str,
        claim: str,
        normalized: str,
        ranked: list[tuple[EvidenceChunk, float]],
    ) -> ClaimResult:
        if not ranked:
            label = Verdict.INSUFFICIENT_EVIDENCE
        else:
            prompt = (
                "Return exactly SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\n"
                f"Claim: {normalized}\nEvidence: {ranked[0][0].text}\nLabel:"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            output = self._model.generate(**inputs, max_new_tokens=8, do_sample=False)
            generated = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).upper()
            label = next(
                (item for item in Verdict if item.value in generated),
                Verdict.INSUFFICIENT_EVIDENCE,
            )
        assessments = [
            EvidenceAssessment(
                chunk=chunk,
                rerank_score=float(score),
                verdict=label,
                probability=1.0 if label != Verdict.INSUFFICIENT_EVIDENCE else 0.0,
                relevance=_relevance(chunk, float(score)),
                strength=EvidenceStrength.MEDIUM if ranked else EvidenceStrength.LOW,
            )
            for chunk, score in ranked[:2]
        ]
        return ClaimResult(
            id=claim_id,
            claim=claim,
            normalized_claim=normalized,
            verdict=label,
            evidence_strength=EvidenceStrength.MEDIUM if ranked else EvidenceStrength.LOW,
            evidence=_evidence_items(assessments),
            explanation="The configured PEFT adapter classified the claim against the top-ranked evidence.",
        )
