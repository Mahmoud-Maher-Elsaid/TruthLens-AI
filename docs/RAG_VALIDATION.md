# Real local RAG validation

Validated at `2026-10-01T06:27:31.764204+00:00` using the bundled controlled knowledge base.
The run used the real Sentence Transformer embedding, CPU FAISS index, and real cross-encoder reranker; no demo lexical fallback was accepted.

| Measurement | Result |
|---|---:|
| Documents | 1 |
| Chunks indexed | 2 |
| Queries | 5 |
| Recall@3 | 1.0000 |
| MRR | 0.8000 |
| Expected-evidence rate | 1.0000 |
| Citation-metadata preservation | 1.0000 |
| Mean retrieval time | 15.26 ms |
| Mean reranking time | 8.52 ms |

## Query results

| Query | Expected evidence found | Retrieval (ms) | Reranking (ms) |
|---|---:|---:|---:|
| Where is the Eiffel Tower located? | yes | 52.11 | 16.57 |
| When was the Eiffel Tower constructed? | yes | 7.49 | 7.30 |
| How tall is the Eiffel Tower with its current antenna? | yes | 5.70 | 5.36 |
| Was the Eiffel Tower built in 1920? | yes | 5.15 | 7.61 |
| Is the Eiffel Tower located in Rome? | yes | 5.84 | 5.75 |

## Scope

This is a controlled smoke validation over the small `data/demo_kb` corpus, not a broad-domain retrieval benchmark. Timings include query embedding and model scoring on the recorded machine and should not be generalized to production hardware.

## Multi-evidence policy

The LOCAL verifier now evaluates each retained reranked passage separately. Exact and near-duplicate passages from the same source are removed; passages from different sources remain so disagreement is observable. Each citation keeps its own relationship. A passage is meaningful only when its NLI score and retrieval/reranking relevance clear the configured internal strength bands. A clear explicit negation tied to the claim can resolve a neutral NLI output as `CONTRADICTED`; related neutral text remains `INSUFFICIENT_EVIDENCE`. When meaningful support and contradiction coexist, the report sets `conflicting_evidence=true` and uses a conservative verdict: explicit or sufficiently strong contradiction takes precedence, otherwise the final verdict is `INSUFFICIENT_EVIDENCE`. These are deterministic aggregation rules, not calibrated probabilities.
