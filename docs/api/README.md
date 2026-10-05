# API

Base path: `/api/v1`. `GET /health` is also available at the root.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health`, `/api/v1/health` | Mode and model/vector readiness |
| POST | `/api/v1/analyze` | Strict non-streaming audit |
| POST | `/api/v1/analyze/stream` | NDJSON progress and final report |
| POST | `/api/v1/documents` | PDF/TXT/Markdown ingestion into the active shared evidence index |
| POST | `/api/v1/compare` | Illustrative variants; not an independent three-system benchmark |
| GET | `/api/v1/benchmarks` | Real JSON artifacts or honest empty state |

Streaming emits `validation_started`, `claims_extracted`, `retrieval_started`, `claim_verified`, then `analysis_complete` or `analysis_failed`. Events announce completed work only. Pydantic forbids unexpected request fields. HTTP errors use `{ "error": { "code", "message", "request_id" } }`; streaming failures are terminal NDJSON events. The request ID is also returned in the `X-Request-ID` response header. The configured timeout applies to cooperative async work; synchronous model inference may delay cancellation.

An upload must return HTTP 201 with a validated `document_id` and positive `chunks_indexed` before the UI starts analysis. `document_id` is provenance metadata; retrieval currently searches the shared index and does not isolate documents by ID. In-memory FAISS documents disappear on service restart. The health endpoint reports `degraded` for unavailable Qdrant, an empty FAISS index, or a model fallback; Demo mode reports its deterministic readiness explicitly.

`AnalyzeRequest.mode` currently does not select a distinct backend algorithm. The service's `TRUTHLENS_MODE` controls Demo versus real providers. The web Demo Example is a fixed local fixture; Standard and Deep currently call the same backend audit path.

When multiple passages are assessed, each citation keeps its own `supports`, `contradicts`, or `related` relationship. `conflicting_evidence` is true when meaningful support and contradiction coexist; the final verdict remains conservative and never treats missing evidence as false.
