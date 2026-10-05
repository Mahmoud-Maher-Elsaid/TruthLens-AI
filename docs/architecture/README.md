# Architecture

TruthLens separates the Next.js interface from a FastAPI service. The browser submits strict JSON or multipart documents and consumes newline-delimited progress events. FastAPI owns validation and safe errors. `AuditOrchestrator` uses a four-way claim concurrency limit and at most one retrieval retry outside Demo mode. A LangGraph topology is defined, but its current pass-through nodes do not execute the audit pipeline.

`VectorStore` is the shared boundary. Demo mode uses an in-memory lexical index so it needs no downloads. LOCAL selects a lazy FAISS/Sentence Transformer implementation. PRODUCTION may select Qdrant, using the same configurable embedding model for collection creation, upsert, and querying. Evidence metadata survives retrieval into claim citations.

The verifier provider contract includes deterministic Demo, pretrained NLI baseline, and opt-in fine-tuned adapter providers. Demo mode does not load real models. LOCAL/PRODUCTION load the configured models when the service starts. See `docs/ML_PIPELINE.md` for the training boundary.
