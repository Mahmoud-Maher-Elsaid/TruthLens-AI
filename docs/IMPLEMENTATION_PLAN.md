# Implementation plan

This is the original implementation sequence, not a current completion checklist. Current measured results and limitations are in `README.md` and `ml/benchmarks/README.md`; automation and conflict-aware verification remain pending.

1. Establish the isolated repository, environment contracts, safety rules, and reproducible tooling.
2. Build the responsive Next.js App Router product shell and forensic design system.
3. Implement FastAPI schemas, safe errors, deterministic demo mode, and versioned endpoints.
4. Add bounded claim extraction, FAISS/Qdrant retrieval adapters, reranking, verification, and orchestration.
5. Connect the Analyze workspace to real streamed backend events and render claim reports and an evidence graph.
6. Add Compare, Benchmarks, Architecture, About, ML training/evaluation/quantization utilities, deployment assets, and documentation.
7. Run automated checks, integrated smoke tests, secret/build-artifact review, and reconcile documentation with observed results.

External model downloads, GPU training, hosted deployment, and authenticated publishing are deliberately separate manual steps.
