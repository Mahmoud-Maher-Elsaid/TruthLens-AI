# Deployment

## Vercel web application

1. Import the repository into Vercel.
2. Set **Root Directory** to `apps/web`.
3. Keep the standard Next.js build command (`npm run build`).
4. Set `NEXT_PUBLIC_API_BASE_URL` to the public HTTPS API origin. The backend's `TRUTHLENS_MODE` determines real or Demo operation.
5. Deploy a preview, test Analyze/Compare/Benchmarks, then promote only with project-owner approval.

## Modal FastAPI service

`services/ai-api/modal_app.py` deploys the real production baseline, not Demo mode. Its image installs `requirements-ml.txt`, includes the API package and compact benchmark summaries, and uses an L4 for the configured public models:

- embedding: `sentence-transformers/all-MiniLM-L6-v2`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- verifier: `cross-encoder/nli-deberta-v3-small`

The Modal Volume `truthlens-huggingface-cache` retains model cache files between cold starts. The current FAISS store is in memory, so documents uploaded through `/api/v1/documents` are available only to the active container and are not durable across cold starts. Use the configured Qdrant backend for durable shared retrieval when that product requirement is enabled.

Before redeploying after Vercel is available, set `TRUTHLENS_CORS_ORIGINS` to the exact HTTPS Vercel origin. The deployment default remains the narrow local origin `http://localhost:3000`; it never uses a wildcard. Public model downloads do not require a Hugging Face token. Configure Modal or Qdrant credentials only through their respective secret/environment mechanisms, never source control.

## Container / other Python host

Build from the repository root: `docker build -f docker/backend.Dockerfile -t truthlens-api .`. Run with an environment file and bind port 8000. `docker compose up api` starts Demo mode; `docker compose --profile qdrant up` also starts Qdrant.

## ngrok for experiments

Start FastAPI locally, install/authenticate ngrok, then run `./scripts/start-ngrok.ps1`. Use the temporary HTTPS URL only for development, notebooks, and demonstrations—not as the recommended production route.

Required manual credentials depend on the selected path: Vercel login/project access, Modal login, optional Hugging Face token for gated models, and Qdrant URL/API key for managed retrieval.
