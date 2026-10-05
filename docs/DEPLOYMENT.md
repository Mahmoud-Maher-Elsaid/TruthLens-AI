# Deployment

## Vercel web application

1. Import the repository into Vercel.
2. Set **Root Directory** to `apps/web`.
3. Keep the standard Next.js build command (`npm run build`).
4. Set `NEXT_PUBLIC_API_BASE_URL` to the public HTTPS API origin. The backend's `TRUTHLENS_MODE` determines real or Demo operation.
5. Deploy a preview, test Analyze/Compare/Benchmarks, then promote only with project-owner approval.

## Modal FastAPI service

The repository includes a Modal ASGI entry point, but no live deployment is verified. Its current image installs the core API requirements and defaults to deterministic Demo mode. Real GPU model serving requires a separately configured image, GPU allocation, model dependencies, retrieval service, and measured validation before deployment. Set secrets through Modal, never source control.

## Container / other Python host

Build from the repository root: `docker build -f docker/backend.Dockerfile -t truthlens-api .`. Run with an environment file and bind port 8000. `docker compose up api` starts Demo mode; `docker compose --profile qdrant up` also starts Qdrant.

## ngrok for experiments

Start FastAPI locally, install/authenticate ngrok, then run `./scripts/start-ngrok.ps1`. Use the temporary HTTPS URL only for development, notebooks, and demonstrations—not as the recommended production route.

Required manual credentials depend on the selected path: Vercel login/project access, Modal login, optional Hugging Face token for gated models, and Qdrant URL/API key for managed retrieval.
