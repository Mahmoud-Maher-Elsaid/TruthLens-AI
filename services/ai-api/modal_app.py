"""Modal ASGI deployment for the real TruthLens production baseline."""

import os

import modal

REPOSITORY_ROOT = "/root/truthlens"
API_ROOT = f"{REPOSITORY_ROOT}/services/ai-api"
MODEL_CACHE_ROOT = f"{REPOSITORY_ROOT}/.cache/huggingface"

# requirements-ml.txt includes the core API plus the measured production
# embedding, reranking, FAISS, and NLI inference dependencies.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(
        "services/ai-api/requirements.txt",
        f"{API_ROOT}/requirements.txt",
        copy=True,
    )
    .add_local_file(
        "services/ai-api/requirements-ml.txt",
        f"{API_ROOT}/requirements-ml.txt",
        copy=True,
    )
    .run_commands(f"cd {API_ROOT} && python -m pip install -r requirements-ml.txt")
    .add_local_dir("services/ai-api/app", f"{API_ROOT}/app", copy=True)
    .add_local_dir("ml/benchmarks", f"{REPOSITORY_ROOT}/ml/benchmarks", copy=True)
)

# This Volume caches public Hugging Face models between cold starts. It stores
# no application data and remains mounted inside the TruthLens repository path.
model_cache_volume = modal.Volume.from_name("truthlens-huggingface-cache", create_if_missing=True)

# Set TRUTHLENS_CORS_ORIGINS to the exact Vercel origin before a production
# redeploy. The localhost default is deliberately narrow rather than wildcard.
runtime_environment = {
    "TRUTHLENS_MODE": "PRODUCTION",
    "TRUTHLENS_VERIFIER_PROVIDER": "baseline",
    "TRUTHLENS_VERIFIER_MODEL": "cross-encoder/nli-deberta-v3-small",
    "TRUTHLENS_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "TRUTHLENS_RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "TRUTHLENS_VECTOR_BACKEND": "faiss",
    "TRUTHLENS_MODEL_CACHE": MODEL_CACHE_ROOT,
    "TRUTHLENS_CORS_ORIGINS": os.getenv("TRUTHLENS_CORS_ORIGINS", "http://localhost:3000"),
}

app = modal.App("truthlens-ai-api")


@app.function(
    image=image,
    gpu="L4",
    timeout=300,
    min_containers=0,
    env=runtime_environment,
    volumes={MODEL_CACHE_ROOT: model_cache_volume},
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, API_ROOT)
    from app.main import app as fastapi_application

    return fastapi_application
