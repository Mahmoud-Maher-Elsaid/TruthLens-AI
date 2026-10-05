"""Project-owned cache locations for Hugging Face model libraries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelCachePaths:
    root: Path
    hub: Path
    sentence_transformers: Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_model_cache(cache_root: str | Path | None = None) -> ModelCachePaths:
    """Resolve a cache path and reject locations outside this repository."""
    root = repository_root()
    configured = Path(cache_root or os.getenv("TRUTHLENS_MODEL_CACHE", ".cache/huggingface"))
    candidate = configured if configured.is_absolute() else root / configured
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("TRUTHLENS_MODEL_CACHE must remain inside the TruthLens repository.") from exc
    return ModelCachePaths(
        root=resolved,
        hub=resolved / "hub",
        sentence_transformers=resolved / "sentence-transformers",
    )


def configure_model_cache(cache_root: str | Path | None = None) -> ModelCachePaths:
    """Force all supported Hugging Face libraries into the project cache."""
    paths = resolve_model_cache(cache_root)
    for path in (paths.root, paths.hub, paths.sentence_transformers):
        path.mkdir(parents=True, exist_ok=True)
    values = {
        "TRUTHLENS_MODEL_CACHE": str(paths.root),
        "HF_HOME": str(paths.root),
        "HF_HUB_CACHE": str(paths.hub),
        "HUGGINGFACE_HUB_CACHE": str(paths.hub),
        "SENTENCE_TRANSFORMERS_HOME": str(paths.sentence_transformers),
    }
    os.environ.update(values)
    # Deprecated by Transformers; supported Hugging Face cache variables above
    # remain authoritative even when the parent shell inherited this setting.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return paths
