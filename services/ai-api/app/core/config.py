from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRUTHLENS_", env_file=".env", extra="ignore", case_sensitive=False
    )

    mode: Literal["DEMO", "LOCAL", "PRODUCTION"] = "DEMO"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"
    max_input_chars: int = Field(50_000, ge=100, le=250_000)
    max_upload_bytes: int = Field(10_485_760, ge=1024, le=25_000_000)
    request_timeout_seconds: int = Field(120, ge=5, le=600)
    model_cache: Path = Path(".cache/huggingface")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    verifier_model: str = "cross-encoder/nli-deberta-v3-small"
    # DEMO mode always selects the deterministic demo provider in the orchestrator.
    # LOCAL/PRODUCTION default to the measured pretrained verifier; fine-tuned
    # adapters remain opt-in research artifacts.
    verifier_provider: Literal["demo", "baseline", "finetuned"] = "baseline"
    finetuned_adapter: str | None = None
    vector_backend: Literal["faiss", "qdrant"] = "faiss"
    faiss_path: Path = Path("./data/indexes/faiss")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "truthlens-evidence"
    chunk_size: int = Field(700, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=1000)
    top_k: int = Field(8, ge=1, le=50)
    rerank_k: int = Field(4, ge=1, le=20)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        return value.upper()

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
