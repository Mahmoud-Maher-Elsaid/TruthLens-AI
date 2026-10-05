import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.model_cache import configure_model_cache
from app.models.schemas import ErrorDetail, ErrorResponse
from app.pipeline.orchestrator import AuditOrchestrator
from app.retrieval.store import FaissStore, LexicalDemoStore, QdrantStore

settings = get_settings()
configure_model_cache(settings.model_cache)
configure_logging(settings.log_level)
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.mode == "DEMO":
        store = LexicalDemoStore()
    elif settings.vector_backend == "qdrant":
        store = QdrantStore(
            settings.qdrant_url,
            settings.qdrant_api_key,
            settings.qdrant_collection,
            settings.embedding_model,
        )
    else:
        store = FaissStore(settings.embedding_model)
    orchestrator = AuditOrchestrator(settings, store)
    await orchestrator.initialize()
    app.state.settings, app.state.store, app.state.orchestrator = settings, store, orchestrator
    app.state.repo_root = Path(__file__).resolve().parents[3]
    log.info("service_started", mode=settings.mode, vector_backend=settings.vector_backend)
    yield
    log.info("service_stopped")


app = FastAPI(title="TruthLens AI API", version=__version__, description="Claim-level AI response auditing and evidence verification.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:100]
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers.update({"X-Request-ID": request_id, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "strict-origin-when-cross-origin"})
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    payload = ErrorResponse(error=ErrorDetail(code="VALIDATION_ERROR", message="The request is invalid.", request_id=request.state.request_id, details=[{"location": list(error["loc"]), "message": error["msg"]} for error in exc.errors()]))
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    code = "REQUEST_TIMEOUT" if exc.status_code == 504 else "HTTP_ERROR"
    payload = ErrorResponse(error=ErrorDetail(code=code, message=str(exc.detail), request_id=request.state.request_id))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.exception("unhandled_error", path=request.url.path, error_type=type(exc).__name__)
    payload = ErrorResponse(error=ErrorDetail(code="INTERNAL_ERROR", message="The service could not complete the request.", request_id=request.state.request_id))
    return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))


app.include_router(router)
app.include_router(router, prefix="/api/v1")
