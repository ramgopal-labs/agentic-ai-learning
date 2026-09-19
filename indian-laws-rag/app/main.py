import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.models import AskRequest, AskResponse
from app.core.config import is_usable_key, settings
from app.core.logging_config import configure_logging, get_logger, set_request_id
from app.core.security import enforce_rate_limit, require_api_key
from app.generation.llm import ANSWER_MODELS, UTILITY_MODELS, LLMService
from app.pipeline.graph import RAGGraph
from app.pipeline.rag_pipeline import IndianLawsRAG
from app.retrieval.retriever import Retriever

configure_logging()
logger = get_logger(__name__)


class AppState:
    """Everything built once at startup and shared by every request."""

    def __init__(self):
        self.retriever: Retriever | None = None
        self.pipeline: IndianLawsRAG | None = None
        self.ready: bool = False
        self.startup_error: str | None = None


state = AppState()


def build_pipeline() -> None:
    """
    Load the models and open the store connections.

    Done at startup rather than on the first request: the embedding and
    reranking models take tens of seconds to load, and a request should never
    pay that. The readiness probe stays false until this finishes.
    """
    started = time.monotonic()
    logger.info("loading models and opening connections")

    # One Retriever per process: the embedded Qdrant client takes an exclusive
    # lock, so every route must share it.
    state.retriever = Retriever()
    state.pipeline = IndianLawsRAG(graph=RAGGraph(retriever=state.retriever))
    state.ready = True
    state.startup_error = None

    logger.info(
        "ready to serve",
        extra={
            "seconds": round(time.monotonic() - started, 2),
            "qdrant": "server" if settings.uses_qdrant_server else "embedded",
            "memory": "postgres" if settings.uses_postgres else "sqlite",
            "auth": settings.auth_required,
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.warmup_on_startup:
        try:
            build_pipeline()
        except Exception as error:
            # Start anyway and report unready, so a container orchestrator can
            # surface the reason through the probe instead of crash-looping
            # with no diagnosis.
            state.startup_error = str(error)
            logger.exception("startup failed; serving as not-ready")
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Indian Laws RAG API",
    description="Backend API for the Indian Laws RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id, echo it back, and log the outcome."""
    request_id = set_request_id(request.headers.get("X-Request-ID"))
    started = time.monotonic()

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, error: Exception):
    """
    Log the detail, return a generic message.

    Exception text can carry connection strings and internal paths, so it goes
    to the logs rather than to the client; the request id ties the two together.
    """
    logger.exception("unhandled error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error.",
            "request_id": request.headers.get("X-Request-ID") or "-",
        },
    )


def get_pipeline() -> IndianLawsRAG:
    if state.pipeline is None:
        build_pipeline()
    return state.pipeline  # type: ignore[return-value]


def get_retriever() -> Retriever:
    if state.retriever is None:
        build_pipeline()
    return state.retriever  # type: ignore[return-value]


# --- probes ----------------------------------------------------------------


@app.get("/")
def home():
    return {"message": "Indian Laws RAG API is running"}


@app.get("/health")
def health_check():
    """Liveness: is the process up? Never touches a dependency."""
    return {"status": "healthy", "service": settings.service_name}


@app.get("/ready")
def readiness_check():
    """
    Readiness: can this replica actually serve a request?

    Distinct from liveness so an orchestrator restarts a wedged process but
    merely withholds traffic from one that is still warming up.
    """
    checks = {
        "models_loaded": state.ready,
        "vector_store": False,
        "memory": False,
    }

    if state.ready and state.retriever is not None:
        checks["vector_store"] = state.retriever.store.ping()
        checks["memory"] = state.pipeline.graph.memory.ping()

    ready = all(checks.values())
    body = {"ready": ready, "checks": checks}
    if state.startup_error:
        body["startup_error"] = state.startup_error

    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body,
    )


@app.get("/info")
def service_info():
    """Configuration a operator needs, with no secret values."""
    try:
        indexed_chunks = get_retriever().store.count()
    except Exception:
        logger.warning("could not read the indexed chunk count", exc_info=True)
        indexed_chunks = None

    return {
        "service": settings.service_name,
        "indexed_chunks": indexed_chunks,
        "providers_configured": LLMService().available_providers(),
        "providers_supported": list(ANSWER_MODELS),
        "models": {"answer": ANSWER_MODELS, "utility": UTILITY_MODELS},
        "web_fallback_configured": is_usable_key(settings.serper_api_key),
        "confidence_threshold": settings.confidence_threshold,
        "vector_store": "server" if settings.uses_qdrant_server else "embedded",
        "memory_backend": "postgres" if settings.uses_postgres else "sqlite",
        "auth_required": settings.auth_required,
        "rate_limit_per_minute": (
            settings.rate_limit_per_minute if settings.rate_limit_enabled else None
        ),
    }


# --- application endpoints --------------------------------------------------


@app.get("/acts", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
def list_acts() -> dict:
    """Distinct Act titles in the index, for the UI's filter dropdown."""
    return {"acts": get_retriever().list_act_titles()}


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def ask_question(request: AskRequest):
    if not state.ready and state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is still starting up. Retry shortly.",
        )

    logger.info(
        "answering question",
        extra={"session_id": request.session_id, "provider": request.provider},
    )
    return get_pipeline().answer(request)
