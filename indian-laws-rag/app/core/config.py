"""
Settings, resolved once from the environment.

Every backend choice is driven by an environment variable, so the same image
runs unchanged on a laptop and in a cluster: no code path branches on "is this
production". Unset means the local-development default.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project folder: indian-laws-rag/. Three levels up from app/core/config.py —
# keep this in step with the file's depth or every data path silently moves.
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Load variables from the private .env file. Real deployments inject real
# environment variables instead, which take precedence over an absent file.
load_dotenv(PROJECT_DIR / ".env")


def _read_int(name: str, default: int) -> int:
    """Read a tunable from the environment, falling back when unset or invalid."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _read_list(name: str, default: list[str]) -> list[str]:
    """Comma-separated env var into a list, e.g. CORS_ORIGINS=https://a,https://b"""
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Credentials -------------------------------------------------------
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    serper_api_key: str | None = os.getenv("SERPER_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    # --- Vector store ------------------------------------------------------
    # Set QDRANT_URL to use a Qdrant server. Left unset, an embedded on-disk
    # instance is used, which takes an exclusive lock and so permits exactly one
    # process — fine for local development, unusable for more than one replica.
    qdrant_url: str | None = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", PROJECT_DIR / "qdrant_data"))
    qdrant_timeout: int = _read_int("QDRANT_TIMEOUT", 30)

    # --- Conversation memory ----------------------------------------------
    # postgresql:// selects Postgres; unset falls back to a local SQLite file.
    # SQLite ties sessions to one machine's disk, so multi-replica deployments
    # need the Postgres path.
    database_url: str | None = os.getenv("DATABASE_URL")
    database_path: Path = Path(
        os.getenv("DATABASE_PATH", PROJECT_DIR / "indian_laws_memory.db")
    )
    database_pool_min: int = _read_int("DATABASE_POOL_MIN", 1)
    database_pool_max: int = _read_int("DATABASE_POOL_MAX", 10)

    # --- API surface -------------------------------------------------------
    # When set, requests must carry it in the X-API-Key header.
    api_key: str | None = os.getenv("API_KEY")
    cors_origins: list[str] = field(
        default_factory=lambda: _read_list("CORS_ORIGINS", ["*"])
    )
    rate_limit_per_minute: int = _read_int("RATE_LIMIT_PER_MINUTE", 60)
    rate_limit_enabled: bool = _read_bool("RATE_LIMIT_ENABLED", True)

    # --- Observability -----------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format: str = os.getenv("LOG_FORMAT", "text").lower()  # "text" | "json"
    service_name: str = os.getenv("SERVICE_NAME", "indian-laws-rag")

    # --- Model behaviour ---------------------------------------------------
    llm_timeout_seconds: float = _read_float("LLM_TIMEOUT_SECONDS", 60.0)
    llm_max_retries: int = _read_int("LLM_MAX_RETRIES", 2)
    warmup_on_startup: bool = _read_bool("WARMUP_ON_STARTUP", True)

    # --- Retrieval tunables ------------------------------------------------
    # Candidates pulled from each of the dense and sparse searches before fusion.
    fused_k: int = _read_int("FUSED_K", 25)
    # Candidates kept after MMR diversification, handed to the reranker.
    mmr_k: int = _read_int("MMR_K", 15)
    # Balance between relevance (1.0) and diversity (0.0) inside MMR.
    mmr_lambda: float = _read_float("MMR_LAMBDA", 0.5)
    # Final chunks handed to the generator as context.
    rerank_top_n: int = _read_int("RERANK_TOP_N", 6)
    # Top rerank score below which we stop trusting the local index.
    confidence_threshold: float = _read_float("CONFIDENCE_THRESHOLD", 0.35)
    # Maximum sub-questions a compound query may be split into.
    max_sub_queries: int = _read_int("MAX_SUB_QUERIES", 3)
    # Conversation turns loaded as context for the query rewriter.
    history_turns: int = _read_int("HISTORY_TURNS", 4)

    # --- Derived -----------------------------------------------------------

    @property
    def uses_qdrant_server(self) -> bool:
        return bool(self.qdrant_url)

    @property
    def uses_postgres(self) -> bool:
        return bool(self.database_url and self.database_url.startswith("postgres"))

    @property
    def auth_required(self) -> bool:
        return is_usable_key(self.api_key)


settings = Settings()


PLACEHOLDER_KEY_MARKERS = ("paste", "your_key", "changeme", "xxx")


def is_usable_key(value: str | None) -> bool:
    """
    True only for a key that looks real.

    The shipped .env.example leaves placeholders like `paste_your_key_here`,
    and a placeholder reaching an SDK produces a confusing 401 rather than a
    clear "you forgot to set this" message.
    """
    if not value or not value.strip():
        return False
    return not any(marker in value.lower() for marker in PLACEHOLDER_KEY_MARKERS)
