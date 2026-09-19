"""API key authentication and per-IP rate limiting."""

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"


async def require_api_key(request: Request) -> None:
    """
    Enforce the API key when one is configured.

    With API_KEY unset the API is open, which is what local development wants;
    setting it is what a deployment does.
    """
    if not settings.auth_required:
        return

    provided = request.headers.get(API_KEY_HEADER)
    if not provided or not _matches(provided, settings.api_key or ""):
        logger.warning(
            "rejected request with missing or invalid API key",
            extra={"path": request.url.path, "client": _client_ip(request)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"A valid {API_KEY_HEADER} header is required.",
        )


def _matches(provided: str, expected: str) -> bool:
    """Constant-time compare, so response timing cannot reveal the key."""
    import hmac

    return hmac.compare_digest(provided.strip(), expected.strip())


def _client_ip(request: Request) -> str:
    # Behind an ingress or load balancer the socket address is the proxy's, so
    # prefer the forwarded header when one is present.
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """
    A sliding-window limiter, per client IP.

    In-process by design: the state lives in this replica's memory, so N
    replicas allow N times the configured rate. Genuinely shared limiting needs
    Redis or a policy at the ingress; this stops a single runaway client, which
    is what it is here for.
    """

    def __init__(self, limit_per_minute: int | None = None):
        self.limit = limit_per_minute or settings.rate_limit_per_minute
        self.window_seconds = 60
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def check(self, client: str) -> bool:
        """True when the request is allowed; records it as a hit."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                return False

            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


async def enforce_rate_limit(request: Request) -> None:
    if not settings.rate_limit_enabled:
        return

    client = _client_ip(request)
    if not rate_limiter.check(client):
        logger.warning(
            "rate limit exceeded",
            extra={"client": client, "limit_per_minute": rate_limiter.limit},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {rate_limiter.limit} requests per minute exceeded.",
            headers={"Retry-After": "60"},
        )
