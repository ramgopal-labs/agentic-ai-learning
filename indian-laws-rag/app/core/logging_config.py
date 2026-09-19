"""
Structured logging with a request id carried through the whole pipeline.

Every log line emitted while answering a question carries the same request id,
so one question's rewrite, retrieval, rerank and generation lines can be
correlated in an aggregator — which is the difference between a debuggable
service and a silent one.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar

from app.core.config import settings

# A ContextVar rather than a global: each request (and each thread FastAPI runs a
# sync endpoint in) sees its own value without any explicit passing.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(request_id: str | None = None) -> str:
    request_id = request_id or new_request_id()
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> str:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Attaches the current request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregators."""

    # Everything the stdlib puts on a LogRecord that we render explicitly or
    # deliberately drop; anything else the caller passed via `extra` is included.
    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "request_id",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.service_name,
            "request_id": getattr(record, "request_id", "-"),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


TEXT_FORMAT = "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"


def configure_logging() -> None:
    """
    Install handlers on the root logger. Safe to call more than once.

    Uvicorn installs its own handlers, so we replace them rather than let every
    line be emitted twice in two different formats.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(noisy)
        logger.handlers = [handler]
        logger.propagate = False

    # These libraries log a line per HTTP call at INFO, which drowns out our own.
    for chatty in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(chatty).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
