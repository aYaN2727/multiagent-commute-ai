"""
utils/logger.py
Structured JSON logging with agent-aware decorators.
"""
import functools
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "agent_name": getattr(record, "agent_name", record.name),
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via the 'extra' kwarg
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "agent_name",
            }:
                log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Factory that returns a JSON-structured logger.
    Calling this multiple times with the same name returns the same logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


def log_agent_call(agent_name: Optional[str] = None) -> Callable:
    """
    Decorator that wraps an agent function and logs:
    - Agent name, start time
    - State keys present at start
    - Keys updated in returned dict
    - Total execution time in ms
    """
    def decorator(func: Callable) -> Callable:
        name = agent_name or func.__name__

        @functools.wraps(func)
        def wrapper(state: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
            logger = get_logger(name)
            start = time.perf_counter()
            logger.info(
                f"Agent starting",
                extra={
                    "agent_name": name,
                    "input_keys": list(state.keys()),
                },
            )
            try:
                result = func(state, *args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    f"Agent completed",
                    extra={
                        "agent_name": name,
                        "updated_keys": list(result.keys()) if result else [],
                        "elapsed_ms": round(elapsed_ms, 2),
                    },
                )
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Agent raised unhandled exception",
                    extra={
                        "agent_name": name,
                        "error": str(exc),
                        "elapsed_ms": round(elapsed_ms, 2),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator
