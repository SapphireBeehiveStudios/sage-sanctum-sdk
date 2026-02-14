"""Centralized structured logging configuration for the Sage Sanctum SDK.

Uses ``structlog`` for machine-readable JSON output in production and
human-readable console output during development.

Usage::

    from sage_sanctum.logging import configure_logging, get_logger

    configure_logging(log_format="json", verbose=True)
    logger = get_logger(__name__)
    logger.info("agent_starting", agent="my-agent", version="1.0.0")
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(
    log_format: str = "text",
    verbose: bool = False,
) -> None:
    """Configure structured logging for the SDK.

    Should be called once at startup (e.g. from ``AgentRunner.run()``).

    Args:
        log_format: ``"json"`` for machine-readable output, ``"text"``
            for human-readable console output.
        verbose: If ``True``, set log level to ``DEBUG``; otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Suppress noisy third-party loggers
    for name in ("langchain", "litellm", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger wrapping a stdlib logger.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A ``structlog.stdlib.BoundLogger`` instance.
    """
    return structlog.get_logger(name)
