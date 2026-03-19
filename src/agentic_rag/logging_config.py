"""Logging configuration for Agentic RAG — structured via structlog."""

import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

MAIN_LOG = LOGS_DIR / f"agentic_rag_{datetime.now().strftime('%Y%m%d')}.log"
ERROR_LOG = LOGS_DIR / f"agentic_rag_errors_{datetime.now().strftime('%Y%m%d')}.log"


def setup_logging(level: str = "INFO", use_file_logging: bool = True) -> None:
    """Configure stdlib logging handlers, then wire structlog to use them."""

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
            "detailed": {
                "format": "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "agentic_rag": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
            "langchain": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    if use_file_logging:
        config["handlers"]["file"] = {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(MAIN_LOG),
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "detailed",
        }
        config["handlers"]["error_file"] = {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(ERROR_LOG),
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "detailed",
        }
        config["loggers"]["agentic_rag"]["handlers"].extend(["file", "error_file"])

    logging.config.dictConfig(config)

    # Configure structlog to render key=value pairs and forward to stdlib
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


setup_logging()
