"""
CuliFeed Logging Configuration
=============================

Structured logging setup with proper formatting, levels, and output handling
for both development and production environments.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Basic log data
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }
        }

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Build formatted message
        formatted = (
            f"{color}[{timestamp}] {record.levelname:8}{reset} "
            f"{record.name}:{record.funcName}:{record.lineno} - "
            f"{record.getMessage()}"
        )

        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logger(
    name: str = "culifeed",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    structured: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logger with appropriate handlers and formatting.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        structured: Whether to use structured JSON logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)

        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ColoredConsoleFormatter())

        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to manage log size
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )

        # Always use structured format for file logging
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """Initialize adapter with extra context.

        Args:
            logger: Base logger instance
            extra: Extra context to add to all log messages
        """
        super().__init__(logger, extra)

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        """Process log message with extra context."""
        # Merge extra context
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra.copy()

        return msg, kwargs


def get_logger_for_component(
    component_name: str,
    chat_id: Optional[str] = None,
    article_id: Optional[str] = None,
    topic_name: Optional[str] = None,
) -> LoggerAdapter:
    """Get a logger adapter with component-specific context.

    Args:
        component_name: Name of the component (e.g., 'ingestion', 'ai_processor')
        chat_id: Associated chat ID (optional)
        article_id: Associated article ID (optional)
        topic_name: Associated topic name (optional)

    Returns:
        Logger adapter with context
    """
    base_logger = logging.getLogger(f"culifeed.{component_name}")

    extra_context = {
        "component": component_name,
    }

    if chat_id:
        extra_context["chat_id"] = chat_id
    if article_id:
        extra_context["article_id"] = article_id
    if topic_name:
        extra_context["topic"] = topic_name

    return LoggerAdapter(base_logger, extra_context)


def configure_application_logging(
    log_level: str = "INFO",
    log_file: str = "logs/culifeed.log",
    enable_console: bool = True,
    structured_logging: bool = False,
) -> None:
    """Configure application-wide logging settings.

    Args:
        log_level: Global log level
        log_file: Path to main log file
        enable_console: Whether to enable console logging
        structured_logging: Whether to use JSON structured logging
    """
    # Set up main application logger
    setup_logger(
        name="culifeed",
        level=log_level,
        log_file=log_file,
        console=enable_console,
        structured=structured_logging,
    )

    # Configure third-party library logging levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("feedparser").setLevel(logging.WARNING)

    # Suppress overly verbose library logs
    logging.getLogger("sqlite3").setLevel(logging.WARNING)


# Pre-configured logger instances for common use cases
def get_ingestion_logger(chat_id: Optional[str] = None) -> LoggerAdapter:
    """Get logger for ingestion components."""
    return get_logger_for_component("ingestion", chat_id=chat_id)


def get_processing_logger(
    chat_id: Optional[str] = None, topic_name: Optional[str] = None
) -> LoggerAdapter:
    """Get logger for processing components."""
    return get_logger_for_component(
        "processing", chat_id=chat_id, topic_name=topic_name
    )


def get_ai_logger(
    article_id: Optional[str] = None, topic_name: Optional[str] = None
) -> LoggerAdapter:
    """Get logger for AI processing components."""
    return get_logger_for_component("ai", article_id=article_id, topic_name=topic_name)


def get_bot_logger(chat_id: Optional[str] = None) -> LoggerAdapter:
    """Get logger for Telegram bot components."""
    return get_logger_for_component("bot", chat_id=chat_id)


def get_delivery_logger(chat_id: Optional[str] = None) -> LoggerAdapter:
    """Get logger for delivery components."""
    return get_logger_for_component("delivery", chat_id=chat_id)


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        """Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Operation being timed
            **kwargs: Additional context for the operation
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None

    def __enter__(self) -> "PerformanceLogger":
        """Start timing the operation."""
        self.start_time = datetime.now(timezone.utc)
        self.logger.debug(f"Starting {self.operation}", extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing and log results."""
        if self.start_time:
            duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            context = {
                **self.context,
                "duration_seconds": duration,
                "success": exc_type is None,
            }

            if exc_type:
                self.logger.error(
                    f"Failed {self.operation} in {duration:.3f}s", extra=context
                )
            else:
                self.logger.info(
                    f"Completed {self.operation} in {duration:.3f}s", extra=context
                )
