"""
Centralized logging configuration for Palm Oil Backend API
Provides structured logging with rotation, JSON formatting, and contextual information
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
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

        # Add extra context if present
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add custom fields
        for key in ["user_id", "request_id", "tree_id", "form_id", "plot_id", "endpoint", "method", "status_code", "duration_ms"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Add contextual information to log records"""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record"""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_json: bool = True,
    enable_console: bool = True,
    enable_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure comprehensive logging for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        enable_json: Use JSON formatting for structured logs
        enable_console: Enable console output
        enable_file: Enable file output with rotation
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance
    """

    # Detect serverless environment (Vercel, AWS Lambda, etc.)
    is_serverless = os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME')

    # Disable file logging in serverless environments (filesystem is ephemeral/read-only)
    if is_serverless:
        enable_file = False

    # Try to create log directory if file logging is enabled
    log_path = None
    if enable_file:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # If we can't create log directory, disable file logging and continue with console only
            print(f"Warning: Could not create log directory '{log_dir}': {e}. Disabling file logging.")
            enable_file = False

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console handler (always enabled in serverless)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handlers with rotation (only if enabled and directory exists)
    if enable_file and log_path:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(formatter)
        logger.addHandler(app_handler)

        # Error log (only errors and critical)
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # Access log (for HTTP requests)
        access_handler = logging.handlers.RotatingFileHandler(
            log_path / "access.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(formatter)
        access_logger = logging.getLogger("access")
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False

        # Database log (for database operations)
        db_handler = logging.handlers.RotatingFileHandler(
            log_path / "database.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        db_handler.setLevel(logging.DEBUG)
        db_handler.setFormatter(formatter)
        db_logger = logging.getLogger("database")
        db_logger.addHandler(db_handler)
        db_logger.setLevel(logging.DEBUG)
        db_logger.propagate = False

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def get_access_logger() -> logging.Logger:
    """Get the access logger for HTTP requests"""
    return logging.getLogger("access")


def get_db_logger() -> logging.Logger:
    """Get the database logger for database operations"""
    return logging.getLogger("database")


# Convenience function to add context to logs
def log_with_context(logger: logging.Logger, level: int, message: str, **context):
    """Log message with additional context"""
    extra = {"context": context}
    extra.update(context)  # Also add as direct attributes
    logger.log(level, message, extra=extra)
