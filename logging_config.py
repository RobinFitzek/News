"""
Logging Configuration for Investment Monitor
Provides structured logging across the application

Development Mode: Set DEV_MODE=True or DEVELOPMENT_MODE setting to enable verbose debug logging
"""
import logging
import logging.handlers
from pathlib import Path
import sys
import os

# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Development mode flag (can be set via environment variable)
DEV_MODE = os.getenv('DEV_MODE', 'false').lower() in ('true', '1', 'yes')

def setup_logging(level=logging.INFO, dev_mode=None):
    """
    Configure logging for the application

    Args:
        level: Logging level (default: INFO)
        dev_mode: Enable development mode with verbose DEBUG logging (default: from DEV_MODE env var)
    """
    # Check development mode
    if dev_mode is None:
        dev_mode = DEV_MODE
        # Also check database setting if available
        try:
            from core.database import db
            db_dev_mode = db.get_setting('development_mode')
            if db_dev_mode is not None:
                dev_mode = bool(db_dev_mode)
        except Exception:
            pass  # Database might not be initialized yet
    
    # Set level based on dev mode
    if dev_mode:
        level = logging.DEBUG
        print("DEVELOPMENT MODE ENABLED - Verbose debug logging active")
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers = []

    # Format for logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler (DEBUG in dev mode, INFO otherwise)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if dev_mode else logging.INFO)
    
    # Enhanced formatter for dev mode
    if dev_mode:
        dev_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(dev_formatter)
    else:
        console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)

    # File Handler - General Application Logs
    app_log_file = LOGS_DIR / "application.log"
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_handler.setLevel(logging.DEBUG)
    # Use dev formatter in dev mode for better debugging in web UI
    if dev_mode:
        app_handler.setFormatter(dev_formatter)
    else:
        app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)

    # File Handler - Error Logs Only
    error_log_file = LOGS_DIR / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Suppress noisy third-party loggers (unless in dev mode)
    third_party_level = logging.DEBUG if dev_mode else logging.WARNING
    logging.getLogger('urllib3').setLevel(third_party_level)
    logging.getLogger('requests').setLevel(third_party_level)
    logging.getLogger('google').setLevel(third_party_level)
    logging.getLogger('httpx').setLevel(third_party_level)
    logging.getLogger('yfinance').setLevel(third_party_level)
    logging.getLogger('asyncio').setLevel(third_party_level)
    logging.getLogger('python_multipart').setLevel(third_party_level)
    logging.getLogger('uvicorn').setLevel(third_party_level)
    logging.getLogger('fastapi').setLevel(third_party_level)

    if dev_mode:
        logging.info("Logging initialized in DEVELOPMENT MODE with DEBUG level")
    else:
        logging.info("Logging initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    # Test logging configuration
    setup_logging(level=logging.DEBUG)

    logger = get_logger(__name__)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    try:
        raise Exception("Test exception")
    except Exception:
        logger.exception("Exception occurred")

    print(f"\nLog files created in: {LOGS_DIR}")
    print(f"- {LOGS_DIR / 'application.log'}")
    print(f"- {LOGS_DIR / 'errors.log'}")
