# code/logger_utils.py
import logging
from pathlib import Path
from typing import Optional
from path_utils import shorten_path_for_logging

_FMT = "%(asctime)s | %(levelname)-8s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"

def init_logger(log_path: Path, level: str = "INFO", console_level: Optional[str] = None) -> None:
    """
        Call it once to set up the root logger for a specific process/run, then any module can do logging.getLogger(__name__).
    """
    # Create the directory for the log file if it doesn't exist.
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # root logger
    root = logging.getLogger()
    
    # If handlers already exist, clear them to avoid duplicate logs
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close() # It's good practice to close handlers

    root.setLevel(level.upper())

    formatter = logging.Formatter(_FMT)

    # Console handler: Use console_level if provided, otherwise default to WARNING
    effective_console_level = console_level.upper() if console_level else "WARNING"
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, effective_console_level))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler: logs everything from the specified level to the file
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level.upper())
    fh.setFormatter(formatter)
    root.addHandler(fh)

    root.propagate = False

    # Log initial message with info about console level
    if console_level:
        root.info("Log file created at %s. Console will show %s level messages and above.", shorten_path_for_logging(str(log_path)), effective_console_level)
    else:
        root.info("Log file created at %s. Console will show WARNING level messages and above (default).", shorten_path_for_logging(str(log_path)))


def get_logger(name: str) -> logging.Logger:
    """
        Return a child logger to inherit all handlers, levels, etc.
    """
    return logging.getLogger(name)