# logger_setup.py

import logging
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes both to logs/app.log and to the console.
    """
    # Make sure logs/ folder exists
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # If handlers already added, just reuse (avoid duplicate logs)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler → logs/app.log
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler → VS Code terminal
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
