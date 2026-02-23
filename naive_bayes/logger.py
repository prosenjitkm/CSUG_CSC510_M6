"""
naive_bayes/logger.py
---------------------
Centralised logging configuration for the Naive Bayes Classifier package.

Usage (in any module):
    from naive_bayes.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")
"""

import logging
import sys


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger that writes to stdout with a consistent format.
    Avoids adding duplicate handlers if called multiple times.

    Args:
        name:  typically __name__ of the calling module.
        level: logging level (default DEBUG so all messages are captured).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist yet (prevents duplicate log lines)
    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent messages from bubbling up to the root logger
        logger.propagate = False

    return logger

