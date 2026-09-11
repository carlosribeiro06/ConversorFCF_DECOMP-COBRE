"""Audit-grade logging: a rich console handler plus a rotating file handler.

The file handler is the audit trail; the console handler is the operator view.
Both hang off the `conversor_fcf` logger, which does not propagate, so no root
handler can duplicate records into the audit log.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from conversor_fcf.config import LoggingSettings

ROOT_LOGGER_NAME = "conversor_fcf"

_FILE_FORMAT = "%(asctime)s.%(msecs)03d+00:00 %(levelname)-8s %(name)s %(message)s"
_FILE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Attach exactly two handlers to the package logger and return it."""
    log_path = Path(settings.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_names = logging.getLevelNamesMapping()
    console_level = level_names[settings.level_console]
    file_level = level_names[settings.level_file]

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    console_handler = RichHandler(level=console_level, show_path=False)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.max_bytes,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(_FILE_FORMAT, _FILE_DATE_FORMAT)
    # UTC at millisecond resolution, so audit records collate against
    # RunManifest.created_at without offset arithmetic.
    file_formatter.converter = time.gmtime
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(min(console_level, file_level))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the package logger."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


@contextmanager
def log_step(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log entry and exit of a named step, reporting elapsed wall time on exit.

    A step that raises is recorded as `fail` with its traceback, so an aborted run
    can never read as a completed one in the audit trail.
    """
    logger.info("start %s", label)
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        # BaseException, not Exception: an interrupt mid-conversion must also leave a
        # failure record. Always re-raised, so control flow is unchanged.
        logger.exception("fail %s elapsed=%.3fs", label, time.perf_counter() - started)
        raise
    else:
        logger.info("end %s elapsed=%.3fs", label, time.perf_counter() - started)
