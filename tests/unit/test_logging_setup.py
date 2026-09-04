import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from conversor_fcf.config import LoggingSettings
from conversor_fcf.logging_setup import (
    ROOT_LOGGER_NAME,
    configure_logging,
    get_logger,
    log_step,
)


@pytest.fixture(autouse=True)
def _detach_handlers() -> Iterator[None]:
    yield
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _settings(tmp_path: Path, console: str = "INFO", file: str = "DEBUG") -> LoggingSettings:
    return LoggingSettings(
        level_console=console,
        level_file=file,
        file_path=str(tmp_path / "logs" / "run.log"),
        max_bytes=1024,
        backup_count=1,
    )


def test_console_and_file_levels_are_independent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(tmp_path, console="WARNING", file="DEBUG")
    logger = configure_logging(settings)
    logger.info("ingested 169 hydros")
    for handler in logger.handlers:
        handler.flush()

    contents = Path(settings.file_path).read_text(encoding="utf-8")
    assert "ingested 169 hydros" in contents
    assert "ingested 169 hydros" not in capsys.readouterr().out


def test_repeated_configuration_does_not_duplicate_handlers(tmp_path: Path) -> None:
    configure_logging(_settings(tmp_path))
    configure_logging(_settings(tmp_path))
    assert len(logging.getLogger(ROOT_LOGGER_NAME).handlers) == 2


def test_parent_directory_is_created(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    configure_logging(settings)
    assert Path(settings.file_path).parent.is_dir()


def test_logger_does_not_propagate(tmp_path: Path) -> None:
    logger = configure_logging(_settings(tmp_path))
    assert logger.propagate is False


def test_log_step_records_entry_and_elapsed_exit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    logger = configure_logging(settings)
    with log_step(logger, "ingest"):
        pass
    for handler in logger.handlers:
        handler.flush()

    records = [
        line
        for line in Path(settings.file_path).read_text(encoding="utf-8").splitlines()
        if "ingest" in line
    ]
    assert len(records) == 2
    assert "elapsed" in records[1]


def test_log_step_records_a_failure_distinguishably_with_traceback(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    logger = configure_logging(settings)
    with pytest.raises(ValueError, match="truncated"), log_step(logger, "ingest"):
        raise ValueError("cobre checkpoint truncated")
    for handler in logger.handlers:
        handler.flush()

    contents = Path(settings.file_path).read_text(encoding="utf-8")
    assert "fail ingest" in contents
    assert "end ingest" not in contents
    assert "ValueError: cobre checkpoint truncated" in contents
    assert "Traceback" in contents


def test_get_logger_namespaces_under_the_package_logger() -> None:
    assert get_logger("decomp").name == f"{ROOT_LOGGER_NAME}.decomp"
