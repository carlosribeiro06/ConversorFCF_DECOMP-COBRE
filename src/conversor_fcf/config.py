"""Settings loading with fail-fast validation.

Every key is required. No default is ever substituted for a missing key, so a
malformed settings file stops the run instead of silently changing behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(Exception):
    """Raised when the settings file is missing, malformed or incomplete."""


def _require(obj: dict[str, Any], key: str, dotted: str) -> Any:
    if key not in obj:
        raise ConfigError(f"missing required key {dotted}")
    return obj[key]


def _require_object(obj: dict[str, Any], key: str, dotted: str) -> dict[str, Any]:
    value = _require(obj, key, dotted)
    if not isinstance(value, dict):
        raise ConfigError(f"{dotted} must be object, got {type(value).__name__}")
    return value


def _require_str(obj: dict[str, Any], key: str, dotted: str) -> str:
    value = _require(obj, key, dotted)
    if not isinstance(value, str):
        raise ConfigError(f"{dotted} must be str, got {type(value).__name__}")
    return value


def _require_non_empty_str(obj: dict[str, Any], key: str, dotted: str) -> str:
    value = _require_str(obj, key, dotted)
    if not value.strip():
        raise ConfigError(f"{dotted} must be a non-empty path, got {value!r}")
    return value


def _require_int(obj: dict[str, Any], key: str, dotted: str) -> int:
    value = _require(obj, key, dotted)
    # bool is a subclass of int, so isinstance would accept `true` here.
    if type(value) is not int:
        raise ConfigError(f"{dotted} must be int, got {type(value).__name__}")
    return value


def _require_bool(obj: dict[str, Any], key: str, dotted: str) -> bool:
    value = _require(obj, key, dotted)
    if type(value) is not bool:
        raise ConfigError(f"{dotted} must be bool, got {type(value).__name__}")
    return value


def _require_positive_int(obj: dict[str, Any], key: str, dotted: str) -> int:
    value = _require_int(obj, key, dotted)
    if value <= 0:
        raise ConfigError(f"{dotted} must be a positive int, got {value}")
    return value


def _require_level(obj: dict[str, Any], key: str, dotted: str) -> str:
    value = _require_str(obj, key, dotted)
    if value not in LOG_LEVELS:
        allowed = ", ".join(sorted(LOG_LEVELS))
        raise ConfigError(f"{dotted} must be one of {allowed}, got {value!r}")
    return value


@dataclass(frozen=True)
class LoggingSettings:
    """Audit-log destinations, levels and rotation policy."""

    level_console: str
    level_file: str
    file_path: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class OutputSettings:
    """Where the converted artifacts are written, relative to the case directory."""

    directory: str
    eco_subdirectory: str
    content_subdirectory: str


@dataclass(frozen=True)
class ConversionSettings:
    """Conversion inputs and switches."""

    hydro_codes_path: str
    include_terminal_pool: bool


@dataclass(frozen=True)
class Settings:
    """Fully validated, immutable project configuration."""

    logging: LoggingSettings
    output: OutputSettings
    conversion: ConversionSettings


def load_settings(path: Path) -> Settings:
    """Load and validate the settings file, raising ConfigError on any problem."""
    if not path.is_file():
        raise ConfigError(f"settings file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"settings file is not valid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"settings file is not valid UTF-8: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"settings root must be a JSON object, got {type(raw).__name__}")

    log_obj = _require_object(raw, "logging", "logging")
    out_obj = _require_object(raw, "output", "output")
    conv_obj = _require_object(raw, "conversion", "conversion")

    return Settings(
        logging=LoggingSettings(
            level_console=_require_level(log_obj, "level_console", "logging.level_console"),
            level_file=_require_level(log_obj, "level_file", "logging.level_file"),
            file_path=_require_non_empty_str(log_obj, "file_path", "logging.file_path"),
            max_bytes=_require_positive_int(log_obj, "max_bytes", "logging.max_bytes"),
            backup_count=_require_positive_int(log_obj, "backup_count", "logging.backup_count"),
        ),
        output=OutputSettings(
            directory=_require_non_empty_str(out_obj, "directory", "output.directory"),
            eco_subdirectory=_require_non_empty_str(
                out_obj, "eco_subdirectory", "output.eco_subdirectory"
            ),
            content_subdirectory=_require_non_empty_str(
                out_obj, "content_subdirectory", "output.content_subdirectory"
            ),
        ),
        conversion=ConversionSettings(
            hydro_codes_path=_require_non_empty_str(
                conv_obj, "hydro_codes_path", "conversion.hydro_codes_path"
            ),
            include_terminal_pool=_require_bool(
                conv_obj, "include_terminal_pool", "conversion.include_terminal_pool"
            ),
        ),
    )
