import json
from pathlib import Path
from typing import Any

import pytest

from conversor_fcf.config import ConfigError, load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_SETTINGS = REPO_ROOT / "settings.json"


ALL_KEYS = [
    ("logging", "level_console", "str"),
    ("logging", "level_file", "str"),
    ("logging", "file_path", "str"),
    ("logging", "max_bytes", "int"),
    ("logging", "backup_count", "int"),
    ("output", "directory", "str"),
    ("output", "eco_subdirectory", "str"),
    ("output", "content_subdirectory", "str"),
    ("conversion", "hydro_codes_path", "str"),
    ("conversion", "include_terminal_pool", "bool"),
]

PATH_KEYS = [
    ("logging", "file_path"),
    ("output", "directory"),
    ("output", "eco_subdirectory"),
    ("output", "content_subdirectory"),
    ("conversion", "hydro_codes_path"),
]

_WRONG_TYPED_VALUE = {"str": 1, "int": "1", "bool": "true"}


def _valid_payload() -> dict[str, Any]:
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_tracked_settings_load_with_documented_values() -> None:
    settings = load_settings(TRACKED_SETTINGS)
    assert settings.logging.level_console == "INFO"
    assert settings.logging.level_file == "DEBUG"
    assert settings.logging.file_path == "logs/conversor-fcf.log"
    assert settings.logging.max_bytes == 10485760
    assert settings.logging.backup_count == 5
    assert settings.output.directory == "output/decomp_fcf"
    assert settings.output.eco_subdirectory == "eco"
    assert settings.output.content_subdirectory == "content"
    assert settings.conversion.hydro_codes_path == "decomp_hydro_codes.json"
    assert settings.conversion.include_terminal_pool is False


def test_missing_key_names_the_dotted_path(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["logging"]["file_path"]
    with pytest.raises(ConfigError) as excinfo:
        load_settings(_write(tmp_path, payload))
    assert "logging.file_path" in str(excinfo.value)


def test_wrong_type_names_the_key_and_expected_type(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["logging"]["max_bytes"] = "10485760"
    with pytest.raises(ConfigError) as excinfo:
        load_settings(_write(tmp_path, payload))
    message = str(excinfo.value)
    assert "logging.max_bytes" in message
    assert "int" in message


def test_bool_is_rejected_for_an_integer_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["logging"]["max_bytes"] = True
    with pytest.raises(ConfigError, match="logging.max_bytes"):
        load_settings(_write(tmp_path, payload))


def test_non_positive_integer_is_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["logging"]["backup_count"] = 0
    with pytest.raises(ConfigError, match="positive"):
        load_settings(_write(tmp_path, payload))


def test_unknown_log_level_is_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["logging"]["level_console"] = "VERBOSE"
    with pytest.raises(ConfigError, match="logging.level_console"):
        load_settings(_write(tmp_path, payload))


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "absent.json")


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_settings(path)


def test_non_object_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="JSON object"):
        load_settings(_write(tmp_path, [1, 2, 3]))


@pytest.mark.parametrize(("section", "key", "expected_type"), ALL_KEYS)
def test_every_key_names_its_dotted_path_when_missing(
    tmp_path: Path, section: str, key: str, expected_type: str
) -> None:
    payload = _valid_payload()
    del payload[section][key]
    with pytest.raises(ConfigError) as excinfo:
        load_settings(_write(tmp_path, payload))
    assert f"{section}.{key}" in str(excinfo.value)


@pytest.mark.parametrize(("section", "key", "expected_type"), ALL_KEYS)
def test_every_key_names_its_dotted_path_and_type_when_mistyped(
    tmp_path: Path, section: str, key: str, expected_type: str
) -> None:
    payload = _valid_payload()
    payload[section][key] = _WRONG_TYPED_VALUE[expected_type]
    with pytest.raises(ConfigError) as excinfo:
        load_settings(_write(tmp_path, payload))
    message = str(excinfo.value)
    assert f"{section}.{key}" in message
    assert expected_type in message


@pytest.mark.parametrize(("section", "key"), PATH_KEYS)
def test_blank_path_is_rejected_by_dotted_name(tmp_path: Path, section: str, key: str) -> None:
    payload = _valid_payload()
    payload[section][key] = "   "
    with pytest.raises(ConfigError) as excinfo:
        load_settings(_write(tmp_path, payload))
    assert f"{section}.{key}" in str(excinfo.value)


def test_non_object_section_is_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["logging"] = []
    with pytest.raises(ConfigError, match="logging must be object"):
        load_settings(_write(tmp_path, payload))


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    # A lone 0xE7 is valid cp1252 and invalid UTF-8: the Windows-editor / WSL-runtime case.
    path.write_bytes(b'{"output": {"directory": "sa\xe7ao"}}')
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_settings(path)


def test_settings_are_immutable() -> None:
    settings = load_settings(TRACKED_SETTINGS)
    with pytest.raises(AttributeError):
        settings.logging.level_console = "DEBUG"  # type: ignore[misc]
