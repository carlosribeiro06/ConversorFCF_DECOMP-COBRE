import json
from pathlib import Path
from typing import Any

import pytest

from conversor_fcf.cobre.inputs_reader import (
    InputReadError,
    anticipated_thermals,
    read_case_inputs,
)

FILES = ("stages.json", "system/hydros.json", "system/thermals.json", "system/buses.json")


def _payloads() -> dict[str, Any]:
    return {
        "stages.json": {
            "policy_graph": {"annual_discount_rate": 0.12},
            "stages": [
                {
                    "id": 0,
                    "start_date": "2026-04-25",
                    "end_date": "2026-05-02",
                    "blocks": [
                        {"id": 0, "name": "HEAVY", "hours": 24.0},
                        {"id": 1, "name": "MEDIUM", "hours": 65.0},
                        {"id": 2, "name": "LIGHT", "hours": 79.0},
                    ],
                    "state_variables": {"storage": True, "inflow_lags": False},
                }
            ],
        },
        "system/hydros.json": {
            "hydros": [
                {
                    "id": 0,
                    "name": "CAMARGOS",
                    "downstream_id": 1,
                    "reservoir": {"min_storage_hm3": 120.0, "max_storage_hm3": 792.0},
                }
            ]
        },
        "system/thermals.json": {
            "thermals": [
                {"id": 0, "name": "ANGRA 1", "bus_id": 0},
                {
                    "id": 113,
                    "name": "PSERGIPE I",
                    "bus_id": 2,
                    "anticipated_config": {"lead_time_hours": 1608.0},
                },
                {
                    "id": 112,
                    "name": "SANTA CRUZ",
                    "bus_id": 0,
                    "anticipated_config": {"lead_time_hours": 1608.0},
                },
            ]
        },
        "system/buses.json": {"buses": [{"id": 0, "name": "SE"}, {"id": 5, "name": "IV"}]},
    }


def _build_case(tmp_path: Path, payloads: dict[str, Any] | None = None) -> Path:
    resolved = _payloads() if payloads is None else payloads
    (tmp_path / "system").mkdir(parents=True, exist_ok=True)
    for relative, payload in resolved.items():
        path = tmp_path / relative
        path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_minimal_case_reads_every_input(tmp_path: Path) -> None:
    inputs = read_case_inputs(_build_case(tmp_path))
    assert len(inputs.stages) == 1
    assert len(inputs.hydros) == 1
    assert len(inputs.thermals) == 3
    assert len(inputs.buses) == 2
    assert inputs.annual_discount_rate == 0.12


@pytest.mark.parametrize("missing", FILES)
def test_missing_file_names_its_absolute_path(tmp_path: Path, missing: str) -> None:
    case = _build_case(tmp_path)
    (case / missing).unlink()
    with pytest.raises(InputReadError) as excinfo:
        read_case_inputs(case)
    message = str(excinfo.value)
    assert Path(missing).name in message
    assert str(case) in message


@pytest.mark.parametrize("corrupt", FILES)
def test_invalid_json_names_the_path_and_wraps_the_decoder_message(
    tmp_path: Path, corrupt: str
) -> None:
    case = _build_case(tmp_path)
    (case / corrupt).write_text("{not json", encoding="utf-8")
    with pytest.raises(InputReadError) as excinfo:
        read_case_inputs(case)
    message = str(excinfo.value)
    assert Path(corrupt).name in message
    assert "not valid JSON" in message


def test_non_utf8_input_is_reported_as_such(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    (case / "system/buses.json").write_bytes(bytes([0x7B, 0x22, 0xE7, 0x22, 0x7D]))
    with pytest.raises(InputReadError, match="not valid UTF-8"):
        read_case_inputs(case)


@pytest.mark.parametrize(
    ("relative", "mutate", "expected_key"),
    [
        ("stages.json", ("policy_graph", "annual_discount_rate"), "annual_discount_rate"),
        ("stages.json", ("stages", 0, "id"), "stages[0].id"),
        ("stages.json", ("stages", 0, "start_date"), "stages[0].start_date"),
        ("stages.json", ("stages", 0, "state_variables", "inflow_lags"), "inflow_lags"),
        ("stages.json", ("stages", 0, "blocks", 0, "hours"), "stages[0].blocks[0].hours"),
        ("system/hydros.json", ("hydros", 0, "name"), "hydros[0].name"),
        ("system/hydros.json", ("hydros", 0, "reservoir", "max_storage_hm3"), "max_storage_hm3"),
        ("system/thermals.json", ("thermals", 0, "bus_id"), "thermals[0].bus_id"),
        ("system/buses.json", ("buses", 0, "name"), "buses[0].name"),
    ],
)
def test_missing_key_names_its_dotted_path(
    tmp_path: Path, relative: str, mutate: tuple[Any, ...], expected_key: str
) -> None:
    payloads = _payloads()
    target: Any = payloads[relative]
    for step in mutate[:-1]:
        target = target[step]
    del target[mutate[-1]]
    with pytest.raises(InputReadError) as excinfo:
        read_case_inputs(_build_case(tmp_path, payloads))
    assert expected_key in str(excinfo.value)


def test_absent_downstream_id_is_none_not_zero(tmp_path: Path) -> None:
    payloads = _payloads()
    del payloads["system/hydros.json"]["hydros"][0]["downstream_id"]
    inputs = read_case_inputs(_build_case(tmp_path, payloads))
    assert inputs.hydros[0].downstream_id is None


def test_null_downstream_id_is_none_not_zero(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["system/hydros.json"]["hydros"][0]["downstream_id"] = None
    inputs = read_case_inputs(_build_case(tmp_path, payloads))
    assert inputs.hydros[0].downstream_id is None


def test_downstream_id_zero_is_preserved_as_zero(tmp_path: Path) -> None:
    """0 is a real hydro id, so it must survive as 0 and never collapse to None."""
    payloads = _payloads()
    payloads["system/hydros.json"]["hydros"][0]["downstream_id"] = 0
    inputs = read_case_inputs(_build_case(tmp_path, payloads))
    assert inputs.hydros[0].downstream_id == 0


def test_non_integer_downstream_id_is_rejected(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["system/hydros.json"]["hydros"][0]["downstream_id"] = "1"
    with pytest.raises(InputReadError, match="downstream_id"):
        read_case_inputs(_build_case(tmp_path, payloads))


def test_bool_is_rejected_where_an_integer_is_required(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["system/buses.json"]["buses"][0]["id"] = True
    with pytest.raises(InputReadError, match="buses\\[0\\].id"):
        read_case_inputs(_build_case(tmp_path, payloads))


def test_thermal_without_anticipated_config_has_no_lead_time(tmp_path: Path) -> None:
    inputs = read_case_inputs(_build_case(tmp_path))
    by_id = {thermal.id: thermal for thermal in inputs.thermals}
    assert by_id[0].lead_time_hours is None
    assert by_id[112].lead_time_hours == 1608.0


def test_anticipated_thermals_are_filtered_and_sorted_by_id(tmp_path: Path) -> None:
    """The fixture lists 113 before 112 on purpose: order comes from the id, not the file."""
    inputs = read_case_inputs(_build_case(tmp_path))
    assert [thermal.id for thermal in anticipated_thermals(inputs)] == [112, 113]


def test_stages_list_may_be_empty_without_error(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["stages.json"]["stages"] = []
    inputs = read_case_inputs(_build_case(tmp_path, payloads))
    assert inputs.stages == ()


def test_non_list_stages_is_rejected(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["stages.json"]["stages"] = {"id": 0}
    with pytest.raises(InputReadError, match="stages must be a list"):
        read_case_inputs(_build_case(tmp_path, payloads))


@pytest.mark.parametrize(
    ("relative", "path_steps", "bad_value", "expected"),
    [
        ("system/buses.json", ("buses", 0, "name"), 5, "buses[0].name must be str"),
        (
            "stages.json",
            ("stages", 0, "blocks", 0, "hours"),
            "24",
            "stages[0].blocks[0].hours must be a number",
        ),
        (
            "stages.json",
            ("stages", 0, "blocks", 0, "hours"),
            True,
            "stages[0].blocks[0].hours must be a number",
        ),
        (
            "stages.json",
            ("stages", 0, "state_variables", "storage"),
            "true",
            "storage must be bool",
        ),
    ],
)
def test_wrong_typed_value_names_the_key_and_the_expected_type(
    tmp_path: Path, relative: str, path_steps: tuple[Any, ...], bad_value: Any, expected: str
) -> None:
    payloads = _payloads()
    target: Any = payloads[relative]
    for step in path_steps[:-1]:
        target = target[step]
    target[path_steps[-1]] = bad_value
    with pytest.raises(InputReadError) as excinfo:
        read_case_inputs(_build_case(tmp_path, payloads))
    assert expected in str(excinfo.value)


def test_non_object_section_is_reported_as_such(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["stages.json"]["policy_graph"] = 5
    with pytest.raises(InputReadError, match="must be an object"):
        read_case_inputs(_build_case(tmp_path, payloads))


def test_relative_case_dir_is_reported_as_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 7 asks for an absolute path, not an echo of what the caller passed."""
    case = _build_case(tmp_path / "case")
    (case / "system/buses.json").unlink()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InputReadError) as excinfo:
        read_case_inputs(Path("case"))
    message = str(excinfo.value)
    assert Path(message.split(": ", 1)[1]).is_absolute()
    assert str(case.resolve()) in message


def test_missing_hydro_id_names_its_dotted_key(tmp_path: Path) -> None:
    """Named literally in acceptance criterion 5, so it gets its own regression test."""
    payloads = _payloads()
    del payloads["system/hydros.json"]["hydros"][0]["id"]
    with pytest.raises(InputReadError, match=r"hydros\[0\].id"):
        read_case_inputs(_build_case(tmp_path, payloads))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(tmp_path: Path, literal: str) -> None:
    """json.loads accepts these non-standard literals; a nan would reach cortdeco."""
    case = _build_case(tmp_path)
    text = (case / "stages.json").read_text(encoding="utf-8")
    (case / "stages.json").write_text(text.replace("0.12", literal), encoding="utf-8")
    with pytest.raises(InputReadError, match="finite"):
        read_case_inputs(case)


def test_non_finite_block_hours_are_rejected(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    text = (case / "stages.json").read_text(encoding="utf-8")
    (case / "stages.json").write_text(text.replace("24.0", "NaN"), encoding="utf-8")
    with pytest.raises(InputReadError, match="hours must be a finite number"):
        read_case_inputs(case)
