"""Readers for the four Cobre JSON inputs the conversion depends on.

Only JSON is read. Because premise P1 zero-fills mapcut record indices 4-17 and
premise P3 sets `n_utv = 0`, no Cobre Parquet artifact is needed, which is what
keeps `pyarrow` out of the dependency set.

Stage durations are **not** uniform in the reference deck: stages 0-5 span 168
hours each while stage 6 spans 600. Anything that weights by time must read the
per-stage block hours rather than assume a week.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conversor_fcf.logging_setup import get_logger

_STAGES_FILE = "stages.json"
_SYSTEM_DIR = "system"
_HYDROS_FILE = "hydros.json"
_THERMALS_FILE = "thermals.json"
_BUSES_FILE = "buses.json"

_logger = get_logger("inputs_reader")


class InputReadError(Exception):
    """Raised when a Cobre input file is missing, malformed or incomplete."""


@dataclass(frozen=True)
class BlockInfo:
    """One load block of a stage."""

    id: int
    name: str
    hours: float


@dataclass(frozen=True)
class StageInfo:
    """One study stage, its load blocks and which state variables it carries."""

    id: int
    start_date: str
    end_date: str
    blocks: tuple[BlockInfo, ...]
    storage_state: bool
    inflow_lags_state: bool


@dataclass(frozen=True)
class HydroInfo:
    """One hydro plant. `downstream_id` is `None` for a terminal plant."""

    id: int
    name: str
    downstream_id: int | None
    min_storage_hm3: float
    max_storage_hm3: float


@dataclass(frozen=True)
class ThermalInfo:
    """One thermal plant. `lead_time_hours` is `None` when it is not GNL-capable."""

    id: int
    name: str
    bus_id: int
    lead_time_hours: float | None


@dataclass(frozen=True)
class BusInfo:
    """One electrical bus, which maps to a DECOMP submarket in `ticket-006`."""

    id: int
    name: str


@dataclass(frozen=True)
class CaseInputs:
    """Everything the conversion reads from the case's JSON inputs."""

    stages: tuple[StageInfo, ...]
    hydros: tuple[HydroInfo, ...]
    thermals: tuple[ThermalInfo, ...]
    buses: tuple[BusInfo, ...]
    annual_discount_rate: float


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise InputReadError(f"Cobre input file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputReadError(f"{path} is not valid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputReadError(f"{path} is not valid UTF-8: {exc}") from exc


def _require(obj: Any, key: str, dotted: str) -> Any:
    if not isinstance(obj, dict):
        raise InputReadError(f"{dotted} must be an object, got {type(obj).__name__}")
    if key not in obj:
        raise InputReadError(f"missing required key {dotted}")
    return obj[key]


def _require_list(obj: Any, key: str, dotted: str) -> list[Any]:
    value = _require(obj, key, dotted)
    if not isinstance(value, list):
        raise InputReadError(f"{dotted} must be a list, got {type(value).__name__}")
    return value


def _require_str(obj: Any, key: str, dotted: str) -> str:
    value = _require(obj, key, dotted)
    if not isinstance(value, str):
        raise InputReadError(f"{dotted} must be str, got {type(value).__name__}")
    return value


def _require_int(obj: Any, key: str, dotted: str) -> int:
    value = _require(obj, key, dotted)
    # bool is a subclass of int, so isinstance would accept `true` here.
    if type(value) is not int:
        raise InputReadError(f"{dotted} must be int, got {type(value).__name__}")
    return value


def _require_float(obj: Any, key: str, dotted: str) -> float:
    value = _require(obj, key, dotted)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputReadError(f"{dotted} must be a number, got {type(value).__name__}")
    number = float(value)
    # json.loads accepts the non-standard NaN / Infinity / -Infinity literals,
    # and block hours and the discount rate both feed cortdeco coefficients.
    if not math.isfinite(number):
        raise InputReadError(f"{dotted} must be a finite number, got {value!r}")
    return number


def _require_bool(obj: Any, key: str, dotted: str) -> bool:
    value = _require(obj, key, dotted)
    if type(value) is not bool:
        raise InputReadError(f"{dotted} must be bool, got {type(value).__name__}")
    return value


def _read_blocks(raw_stage: Any, stage_dotted: str) -> tuple[BlockInfo, ...]:
    blocks = _require_list(raw_stage, "blocks", f"{stage_dotted}.blocks")
    return tuple(
        BlockInfo(
            id=_require_int(block, "id", f"{stage_dotted}.blocks[{position}].id"),
            name=_require_str(block, "name", f"{stage_dotted}.blocks[{position}].name"),
            hours=_require_float(block, "hours", f"{stage_dotted}.blocks[{position}].hours"),
        )
        for position, block in enumerate(blocks)
    )


def _read_stages(path: Path) -> tuple[tuple[StageInfo, ...], float]:
    raw = _load_json(path)
    policy_graph = _require(raw, "policy_graph", "policy_graph")
    # The discount rate lives under policy_graph, not at the top level.
    discount_rate = _require_float(
        policy_graph, "annual_discount_rate", "policy_graph.annual_discount_rate"
    )

    stages = []
    for position, raw_stage in enumerate(_require_list(raw, "stages", "stages")):
        dotted = f"stages[{position}]"
        state_variables = _require(raw_stage, "state_variables", f"{dotted}.state_variables")
        stages.append(
            StageInfo(
                id=_require_int(raw_stage, "id", f"{dotted}.id"),
                start_date=_require_str(raw_stage, "start_date", f"{dotted}.start_date"),
                end_date=_require_str(raw_stage, "end_date", f"{dotted}.end_date"),
                blocks=_read_blocks(raw_stage, dotted),
                storage_state=_require_bool(
                    state_variables, "storage", f"{dotted}.state_variables.storage"
                ),
                inflow_lags_state=_require_bool(
                    state_variables, "inflow_lags", f"{dotted}.state_variables.inflow_lags"
                ),
            )
        )
    return tuple(stages), discount_rate


def _read_hydros(path: Path) -> tuple[HydroInfo, ...]:
    raw = _load_json(path)
    hydros = []
    for position, entry in enumerate(_require_list(raw, "hydros", "hydros")):
        dotted = f"hydros[{position}]"
        reservoir = _require(entry, "reservoir", f"{dotted}.reservoir")
        # Absent or null both denote a terminal plant. 0 is a valid hydro id, so
        # coercing to 0 would silently reroute the plant into CAMARGOS.
        downstream = entry.get("downstream_id") if isinstance(entry, dict) else None
        if downstream is not None and type(downstream) is not int:
            raise InputReadError(
                f"{dotted}.downstream_id must be int or null, got {type(downstream).__name__}"
            )
        hydros.append(
            HydroInfo(
                id=_require_int(entry, "id", f"{dotted}.id"),
                name=_require_str(entry, "name", f"{dotted}.name"),
                downstream_id=downstream,
                min_storage_hm3=_require_float(
                    reservoir, "min_storage_hm3", f"{dotted}.reservoir.min_storage_hm3"
                ),
                max_storage_hm3=_require_float(
                    reservoir, "max_storage_hm3", f"{dotted}.reservoir.max_storage_hm3"
                ),
            )
        )
    return tuple(hydros)


def _read_thermals(path: Path) -> tuple[ThermalInfo, ...]:
    raw = _load_json(path)
    thermals = []
    for position, entry in enumerate(_require_list(raw, "thermals", "thermals")):
        dotted = f"thermals[{position}]"
        # The GNL discriminator is the presence of anticipated_config, never the
        # plant name and never its cost.
        anticipated = entry.get("anticipated_config") if isinstance(entry, dict) else None
        lead_time = (
            None
            if anticipated is None
            else _require_float(
                anticipated, "lead_time_hours", f"{dotted}.anticipated_config.lead_time_hours"
            )
        )
        thermals.append(
            ThermalInfo(
                id=_require_int(entry, "id", f"{dotted}.id"),
                name=_require_str(entry, "name", f"{dotted}.name"),
                bus_id=_require_int(entry, "bus_id", f"{dotted}.bus_id"),
                lead_time_hours=lead_time,
            )
        )
    return tuple(thermals)


def _read_buses(path: Path) -> tuple[BusInfo, ...]:
    raw = _load_json(path)
    return tuple(
        BusInfo(
            id=_require_int(entry, "id", f"buses[{position}].id"),
            name=_require_str(entry, "name", f"buses[{position}].name"),
        )
        for position, entry in enumerate(_require_list(raw, "buses", "buses"))
    )


def anticipated_thermals(inputs: CaseInputs) -> tuple[ThermalInfo, ...]:
    """The GNL-capable thermals, ascending by id."""
    return tuple(
        sorted(
            (thermal for thermal in inputs.thermals if thermal.lead_time_hours is not None),
            key=lambda thermal: thermal.id,
        )
    )


def read_case_inputs(case_dir: Path) -> CaseInputs:
    """Read `stages.json` and the three `system/` inputs the conversion needs."""
    # Resolved up front so every error message names an absolute path, whatever
    # the caller passed.
    case_dir = case_dir.resolve()
    system_dir = case_dir / _SYSTEM_DIR
    stages, annual_discount_rate = _read_stages(case_dir / _STAGES_FILE)
    inputs = CaseInputs(
        stages=stages,
        hydros=_read_hydros(system_dir / _HYDROS_FILE),
        thermals=_read_thermals(system_dir / _THERMALS_FILE),
        buses=_read_buses(system_dir / _BUSES_FILE),
        annual_discount_rate=annual_discount_rate,
    )

    _logger.info(
        "read case inputs %s stages=%d hydros=%d thermals=%d buses=%d anticipated_thermals=%d "
        "annual_discount_rate=%s",
        case_dir,
        len(inputs.stages),
        len(inputs.hydros),
        len(inputs.thermals),
        len(inputs.buses),
        len(anticipated_thermals(inputs)),
        inputs.annual_discount_rate,
    )
    terminal = sum(1 for hydro in inputs.hydros if hydro.downstream_id is None)
    _logger.info("hydro topology: %d terminal plants with no downstream", terminal)
    durations = {sum(block.hours for block in stage.blocks) for stage in inputs.stages}
    if len(durations) > 1:
        _logger.info(
            "stage durations are not uniform: %s hours; anything weighted by time must read "
            "per-stage block hours",
            sorted(durations),
        )
    return inputs
