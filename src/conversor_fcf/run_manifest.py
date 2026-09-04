"""Run manifest: the record of which premise set and inputs produced an output pair.

`PREMISES` is the single source of truth for the ten v1 premises. Documentation
quotes it rather than restating it, so code and prose cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from conversor_fcf import __version__
from conversor_fcf.config import Settings

_P1 = "P1: mapcut record indices 4-17 emitted as zeros; WARNING logged; deferred to ticket-015"
_P2 = (
    "P2: intercept and every coefficient divided by 1000 (DECOMP is in 10^3 R$); "
    "verified to 1 ulp on a matched pair; not configurable"
)
_P3 = (
    "P3: n_utv = 0, so no mapcut regs 7/8 and NCOEF omits the pi_qdefp block; "
    "Cobre carried no HydroTransitBucket state"
)
_P4 = (
    "P4: GNL coefficients negated (Cobre negative, DECOMP positive), isolated in one "
    "named, tested, documented function"
)
_P5 = "P5: GNL disaggregated across the 3 load blocks weighted by hours (24 / 65 / 79, non-uniform)"
_P6 = "P6: submarket = bus_id + 1; bus 5 (IV) excluded; n_submercados = 5"
_P7 = (
    "P7: only trunk pools 0-5 converted; pool 6 (267-node terminal fan, 10 000 "
    "warm-start cuts) excluded"
)
_P8 = "P8: inflow-lag coefficients dropped with a counted audit report, never a silent truncation"
_P9 = (
    "P9: discount rate duration-proportional (1+r)^(-days/365); diverges from the "
    "reference deck's constant per-stage ratio for mixed-duration stages"
)
_P10 = (
    "P10: all populated cuts emitted; is_active and active_cut_indices recorded as ECO "
    "columns, not used as filters"
)

PREMISES: tuple[str, ...] = (_P1, _P2, _P3, _P4, _P5, _P6, _P7, _P8, _P9, _P10)

_TRACKED_LIBRARIES = ("numpy", "pandas", "flatbuffers")


@dataclass(frozen=True)
class RunManifest:
    """Provenance of a single conversion run."""

    tool_version: str
    created_at: str
    case_path: str
    revision: str
    settings_snapshot: dict[str, Any]
    premises: tuple[str, ...]
    library_versions: dict[str, str]
    outputs: dict[str, str]


def _library_versions() -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "unknown"
    return resolved


def build_run_manifest(
    case_path: Path,
    revision: str,
    settings: Settings,
    outputs: Mapping[str, Path],
) -> RunManifest:
    """Assemble the manifest for a run over `case_path` at `revision`."""
    return RunManifest(
        tool_version=__version__,
        created_at=datetime.now(UTC).isoformat(),
        case_path=str(case_path),
        revision=revision,
        settings_snapshot=asdict(settings),
        premises=PREMISES,
        library_versions=_library_versions(),
        outputs={name: str(path) for name, path in outputs.items()},
    )


def write_run_manifest(manifest: RunManifest, path: Path) -> None:
    """Write the manifest as deterministic, key-sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
