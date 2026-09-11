"""Run manifest: the record of which premise set and inputs produced an output pair.

`PREMISES` is the single source of truth for the v1 premises. Documentation
quotes it rather than restating it, so code and prose cannot drift apart. No
entry here states how many premises there are: a count in prose next to the
tuple that defines it goes stale on the next addition, and has once already.

A premise that another premise has taken off the code path is marked **DORMANT**
in its own text rather than deleted. A manifest that asserts a rule with no
effect on the output pair is an auditability defect, and so is one that drops a
rule the code still carries: naming the dormancy is what distinguishes the two.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from conversor_fcf import __version__
from conversor_fcf.config import Settings
from conversor_fcf.decomp.layout import sync_directory, write_durably

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
    "named, tested, documented function. DORMANT while P14 holds: cortdeco emits its whole "
    "pi_gnl block as zeros, so no real coefficient reaches this rule and it changes no byte of "
    "either output file. negate_gnl and negate_gnl_array are retained, tested and reachable "
    "because ticket-016 needs them the moment the reduction is settled"
)
_P5 = (
    "P5: GNL disaggregated across the 3 load blocks weighted by each stage's own hours. The "
    "weights are per-stage, not one triple for the study: stage 0 is 24/65/79, stages 1-4 are "
    "15/64/89, stage 5 is 12/61/95 and stage 6 is 51/226/323 over 600 hours rather than 168. "
    "DORMANT while P14 holds: with the pi_gnl block emitted as zeros there is no coefficient to "
    "spread across load blocks, so this rule changes no byte of either output file. "
    "gnl_block_weights is retained and tested for ticket-016"
)
_P6 = "P6: submarket = bus_id + 1; bus 5 (IV) excluded; n_submercados = 5"
_P7 = (
    "P7: only trunk pools 0-5 converted; pool 6 (267-node terminal fan, 10 000 "
    "warm-start cuts) excluded"
)
_P8 = "P8: inflow-lag coefficients dropped with a counted audit report, never a silent truncation"
_P9 = (
    "P9: discount rate duration-proportional (1+r)^(-cumulative_days/365.25); the day-count "
    "basis is 365.25, solved from the reference series and exact to 3.7e-13, not 365"
)
_P10 = (
    "P10: all populated cuts emitted; is_active and active_cut_indices recorded as ECO "
    "columns, not used as filters"
)
_P11 = (
    "P11: mapcut reg 10 emits zeros for parcela_custo_geracao_termica_minima, "
    "parcela_custo_contrato_importacao_minimo, parcela_custo_contrato_exportacao_minimo, "
    "geracao_termica_minima_sinalizada_gnl and geracao_termica_minima_gerada_gnl; "
    "they are DECOMP operational quantities from the deck's own data, non-zero in the reference, "
    "and are not derivable from a Cobre policy checkpoint, whose cut intercept already embeds "
    "the constant term. Only taxa_desconto is filled."
)

_P12 = (
    "P12: mapcut reg 9's trailing float64 block emitted as zeros. The reference populates it with "
    "the GNL lag month's hours split across load blocks, three values per submarket summing to "
    "730.5 = 365.25*24/12, one average month on the same day-count basis as P9. Two independent "
    "reasons, the first decisive: the values are not derivable from a Cobre case, because that "
    "split is a monthly load-block structure belonging to the DECOMP deck while Cobre supplies "
    "weekly stage blocks, and no aggregation of this case's own blocks reproduces the reference "
    "triple (the closest, stages 0-5, is off by 0.0054 in proportion). Second, the block's axis is "
    "itself unsettled: the populated width is ngnl*npat while idecomp's reader consumes "
    "ngnl*n_estagios. Populating this block needs a DECOMP-side input, not better Cobre parsing."
)

_P13 = (
    "P13: cortdeco holds numero_cortes + 1 records, and the extra one duplicates the last "
    "cut-building node's last cut. The reference deck holds the next backward-pass cut there, "
    "written but not yet exposed in the head table, which a checkpoint with a whole number of "
    "completed iterations cannot supply. Nothing points to that record, since the chains only "
    "step backwards from the heads. A duplicate is mathematically inert, because a repeated "
    "hyperplane adds nothing to an FCF, while a zero-filled record would fabricate theta >= 0."
)

_P14 = (
    "P14: cortdeco's pi_gnl coefficients emitted as zeros. Only the values are zero: the block "
    "stays dimensioned by the NCOEF formula and occupies its full "
    "n_sbm_gnl*n_estagios*n_patamares span, and n_sbm_gnl, codigos_submercados_gnl (int32 on "
    "disk despite idecomp surfacing floats), NCOEF, the record size and the file size are all "
    "unchanged. Reducing Cobre's anticipated-thermal ring positions to DECOMP's "
    "(submarket, stage, block) address is unresolved: ring positions sharing a delivery month "
    "carry different coefficients, so they are distinct state variables rather than copies, and "
    "the reference deck comes from an independent run whose GNL magnitudes span seven orders of "
    "magnitude against the oracle's 15% band and therefore cannot arbitrate between candidate "
    "reductions. Every candidate writes a structurally valid file that DESSEM reads without "
    "complaint, so a wrong one would corrupt the cut's GNL slope invisibly - unlike P1, P11 and "
    "P12, whose divergence is a declared zero. Deferred to ticket-016; P4 and P5 are DORMANT "
    "while this premise holds"
)

_P15 = (
    "P15: mapcut reg 9's lag_meses_gnl emitted as 2 for every GNL plant. Evidenced only for the "
    "reference case's own lead_time_hours of 1608.0 (1608.0/730.5 = 2.2012): floor and round "
    "both map that single known point to 2, and the two formulas first diverge at 2.5 (any "
    "fractional part of 0.5 or more), so one evidenced point cannot decide between them. "
    "Guessing either would repeat the error premise P14 exists to avoid. "
    "assert_gnl_lead_time_is_evidenced refuses any GNL plant whose lead_time_hours differs from "
    "1608.0 rather than guess at an unevidenced case."
)

PREMISES: tuple[str, ...] = (
    _P1,
    _P2,
    _P3,
    _P4,
    _P5,
    _P6,
    _P7,
    _P8,
    _P9,
    _P10,
    _P11,
    _P12,
    _P13,
    _P14,
    _P15,
)

_TRACKED_LIBRARIES = ("numpy", "pandas", "flatbuffers")


@dataclass(frozen=True)
class RunManifest:
    """Provenance of a single conversion run.

    `status` and `failed_step` (M7) make the manifest a real audit record for a
    failed run, not only a successful one: a run that raised must still leave a
    trace of which step it reached, because a missing manifest tells an auditor
    nothing about what was attempted.
    """

    tool_version: str
    created_at: str
    case_path: str
    revision: str
    settings_snapshot: dict[str, Any]
    premises: tuple[str, ...]
    library_versions: dict[str, str]
    outputs: dict[str, str]
    status: str
    failed_step: str | None


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
    status: str = "ok",
    failed_step: str | None = None,
) -> RunManifest:
    """Assemble the manifest for a run over `case_path` at `revision`.

    `status`/`failed_step` default to the success case, so every existing caller
    that predates M7 keeps building an `"ok"` manifest unchanged; the pipeline's
    failure path passes `status="failed"` and the `log_step` label that raised.
    """
    return RunManifest(
        tool_version=__version__,
        created_at=datetime.now(UTC).isoformat(),
        case_path=str(case_path),
        revision=revision,
        settings_snapshot=asdict(settings),
        premises=PREMISES,
        library_versions=_library_versions(),
        outputs={name: str(path) for name, path in outputs.items()},
        status=status,
        failed_step=failed_step,
    )


def write_run_manifest(manifest: RunManifest, path: Path) -> None:
    """Write the manifest as deterministic, key-sorted JSON, atomically (M7).

    `.partial`-then-rename, the same idiom the two binary writers use: a
    manifest is a production artifact too, and a truncated one is an unreadable
    audit record, which matters most on exactly the failure path this write
    also serves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    try:
        write_durably(temporary, payload)
        os.replace(temporary, path)
        sync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
