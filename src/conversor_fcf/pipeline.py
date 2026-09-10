"""The single end-to-end conversion pipeline behind `cli.run_conversion`'s seam.

`run_conversion` is the **only** call site of `write_mapcut` and `write_cortdeco`
in this package (outside tests), which is what turns "premise P13/P14 logged
once per run" from `ticket-008`'s per-call caveat (item C8) into a real
guarantee: each writer logs its own premise exactly once, and each writer runs
exactly once per run.

**Why the content-CSV offsets are re-derived, not reused (load-bearing).**
`cross_check_pair` recomputes `CutBlockOffsets` from the scalars the read-back
`mapcut` recovers, via `layout.cortdeco_block_offsets`, instead of reusing the
`CutBlockOffsets` object `write_cortdeco` was handed. Acceptance criterion 2
requires `NCOEF` to be *witnessed*, not echoed: passing the writer's own offsets
back in would leave the content CSV self-consistent under an `NCOEF` bug — the
one failure mode this whole read-back design exists to catch (see
`decomp.reader`'s module docstring for the general argument).

**Pool loading (a documented decision, not a ticket requirement).** Only the
structurally-trunk pools (every declared pool id but the highest) are read
unconditionally; the terminal pool is read only when `include_terminal` is set,
matching `reporting.eco_csv.emit_eco_csvs`'s own tolerance for a missing
terminal pool and premise P7's "pool 6 excluded" (it never builds cuts, so it
is never needed for `mapcut`/`cortdeco`). `cut_building_pools` (Requirement 5)
is given the true declared pool count regardless, so its `n_pools - 1` sanity
check stays meaningful even when the terminal pool was not loaded.

**Any failure before the pair is validated unpublishes it (Fix 1).** The
window runs from the moment `write_mapcut` starts until `cross_check_pair`
has *passed* - not from a step name, which the first version of this fix
picked and which a `--force` re-run failing at `write cortdeco` (or later)
proved wrong: `write_mapcut` had already replaced the published mapcut, so a
step-scoped condition left that fresh mapcut sitting beside the *previous*
run's cortdeco, an uncross-checked mixed pair, at both published names. The
condition is now `unvalidated_pair_published`, a plain flag set the moment
`write_mapcut` begins and cleared only once `cross_check_pair` returns
without raising. On any failure while it is set, `_reject_pair` renames
*whatever currently sits* at the two published paths to `<name>.rejected` -
including a stale file from an earlier run, which is exactly half of a mixed
pair. Once the cross-check has validated the pair, the flag is clear and nothing
downstream (the content CSVs, the manifest) can un-validate it - that half of
the asymmetry is unchanged; only what triggers the other half is now the fact
of validation, not a step's name.

One consequence of the wider window, judged fail-safe rather than a defect: a
`--force` re-run that fails *inside* `write_mapcut` itself, before it replaces
anything at the published mapcut path, still moves the *previous, already
cross-checked* pair aside to `.rejected` — the flag is set before
`write_mapcut` is even called, so it cannot distinguish "about to overwrite"
from "failed before overwriting anything". `--force`'s whole premise is that
the previous pair is about to be replaced, and both binaries' bytes are
preserved under `.rejected` rather than deleted, so nothing is lost; a
`--force` re-run of a `--force` re-run reconverts the same case from the same
inputs regardless. Documented in the README's own `.rejected` paragraph rather
than left to be discovered from the source.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from conversor_fcf.cobre.inputs_reader import CaseInputs, anticipated_thermals, read_case_inputs
from conversor_fcf.cobre.policy_reader import (
    EntitySlotRecord,
    PolicyManifest,
    StageCutPool,
    nodes_by_pool,
    read_policy_manifest,
    read_stage_cuts,
)
from conversor_fcf.config import Settings
from conversor_fcf.decomp.cortdeco_writer import (
    CutInput,
    storage_coefficients,
    write_cortdeco,
    zeroed_gnl_block,
)
from conversor_fcf.decomp.layout import (
    CutBlockOffsets,
    LayoutError,
    assert_no_travel_time,
    assert_uniform_blocks,
    cortdeco_block_offsets,
    cut_head_indices,
    sync_directory,
)
from conversor_fcf.decomp.mapcut_writer import MapcutHeader, write_mapcut
from conversor_fcf.decomp.reader import CortdecoContents, MapcutContents, read_cortdeco, read_mapcut
from conversor_fcf.logging_setup import get_logger, log_step
from conversor_fcf.mapping.rules import (
    EVIDENCED_GNL_LAG_MESES,
    SUBMARKET_COUNT,
    MappingError,
    assert_gnl_lead_time_is_evidenced,
    cut_building_pools,
    discount_factors,
    inflow_lag_drop_audit,
    load_hydro_codes,
    submarket_for_bus,
    tree_indices,
)
from conversor_fcf.paths import POLICY_MANIFEST_RELATIVE, OutputPaths
from conversor_fcf.reporting.content_csv import (
    content_csv_path,
    write_cortdeco_content_csv,
    write_mapcut_content_csv,
)
from conversor_fcf.reporting.eco_csv import emit_eco_csvs
from conversor_fcf.run_manifest import build_run_manifest, write_run_manifest

_logger = get_logger("pipeline")

MAPCUT_CONTENT_CSV_NAME = "mapcut_content.csv"
CORTDECO_CONTENT_CSV_NAME = "cortdeco_content.csv"

# The cross-check step's own label, named once so its `current_step`
# assignment and its `log_step` call cannot drift apart. No longer the
# unpublish trigger itself (Fix 1) - see `unvalidated_pair_published` in
# `run_conversion`.
_CROSS_CHECK_STEP = "cross-check the pair"

_CUTS_SUBDIR = POLICY_MANIFEST_RELATIVE.parent / "cuts"


def _cuts_path(case_dir: Path, pool_id: int) -> Path:
    return case_dir / _CUTS_SUBDIR / f"{pool_id:03d}.bin"


def load_pools(
    case_dir: Path, pool_ids: Sequence[int], include_terminal: bool
) -> dict[int, StageCutPool]:
    """Read every structurally-trunk pool, plus the terminal pool only when asked.

    "Structurally trunk" is every declared pool id but the highest — premise
    P7's own definition of the trunk, independent of `cut_building_pools`'
    semantic check, which runs on this function's *result* (Requirement 3's
    "read the trunk pools" step). Extracted from `run_conversion` so the
    `include_terminal=True` branch is testable on its own, without paying for
    the ECO CSV write that also follows from that flag in a full run.
    """
    terminal_pool_id = pool_ids[-1]
    structural_trunk_ids = pool_ids[:-1]
    pools: dict[int, StageCutPool] = {
        pool_id: read_stage_cuts(_cuts_path(case_dir, pool_id)) for pool_id in structural_trunk_ids
    }
    if include_terminal:
        pools[terminal_pool_id] = read_stage_cuts(_cuts_path(case_dir, terminal_pool_id))
    return pools


def assemble_mapcut_header(
    inputs: CaseInputs,
    manifest: PolicyManifest,
    trunk_slots: Sequence[EntitySlotRecord],
    hydro_codes: Sequence[int],
    trunk_pool_ids: Sequence[int],
) -> MapcutHeader:
    """Assemble `mapcut`'s 22 header fields from the ingested case (Requirement 4).

    `trunk_pool_ids` is `cut_building_pools`'s own result (Requirement 5), which
    `assemble_cut_inputs` also consumes: both the head table here and
    `cortdeco`'s chains derive from the same `len(trunk_pool_ids)` through
    `cut_head_indices`, the single source for both (M6).

    `trunk_slots` may be any one trunk pool's entity manifest; they are
    structurally identical (finding F-established, one pool is enough to check
    premise P3 for the whole case). `assert_gnl_lead_time_is_evidenced` guards
    `lag_meses_gnl` the same way, for premise P15: it refuses a GNL plant whose
    `lead_time_hours` the lag-month mapping has no evidence for, rather than
    guess at a formula a single data point cannot decide between.
    """
    assert_no_travel_time(trunk_slots)

    node_count = len(manifest.nodes)
    n_nodes = len(trunk_pool_ids)
    total_cuts = n_nodes * manifest.completed_iterations
    heads = cut_head_indices(total_cuts, n_nodes, node_count)

    first_node_by_stage: dict[int, int] = {}
    for node in manifest.nodes:
        first_node_by_stage.setdefault(node.stage_id, node.id + 1)

    gnl = anticipated_thermals(inputs)
    assert_gnl_lead_time_is_evidenced(gnl)
    start = date.fromisoformat(inputs.stages[0].start_date)
    return MapcutHeader(
        numero_iteracoes=manifest.completed_iterations,
        numero_cortes=total_cuts,
        numero_submercados=SUBMARKET_COUNT,
        numero_uhes=len(inputs.hydros),
        numero_cenarios=node_count,
        numero_estagios=manifest.num_stages,
        numero_semanas=n_nodes,
        # Guaranteed 0 by assert_no_travel_time above: premise P3 is a claim
        # about the case, and n_utv has nothing left to bound once it holds.
        n_utv=0,
        dia=start.day,
        mes=start.month,
        ano=start.year,
        codigos_uhes=tuple(hydro_codes),
        codigos_uhes_jusante=tuple(
            (hydro.downstream_id + 1) if hydro.downstream_id is not None else 0
            for hydro in inputs.hydros
        ),
        indice_no_arvore=tree_indices(manifest),
        indice_primeiro_no_estagio=tuple(first_node_by_stage.values()),
        patamares_por_estagio=tuple(len(stage.blocks) for stage in inputs.stages),
        registro_ultimo_corte_no=heads,
        codigos_submercados_gnl=tuple(submarket_for_bus(t.bus_id) for t in gnl),
        lag_meses_gnl=tuple(EVIDENCED_GNL_LAG_MESES for _ in gnl),
        patamares_gnl=tuple(len(inputs.stages[0].blocks) for _ in gnl),
        taxa_desconto=discount_factors(inputs.stages, inputs.annual_discount_rate),
    )


def assemble_cut_inputs(
    trunk_pools: Mapping[int, StageCutPool],
    trunk_pool_ids: Sequence[int],
    hydro_codes: Sequence[int],
    offsets: CutBlockOffsets,
) -> list[list[CutInput]]:
    """Per-node `CutInput` sequences for `write_cortdeco` (Requirement 5).

    `trunk_pool_ids` ascending puts `cuts[0]` at the lowest pool id, whose head
    is the highest one (`cut_head_indices`: `head(j) = numero_cortes - j`
    descends as `j` ascends) — `write_cortdeco`'s contract that `cuts[0]` is the
    node whose head is highest.

    `pi_varm` comes from `storage_coefficients`, `pi_gnl` from
    `zeroed_gnl_block(offsets)` under premise P14. Pieces are sorted by
    `iteration` rather than trusted in array order: on the reference deck they
    are already ascending, but nothing in `read_stage_cuts`'s contract
    guarantees that, and an out-of-order chain would silently misplace which
    cut sits at which depth.
    """
    return [
        [
            CutInput(
                intercept=piece.intercept,
                pi_varm=storage_coefficients(piece, trunk_pools[pool_id].slots, hydro_codes),
                pi_gnl=zeroed_gnl_block(offsets),
            )
            for piece in sorted(trunk_pools[pool_id].pieces, key=lambda piece: piece.iteration)
        ]
        for pool_id in trunk_pool_ids
    ]


def cross_check_pair(
    mapcut_path: Path, cortdeco_path: Path
) -> tuple[MapcutContents, CortdecoContents]:
    """Read both artifacts back and refuse a `mapcut`/`cortdeco` disagreement (M6).

    Both writers already validate their own internal layout before renaming
    into place; what neither can see is whether the *other* file was built from
    the same `numero_cortes` and node count. This check reads both back through
    the native reader and compares `numero_cortes` and the head table, so a
    wiring defect between the two call sites is caught here rather than
    reaching `ticket-013`'s oracle-based check, which is a different, more
    independent witness (a different check, not a duplicate).

    `NCOEF` is absent from `cortdeco` itself, so its `CutBlockOffsets` are
    re-derived from the read-back `mapcut`'s own scalars via
    `cortdeco_block_offsets` — never the offsets `write_cortdeco` was handed —
    for the reason the module docstring records: reusing the writer's own
    offsets would make the check self-referential.
    """
    mapcut_contents = read_mapcut(mapcut_path)
    scalars = mapcut_contents.scalars
    n_patamares = assert_uniform_blocks(mapcut_contents.patamares_por_estagio)
    offsets = cortdeco_block_offsets(
        n_uhes=scalars.numero_uhes,
        n_utv=scalars.n_utv,
        max_lag=scalars.max_lag,
        n_sbm_gnl=len(mapcut_contents.codigos_submercados_gnl),
        n_estagios=scalars.numero_estagios,
        n_patamares=n_patamares,
    )
    # numero_semanas doubles as the cut-building-node count in this project's
    # own convention (assemble_mapcut_header sets it to len(trunk_pool_ids)).
    n_nodes = scalars.numero_semanas
    cortdeco_contents = read_cortdeco(cortdeco_path, offsets, n_nodes)

    if scalars.numero_cortes != cortdeco_contents.numero_cortes:
        raise LayoutError(
            f"mapcut ({mapcut_path}) declares numero_cortes={scalars.numero_cortes} but "
            f"cortdeco ({cortdeco_path}) holds numero_cortes={cortdeco_contents.numero_cortes} "
            f"(record_count - 1): the pair disagrees"
        )

    expected_heads = cut_head_indices(scalars.numero_cortes, n_nodes, scalars.numero_cenarios)
    if mapcut_contents.cut_heads != expected_heads:
        raise LayoutError(
            f"mapcut's cut heads {mapcut_contents.cut_heads[:n_nodes]} disagree with "
            f"cut_head_indices({scalars.numero_cortes}, {n_nodes}, {scalars.numero_cenarios}) = "
            f"{expected_heads[:n_nodes]}: the pair disagrees"
        )

    _logger.info(
        "cross-checked mapcut/cortdeco: numero_cortes=%d n_nodes=%d heads=%s",
        scalars.numero_cortes,
        n_nodes,
        list(expected_heads[:n_nodes]),
    )
    return mapcut_contents, cortdeco_contents


def _reject_pair(paths: OutputPaths) -> None:
    """Rename whatever sits at the two published paths to `<name>.rejected` (Fix 1 / M6).

    Called by `run_conversion` whenever `unvalidated_pair_published` is still
    set at the point of failure - any failure from the moment `write_mapcut`
    starts until `cross_check_pair` has passed, not a specific step name. A
    step-scoped condition (the first version of this fix) left a `--force`
    re-run's freshly-written mapcut sitting beside the *previous* run's
    cortdeco when the second run failed after the mapcut write: neither
    writer's own guards catch that, because each is self-consistent on its
    own, and the mismatch is only visible by comparing the two. Renaming
    *whatever currently occupies* the two published names - rather than only
    what this run itself wrote - is deliberate: a stale file from an earlier
    run is exactly half of a mixed, uncross-checked pair, and it must not be
    mistaken for the converted result either.

    Both renames are attempted even if the first one raises. A genuine rename
    failure is logged, never propagated: it must not mask the original error
    the caller is already handling, which is why this function itself never
    raises and runs entirely inside the caller's `except` block. `os.replace`
    overwrites an existing `.rejected` from a previous run rather than
    failing on it, the same tolerance `assert_outputs_absent`'s `--force`
    path gives the two binaries themselves.

    A *first* run (clean output directory) that fails between the mapcut
    write and the cortdeco write has nothing at `paths.cortdeco` yet -
    `os.replace` raising `FileNotFoundError` there is the ordinary shape of
    that failure, not a rename defect, and is logged at DEBUG rather than
    ERROR: an ERROR here, on what is a routine failure path, is the same
    "an alarm that always fires trains an auditor to ignore alarms" mistake
    `cut_building_pools`'s own WARNING was fixed for. Any other `OSError` -
    the target exists but cannot be renamed for some other reason - is a
    real rename failure and is still logged at ERROR.
    """
    for path in (paths.mapcut, paths.cortdeco):
        rejected = path.with_name(path.name + ".rejected")
        try:
            os.replace(path, rejected)
            sync_directory(path.parent)
        except FileNotFoundError:
            _logger.debug(
                "nothing to reject at %s: the validation window failed before this file was "
                "ever written",
                path,
            )
        except OSError:
            _logger.exception(
                "failed to rename %s to %s after a failure in the validation window (Fix 1 / "
                "M6); the original error is still raised",
                path,
                rejected,
            )


def run_conversion(
    case_dir: Path,
    revision: str,
    paths: OutputPaths,
    settings: Settings,
    include_terminal: bool,
) -> int:
    """Run the whole conversion: ingest, map, write, cross-check, report.

    The single call site of `write_mapcut`/`write_cortdeco` in this package
    (`cli.py`'s own seam calls only this function), so "premise logged once per
    run" is a real guarantee here (Requirement 7, closing `ticket-008` item C8).

    Every step runs inside `log_step`, which already records a `fail` line with
    the traceback on any exception. This function additionally tracks which
    step was active, so the run manifest can be written with `status="failed"`
    and that step's label (M7) before the exception propagates to `cli.main`'s
    exit-code mapping. A failed run therefore still leaves an audit-grade
    manifest naming what was attempted.
    """
    _logger.info(
        "resolved plan: case=%s revision=%s root=%s mapcut=%s cortdeco=%s eco=%s content=%s "
        "manifest=%s log=%s include_terminal=%s",
        case_dir,
        revision,
        paths.root,
        paths.mapcut.name,
        paths.cortdeco.name,
        paths.eco_dir,
        paths.content_dir,
        paths.run_manifest,
        paths.log,
        include_terminal,
    )

    mapcut_content_path = content_csv_path(
        paths.root, settings.output.content_subdirectory, MAPCUT_CONTENT_CSV_NAME
    )
    cortdeco_content_path = content_csv_path(
        paths.root, settings.output.content_subdirectory, CORTDECO_CONTENT_CSV_NAME
    )
    outputs = {
        "mapcut": paths.mapcut,
        "cortdeco": paths.cortdeco,
        "eco_dir": paths.eco_dir,
        "content_dir": paths.content_dir,
        "mapcut_content": mapcut_content_path,
        "cortdeco_content": cortdeco_content_path,
        "log": paths.log,
    }

    current_step = "read settings-derived inputs"
    # Fix 1 / M6: the fact that unpublishes the pair, not a step name. Set the
    # moment `write_mapcut` starts, cleared only once `cross_check_pair`
    # returns without raising - see the module docstring for why a step-scoped
    # condition was wrong.
    unvalidated_pair_published = False
    try:
        with log_step(_logger, current_step):
            inputs = read_case_inputs(case_dir)
            hydro_codes = load_hydro_codes(Path(settings.conversion.hydro_codes_path))
            _logger.info(
                "read %d hydro code(s) from %s",
                len(hydro_codes),
                settings.conversion.hydro_codes_path,
            )

        current_step = "read the policy manifest"
        with log_step(_logger, current_step):
            manifest = read_policy_manifest(case_dir / POLICY_MANIFEST_RELATIVE)

        current_step = "read the trunk pools"
        with log_step(_logger, current_step):
            pool_ids = sorted(nodes_by_pool(manifest))
            terminal_pool_id = pool_ids[-1]
            structural_trunk_ids = pool_ids[:-1]
            pools = load_pools(case_dir, pool_ids, include_terminal)
            _logger.info(
                "read %d pool(s): %s",
                len(pools),
                {pid: (len(pool.pieces), len(pool.slots)) for pid, pool in pools.items()},
            )

            # cut_building_pools sees only the structurally-trunk pools, never
            # `pools` whole: the terminal pool must never be a candidate, or
            # --include-terminal-pool could let it into trunk_pool_ids and
            # drive numero_semanas, the head table and the chain count -
            # against premise P7, which excludes it unconditionally. The
            # expected count still comes from the manifest's own declared pool
            # list, not len(pools), which would undercount whenever the
            # terminal pool was not loaded (the common, default-settings case).
            trunk_pool_ids = cut_building_pools(
                {pid: pools[pid] for pid in structural_trunk_ids},
                expected_count=len(pool_ids) - 1,
            )
            if not trunk_pool_ids:
                raise MappingError(
                    "no pool built cuts (every pool's populated_count equals its "
                    "warm_start_count): this checkpoint carries only seeded or warm-start "
                    "cuts, so there is no FCF to convert"
                )
            trunk_pools = {pool_id: pools[pool_id] for pool_id in trunk_pool_ids}
            inflow_lag_drop_audit(trunk_pools)

        current_step = "emit the ECO CSVs"
        with log_step(_logger, current_step):
            emit_eco_csvs(
                pools,
                pool_ids,
                paths.root,
                settings.output.eco_subdirectory,
                terminal_pool_id,
                include_terminal,
            )

        current_step = "build the mapping quantities"
        with log_step(_logger, current_step):
            n_nodes = len(trunk_pool_ids)
            numero_cortes = n_nodes * manifest.completed_iterations
            gnl = anticipated_thermals(inputs)
            n_patamares = assert_uniform_blocks(tuple(len(stage.blocks) for stage in inputs.stages))
            offsets = cortdeco_block_offsets(
                n_uhes=len(hydro_codes),
                n_utv=0,
                max_lag=0,
                n_sbm_gnl=len(gnl),
                n_estagios=manifest.num_stages,
                n_patamares=n_patamares,
            )
            heads = cut_head_indices(numero_cortes, n_nodes, len(manifest.nodes))
            _logger.info(
                "mapping quantities: n_uhes=%d n_estagios=%d n_patamares=%d n_sbm_gnl=%d ncoef=%d "
                "heads=%s",
                len(hydro_codes),
                manifest.num_stages,
                n_patamares,
                len(gnl),
                offsets.ncoef,
                list(heads[:n_nodes]),
            )

        current_step = "assemble the mapcut header"
        with log_step(_logger, current_step):
            header = assemble_mapcut_header(
                inputs, manifest, trunk_pools[trunk_pool_ids[0]].slots, hydro_codes, trunk_pool_ids
            )

        current_step = "write mapcut"
        unvalidated_pair_published = True
        with log_step(_logger, current_step):
            write_mapcut(header, paths.mapcut)

        current_step = "assemble the cut inputs"
        with log_step(_logger, current_step):
            cuts = assemble_cut_inputs(trunk_pools, trunk_pool_ids, hydro_codes, offsets)

        current_step = "write cortdeco"
        with log_step(_logger, current_step):
            write_cortdeco(
                cuts, paths.cortdeco, numero_cortes=header.numero_cortes, offsets=offsets
            )

        current_step = _CROSS_CHECK_STEP
        with log_step(_logger, current_step):
            mapcut_contents, cortdeco_contents = cross_check_pair(paths.mapcut, paths.cortdeco)
            # Reached only on success: cross_check_pair raising leaves this
            # unset, and the pair is still unpublished on failure (Fix 1).
            unvalidated_pair_published = False

        current_step = "read both back and write the content CSVs"
        with log_step(_logger, current_step):
            mapcut_rows = write_mapcut_content_csv(mapcut_contents, mapcut_content_path)
            cortdeco_rows = write_cortdeco_content_csv(
                cortdeco_contents,
                cortdeco_content_path,
                mapcut_contents.codigos_uhes,
                mapcut_contents.codigos_submercados_gnl,
                mapcut_contents.scalars.numero_estagios,
                assert_uniform_blocks(mapcut_contents.patamares_por_estagio),
                mapcut_contents.scalars.n_utv,
                mapcut_contents.scalars.max_lag,
            )
            _logger.info(
                "wrote content CSVs: mapcut_content rows=%d cortdeco_content rows=%d",
                mapcut_rows,
                cortdeco_rows,
            )
    except BaseException:
        # BaseException, not Exception: an interrupt mid-conversion must still
        # leave a manifest naming what was attempted (M7).
        if unvalidated_pair_published:
            # Fix 1 / M6: any failure between the mapcut write and a passed
            # cross-check unpublishes the pair (see the module docstring for
            # why this is a fact, not a step name).
            _reject_pair(paths)
        failure_manifest = build_run_manifest(
            case_path=case_dir,
            revision=revision,
            settings=settings,
            outputs=outputs,
            status="failed",
            failed_step=current_step,
        )
        try:
            write_run_manifest(failure_manifest, paths.run_manifest)
        except OSError:
            # Fix 2: a manifest-write failure on the failure path (ENOSPC is
            # plausible exactly when the conversion just failed on a full
            # disk) must never replace the original error - that would stop
            # it from ever reaching `cli.main`'s classified exit code and the
            # audit log's ERROR line. Logged, then fall through to the
            # original `raise`, the same non-masking discipline `_reject_pair`
            # already follows for its own rename.
            _logger.exception(
                "failed to write the failure manifest to %s; the original error is still raised",
                paths.run_manifest,
            )
        raise

    with log_step(_logger, "write the run manifest"):
        success_manifest = build_run_manifest(
            case_path=case_dir,
            revision=revision,
            settings=settings,
            outputs=outputs,
            status="ok",
            failed_step=None,
        )
        write_run_manifest(success_manifest, paths.run_manifest)

    return 0
