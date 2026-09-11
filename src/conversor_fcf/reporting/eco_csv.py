"""ECO CSVs: the Cobre cuts exactly as read, before any transformation.

This is the pre-transformation audit artifact. No division by 1000, no sign
flip, no axis remapping, no reordering: one row per affine piece, one column per
coefficient in `entity_manifest` order. `is_active` and `in_active_cut_indices`
are recorded as columns and never used as filters (premise P10).

Floats are written through `repr(float(value))`. The `float()` is load-bearing:
`repr` of a `numpy.float64` yields `np.float64(...)` rather than a bare literal,
and `pandas.DataFrame.to_csv` would truncate to six significant digits, either of
which would break the exact round-trip this artifact exists to guarantee.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TextIO

from conversor_fcf.cobre.entities import slot_label
from conversor_fcf.cobre.policy_reader import StageCutPool
from conversor_fcf.logging_setup import get_logger

FIXED_COLUMNS = (
    "pool_stage_id",
    "node_id",
    "piece_id",
    "slot_index",
    "iteration",
    "forward_pass_index",
    "is_active",
    "in_active_cut_indices",
    "intercept",
)

_logger = get_logger("eco_csv")


def eco_csv_path(output_dir: Path, eco_subdir: str, pool_id: int) -> Path:
    """Destination of a pool's plain-CSV ECO artifact."""
    return output_dir / eco_subdir / f"eco_cuts_pool_{pool_id:03d}.csv"


def eco_csv_gzip_path(output_dir: Path, eco_subdir: str, pool_id: int) -> Path:
    """Destination of a pool's gzipped ECO artifact."""
    plain = eco_csv_path(output_dir, eco_subdir, pool_id)
    return plain.with_name(plain.name + ".gz")


def _format_float(value: float) -> str:
    return repr(float(value))


def _write_rows(pool: StageCutPool, handle: TextIO) -> int:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow([*FIXED_COLUMNS, *(slot_label(slot) for slot in pool.slots)])

    active = set(pool.active_cut_indices)
    slot_count = len(pool.slots)
    for index, piece in enumerate(pool.pieces):
        coefficient_count = len(piece.coefficients)
        if coefficient_count != slot_count:
            raise ValueError(
                f"piece index {index} of pool stage_id={pool.stage_id} node_id={pool.node_id} "
                f"carries {coefficient_count} coefficients but the pool has {slot_count} slots"
            )
        writer.writerow(
            [
                pool.stage_id,
                pool.node_id,
                piece.piece_id,
                piece.slot_index,
                piece.iteration,
                piece.forward_pass_index,
                piece.is_active,
                index in active,
                _format_float(piece.intercept),
                *(_format_float(value) for value in piece.coefficients),
            ]
        )
    return len(pool.pieces)


def _log_written(pool: StageCutPool, path: Path, rows: int) -> None:
    _logger.info(
        "wrote ECO CSV %s stage_id=%d node_id=%d rows=%d columns=%d",
        path,
        pool.stage_id,
        pool.node_id,
        rows,
        len(FIXED_COLUMNS) + len(pool.slots),
    )


def write_eco_csv(pool: StageCutPool, path: Path) -> int:
    """Write the pool verbatim as plain CSV, returning the number of data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        rows = _write_rows(pool, handle)
    _log_written(pool, path, rows)
    return rows


def write_eco_csv_gzip(pool: StageCutPool, path: Path) -> int:
    """Write the pool verbatim as gzipped CSV, returning the number of data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Text mode with newline="" so the csv writer does not emit \r\r\n.
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        rows = _write_rows(pool, handle)
    _log_written(pool, path, rows)
    return rows


def select_eco_pools(
    pool_ids: Iterable[int], terminal_pool_id: int, include_terminal: bool
) -> tuple[int, ...]:
    """The pool ids to emit, ascending, with the terminal pool excluded by default."""
    ordered = tuple(sorted(set(pool_ids)))
    if not ordered:
        raise ValueError("pool_ids must not be empty")
    if include_terminal:
        return ordered
    return tuple(pool_id for pool_id in ordered if pool_id != terminal_pool_id)


def emit_eco_csvs(
    pools: Mapping[int, StageCutPool],
    declared_pool_ids: Iterable[int],
    output_dir: Path,
    eco_subdir: str,
    terminal_pool_id: int,
    include_terminal: bool,
) -> dict[int, Path]:
    """Write the selected pools' ECO artifacts, gzip for the terminal pool only.

    Selection comes from `declared_pool_ids` (the case's own pool count), not
    from the keys of `pools`. Deriving it from the loaded mapping would make a
    trunk pool that failed to load indistinguishable from a case that never had
    it, and silently writing a short audit trail is the failure mode that
    matters for an artifact whose purpose is to prove what was ingested.
    A missing non-terminal pool therefore raises `KeyError`; only the terminal
    pool may be absent, and only when `include_terminal` is false.
    """
    selected = select_eco_pools(declared_pool_ids, terminal_pool_id, include_terminal)
    _logger.info("emitting ECO CSVs for pools %s", list(selected))
    if not include_terminal:
        _logger.info(
            "skipping terminal pool %d: DECOMP's last stage builds no cuts, so the pool is not "
            "converted and its ECO CSV is an opt-in inspection aid",
            terminal_pool_id,
        )

    written: dict[int, Path] = {}
    for pool_id in selected:
        # KeyError here means the caller selected a pool it did not load, which is
        # a caller bug rather than a data condition.
        pool = pools[pool_id]
        if pool_id == terminal_pool_id:
            _logger.warning(
                "writing the terminal pool ECO CSV for pool %d: %d pieces x %d coefficients, "
                "gzipped",
                pool_id,
                len(pool.pieces),
                len(pool.slots),
            )
            path = eco_csv_gzip_path(output_dir, eco_subdir, pool_id)
            write_eco_csv_gzip(pool, path)
        else:
            path = eco_csv_path(output_dir, eco_subdir, pool_id)
            write_eco_csv(pool, path)
        written[pool_id] = path
    return written
