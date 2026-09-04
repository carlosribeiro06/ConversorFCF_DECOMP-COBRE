"""Cobre-to-DECOMP mapping rules, one named function per premise.

Each premise lives in exactly one function so it stays individually testable,
documented and revertible. Nothing here performs I/O except `load_hydro_codes`,
and nothing here writes a byte of either binary.

Two of these rules encode findings that contradict what the plan first assumed,
and both are load-bearing:

- The discount day-count basis is 365.25, not 365. Solving the basis from the
  reference series' per-stage ratio yields exactly 365.25, and the formula is
  then exact rather than approximate.
- The GNL stage axis is keyed off `delivery_date`, never off `subindex`. The
  anticipated-thermal slots are a ring buffer that rotates between stages, so a
  ring position means a different delivery month in different pools.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conversor_fcf.cobre.entities import has_delivery_date, index_slots
from conversor_fcf.cobre.inputs_reader import BlockInfo, BusInfo, StageInfo
from conversor_fcf.cobre.policy_reader import (
    EntitySlotRecord,
    PolicyManifest,
    StageCutPool,
)
from conversor_fcf.logging_setup import get_logger

# Premise P2: DECOMP works in 10^3 R$ while Cobre works in R$. Verified to 1 ulp
# on a matched pair. Not configurable.
DECOMP_COST_DIVISOR = 1000.0

# Premise P9. Solving the basis from the reference ratio 0.997830417741 gives
# exactly 365.250000; with 365 the series is wrong by up to 9e-6.
DAYS_PER_YEAR = 365.25

# Premise P6. Bus 5 is the interconnection node, not a submarket.
EXCLUDED_BUS_ID = 5
EXCLUDED_BUS_NAME = "IV"
SUBMARKET_COUNT = 5

_logger = get_logger("mapping")


class MappingError(Exception):
    """Raised when an input violates a mapping rule."""


@dataclass(frozen=True)
class DeliveryGroup:
    """A GNL delivery month and its rank within one pool's own set of months.

    `pool_rank` is **not** a study-global DECOMP stage index. The same month
    ranks differently in different pools: `20260501` is rank 2 in pool 0 and
    rank 1 in pools 1-5, because each pool sees a different set of months. A
    writer that needs a study stage axis must key off `delivery_date`.
    """

    delivery_date: int
    pool_rank: int


@dataclass(frozen=True)
class InflowLagAudit:
    """Counted record of the inflow-lag coefficients premise P8 drops."""

    per_pool: tuple[tuple[int, int], ...]
    total: int


def to_decomp_cost(value: float) -> float:
    """Convert one Cobre currency value to DECOMP's 10^3 R$ (premise P2)."""
    return value / DECOMP_COST_DIVISOR


def to_decomp_costs(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a coefficient vector to DECOMP's 10^3 R$ (premise P2)."""
    return np.asarray(values, dtype=np.float64) / DECOMP_COST_DIVISOR


def negate_gnl(value: float) -> float:
    """Flip a GNL coefficient's sign (premise P4).

    The single site of the GNL sign convention: Cobre reports these negative and
    DECOMP positive. No rule other than this one and its vectorised twin below
    changes a sign anywhere in the package.
    """
    return -value


def negate_gnl_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip a GNL coefficient vector's sign (premise P4)."""
    return -np.asarray(values, dtype=np.float64)


def _reject_repeated_keys(path: Path) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """A json object hook that refuses a repeated key instead of collapsing it.

    `json.loads` keeps the last value for a repeated key and discards the rest,
    so without this hook a map holding `"0"` twice loads as a shorter, wrong map
    with no error at all.
    """

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise MappingError(f"{path} repeats key {key!r}")
            seen.add(key)
        return dict(pairs)

    return hook


def load_hydro_codes(path: Path) -> tuple[int, ...]:
    """Read the Cobre-position-to-DECOMP-plant-code map in positional order."""
    if not path.is_file():
        raise MappingError(f"hydro code map not found: {path}")
    try:
        raw: Any = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_repeated_keys(path)
        )
    except json.JSONDecodeError as exc:
        raise MappingError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise MappingError(f"{path} must hold a JSON object, got {type(raw).__name__}")
    if not raw:
        raise MappingError(f"{path} holds no entries")

    positions: dict[int, int] = {}
    for key, value in raw.items():
        try:
            position = int(key)
        except ValueError as exc:
            raise MappingError(f"{path} key {key!r} is not an integer position") from exc
        # Checked against the round-trip, not just parseability: "00", "+0", " 1"
        # and non-ASCII digits all parse to a position that already exists, so
        # N+1 entries would collapse into N with a still-contiguous range.
        if key != str(position):
            raise MappingError(
                f"{path} key {key!r} is not the canonical form of position {position}; "
                f"keys must be exactly '0'..'N-1'"
            )
        if type(value) is not int:
            raise MappingError(
                f"{path} entry {key!r} must map to an int, got {type(value).__name__}"
            )
        positions[position] = value

    expected = set(range(len(positions)))
    missing = sorted(expected - positions.keys())
    if missing:
        raise MappingError(
            f"{path} is not contiguous from 0: missing position(s) {missing[:10]} "
            f"of {len(positions)} entries"
        )
    codes = tuple(positions[position] for position in sorted(positions))
    # Two positions mapping to one DECOMP plant code would silently merge two
    # reservoirs into one column of pi_varm.
    repeated = sorted(code for code, count in Counter(codes).items() if count > 1)
    if repeated:
        raise MappingError(
            f"{path} maps more than one position to DECOMP plant code(s) {repeated[:10]}"
        )
    return codes


def hydro_code_for_position(codes: Sequence[int], position: int) -> int:
    """The DECOMP plant code for a Cobre entity-manifest position (finding F6)."""
    if not 0 <= position < len(codes):
        raise MappingError(f"hydro position {position} is outside the {len(codes)}-entry code map")
    return codes[position]


def submarket_for_bus(bus_id: int) -> int:
    """Map a Cobre bus id to a DECOMP submarket number (premise P6)."""
    if bus_id == EXCLUDED_BUS_ID:
        raise MappingError(f"bus {bus_id} is the interconnection node and has no submarket")
    if not 0 <= bus_id < EXCLUDED_BUS_ID:
        raise MappingError(f"bus {bus_id} is outside the {SUBMARKET_COUNT} submarket buses")
    return bus_id + 1


def select_submarket_buses(buses: Sequence[BusInfo]) -> tuple[BusInfo, ...]:
    """The buses that become DECOMP submarkets, excluding the interconnection node.

    Excluded by id, but the name is asserted too: a reordered bus list would
    otherwise shift every submarket silently.
    """
    excluded = [bus for bus in buses if bus.id == EXCLUDED_BUS_ID]
    if len(excluded) != 1:
        raise MappingError(
            f"expected exactly one bus with id {EXCLUDED_BUS_ID}, found {len(excluded)}"
        )
    if excluded[0].name != EXCLUDED_BUS_NAME:
        raise MappingError(
            f"bus {EXCLUDED_BUS_ID} is named {excluded[0].name!r}, expected "
            f"{EXCLUDED_BUS_NAME!r}; the bus list may have been reordered"
        )
    selected = tuple(bus for bus in buses if bus.id != EXCLUDED_BUS_ID)
    if len(selected) != SUBMARKET_COUNT:
        raise MappingError(f"expected {SUBMARKET_COUNT} submarket buses, found {len(selected)}")
    return selected


def gnl_block_weights(blocks: Sequence[BlockInfo]) -> tuple[float, ...]:
    """Hours-weighted partition of a stage across its load blocks (premise P5).

    A partition summing to 1.0, not a scaling, so the disaggregated total is
    preserved. The block split differs in every stage of the reference deck, so
    this must be called with the stage's own blocks and can carry no constant.
    """
    if not blocks:
        raise MappingError("cannot weight an empty block list")
    # Each block is checked, not just the sum: a negative hour yields a negative
    # weight that still sums to 1.0, which would flip that block's GNL sign
    # outside negate_gnl, the only place allowed to change a sign.
    for block in blocks:
        if not math.isfinite(block.hours) or block.hours < 0.0:
            raise MappingError(
                f"block {block.id} ({block.name}) has hours {block.hours!r}; block hours must be "
                f"finite and non-negative"
            )
    total = sum(block.hours for block in blocks)
    if total <= 0.0:
        raise MappingError(f"block hours must sum to a positive value, got {total}")
    return tuple(block.hours / total for block in blocks)


def delivery_slot_map(slots: Sequence[EntitySlotRecord]) -> dict[int, DeliveryGroup]:
    """Group anticipated-thermal positions by delivery month, per pool.

    Keyed off `delivery_date`, never off `subindex` (finding F2): the slots are a
    ring buffer whose positions rotate between stages, so a ring position means a
    different delivery month in different pools.

    Each position gets its month and that month's ascending rank **within this
    pool**. The rank is deliberately not a study stage index — see
    `DeliveryGroup` — and the month travels with it so a writer never has to
    re-read the slots to place a coefficient on a global axis.
    """
    positions = index_slots(slots).anticipated_thermal
    if not positions:
        return {}

    for position in positions:
        if not has_delivery_date(slots[position]):
            raise MappingError(
                f"anticipated-thermal slot at position {position} carries no delivery date "
                f"({slots[position].delivery_date}), so its DECOMP stage slot is undefined"
            )

    dates = sorted({slots[position].delivery_date for position in positions})
    ranked = {delivery_date: rank for rank, delivery_date in enumerate(dates, start=1)}
    return {
        position: DeliveryGroup(
            delivery_date=slots[position].delivery_date,
            pool_rank=ranked[slots[position].delivery_date],
        )
        for position in positions
    }


def _parse_start_date(stage: StageInfo) -> date:
    try:
        return date.fromisoformat(stage.start_date)
    except ValueError as exc:
        raise MappingError(
            f"stage {stage.id} start_date {stage.start_date!r} is not ISO-8601"
        ) from exc


def discount_factors(stages: Sequence[StageInfo], annual_rate: float) -> tuple[float, ...]:
    """Per-stage discount factors, duration-proportional on a 365.25-day year.

    Premise P9, corrected. The exponent is the cumulative day count from the first
    stage's start to each stage's start, so a stage's own duration affects only
    the stages that follow it.
    """
    if not stages:
        return ()
    if annual_rate <= -1.0:
        raise MappingError(f"annual discount rate must exceed -1, got {annual_rate}")

    origin = _parse_start_date(stages[0])
    factors = []
    for stage in stages:
        elapsed_days = (_parse_start_date(stage) - origin).days
        if elapsed_days < 0:
            raise MappingError(
                f"stage {stage.id} starts {abs(elapsed_days)} days before stage 0; "
                f"the stage list must be chronological"
            )
        factors.append((1.0 + annual_rate) ** (-elapsed_days / DAYS_PER_YEAR))
    return tuple(factors)


def tree_indices(manifest: PolicyManifest) -> tuple[int, ...]:
    """DECOMP's 1-based parent index per node, from the policy graph edges.

    Finding F7: a `target_id -> source_id` parent map, plus one for DECOMP's
    1-based indexing, with the root as its own parent.

    Every degenerate graph raises rather than being narrowed. `indice_no_arvore`
    holds exactly one parent per node, so a node with two incoming edges is an
    unrepresentable input, and picking one of them would emit a structurally
    valid, plausible, wrong tree whose content depended on edge order in the
    file. Node ids are also used as 1-based positions, so they must be the
    contiguous range `0..len(nodes)-1`.
    """
    node_ids = [node.id for node in manifest.nodes]
    if node_ids != list(range(len(node_ids))):
        raise MappingError(
            f"node ids must be the contiguous range 0..{len(node_ids) - 1} because they are "
            f"written as 1-based positions; got {node_ids[:8]}..."
        )
    known = set(node_ids)

    parents: dict[int, int] = {}
    for edge in manifest.edges:
        if edge.source_id not in known:
            raise MappingError(
                f"edge {edge.source_id} -> {edge.target_id} names an unknown source node"
            )
        if edge.target_id not in known:
            raise MappingError(
                f"edge {edge.source_id} -> {edge.target_id} names an unknown target node"
            )
        if edge.target_id in parents:
            raise MappingError(
                f"node {edge.target_id} has more than one parent ({parents[edge.target_id]} and "
                f"{edge.source_id}); DECOMP records exactly one parent per node"
            )
        parents[edge.target_id] = edge.source_id

    roots = [node_id for node_id in node_ids if node_id not in parents]
    if len(roots) != 1:
        raise MappingError(f"expected exactly one root node, found {len(roots)}: {roots[:8]}")
    return tuple(parents.get(node_id, node_id) + 1 for node_id in node_ids)


def cut_building_pools(pools: Mapping[int, StageCutPool]) -> tuple[int, ...]:
    """The pools that actually built cuts, ascending (finding F8).

    A pool whose populated count merely equals its warm-start count carries only
    seeded cuts. That is the semantic rule; it coincides with `n_pools - 1` on the
    reference deck, and a disagreement is logged rather than silently preferred.
    """
    selected = tuple(
        sorted(
            pool_id
            for pool_id, pool in pools.items()
            if pool.populated_count > pool.warm_start_count
        )
    )
    if not pools:
        return ()
    expected = len(pools) - 1
    if len(selected) != expected:
        _logger.warning(
            "cut-building pools by populated_count > warm_start_count is %s, %d pool(s), but "
            "n_pools - 1 is %d; the populated-count rule governs and those pools are what gets "
            "written",
            list(selected),
            len(selected),
            expected,
        )
    return selected


def inflow_lag_drop_audit(pools: Mapping[int, StageCutPool]) -> InflowLagAudit:
    """Count the inflow-lag coefficients premise P8 drops, per pool.

    A counted audit, never a silent truncation. On the reference deck only the
    terminal pool carries inflow-lag slots, and premise P7 excludes it from
    conversion, so the conversion path itself drops nothing.
    """
    per_pool = []
    for pool_id in sorted(pools):
        pool = pools[pool_id]
        lag_slots = len(index_slots(pool.slots).inflow_lag)
        per_pool.append((pool_id, lag_slots * len(pool.pieces)))

    total = sum(count for _, count in per_pool)
    if total:
        _logger.warning(
            "premise P8 drops %d inflow-lag coefficients: %s",
            total,
            ", ".join(f"pool {pool_id}={count}" for pool_id, count in per_pool if count),
        )
    else:
        _logger.info("premise P8 drops no inflow-lag coefficients on this case")
    return InflowLagAudit(per_pool=tuple(per_pool), total=total)
