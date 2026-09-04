import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.cobre.entities import DELIVERY_DATE_SENTINEL
from conversor_fcf.cobre.inputs_reader import BlockInfo, BusInfo, StageInfo
from conversor_fcf.cobre.policy_reader import (
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyManifest,
    StageCutPool,
)
from conversor_fcf.mapping.rules import (
    DAYS_PER_YEAR,
    DECOMP_COST_DIVISOR,
    InflowLagAudit,
    MappingError,
    cut_building_pools,
    delivery_slot_map,
    discount_factors,
    gnl_block_weights,
    hydro_code_for_position,
    inflow_lag_drop_audit,
    load_hydro_codes,
    negate_gnl,
    negate_gnl_array,
    select_submarket_buses,
    submarket_for_bus,
    to_decomp_cost,
    to_decomp_costs,
    tree_indices,
)
from conversor_fcf.run_manifest import PREMISES


@pytest.fixture(autouse=True)
def propagating_package_logger() -> Iterator[None]:
    """caplog reads through the root logger, so propagation must be on.

    Without this, a test asserting that *no* WARNING was emitted passes trivially
    whenever an earlier test left `propagate` false.
    """
    logger = logging.getLogger("conversor_fcf")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


REFERENCE_BUSES = (
    BusInfo(id=0, name="SE"),
    BusInfo(id=1, name="S"),
    BusInfo(id=2, name="NE"),
    BusInfo(id=3, name="N"),
    BusInfo(id=4, name="FC"),
    BusInfo(id=5, name="IV"),
)


def _slot(entity_type: int, entity_id: int, subindex: int, delivery_date: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        subindex=subindex,
        was_active=True,
        delivery_date=delivery_date,
    )


def _stage(stage_id: int, start: str, end: str, hours: tuple[float, ...]) -> StageInfo:
    return StageInfo(
        id=stage_id,
        start_date=start,
        end_date=end,
        blocks=tuple(BlockInfo(id=i, name=f"B{i}", hours=h) for i, h in enumerate(hours)),
        storage_state=True,
        inflow_lags_state=False,
    )


def _pool(
    pool_id: int,
    populated: int,
    warm_start: int,
    slots: tuple[EntitySlotRecord, ...] = (),
    pieces: int = 1,
) -> StageCutPool:
    piece = AffinePieceRecord(
        piece_id=0,
        slot_index=0,
        iteration=0,
        forward_pass_index=0,
        intercept=0.0,
        coefficients=np.zeros(len(slots), dtype=np.float64),
        is_active=True,
    )
    return StageCutPool(
        stage_id=pool_id,
        node_id=pool_id,
        graph_stage_id=pool_id,
        state_dimension=2211,
        capacity=100,
        warm_start_count=warm_start,
        populated_count=populated,
        cost_scale_factor=1000000.0,
        slots=slots,
        pieces=tuple(piece for _ in range(pieces)),
        active_cut_indices=(),
    )


# --- premise P2 --------------------------------------------------------------


def test_cost_divisor_is_exactly_one_thousand() -> None:
    assert DECOMP_COST_DIVISOR == 1000.0
    assert to_decomp_cost(2312687234790.0674) == 2312687234790.0674 / 1000.0


def test_cost_conversion_is_vectorised_identically() -> None:
    values = np.array([-5.028946e7, 3.181296e5, 0.0], dtype=np.float64)
    converted = to_decomp_costs(values)
    # Anchored on literals, not only on agreement with the scalar function.
    assert [float(v) for v in converted] == [-50289.46, 318.1296, 0.0]
    assert [float(v) for v in converted] == [to_decomp_cost(float(v)) for v in values]
    assert converted.dtype == np.float64


# --- premise P4 --------------------------------------------------------------


def test_gnl_negation_is_a_pure_sign_flip() -> None:
    assert negate_gnl(-542599.0116637845) == 542599.0116637845
    assert negate_gnl(0.0) == 0.0
    flipped = negate_gnl_array(np.array([-1.5, 0.0, 2.5], dtype=np.float64))
    assert [float(v) for v in flipped] == [1.5, 0.0, -2.5]


def test_gnl_negation_does_not_scale() -> None:
    """P4 and P2 are separate rules; neither may absorb the other."""
    assert abs(negate_gnl(-1000.0)) == 1000.0


# --- premise P6 --------------------------------------------------------------


@pytest.mark.parametrize(("bus_id", "submarket"), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
def test_submarket_is_bus_id_plus_one(bus_id: int, submarket: int) -> None:
    assert submarket_for_bus(bus_id) == submarket


@pytest.mark.parametrize("bus_id", [5, 6, -1])
def test_non_submarket_buses_are_rejected(bus_id: int) -> None:
    with pytest.raises(MappingError):
        submarket_for_bus(bus_id)


def test_selecting_submarket_buses_drops_the_interconnection_node() -> None:
    selected = select_submarket_buses(REFERENCE_BUSES)
    assert [bus.id for bus in selected] == [0, 1, 2, 3, 4]
    assert [submarket_for_bus(bus.id) for bus in selected] == [1, 2, 3, 4, 5]


def test_a_renamed_bus_five_is_rejected_rather_than_shifting_every_submarket() -> None:
    reordered = REFERENCE_BUSES[:5] + (BusInfo(id=5, name="SUL"),)
    with pytest.raises(MappingError, match="reordered"):
        select_submarket_buses(reordered)


def test_missing_interconnection_bus_is_rejected() -> None:
    with pytest.raises(MappingError, match="exactly one bus with id 5"):
        select_submarket_buses(REFERENCE_BUSES[:5])


def test_wrong_submarket_count_is_rejected() -> None:
    with pytest.raises(MappingError, match="expected 5 submarket buses"):
        select_submarket_buses(REFERENCE_BUSES[:3] + (REFERENCE_BUSES[5],))


# --- premise P5 --------------------------------------------------------------


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        ((24.0, 65.0, 79.0), (24 / 168, 65 / 168, 79 / 168)),
        ((15.0, 64.0, 89.0), (15 / 168, 64 / 168, 89 / 168)),
        ((12.0, 61.0, 95.0), (12 / 168, 61 / 168, 95 / 168)),
        ((51.0, 226.0, 323.0), (51 / 600, 226 / 600, 323 / 600)),
    ],
)
def test_block_weights_partition_each_real_stage_split(
    hours: tuple[float, ...], expected: tuple[float, ...]
) -> None:
    """Every split the reference deck actually uses, including stage 6's 600 hours."""
    weights = gnl_block_weights(_stage(0, "2026-04-25", "2026-05-02", hours).blocks)
    assert weights == expected
    assert sum(weights) == pytest.approx(1.0, abs=1e-15)


def test_block_weights_reject_an_empty_or_zero_stage() -> None:
    with pytest.raises(MappingError, match="empty block list"):
        gnl_block_weights(())
    with pytest.raises(MappingError, match="positive value"):
        gnl_block_weights(_stage(0, "2026-04-25", "2026-05-02", (0.0, 0.0)).blocks)


# --- premise P9 --------------------------------------------------------------

REFERENCE_DISCOUNTS = (
    1.0,
    0.997830417741,
    0.995665542570,
    0.993505364273,
    0.991349872661,
    0.989199057565,
    0.987052908840,
)


def _seven_weekly_stages() -> tuple[StageInfo, ...]:
    starts = [
        "2026-04-25",
        "2026-05-02",
        "2026-05-09",
        "2026-05-16",
        "2026-05-23",
        "2026-05-30",
        "2026-06-06",
    ]
    return tuple(_stage(k, start, start, (168.0,)) for k, start in enumerate(starts))


def test_days_per_year_is_the_solved_basis() -> None:
    assert DAYS_PER_YEAR == 365.25


def test_discount_series_matches_the_reference_deck() -> None:
    factors = discount_factors(_seven_weekly_stages(), 0.12)
    assert len(factors) == 7
    for computed, reference in zip(factors, REFERENCE_DISCOUNTS, strict=True):
        assert abs(computed - reference) < 1e-12


def test_discount_factors_reject_the_365_day_basis() -> None:
    """The deliberate mutation this ticket's suite must catch.

    It compares `discount_factors`' real output against the 365-basis series, so
    it fails if the function is broken in any way. An earlier version of this
    test compared two literal series and never called the function at all, which
    made it an arithmetic tautology that no change to the module could break.
    """
    factors = discount_factors(_seven_weekly_stages(), 0.12)
    wrong_basis = tuple(1.12 ** (-(7 * k) / 365.0) for k in range(7))
    worst = max(abs(computed - wrong) for computed, wrong in zip(factors, wrong_basis, strict=True))
    assert worst > 1e-6, "a 365-day basis must be distinguishable from what we compute"
    assert factors[0] == 1.0


def test_discount_factors_handle_the_empty_and_degenerate_cases() -> None:
    assert discount_factors((), 0.12) == ()
    with pytest.raises(MappingError, match="must exceed -1"):
        discount_factors(_seven_weekly_stages(), -1.0)


def test_non_chronological_stages_are_rejected() -> None:
    stages = (
        _stage(0, "2026-05-02", "2026-05-09", (168.0,)),
        _stage(1, "2026-04-25", "2026-05-02", (168.0,)),
    )
    with pytest.raises(MappingError, match="before stage 0"):
        discount_factors(stages, 0.12)


def test_malformed_start_date_is_rejected() -> None:
    with pytest.raises(MappingError, match="ISO-8601"):
        discount_factors((_stage(0, "25/04/2026", "", (168.0,)),), 0.12)


def test_malformed_start_date_in_a_later_stage_names_that_stage() -> None:
    stages = (
        _stage(0, "2026-04-25", "2026-05-02", (168.0,)),
        _stage(1, "not-a-date", "", (168.0,)),
    )
    with pytest.raises(MappingError, match="stage 1 start_date"):
        discount_factors(stages, 0.12)


def test_premise_p9_names_the_corrected_basis() -> None:
    p9 = next(entry for entry in PREMISES if entry.startswith("P9:"))
    assert "365.25" in p9
    assert "mixed-duration" not in p9


# --- finding F2: the rotating ring buffer ------------------------------------


def _thermal_slots(dates: Sequence[int]) -> tuple[EntitySlotRecord, ...]:
    return tuple(
        _slot(2, entity_id, ring, delivery)
        for ring, delivery in enumerate(dates)
        for entity_id in (112, 113)
    )


def _pool_zero_thermal_slots() -> tuple[EntitySlotRecord, ...]:
    """Pool 0's real layout: ring 0 in April, rings 1-5 in May, ring 6 in June."""
    return _thermal_slots((20260401, 20260501, 20260501, 20260501, 20260501, 20260501, 20260601))


def _pool_five_thermal_slots() -> tuple[EntitySlotRecord, ...]:
    """Pool 5's real layout: the ring has rotated, so the same positions differ."""
    return _thermal_slots((20260701, 20260701, 20260701, 20260701, 20260701, 20260501, 20260601))


def test_delivery_slot_map_ranks_distinct_dates_ascending() -> None:
    mapped = delivery_slot_map(_pool_zero_thermal_slots())
    assert mapped[0].pool_rank == 1 and mapped[1].pool_rank == 1
    assert all(mapped[position].pool_rank == 2 for position in range(2, 12))
    assert mapped[12].pool_rank == 3 and mapped[13].pool_rank == 3
    assert mapped[0].delivery_date == 20260401
    assert mapped[2].delivery_date == 20260501
    assert mapped[12].delivery_date == 20260601


def test_delivery_slot_map_follows_the_rotation_not_the_ring_position() -> None:
    """Identical ring positions map to different stage slots once the ring rotates."""
    pool_zero = delivery_slot_map(_pool_zero_thermal_slots())
    pool_five = delivery_slot_map(_pool_five_thermal_slots())
    assert pool_zero != pool_five
    # Ring 0 is the earliest month in pool 0 but the latest in pool 5.
    assert pool_zero[0].pool_rank == 1
    assert pool_five[0].pool_rank == 3
    assert pool_five[10].pool_rank == 1


def test_delivery_slot_map_rejects_a_slot_with_no_delivery_date() -> None:
    slots = (_slot(2, 112, 0, DELIVERY_DATE_SENTINEL),)
    with pytest.raises(MappingError, match="carries no delivery date"):
        delivery_slot_map(slots)


def test_delivery_slot_map_rejects_the_absent_field_default() -> None:
    with pytest.raises(MappingError, match="carries no delivery date"):
        delivery_slot_map((_slot(2, 112, 0, 0),))


def test_delivery_slot_map_is_empty_without_thermal_slots() -> None:
    assert delivery_slot_map((_slot(0, 0, 0, DELIVERY_DATE_SENTINEL),)) == {}


# --- finding F6: hydro codes -------------------------------------------------


def _write_codes(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "codes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hydro_codes_load_in_positional_order(tmp_path: Path) -> None:
    path = _write_codes(tmp_path, {"2": 30, "0": 10, "1": 20})
    assert load_hydro_codes(path) == (10, 20, 30)


def test_hydro_code_for_position_is_the_identity_on_position(tmp_path: Path) -> None:
    codes = load_hydro_codes(_write_codes(tmp_path, {"0": 10, "1": 20}))
    assert hydro_code_for_position(codes, 1) == 20
    with pytest.raises(MappingError, match="outside the 2-entry"):
        hydro_code_for_position(codes, 2)


def test_a_gap_in_the_code_map_is_rejected(tmp_path: Path) -> None:
    path = _write_codes(tmp_path, {"0": 10, "1": 20, "3": 40})
    with pytest.raises(MappingError, match="not contiguous"):
        load_hydro_codes(path)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([1, 2, 3], "must hold a JSON object"),
        ({}, "no entries"),
        ({"a": 1}, "not an integer position"),
        ({"0": "10"}, "must map to an int"),
        ({"0": True}, "must map to an int"),
    ],
)
def test_malformed_code_maps_are_rejected(tmp_path: Path, payload: object, match: str) -> None:
    with pytest.raises(MappingError, match=match):
        load_hydro_codes(_write_codes(tmp_path, payload))


def test_a_repeated_json_key_is_rejected(tmp_path: Path) -> None:
    """json.loads keeps the last value for a repeated key and drops the rest."""
    path = tmp_path / "codes.json"
    path.write_text('{"0": 10, "1": 20, "0": 99}', encoding="utf-8")
    with pytest.raises(MappingError, match="repeats key"):
        load_hydro_codes(path)


def test_a_repeated_decomp_code_is_rejected(tmp_path: Path) -> None:
    """Two positions sharing one plant code would merge two reservoirs."""
    path = _write_codes(tmp_path, {"0": 10, "1": 20, "2": 10})
    with pytest.raises(MappingError, match="more than one position to DECOMP plant code"):
        load_hydro_codes(path)


def test_distinct_codes_in_any_key_order_are_accepted(tmp_path: Path) -> None:
    assert load_hydro_codes(_write_codes(tmp_path, {"1": 20, "0": 10})) == (10, 20)


def test_missing_and_invalid_code_map_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(MappingError, match="not found"):
        load_hydro_codes(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(MappingError, match="not valid JSON"):
        load_hydro_codes(bad)


# --- findings F7 and F8 ------------------------------------------------------


def test_tree_indices_make_the_root_its_own_parent() -> None:
    manifest = PolicyManifest(
        format_version=1,
        cobre_version="0.15.0",
        created_at="",
        num_stages=3,
        n_pools=3,
        completed_iterations=1,
        cost_scale_factor=1e6,
        nodes=tuple(ManifestNodeRecord(id=i, stage_id=i, pool_id=i) for i in range(4)),
        edges=(
            ManifestEdgeRecord(source_id=0, target_id=1, probability=1.0),
            ManifestEdgeRecord(source_id=1, target_id=2, probability=0.5),
            ManifestEdgeRecord(source_id=1, target_id=3, probability=0.5),
        ),
    )
    assert tree_indices(manifest) == (1, 1, 2, 2)


def test_cut_building_pools_exclude_a_warm_start_only_pool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pools = {
        0: _pool(0, populated=48, warm_start=0),
        1: _pool(1, populated=48, warm_start=0),
        2: _pool(2, populated=10000, warm_start=10000),
    }
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        assert cut_building_pools(pools) == (0, 1)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_cut_building_pools_warn_when_the_two_rules_disagree(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pools = {
        0: _pool(0, populated=48, warm_start=0),
        1: _pool(1, populated=10, warm_start=10),
        2: _pool(2, populated=10, warm_start=10),
    }
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        assert cut_building_pools(pools) == (0,)
    assert any("n_pools - 1" in record.getMessage() for record in caplog.records)


# --- premise P8 --------------------------------------------------------------


def test_inflow_lag_audit_counts_nothing_when_no_pool_carries_lags() -> None:
    pools = {0: _pool(0, 48, 0, slots=(_slot(0, 0, 0, DELIVERY_DATE_SENTINEL),))}
    audit = inflow_lag_drop_audit(pools)
    assert audit == InflowLagAudit(per_pool=((0, 0),), total=0)


def test_inflow_lag_audit_counts_coefficients_not_slots(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P8 demands a counted audit, so the count is slots times pieces."""
    slots = (
        _slot(0, 0, 0, DELIVERY_DATE_SENTINEL),
        _slot(1, 5, 0, DELIVERY_DATE_SENTINEL),
        _slot(1, 5, 1, DELIVERY_DATE_SENTINEL),
    )
    pools = {0: _pool(0, 4, 0, slots=slots, pieces=4)}
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        audit = inflow_lag_drop_audit(pools)
    assert audit.total == 8
    assert audit.per_pool == ((0, 8),)
    assert any("drops 8 inflow-lag" in record.getMessage() for record in caplog.records)


def test_the_same_month_ranks_differently_in_different_pools() -> None:
    """`pool_rank` is per-pool, never a study stage index.

    2026-05-01 is rank 2 in pool 0 and rank 1 in pool 5, so a writer that read
    the rank as a global stage index would place the two pools' coefficients on
    different axes. The month travels with the rank precisely so it does not
    have to.
    """
    pool_zero = delivery_slot_map(_pool_zero_thermal_slots())
    pool_five = delivery_slot_map(_pool_five_thermal_slots())

    may_in_pool_zero = {g.pool_rank for g in pool_zero.values() if g.delivery_date == 20260501}
    may_in_pool_five = {g.pool_rank for g in pool_five.values() if g.delivery_date == 20260501}
    assert may_in_pool_zero == {2}
    assert may_in_pool_five == {1}
    assert may_in_pool_zero != may_in_pool_five


# --- Finding 1: keys must be the canonical form of their position -----------


@pytest.mark.parametrize("key", ["00", "+0", " 0", "0 "])
def test_a_non_canonical_key_is_rejected(tmp_path: Path, key: str) -> None:
    """These all parse to position 0, so N+1 entries would collapse into N."""
    path = tmp_path / "codes.json"
    path.write_text(json.dumps({"0": 10, key: 20, "1": 30}), encoding="utf-8")
    with pytest.raises(MappingError, match="canonical form"):
        load_hydro_codes(path)


def test_a_non_ascii_digit_key_is_rejected(tmp_path: Path) -> None:
    """int() accepts Arabic-Indic digits, which would silently alias a position."""
    path = tmp_path / "codes.json"
    path.write_text(json.dumps({"0": 10, "\u0661": 20, "1": 30}), encoding="utf-8")
    with pytest.raises(MappingError, match="canonical form"):
        load_hydro_codes(path)


# --- Finding 2: degenerate graphs raise rather than being narrowed ----------


def _manifest(nodes: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> PolicyManifest:
    return PolicyManifest(
        format_version=1,
        cobre_version="0.15.0",
        created_at="",
        num_stages=len(nodes),
        n_pools=len(nodes),
        completed_iterations=1,
        cost_scale_factor=1e6,
        nodes=tuple(ManifestNodeRecord(id=i, stage_id=i, pool_id=i) for i in nodes),
        edges=tuple(
            ManifestEdgeRecord(source_id=s, target_id=t, probability=1.0) for s, t in edges
        ),
    )


def test_a_multi_parent_node_is_rejected() -> None:
    """DECOMP records one parent per node, so narrowing would invent a tree."""
    with pytest.raises(MappingError, match="more than one parent"):
        tree_indices(_manifest((0, 1, 2, 3), ((0, 1), (0, 2), (1, 3), (2, 3))))


def test_edge_order_cannot_change_the_result() -> None:
    """The old setdefault made the answer depend on edge order in the file."""
    forward = _manifest((0, 1, 2, 3), ((0, 1), (0, 2), (1, 3), (2, 3)))
    swapped = _manifest((0, 1, 2, 3), ((0, 1), (0, 2), (2, 3), (1, 3)))
    with pytest.raises(MappingError, match="more than one parent"):
        tree_indices(forward)
    with pytest.raises(MappingError, match="more than one parent"):
        tree_indices(swapped)


def test_a_dangling_edge_source_is_rejected() -> None:
    with pytest.raises(MappingError, match="unknown source node"):
        tree_indices(_manifest((0, 1), ((99, 1),)))


def test_a_dangling_edge_target_is_rejected() -> None:
    with pytest.raises(MappingError, match="unknown target node"):
        tree_indices(_manifest((0, 1), ((0, 99),)))


def test_more_than_one_root_is_rejected() -> None:
    with pytest.raises(MappingError, match="exactly one root"):
        tree_indices(_manifest((0, 1, 2), ((0, 2),)))


def test_non_positional_node_ids_are_rejected() -> None:
    """Node ids are written as 1-based positions, so they must be 0..n-1."""
    with pytest.raises(MappingError, match="contiguous range"):
        tree_indices(_manifest((10, 11), ((10, 11),)))


def test_the_root_is_its_own_parent_not_node_zero() -> None:
    """Pins the documented rule against the plausible alternative.

    With the root at node 0 the two rules are indistinguishable, so the root here
    is node 1 in a graph whose ids are still positional.
    """
    indices = tree_indices(_manifest((0, 1), ((1, 0),)))
    assert indices == (2, 2)


# --- Finding 4: a negative block hour is a sign flip in disguise ------------


@pytest.mark.parametrize("hours", [(-24.0, 65.0, 79.0), (24.0, -65.0, 79.0)])
def test_a_negative_block_hour_is_rejected(hours: tuple[float, ...]) -> None:
    """A negative hour gives a negative weight that still sums to 1.0."""
    with pytest.raises(MappingError, match="finite and non-negative"):
        gnl_block_weights(_stage(0, "2026-04-25", "2026-05-02", hours).blocks)


# --- Finding 6: the documented ascending order ------------------------------


def test_cut_building_pools_sort_a_non_ascending_mapping() -> None:
    """Removing the sort survived the whole suite, because every test was ordered."""
    pools = {
        5: _pool(5, populated=48, warm_start=0),
        0: _pool(0, populated=48, warm_start=0),
        3: _pool(3, populated=48, warm_start=0),
        1: _pool(1, populated=10, warm_start=10),
    }
    assert cut_building_pools(pools) == (0, 3, 5)


def test_cut_building_pools_of_an_empty_mapping_is_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty mapping previously warned that 0 disagreed with -1."""
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        assert cut_building_pools({}) == ()
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
