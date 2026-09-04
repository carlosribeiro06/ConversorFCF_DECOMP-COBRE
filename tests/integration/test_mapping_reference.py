"""Acceptance criteria for ticket-006 against the real Cobre reference case.

Every anchor here is transcribed independently of the code under test: the
discount series comes from the reference DECOMP deck, the tree-index prefix from
the reference `mapcut`, and the delivery-date groupings from reading the
checkpoint directly. Per the plan's Numeric Path Testing Policy, no assertion
compares an output against the value that produced it.
"""

import logging
from pathlib import Path

import pytest

from conversor_fcf.cobre.inputs_reader import CaseInputs, read_case_inputs
from conversor_fcf.cobre.policy_reader import (
    PolicyManifest,
    StageCutPool,
    read_policy_manifest,
    read_stage_cuts,
)
from conversor_fcf.mapping.rules import (
    MappingError,
    cut_building_pools,
    delivery_slot_map,
    discount_factors,
    gnl_block_weights,
    hydro_code_for_position,
    inflow_lag_drop_audit,
    load_hydro_codes,
    select_submarket_buses,
    submarket_for_bus,
    tree_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
POLICY_DIR = REFERENCE_CASE / "output" / "policy"
TRUNK_POOL_IDS = (0, 1, 2, 3, 4, 5)

# The verified reference discount series from the DECOMP deck.
REFERENCE_DISCOUNTS = (
    1.0,
    0.997830417741,
    0.995665542570,
    0.993505364273,
    0.991349872661,
    0.989199057565,
    0.987052908840,
)

pytestmark = pytest.mark.skipif(
    not POLICY_DIR.is_dir(),
    reason=f"Cobre reference case not present at {REFERENCE_CASE}",
)


@pytest.fixture(scope="module")
def inputs() -> CaseInputs:
    return read_case_inputs(REFERENCE_CASE)


@pytest.fixture(scope="module")
def manifest() -> PolicyManifest:
    return read_policy_manifest(POLICY_DIR / "manifest.bin")


@pytest.fixture(scope="module")
def trunk_pools() -> dict[int, StageCutPool]:
    return {
        pool_id: read_stage_cuts(POLICY_DIR / "cuts" / f"{pool_id:03d}.bin")
        for pool_id in TRUNK_POOL_IDS
    }


def test_discount_series_reproduces_the_reference_deck(inputs: CaseInputs) -> None:
    factors = discount_factors(inputs.stages, inputs.annual_discount_rate)
    assert len(factors) == 7
    for stage, (computed, reference) in enumerate(zip(factors, REFERENCE_DISCOUNTS, strict=True)):
        assert abs(computed - reference) < 1e-12, f"stage {stage}"


def test_discount_factors_reject_the_365_day_basis(inputs: CaseInputs) -> None:
    """The deliberate mutation for this ticket, asserted against real output.

    The comparison runs against `discount_factors`' own result rather than
    against the transcribed series, so a broken implementation fails here too.
    """
    rate = inputs.annual_discount_rate
    factors = discount_factors(inputs.stages, rate)
    wrong_basis = tuple((1.0 + rate) ** (-(7 * k) / 365.0) for k in range(len(factors)))
    worst = max(abs(computed - wrong) for computed, wrong in zip(factors, wrong_basis, strict=True))
    assert worst > 1e-6, "a 365-day basis must be distinguishable from what we compute"


def test_delivery_slot_map_groups_pool_zero_by_month(
    trunk_pools: dict[int, StageCutPool],
) -> None:
    mapped = delivery_slot_map(trunk_pools[0].slots)
    # Positions 169-170 are April, 171-180 May, 181-182 June.
    assert mapped[169].pool_rank == 1 and mapped[170].pool_rank == 1
    assert {mapped[position].pool_rank for position in range(171, 181)} == {2}
    assert mapped[181].pool_rank == 3 and mapped[182].pool_rank == 3
    assert mapped[169].delivery_date == 20260401
    assert {mapped[position].delivery_date for position in range(171, 181)} == {20260501}
    assert mapped[181].delivery_date == 20260601


def test_delivery_slot_map_differs_between_pools_because_the_ring_rotates(
    trunk_pools: dict[int, StageCutPool],
) -> None:
    """If subindex were used as the stage axis the two maps would be identical."""
    pool_zero = delivery_slot_map(trunk_pools[0].slots)
    pool_five = delivery_slot_map(trunk_pools[5].slots)
    assert pool_zero != pool_five
    assert pool_zero[169].pool_rank == 1
    assert pool_five[169].pool_rank == 3
    assert pool_five[179].pool_rank == 1
    # The same month ranks differently, which is why the rank is not a stage index.
    may_zero = {g.pool_rank for g in pool_zero.values() if g.delivery_date == 20260501}
    may_five = {g.pool_rank for g in pool_five.values() if g.delivery_date == 20260501}
    assert may_zero == {2}
    assert may_five == {1}


def test_submarket_selection_on_the_reference_buses(inputs: CaseInputs) -> None:
    selected = select_submarket_buses(inputs.buses)
    assert [bus.name for bus in selected] == ["SE", "S", "NE", "N", "FC"]
    assert [submarket_for_bus(bus.id) for bus in selected] == [1, 2, 3, 4, 5]


def test_block_weights_partition_every_stage_of_the_deck(inputs: CaseInputs) -> None:
    for stage in inputs.stages:
        weights = gnl_block_weights(stage.blocks)
        assert len(weights) == len(stage.blocks)
        assert sum(weights) == pytest.approx(1.0, abs=1e-15)

    assert gnl_block_weights(inputs.stages[0].blocks) == (24 / 168, 65 / 168, 79 / 168)
    assert gnl_block_weights(inputs.stages[1].blocks) == (15 / 168, 64 / 168, 89 / 168)
    assert gnl_block_weights(inputs.stages[6].blocks) == (51 / 600, 226 / 600, 323 / 600)


def test_hydro_codes_cover_every_reference_plant() -> None:
    """Codes transcribed from the map itself, not read back through the loader."""
    codes = load_hydro_codes(REPO_ROOT / "decomp_hydro_codes.json")
    assert len(codes) == 169
    assert hydro_code_for_position(codes, 0) == 1
    assert hydro_code_for_position(codes, 1) == 2
    assert hydro_code_for_position(codes, 2) == 4
    assert hydro_code_for_position(codes, 84) == 119
    assert hydro_code_for_position(codes, 168) == 315
    assert len(set(codes)) == 169, "DECOMP plant codes must be distinct"
    with pytest.raises(MappingError, match="outside the 169-entry"):
        hydro_code_for_position(codes, 169)


def test_hydro_codes_are_positionally_aligned_with_the_storage_slots(
    inputs: CaseInputs, trunk_pools: dict[int, StageCutPool]
) -> None:
    """Finding F6: manifest position i is hydro id i, so the map is an identity on position."""
    codes = load_hydro_codes(REPO_ROOT / "decomp_hydro_codes.json")
    storage = [slot for slot in trunk_pools[0].slots if slot.entity_type == 0]
    assert len(storage) == len(codes) == len(inputs.hydros)
    assert [slot.entity_id for slot in storage] == list(range(169))


def test_tree_indices_match_the_reference_prefix(manifest: PolicyManifest) -> None:
    assert tree_indices(manifest)[:14] == (1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6)


@pytest.mark.slow
def test_cut_building_pools_are_the_six_trunk_pools(
    trunk_pools: dict[int, StageCutPool], caplog: pytest.LogCaptureFixture
) -> None:
    terminal = read_stage_cuts(POLICY_DIR / "cuts" / "006.bin")
    pools = {**trunk_pools, 6: terminal}
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        assert cut_building_pools(pools) == TRUNK_POOL_IDS
    assert not [record for record in caplog.records if "n_pools - 1" in record.getMessage()]


def test_inflow_lag_audit_reports_nothing_on_the_converted_pools(
    trunk_pools: dict[int, StageCutPool],
) -> None:
    """Premise P7 excludes the only pool that carries inflow-lag slots."""
    audit = inflow_lag_drop_audit(trunk_pools)
    assert audit.total == 0
    assert [count for _, count in audit.per_pool] == [0] * 6
