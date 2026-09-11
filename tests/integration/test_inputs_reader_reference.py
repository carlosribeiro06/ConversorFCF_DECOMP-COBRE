"""Acceptance criteria for ticket-004 against the real Cobre reference case."""

import sys
from pathlib import Path

import pytest

from conversor_fcf.cobre.inputs_reader import (
    CaseInputs,
    anticipated_thermals,
    read_case_inputs,
)

REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")

pytestmark = pytest.mark.skipif(
    not (REFERENCE_CASE / "stages.json").is_file(),
    reason=f"Cobre reference case not present at {REFERENCE_CASE}",
)


@pytest.fixture(scope="module")
def inputs() -> CaseInputs:
    return read_case_inputs(REFERENCE_CASE)


def test_counts_and_discount_rate_match_the_reference_case(inputs: CaseInputs) -> None:
    assert len(inputs.stages) == 7
    assert len(inputs.hydros) == 169
    assert len(inputs.thermals) == 114
    assert len(inputs.buses) == 6
    assert inputs.annual_discount_rate == 0.12


def test_first_stage_carries_three_blocks_of_24_65_and_79_hours(inputs: CaseInputs) -> None:
    blocks = inputs.stages[0].blocks
    assert len(blocks) == 3
    assert [block.hours for block in blocks] == [24.0, 65.0, 79.0]
    assert sum(block.hours for block in blocks) == 168.0
    assert inputs.stages[0].inflow_lags_state is False
    assert inputs.stages[0].storage_state is True


def test_stage_durations_are_not_uniform(inputs: CaseInputs) -> None:
    """Stages 0-5 span a week; stage 6 spans 600 hours.

    The ticket's prose says every stage totals 168 hours, which is not what the
    deck holds. Anything weighting by time must read per-stage block hours.
    """
    durations = [sum(block.hours for block in stage.blocks) for stage in inputs.stages]
    assert durations[:6] == [168.0] * 6
    assert durations[6] == 600.0


def test_block_composition_varies_between_stages(inputs: CaseInputs) -> None:
    """24/65/79 is stage 0 alone; every stage splits its hours differently.

    This is what premise P5's hours weighting must read per stage. Asserting only
    that the totals are 168 would hide the composition drift that actually matters.
    """
    splits = [tuple(block.hours for block in stage.blocks) for stage in inputs.stages]
    assert splits[0] == (24.0, 65.0, 79.0)
    assert splits[1] == (15.0, 64.0, 89.0)
    assert splits[5] == (12.0, 61.0, 95.0)
    assert splits[6] == (51.0, 226.0, 323.0)
    assert len(set(splits)) > 1, "the deck does not use one uniform block split"


def test_only_thermals_112_and_113_are_gnl_capable(inputs: CaseInputs) -> None:
    gnl = anticipated_thermals(inputs)
    assert [thermal.id for thermal in gnl] == [112, 113]
    assert [thermal.bus_id for thermal in gnl] == [0, 2]
    assert all(thermal.lead_time_hours == 1608.0 for thermal in gnl)


def test_first_hydro_is_camargos_flowing_into_plant_one(inputs: CaseInputs) -> None:
    camargos = inputs.hydros[0]
    assert camargos.id == 0
    assert camargos.name == "CAMARGOS"
    assert camargos.downstream_id == 1
    assert camargos.min_storage_hm3 == 120.0
    assert camargos.max_storage_hm3 == 792.0


def test_terminal_plants_are_recorded_with_no_downstream(inputs: CaseInputs) -> None:
    terminal = [hydro.id for hydro in inputs.hydros if hydro.downstream_id is None]
    assert terminal, "the reference deck does contain terminal plants"
    assert 47 in terminal
    # 0 is a valid hydro id, so no terminal plant may be recorded as flowing into it.
    assert all(hydro.downstream_id != 0 or hydro.id != 47 for hydro in inputs.hydros)


def test_buses_are_the_five_submarkets_plus_the_interconnection_node(
    inputs: CaseInputs,
) -> None:
    assert [(bus.id, bus.name) for bus in inputs.buses] == [
        (0, "SE"),
        (1, "S"),
        (2, "NE"),
        (3, "N"),
        (4, "FC"),
        (5, "IV"),
    ]


def test_no_parquet_is_read_and_pyarrow_is_never_imported() -> None:
    read_case_inputs(REFERENCE_CASE)
    assert "pyarrow" not in sys.modules
