"""Unit tests for the `cortdeco` cut record layout and serializer.

Per the master plan's Numeric Path Testing Policy, every anchor here (218, 212,
176, 197, 217, 1000.0) is transcribed from `ticket-009`'s findings, never
computed by the code under test. The two named deliberate mutations for this
ticket, **a stage-major GNL ordering** and **a doubled ÷1000**, were applied to
the source by hand, observed to fail this suite, reverted, and the reversion
confirmed byte-identical. The evidence is recorded in the plan's
`.implementation-state.json` entry for this ticket, which names the failure
counts and the tests that caught each.

Every mutation guard here calls the function under mutation. An earlier test in
this module claimed to guard the doubled division but only compared two literals,
so it passed under that mutation; it was deleted rather than kept as a comment
wearing a `def test_` prefix.
"""

import logging
from collections.abc import Iterator

import numpy as np
import pytest

from conversor_fcf.cobre.policy_reader import AffinePieceRecord, EntitySlotRecord
from conversor_fcf.decomp.cortdeco_writer import CutInput, serialize_cut, storage_coefficients
from conversor_fcf.decomp.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_uniform_blocks,
    cortdeco_block_offsets,
    cortdeco_gnl_offset,
    cortdeco_ncoef,
)
from conversor_fcf.mapping.rules import MappingError

# The reference deck's own scalars (I2, I3, I4) and this project's own (I3).
REFERENCE_SCALARS = {
    "n_uhes": 169,
    "n_utv": 2,
    "max_lag": 3,
    "n_sbm_gnl": 2,
    "n_estagios": 7,
    "n_patamares": 3,
}
PROJECT_SCALARS = {
    "n_uhes": 169,
    "n_utv": 0,
    "max_lag": 0,
    "n_sbm_gnl": 2,
    "n_estagios": 7,
    "n_patamares": 3,
}

REFERENCE_NCOEF = 218
PROJECT_NCOEF = 212
REFERENCE_PI_GNL = 176
PROJECT_PI_GNL = 170

# The reference file's own last non-zero coefficient position (+1) and non-zero
# count (I3): neither is NCOEF, and a formula that produced either would be wrong.
REFERENCE_LAST_NONZERO_PLUS_ONE = 200
REFERENCE_NONZERO_COUNT = 182


@pytest.fixture(autouse=True)
def propagating_package_logger() -> Iterator[None]:
    """caplog reads through the root logger, so propagation must be on."""
    logger = logging.getLogger("conversor_fcf")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


def _slot(entity_type: int, entity_id: int = 0, subindex: int = 0) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        subindex=subindex,
        was_active=True,
        delivery_date=0,
    )


def _piece(intercept: float, coefficients: tuple[float, ...]) -> AffinePieceRecord:
    return AffinePieceRecord(
        piece_id=0,
        slot_index=0,
        iteration=1,
        forward_pass_index=0,
        intercept=intercept,
        coefficients=np.array(coefficients, dtype=np.float64),
        is_active=True,
    )


# --- Requirement 1: cortdeco_ncoef -----------------------------------------


def test_ncoef_reproduces_the_reference_deck() -> None:
    """218, transcribed from the reference file's own declared NCOEF (I3)."""
    assert cortdeco_ncoef(**REFERENCE_SCALARS) == REFERENCE_NCOEF
    assert REFERENCE_NCOEF != REFERENCE_LAST_NONZERO_PLUS_ONE
    assert REFERENCE_NCOEF != REFERENCE_NONZERO_COUNT


def test_ncoef_for_this_project_omits_the_travel_time_block() -> None:
    """212, this project's own NCOEF with n_utv = 0 (premise P3)."""
    assert cortdeco_ncoef(**PROJECT_SCALARS) == PROJECT_NCOEF


@pytest.mark.parametrize(
    "field", ["n_uhes", "n_utv", "max_lag", "n_sbm_gnl", "n_estagios", "n_patamares"]
)
def test_ncoef_rejects_a_negative_scalar(field: str) -> None:
    kwargs = dict(REFERENCE_SCALARS)
    kwargs[field] = -1
    with pytest.raises(LayoutError, match=f"{field} must be non-negative"):
        cortdeco_ncoef(**kwargs)


# --- Requirement 2: CutBlockOffsets and the pi_gnl helper -------------------


def test_block_offsets_reproduce_the_reference_deck() -> None:
    offsets = cortdeco_block_offsets(**REFERENCE_SCALARS)
    assert offsets == CutBlockOffsets(rhs=0, pi_varm=1, pi_gnl=REFERENCE_PI_GNL, ncoef=218)


def test_block_offsets_reproduce_this_project() -> None:
    offsets = cortdeco_block_offsets(**PROJECT_SCALARS)
    assert offsets == CutBlockOffsets(rhs=0, pi_varm=1, pi_gnl=PROJECT_PI_GNL, ncoef=212)


def test_gnl_offset_reproduces_the_two_populated_reference_positions() -> None:
    """176 and 197: the only two positions the reference file actually populates (I4)."""
    offsets = cortdeco_block_offsets(**REFERENCE_SCALARS)
    common = {"n_sbm_gnl": 2, "n_estagios": 7, "n_patamares": 3}
    assert cortdeco_gnl_offset(offsets, sbm=0, stage=0, block=0, **common) == 176
    assert cortdeco_gnl_offset(offsets, sbm=1, stage=0, block=0, **common) == 197


def test_gnl_offset_at_the_last_valid_position() -> None:
    """217 = ncoef - 1: the last coefficient slot the reference vector holds."""
    offsets = cortdeco_block_offsets(**REFERENCE_SCALARS)
    position = cortdeco_gnl_offset(
        offsets, sbm=1, stage=6, block=2, n_sbm_gnl=2, n_estagios=7, n_patamares=3
    )
    assert position == 217 == offsets.ncoef - 1


def test_a_stage_major_reading_would_disagree_with_the_reference() -> None:
    """The ticket's own counter-example: stage-major would put sbm=1/stage=0 at 179, not 197."""
    offsets = cortdeco_block_offsets(**REFERENCE_SCALARS)
    submarket_major = cortdeco_gnl_offset(
        offsets, sbm=1, stage=0, block=0, n_sbm_gnl=2, n_estagios=7, n_patamares=3
    )
    stage_major_reading = offsets.pi_gnl + 0 * (2 * 3) + 1 * 3 + 0
    assert submarket_major == 197
    assert stage_major_reading == 179
    assert submarket_major != stage_major_reading


@pytest.mark.parametrize(
    ("sbm", "stage", "block"), [(-1, 0, 0), (2, 0, 0), (0, -1, 0), (0, 7, 0), (0, 0, -1), (0, 0, 3)]
)
def test_gnl_offset_rejects_an_out_of_range_index(sbm: int, stage: int, block: int) -> None:
    offsets = cortdeco_block_offsets(**REFERENCE_SCALARS)
    with pytest.raises(LayoutError):
        cortdeco_gnl_offset(
            offsets, sbm=sbm, stage=stage, block=block, n_sbm_gnl=2, n_estagios=7, n_patamares=3
        )


# --- Requirement 3: assert_uniform_blocks -----------------------------------


def test_uniform_blocks_returns_the_single_scalar() -> None:
    assert assert_uniform_blocks((3, 3, 3, 3, 3, 3, 3)) == 3


def test_non_uniform_blocks_are_refused_naming_both_values() -> None:
    with pytest.raises(LayoutError) as excinfo:
        assert_uniform_blocks((3, 3, 2))
    message = str(excinfo.value)
    assert "3" in message
    assert "2" in message


# --- Requirements 4-5: serialize_cut, layout and conversions ---------------


def _offsets(
    n_uhes: int = 3, n_sbm_gnl: int = 1, n_estagios: int = 2, n_patamares: int = 2
) -> CutBlockOffsets:
    return cortdeco_block_offsets(
        n_uhes=n_uhes,
        n_utv=0,
        max_lag=0,
        n_sbm_gnl=n_sbm_gnl,
        n_estagios=n_estagios,
        n_patamares=n_patamares,
    )


def test_a_serialized_cut_is_exactly_tamanho_corte_bytes() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.zeros(3), pi_gnl=np.zeros(4))
    data = serialize_cut(cut, pointer=0, offsets=offsets)
    assert len(data) == TAMANHO_CORTE


def test_the_pointer_round_trips_at_offset_zero() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.zeros(3), pi_gnl=np.zeros(4))
    data = serialize_cut(cut, pointer=437, offsets=offsets)
    pointer = np.frombuffer(data, dtype="<i4", count=1)[0]
    assert int(pointer) == 437


def test_every_byte_after_the_coefficient_span_is_zero() -> None:
    offsets = _offsets()
    cut = CutInput(
        intercept=1.0, pi_varm=np.array([2.0, 3.0, 4.0]), pi_gnl=np.array([5.0, 6.0, 7.0, 8.0])
    )
    data = serialize_cut(cut, pointer=1, offsets=offsets)
    tail_start = 4 + 8 * offsets.ncoef
    assert data[tail_start:] == b"\x00" * (TAMANHO_CORTE - tail_start)
    assert any(data[4:tail_start])


def test_the_intercept_is_divided_by_one_thousand_exactly_once() -> None:
    """1,000,000 -> 1000.0, not 1.0 (doubled) and not 1,000,000.0 (missed).

    One of the guards for the ticket's second named mutation, a doubled division,
    which was demonstrated red here and in the two block tests below plus the
    integration round trip.
    """
    offsets = _offsets()
    cut = CutInput(intercept=1_000_000.0, pi_varm=np.zeros(3), pi_gnl=np.zeros(4))
    data = serialize_cut(cut, pointer=0, offsets=offsets)
    rhs = np.frombuffer(data, dtype="<f8", count=1, offset=4)[0]
    assert float(rhs) == 1000.0


def test_pi_varm_is_divided_but_not_negated() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.array([1000.0, -2000.0, 3000.0]), pi_gnl=np.zeros(4))
    data = serialize_cut(cut, pointer=0, offsets=offsets)
    pi_varm = np.frombuffer(data, dtype="<f8", count=3, offset=4 + 8 * offsets.pi_varm)
    assert list(pi_varm) == [1.0, -2.0, 3.0]


def test_pi_gnl_is_divided_and_negated() -> None:
    offsets = _offsets()
    cut = CutInput(
        intercept=0.0, pi_varm=np.zeros(3), pi_gnl=np.array([-1000.0, 2000.0, -3000.0, 4000.0])
    )
    data = serialize_cut(cut, pointer=0, offsets=offsets)
    pi_gnl = np.frombuffer(data, dtype="<f8", count=4, offset=4 + 8 * offsets.pi_gnl)
    assert list(pi_gnl) == [1.0, -2.0, 3.0, -4.0]


# --- Requirement 6: non-finite rejection ------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("block", ["pi_varm", "pi_gnl"])
def test_every_non_finite_value_in_a_coefficient_block_is_rejected(block: str, bad: float) -> None:
    """nan and both infinities, in both blocks: the guard is finiteness, not nan alone."""
    pi_varm = np.zeros(3)
    pi_gnl = np.zeros(4)
    if block == "pi_varm":
        pi_varm[0] = bad
    else:
        pi_gnl[0] = bad

    cut = CutInput(intercept=0.0, pi_varm=pi_varm, pi_gnl=pi_gnl)
    with pytest.raises(LayoutError, match=f"{block} position 0 is not finite"):
        serialize_cut(cut, pointer=0, offsets=_offsets())


def test_a_nan_in_pi_varm_is_rejected_naming_the_block_and_position() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.array([1.0, float("nan"), 3.0]), pi_gnl=np.zeros(4))
    with pytest.raises(LayoutError) as excinfo:
        serialize_cut(cut, pointer=0, offsets=offsets)
    message = str(excinfo.value)
    assert "pi_varm" in message
    assert "position 1" in message


def test_a_nan_in_pi_gnl_is_rejected_naming_the_block_and_position() -> None:
    offsets = _offsets()
    cut = CutInput(
        intercept=0.0, pi_varm=np.zeros(3), pi_gnl=np.array([1.0, 2.0, float("nan"), 4.0])
    )
    with pytest.raises(LayoutError) as excinfo:
        serialize_cut(cut, pointer=0, offsets=offsets)
    message = str(excinfo.value)
    assert "pi_gnl" in message
    assert "position 2" in message


def test_an_infinite_rhs_is_rejected() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=float("inf"), pi_varm=np.zeros(3), pi_gnl=np.zeros(4))
    with pytest.raises(LayoutError, match="rhs is not finite"):
        serialize_cut(cut, pointer=0, offsets=offsets)


def test_no_bytes_are_returned_when_validation_fails() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=float("nan"), pi_varm=np.zeros(3), pi_gnl=np.zeros(4))
    with pytest.raises(LayoutError):
        serialize_cut(cut, pointer=0, offsets=offsets)


# --- Requirement 7: coefficient-count mismatch ------------------------------


def test_a_pi_varm_one_short_is_rejected_naming_both_lengths() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.zeros(2), pi_gnl=np.zeros(4))
    with pytest.raises(LayoutError) as excinfo:
        serialize_cut(cut, pointer=0, offsets=offsets)
    message = str(excinfo.value)
    assert "pi_varm has 2 entries" in message
    assert "expects 3" in message


def test_a_pi_gnl_one_short_is_rejected_naming_both_lengths() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.zeros(3), pi_gnl=np.zeros(3))
    with pytest.raises(LayoutError) as excinfo:
        serialize_cut(cut, pointer=0, offsets=offsets)
    message = str(excinfo.value)
    assert "pi_gnl has 3 entries" in message
    assert "expects 4" in message


def test_a_pi_varm_one_long_is_also_rejected() -> None:
    offsets = _offsets()
    cut = CutInput(intercept=0.0, pi_varm=np.zeros(4), pi_gnl=np.zeros(4))
    with pytest.raises(LayoutError, match="pi_varm has 4 entries"):
        serialize_cut(cut, pointer=0, offsets=offsets)


# --- storage_coefficients: I9, the entity-manifest index and the code map --


def test_storage_coefficients_extracts_by_manifest_position_not_contiguity() -> None:
    """Storage and GNL slots interleaved, so a positional axis boundary is exercised."""
    slots = (
        _slot(entity_type=0, entity_id=5),
        _slot(entity_type=2, entity_id=112),
        _slot(entity_type=0, entity_id=6),
        _slot(entity_type=2, entity_id=113),
    )
    piece = _piece(intercept=0.0, coefficients=(11.0, 22.0, 33.0, 44.0))
    hydro_codes = (100, 101, 102)

    result = storage_coefficients(piece, slots, hydro_codes)
    assert list(result) == [11.0, 33.0]


def test_storage_coefficients_raises_when_a_position_is_absent_from_the_code_map() -> None:
    """I9: a plant missing from the code map already raises, unchanged."""
    slots = (
        _slot(entity_type=0, entity_id=5),
        _slot(entity_type=0, entity_id=6),
        _slot(entity_type=0, entity_id=7),
    )
    piece = _piece(intercept=0.0, coefficients=(1.0, 2.0, 3.0))
    hydro_codes = (100,)

    with pytest.raises(MappingError, match="position 1"):
        storage_coefficients(piece, slots, hydro_codes)
