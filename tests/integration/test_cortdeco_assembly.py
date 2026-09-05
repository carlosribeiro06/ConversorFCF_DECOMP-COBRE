"""Acceptance criteria for `ticket-009` against the real Cobre case.

`NCOEF` and the byte-level round trip are checked against this project's own
transcribed scalars and against an independent recomputation of the converted
values, never against the in-memory array `serialize_cut` was given: per the
master plan's Numeric Path Testing Policy, a self-referential round trip
proves only internal consistency.

`pi_gnl` is written as zeros here. Placing a real `AffinePiece`'s
anticipated-thermal coefficients onto `pi_gnl`'s `(submarket, stage, block)`
address needs a thermal-to-bus lookup and a delivery-date-to-stage lookup that
live in `CaseInputs`, outside `ticket-009`'s scope (see the module docstring
of `cortdeco_writer.py`); this test exercises the byte-level record mechanics
and the `rhs`/`pi_varm` conversions, which need no such lookup, and `ticket-010`
extends this file once chaining exists.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.cobre.entities import EntityType
from conversor_fcf.cobre.policy_reader import StageCutPool, read_stage_cuts
from conversor_fcf.decomp.cortdeco_writer import CutInput, serialize_cut, storage_coefficients
from conversor_fcf.decomp.layout import TAMANHO_CORTE, cortdeco_block_offsets, cortdeco_ncoef
from conversor_fcf.mapping.rules import load_hydro_codes

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
POLICY_DIR = REFERENCE_CASE / "output" / "policy"

# This project's own scalars (I3): n_utv = 0 under premise P3.
PROJECT_NCOEF = 212
PROJECT_PI_GNL = 170
N_SBM_GNL = 2
N_ESTAGIOS = 7
N_PATAMARES = 3

# I8: the reference deck's own rhs envelope. A converted rhs three orders of
# magnitude outside this range would indicate a missed or doubled division.
REFERENCE_RHS_ENVELOPE = (-4.211769e8, 8.321719e9)

pytestmark = pytest.mark.skipif(
    not POLICY_DIR.is_dir(),
    reason=f"Cobre reference case not present at {REFERENCE_CASE}",
)


@pytest.fixture(scope="module")
def pool() -> StageCutPool:
    return read_stage_cuts(POLICY_DIR / "cuts" / "000.bin")


@pytest.fixture(scope="module")
def hydro_codes() -> tuple[int, ...]:
    return load_hydro_codes(REPO_ROOT / "decomp_hydro_codes.json")


@pytest.fixture(autouse=True)
def _propagating_package_logger() -> Iterator[None]:
    logger = logging.getLogger("conversor_fcf")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


def test_ncoef_for_this_projects_own_scalars(hydro_codes: tuple[int, ...]) -> None:
    """212, transcribed independently of the case (I3); the code map's own length
    confirms n_uhes = 169 rather than assuming it."""
    assert len(hydro_codes) == 169
    offsets = cortdeco_block_offsets(
        n_uhes=len(hydro_codes),
        n_utv=0,
        max_lag=0,
        n_sbm_gnl=N_SBM_GNL,
        n_estagios=N_ESTAGIOS,
        n_patamares=N_PATAMARES,
    )
    assert (
        cortdeco_ncoef(
            n_uhes=len(hydro_codes),
            n_utv=0,
            max_lag=0,
            n_sbm_gnl=N_SBM_GNL,
            n_estagios=N_ESTAGIOS,
            n_patamares=N_PATAMARES,
        )
        == PROJECT_NCOEF
    )
    assert offsets.ncoef == PROJECT_NCOEF
    assert offsets.pi_gnl == PROJECT_PI_GNL


def test_a_real_affine_piece_round_trips_at_the_byte_level(
    pool: StageCutPool, hydro_codes: tuple[int, ...]
) -> None:
    piece = pool.pieces[0]
    offsets = cortdeco_block_offsets(
        n_uhes=len(hydro_codes),
        n_utv=0,
        max_lag=0,
        n_sbm_gnl=N_SBM_GNL,
        n_estagios=N_ESTAGIOS,
        n_patamares=N_PATAMARES,
    )
    pi_varm = storage_coefficients(piece, pool.slots, hydro_codes)
    pi_gnl = np.zeros(offsets.ncoef - offsets.pi_gnl, dtype=np.float64)

    cut = CutInput(intercept=piece.intercept, pi_varm=pi_varm, pi_gnl=pi_gnl)
    data = serialize_cut(cut, pointer=283, offsets=offsets)

    assert len(data) == TAMANHO_CORTE
    pointer = int(np.frombuffer(data, dtype="<i4", count=1)[0])
    assert pointer == 283

    # rhs, recomputed independently of `to_decomp_cost` and checked within I8's envelope.
    rhs = float(np.frombuffer(data, dtype="<f8", count=1, offset=4)[0])
    assert rhs == pytest.approx(piece.intercept / 1000.0, rel=0, abs=1e-6)
    lower, upper = REFERENCE_RHS_ENVELOPE
    assert lower <= rhs <= upper

    # pi_varm, recomputed by a raw manifest scan independent of `storage_coefficients`.
    storage_positions = [
        position
        for position, slot in enumerate(pool.slots)
        if slot.entity_type == EntityType.HYDRO_STORAGE
    ]
    expected_pi_varm = np.asarray(piece.coefficients, dtype=np.float64)[storage_positions] / 1000.0
    written_pi_varm = np.frombuffer(
        data, dtype="<f8", count=len(storage_positions), offset=4 + 8 * offsets.pi_varm
    )
    np.testing.assert_allclose(written_pi_varm, expected_pi_varm)

    # Every byte beyond the declared coefficient span is padding.
    tail_start = 4 + 8 * offsets.ncoef
    assert data[tail_start:] == b"\x00" * (TAMANHO_CORTE - tail_start)
