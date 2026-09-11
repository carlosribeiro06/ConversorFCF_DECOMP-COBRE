"""Bound every emitted `cortdeco` coefficient by the reference deck's own envelopes.

Premise P2 divides every intercept and coefficient by 1000 (DECOMP works in 10^3 R$).
Missing or doubling that division changes no structural byte - record count, `NCOEF`,
the chain and the file size are all identical - so a magnitude envelope taken from the
real CEPEL deck is the only whole-file detector available for it. This module is also an
honest accounting of what such an envelope cannot do: measured below, the `rhs` envelope
catches a missed division but not a doubled one, and the `pi_varm` envelope catches
neither. `CLAUDE.md` records both facts so no future ticket mistakes either envelope for
a division witness.

Premise P14 holds throughout: `pi_gnl` is out of scope here (see the ticket's "Out of
Scope"), so nothing below gates on a GNL value or on GNL magnitude agreement between the
two independent runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from conversor_fcf.decomp.layout import (
    CutBlockOffsets,
    assert_uniform_blocks,
    cortdeco_block_offsets,
)
from conversor_fcf.decomp.reader import (
    CortdecoContents,
    MapcutContents,
    read_cortdeco,
    read_mapcut,
)
from conversor_fcf.paths import OutputPaths

REFERENCE_CORTDECO = Path("/home/carlosribeiro/git/cortdeco.rv0")
REFERENCE_NODES = 6
REFERENCE_OFFSETS = CutBlockOffsets(rhs=0, pi_varm=1, pi_gnl=176, ncoef=218)
# Positions 169-174 of the reference deck's own pi_varm span are its pi_qdefp
# (travel-time) block, which this project never emits under premise P3; excluded here so
# the envelope means only the storage block this project actually writes.
REFERENCE_STORAGE_WIDTH = 169

# I8, extended: both derived from and verified against the reference deck's own bytes by
# test_the_envelope_constants_are_derived_from_the_reference_deck below. Neither is a
# premise-P2 witness in every direction - see that fact recorded in CLAUDE.md and the two
# mutation tests at the bottom of this module.
REFERENCE_RHS_ENVELOPE = (-421176891.47949135, 8321718752.484832)
REFERENCE_PI_VARM_ENVELOPE = (-50289462.50082922, 318129.5788995588)


def _assert_within_envelope(
    values: NDArray[np.float64], envelope: tuple[float, float], label: str
) -> None:
    """Fail naming `label`, the offending record and the value - never a bare boolean.

    The two bounds are checked as independent branches rather than a single `any()`, so
    a failure always names which bound and which record, per the ticket's Error Handling.
    """
    lower, upper = envelope
    minimum = float(values.min())
    if minimum < lower:
        record, *_ = np.unravel_index(int(np.argmin(values)), values.shape)
        raise AssertionError(
            f"{label} record {record} = {minimum!r} is below the reference envelope's "
            f"lower bound {lower!r}"
        )
    maximum = float(values.max())
    if maximum > upper:
        record, *_ = np.unravel_index(int(np.argmax(values)), values.shape)
        raise AssertionError(
            f"{label} record {record} = {maximum!r} exceeds the reference envelope's "
            f"upper bound {upper!r}"
        )


@pytest.mark.skipif(
    not REFERENCE_CORTDECO.is_file(),
    reason=f"reference cortdeco not present at {REFERENCE_CORTDECO}",
)
def test_the_envelope_constants_are_derived_from_the_reference_deck() -> None:
    """Both constants recomputed from the deck's own 439 records, to the last digit.

    Exact equality, not `pytest.approx`: the same bytes are read into the same float64
    values here as when the constants above were first measured, so a tolerance would
    let a genuinely different envelope pass unnoticed.
    """
    contents = read_cortdeco(REFERENCE_CORTDECO, REFERENCE_OFFSETS, REFERENCE_NODES)
    rhs = np.array([cut.rhs for cut in contents.cuts], dtype=np.float64)
    pi_varm = np.array(
        [cut.pi_varm[:REFERENCE_STORAGE_WIDTH] for cut in contents.cuts], dtype=np.float64
    )
    assert (float(rhs.min()), float(rhs.max())) == REFERENCE_RHS_ENVELOPE
    assert (float(pi_varm.min()), float(pi_varm.max())) == REFERENCE_PI_VARM_ENVELOPE


@pytest.fixture(scope="module")
def emitted_mapcut(converted_pair: OutputPaths) -> MapcutContents:
    return read_mapcut(converted_pair.mapcut)


@pytest.fixture(scope="module")
def emitted_n_patamares(emitted_mapcut: MapcutContents) -> int:
    return assert_uniform_blocks(emitted_mapcut.patamares_por_estagio)


@pytest.fixture(scope="module")
def emitted_offsets(emitted_mapcut: MapcutContents, emitted_n_patamares: int) -> CutBlockOffsets:
    scalars = emitted_mapcut.scalars
    return cortdeco_block_offsets(
        n_uhes=scalars.numero_uhes,
        n_utv=scalars.n_utv,
        max_lag=scalars.max_lag,
        n_sbm_gnl=len(emitted_mapcut.codigos_submercados_gnl),
        n_estagios=scalars.numero_estagios,
        n_patamares=emitted_n_patamares,
    )


@pytest.fixture(scope="module")
def emitted_cortdeco(
    converted_pair: OutputPaths,
    emitted_mapcut: MapcutContents,
    emitted_offsets: CutBlockOffsets,
) -> CortdecoContents:
    return read_cortdeco(
        converted_pair.cortdeco, emitted_offsets, emitted_mapcut.scalars.numero_semanas
    )


@pytest.fixture(scope="module")
def emitted_rhs(emitted_cortdeco: CortdecoContents) -> NDArray[np.float64]:
    return np.array([cut.rhs for cut in emitted_cortdeco.cuts], dtype=np.float64)


@pytest.fixture(scope="module")
def emitted_pi_varm(emitted_cortdeco: CortdecoContents) -> NDArray[np.float64]:
    return np.array([cut.pi_varm for cut in emitted_cortdeco.cuts], dtype=np.float64)


def test_every_emitted_rhs_lies_within_the_reference_rhs_envelope(
    emitted_rhs: NDArray[np.float64],
) -> None:
    assert emitted_rhs.shape == (289,)
    _assert_within_envelope(emitted_rhs, REFERENCE_RHS_ENVELOPE, "rhs")


def test_every_emitted_pi_varm_lies_within_the_reference_pi_varm_envelope(
    emitted_pi_varm: NDArray[np.float64],
) -> None:
    assert emitted_pi_varm.size == 48_841
    _assert_within_envelope(emitted_pi_varm, REFERENCE_PI_VARM_ENVELOPE, "pi_varm")


def test_a_missed_division_mutation_on_rhs_escapes_the_rhs_envelope(
    emitted_rhs: NDArray[np.float64],
) -> None:
    """Multiplying by 1000 simulates forgetting premise P2's own `/1000`.

    This is the one direction, of the four this ticket measures, that the `rhs` envelope
    actually witnesses (Requirement 4): the mutated maximum lands three orders of
    magnitude past `REFERENCE_RHS_ENVELOPE`'s upper bound, so the same check the bounding
    test above runs must fail here - demonstrating that check is not vacuous.
    """
    mutated = emitted_rhs * 1000.0
    with pytest.raises(AssertionError, match="exceeds the reference envelope's upper bound"):
        _assert_within_envelope(mutated, REFERENCE_RHS_ENVELOPE, "rhs")


def test_a_doubled_rhs_division_and_a_missed_pi_varm_division_stay_inside(
    emitted_rhs: NDArray[np.float64],
    emitted_pi_varm: NDArray[np.float64],
) -> None:
    """Requirement 5: the two mutations neither envelope in this module can catch.

    Dividing `rhs` by 1000 again (a doubled premise-P2 division) leaves it around 2.29e6,
    comfortably inside `REFERENCE_RHS_ENVELOPE`. Multiplying `pi_varm` by 1000 (a missed
    one) leaves it inside `REFERENCE_PI_VARM_ENVELOPE` too, with margin to spare. Neither
    envelope witnesses premise P2 in this direction; that witness is
    `tests/integration/test_cortdeco_assembly.py`'s independent
    `piece.intercept / 1000.0` recomputation, which this module does not duplicate.
    """
    doubled_rhs = emitted_rhs / 1000.0
    missed_pi_varm = emitted_pi_varm * 1000.0
    _assert_within_envelope(doubled_rhs, REFERENCE_RHS_ENVELOPE, "rhs")
    _assert_within_envelope(missed_pi_varm, REFERENCE_PI_VARM_ENVELOPE, "pi_varm")
