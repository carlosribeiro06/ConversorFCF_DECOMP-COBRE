"""DECOMP binary layout constants and the invariants that check them.

The record map was established by reading the reference `mapcut.rv0` (356
records, 17,095,120 bytes) both through `idecomp` and at the byte level, and by
reading `idecomp`'s own section definitions, which are the authority on what a
readable file looks like:

| Records | Content | Count |
|---------|---------|-------|
| 0-3 | regs 1-4 | 4 |
| 4-17 | undocumented per-plant physical records | 14 |
| 18 | reg 5, parent-node indices | 1 |
| 19 | reg 6, stage data | 1 |
| 20.. | regs 7/8 travel time | `n_utv * n_estagios * (max_lag+1)` |
| .. | reg 9 per node and reg 10 per stage | `n_cenarios + n_estagios` |

Reg 9 and reg 10 interleave across the first `n_estagios` pairs, then reg 9
continues alone. Every reg-9 record is byte-identical, because it carries the
GNL configuration rather than per-node data; `idecomp` reads only the first
`n_estagios` of them and never touches the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from conversor_fcf.cobre.entities import EntityType, index_slots
from conversor_fcf.cobre.policy_reader import EntitySlotRecord
from conversor_fcf.logging_setup import get_logger

RECORD_SIZE = 48020

# A fixed DECOMP maximum, not a function of this case.
TAMANHO_CORTE = 26976

# The undocumented per-plant physical span, zero-filled under premise P1.
PHYSICAL_RECORD_COUNT = 14
PHYSICAL_RECORD_FIRST = 4
PHYSICAL_RECORD_LAST = PHYSICAL_RECORD_FIRST + PHYSICAL_RECORD_COUNT - 1

_HEADER_RECORD_COUNT = 4
_TREE_RECORD_COUNT = 1
_STAGE_RECORD_COUNT = 1

_logger = get_logger("layout")


class LayoutError(Exception):
    """Raised when a DECOMP binary does not satisfy its own layout invariant."""


class MapcutDimensions(Protocol):
    """The four header fields that decide a `mapcut`'s record count.

    Structural rather than an import of `MapcutHeader`: the writer depends on
    this module, so naming its dataclass here would invert the layering. Any
    object carrying these four scalars can be checked against a file.
    """

    @property
    def n_utv(self) -> int: ...
    @property
    def numero_estagios(self) -> int: ...
    @property
    def max_lag(self) -> int: ...
    @property
    def numero_cenarios(self) -> int: ...


def mapcut_record_count(n_utv: int, n_estagios: int, max_lag: int, n_cenarios: int) -> int:
    """The record count a `mapcut` must have for these dimensions.

    Reproduces 356 for the reference file (`n_utv=2, n_estagios=7, max_lag=3,
    n_cenarios=273`) and 300 for this project's case with `n_utv=0`. That
    equality against a real file is the strongest available check on the layout
    understanding, which is why the writer sizes itself from here rather than
    from a hand-counted total.
    """
    for name, value in (
        ("n_utv", n_utv),
        ("n_estagios", n_estagios),
        ("max_lag", max_lag),
        ("n_cenarios", n_cenarios),
    ):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    return (
        _HEADER_RECORD_COUNT
        + PHYSICAL_RECORD_COUNT
        + _TREE_RECORD_COUNT
        + _STAGE_RECORD_COUNT
        + n_utv * n_estagios * (max_lag + 1)
        + n_cenarios
        + n_estagios
    )


def derive_n_utv(slots: Sequence[EntitySlotRecord]) -> int:
    """Count the travel-time axes the Cobre case actually carries.

    Derived rather than assumed zero: the reference DECOMP deck has `n_utv = 2`
    with plants 156 and 162, so a constant would be wrong on the very file this
    project validates against. Premise P3 is a claim about the Cobre input, not
    about the format.
    """
    return len(index_slots(slots).transit_bucket)


def assert_no_travel_time(slots: Sequence[EntitySlotRecord]) -> None:
    """Refuse to convert a case whose travel-time axis would be dropped.

    Premise P3 holds only while the case carries no `HydroTransitBucket` state.
    Raising here rather than silently emitting `n_utv = 0` is what stops a future
    case from having a whole state axis discarded without a trace.
    """
    count = derive_n_utv(slots)
    if count:
        raise LayoutError(
            f"the case carries {count} {EntityType.HYDRO_TRANSIT_BUCKET.name} slot(s), so premise "
            f"P3 does not hold: mapcut regs 7/8 are unimplemented and cortdeco's NCOEF would gain "
            f"the pi_qdefp block. Refusing to drop the travel-time axis silently."
        )
    _logger.info(
        "premise P3 holds: no %s state in the case, so n_utv = 0, mapcut regs 7/8 are absent, "
        "and NCOEF omits the pi_qdefp block",
        EntityType.HYDRO_TRANSIT_BUCKET.name,
    )


def assert_record_multiple(size_bytes: int, path_label: str) -> int:
    """Check a file size is a whole number of records and return that count."""
    records, remainder = divmod(size_bytes, RECORD_SIZE)
    if remainder:
        raise LayoutError(
            f"{path_label} is {size_bytes} bytes, not a whole number of {RECORD_SIZE}-byte "
            f"records: {records} records plus {remainder} trailing bytes"
        )
    return records


def assert_mapcut_layout(path: Path, header: MapcutDimensions) -> None:
    """Check a written `mapcut` against the layout its own header declares.

    Three independent consequences of one layout, in ascending cost: the size is
    a whole number of records, the record count is the one the header's scalars
    imply, and the undocumented physical span is untouched. A writer that
    miscounts records and a writer that mis-sizes one are different failures, so
    asserting only one leaves the other invisible.

    The count is checked against **the header** rather than against a count the
    writer derived for itself, which is the whole point: a wrong header and a
    wrong file otherwise agree with each other.
    """
    expected = mapcut_record_count(
        header.n_utv, header.numero_estagios, header.max_lag, header.numero_cenarios
    )
    observed = assert_record_multiple(path.stat().st_size, str(path))
    if observed != expected:
        raise LayoutError(
            f"{path} holds {observed} records but its header declares {expected} "
            f"(n_utv={header.n_utv}, n_estagios={header.numero_estagios}, "
            f"max_lag={header.max_lag}, n_cenarios={header.numero_cenarios})"
        )

    with path.open("rb") as handle:
        handle.seek(PHYSICAL_RECORD_FIRST * RECORD_SIZE)
        for index in range(PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST + 1):
            body = handle.read(RECORD_SIZE)
            # strip runs in C; the byte-by-byte search happens only once, on the
            # record that already failed.
            if body.strip(b"\x00"):
                offset = next(position for position, byte in enumerate(body) if byte)
                raise LayoutError(
                    f"record {index} is inside the zero-filled physical span (records "
                    f"{PHYSICAL_RECORD_FIRST}-{PHYSICAL_RECORD_LAST}) but byte {offset} "
                    f"is {body[offset]}"
                )
