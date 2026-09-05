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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

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

# The chain pointer is an int32 field, so a wrong pointer must be refused by name
# rather than reaching numpy as an overflow.
_MAX_POINTER = 2**31 - 1

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


def cortdeco_ncoef(
    n_uhes: int, n_utv: int, max_lag: int, n_sbm_gnl: int, n_estagios: int, n_patamares: int
) -> int:
    """`cortdeco`'s coefficient count, from the formula alone (I3).

    `1 + n_uhes + n_utv*max_lag + n_sbm_gnl*n_estagios*n_patamares`. Reproduces
    218 for the reference deck's own scalars and 212 for this project's. 218 is
    neither the reference file's last non-zero coefficient position (199) nor
    its non-zero coefficient count (182): the GNL block is dimensioned by stage
    count but only stage 0 is ever populated, so both shortcuts undercount it.
    """
    for name, value in (
        ("n_uhes", n_uhes),
        ("n_utv", n_utv),
        ("max_lag", max_lag),
        ("n_sbm_gnl", n_sbm_gnl),
        ("n_estagios", n_estagios),
        ("n_patamares", n_patamares),
    ):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    return 1 + n_uhes + n_utv * max_lag + n_sbm_gnl * n_estagios * n_patamares


@dataclass(frozen=True)
class CutBlockOffsets:
    """The three block starts inside one `cortdeco` cut's `NCOEF`-length vector.

    Built by `cortdeco_block_offsets` alongside `NCOEF` itself, so a record
    serializer can never compute an offset independently of the count it must
    agree with. `pi_qdefp` (travel time) has no offset here: this project holds
    `n_utv = 0` under premise P3, so `pi_varm` and `pi_gnl` are the only blocks a
    caller ever addresses, and `pi_varm`'s own width silently grows to
    `n_uhes + n_utv*max_lag` if a caller ever passes `n_utv > 0` in the formula
    that produced `pi_gnl` below — this module does not refuse that case, since
    `assert_no_travel_time` already does, upstream, for the whole package.
    """

    rhs: int
    pi_varm: int
    pi_gnl: int
    ncoef: int


def cortdeco_block_offsets(
    n_uhes: int, n_utv: int, max_lag: int, n_sbm_gnl: int, n_estagios: int, n_patamares: int
) -> CutBlockOffsets:
    """`NCOEF` and its three block starts, computed together (Requirement 2).

    Reproduces `CutBlockOffsets(rhs=0, pi_varm=1, pi_gnl=170, ncoef=212)` for
    this project's scalars and `pi_gnl=176, ncoef=218` for the reference deck's,
    both transcribed from the reference file's own populated positions (I4).
    """
    ncoef = cortdeco_ncoef(n_uhes, n_utv, max_lag, n_sbm_gnl, n_estagios, n_patamares)
    return CutBlockOffsets(rhs=0, pi_varm=1, pi_gnl=1 + n_uhes + n_utv * max_lag, ncoef=ncoef)


def cortdeco_gnl_offset(
    offsets: CutBlockOffsets,
    sbm: int,
    stage: int,
    block: int,
    n_sbm_gnl: int,
    n_estagios: int,
    n_patamares: int,
) -> int:
    """The submarket-major `pi_gnl` address for one `(sbm, stage, block)` triple (I4).

    `offsets.pi_gnl + sbm*(n_estagios*n_patamares) + stage*n_patamares + block`.
    Reproduces 176 for `(sbm=0, stage=0, block=0)` and 197 for `(sbm=1, stage=0,
    block=0)` against the reference deck's own `CutBlockOffsets` — the two
    positions the reference file actually populates. No caller computes this
    arithmetic inline (Requirement 2).
    """
    for name, value, bound in (
        ("sbm", sbm, n_sbm_gnl),
        ("stage", stage, n_estagios),
        ("block", block, n_patamares),
    ):
        if not 0 <= value < bound:
            raise LayoutError(f"{name} must be in 0..{bound - 1}, got {value}")
    return offsets.pi_gnl + sbm * (n_estagios * n_patamares) + stage * n_patamares + block


def cut_head_indices(numero_cortes: int, n_nodes: int, n_nodes_total: int) -> tuple[int, ...]:
    """The 1-based index of each node's last cut, zero for non-cut-building nodes (J2).

    `head(j) = numero_cortes - j`, descending, transcribed from the reference
    deck's own `[438, 437, 436, 435, 434, 433]` against its `numero_cortes` of
    438. This is the single source for both artifacts: `mapcut` reg 1 writes
    these and `cortdeco` chains back from them, so they cannot be allowed to
    disagree.
    """
    for name, value in (
        ("numero_cortes", numero_cortes),
        ("n_nodes", n_nodes),
        ("n_nodes_total", n_nodes_total),
    ):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    if n_nodes > n_nodes_total:
        raise LayoutError(
            f"n_nodes is {n_nodes} but n_nodes_total is {n_nodes_total}: there cannot be more "
            f"cut-building nodes than nodes"
        )
    if n_nodes and numero_cortes % n_nodes:
        raise LayoutError(
            f"numero_cortes {numero_cortes} is not divisible by n_nodes {n_nodes}, so the "
            f"per-node chains cannot have equal length (J2: numero_cortes = "
            f"numero_iteracoes * n_cut_building_nodes)"
        )
    return tuple(numero_cortes - node if node < n_nodes else 0 for node in range(n_nodes_total))


def cortdeco_record_count(numero_cortes: int) -> int:
    """`numero_cortes + 1`: CEPEL writes one cut beyond the head table (J4).

    Reproduces 439 for the reference deck's `numero_cortes` of 438, which is
    that file's real record count (11,842,464 / 26,976).
    """
    if numero_cortes < 0:
        raise LayoutError(f"numero_cortes must be non-negative, got {numero_cortes}")
    return numero_cortes + 1


def chain_pointer(record_index: int, n_nodes: int) -> int:
    """The 1-based pointer to the same node's previous cut, or 0 to terminate (J1).

    The on-disk pointer is 1-based, so record `i` (0-based) points at
    `i - n_nodes + 1`; the first `n_nodes` records have no predecessor and carry
    0. Verified against the reference: `chain_pointer(437, 6)` is 432, exactly
    what that file holds, and its first six records carry 0.

    The extra record needs no special case. At `record_index == numero_cortes`
    this returns `numero_cortes - n_nodes + 1`, which is the last node's head, so
    the extra record chains itself correctly by the same rule as every other
    (J4).
    """
    if record_index < 0:
        raise LayoutError(f"record_index must be non-negative, got {record_index}")
    if n_nodes <= 0:
        raise LayoutError(f"n_nodes must be positive, got {n_nodes}")
    if record_index < n_nodes:
        return 0
    return record_index - n_nodes + 1


def assert_pointer_in_range(pointer: int) -> None:
    """Refuse a pointer int32 cannot hold, with a named error rather than overflow."""
    if not 0 <= pointer <= _MAX_POINTER:
        raise LayoutError(
            f"chain pointer {pointer} is outside the int32 range 0..{_MAX_POINTER} that the "
            f"record's own header field can hold"
        )


def _read_pointers(path: Path, records: int) -> list[int]:
    """Every record's chain pointer, reading 4 bytes each rather than the file."""
    pointers: list[int] = []
    with path.open("rb") as handle:
        for index in range(records):
            handle.seek(index * TAMANHO_CORTE)
            pointers.append(int(np.frombuffer(handle.read(4), dtype="<i4")[0]))
    return pointers


def assert_cortdeco_layout(path: Path, numero_cortes: int, n_nodes: int) -> None:
    """Check a written `cortdeco` by walking its own bytes (Requirement 5).

    Five independent consequences of one layout, in ascending cost: the size is
    a whole number of records; the count is `numero_cortes + 1`; each node's
    chain, followed through the **on-disk** pointers rather than recomputed,
    holds exactly `numero_cortes // n_nodes` records and terminates at 0; those
    chains cover every cut record exactly once; and the extra record carries the
    last node's head as its pointer.

    The chains are walked rather than rebuilt from `chain_pointer` on purpose: a
    test or check that regenerates the arithmetic it is verifying proves only
    that the formula equals itself.
    """
    if n_nodes <= 0:
        raise LayoutError(f"n_nodes must be positive, got {n_nodes}")
    expected = cortdeco_record_count(numero_cortes)
    records, remainder = divmod(path.stat().st_size, TAMANHO_CORTE)
    if remainder:
        raise LayoutError(
            f"{path} is {path.stat().st_size} bytes, not a whole number of {TAMANHO_CORTE}-byte "
            f"records: {records} records plus {remainder} trailing bytes"
        )
    if records != expected:
        raise LayoutError(
            f"{path} holds {records} records but numero_cortes {numero_cortes} over {n_nodes} "
            f"nodes implies {expected} (numero_cortes + 1, J4)"
        )

    pointers = _read_pointers(path, records)
    heads = cut_head_indices(numero_cortes, n_nodes, n_nodes)
    per_node = numero_cortes // n_nodes
    visits: dict[int, int] = {}

    for node, head in enumerate(heads):
        index = head - 1
        walked = 0
        while True:
            if not 0 <= index < numero_cortes:
                raise LayoutError(
                    f"node {node}'s chain reached record {index}, outside the cut records "
                    f"0..{numero_cortes - 1}"
                )
            visits[index] = visits.get(index, 0) + 1
            if visits[index] > 1:
                raise LayoutError(
                    f"record {index} is reached by more than one chain, so the chains do not "
                    f"partition the file"
                )
            walked += 1
            pointer = pointers[index]
            if pointer == 0:
                break
            index = pointer - 1
        if walked != per_node:
            raise LayoutError(
                f"node {node}'s chain holds {walked} cuts but numero_cortes {numero_cortes} over "
                f"{n_nodes} nodes implies {per_node}"
            )

    # No separate "every record is covered" check: the two loops above already
    # imply it. n_nodes chains, each of exactly per_node records, none of which is
    # reached twice, hold n_nodes * per_node == numero_cortes distinct indices, all
    # within 0..numero_cortes-1 — so they exhaust the cut records. A third check
    # would be a branch nothing can reach.
    extra = pointers[numero_cortes]
    last_head = heads[n_nodes - 1]
    if extra != last_head:
        raise LayoutError(
            f"the extra record's pointer is {extra} but should be {last_head}, the last "
            f"cut-building node's head: it continues that node's chain rather than starting a "
            f"new one (J4)"
        )


def assert_uniform_blocks(patamares_por_estagio: Sequence[int]) -> int:
    """The single `n_patamares` this deck uses everywhere, or raise if it varies (I6).

    A ragged load-block count per stage has no defined `pi_gnl` layout: the
    offset formula in `cortdeco_gnl_offset` needs one scalar `n_patamares`, not
    a per-stage one, so a non-uniform deck must raise rather than pick a value.
    """
    distinct = sorted(set(patamares_por_estagio))
    if len(distinct) != 1:
        raise LayoutError(
            f"patamares_por_estagio is not uniform: distinct values {distinct}; cortdeco's "
            f"pi_gnl block has no defined layout for a ragged load-block count"
        )
    return distinct[0]
