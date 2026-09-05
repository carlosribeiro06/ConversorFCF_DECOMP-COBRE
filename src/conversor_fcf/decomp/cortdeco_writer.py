"""Native serializer for one DECOMP `cortdeco` cut record.

`idecomp.Cortdeco.cortes` has a no-op setter, so every byte here is written
natively, exactly as in `mapcut_writer`. One record is: an int32 chain pointer
at offset 0, then `NCOEF` float64 in block order `rhs | pi_varm | pi_gnl`, then
zero padding to `TAMANHO_CORTE`. This module owns that one record; `ticket-010`
owns the pointer's value, the record order, the extra trailing record and the
whole file.

`CutInput.pi_varm` and `pi_gnl` are already placed in `CutBlockOffsets`' block
order, in Cobre's raw sign and units. Placing a Cobre entity manifest's
anticipated-thermal coefficients onto `pi_gnl`'s `(submarket, stage, block)`
address (I4) needs a thermal-to-bus lookup and a delivery-date-to-stage
lookup, both of which live in `CaseInputs` (`inputs_reader.py`) and are not
part of this module's inputs; building that placement is therefore left to
whichever pipeline stage has `CaseInputs` in hand. `storage_coefficients`
below extracts the `pi_varm` sub-array, which needs no such lookup, because a
trunk pool's storage positions are ascending and positionally identical to
`codigos_uhes` (I3, I9).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from conversor_fcf.cobre.entities import index_slots
from conversor_fcf.cobre.policy_reader import AffinePieceRecord, EntitySlotRecord
from conversor_fcf.decomp.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_cortdeco_layout,
    assert_pointer_in_range,
    chain_pointer,
    cortdeco_record_count,
    cut_head_indices,
)
from conversor_fcf.logging_setup import get_logger
from conversor_fcf.mapping.rules import (
    hydro_code_for_position,
    negate_gnl_array,
    to_decomp_cost,
    to_decomp_costs,
)

_logger = get_logger("cortdeco_writer")


@dataclass(frozen=True)
class CutInput:
    """One `cortdeco` cut, in Cobre's raw sign and units, in DECOMP block order.

    `pi_varm[i]` is hydro position `i`'s storage coefficient, ordered like
    `codigos_uhes`. `pi_gnl[k]` is the `k`-th slot of `CutBlockOffsets`'
    submarket-major GNL block, already placed by the caller (see the module
    docstring); this dataclass does not know about submarkets, stages or load
    blocks, only about the flat vector `serialize_cut` writes.
    """

    intercept: float
    pi_varm: NDArray[np.float64]
    pi_gnl: NDArray[np.float64]


def storage_coefficients(
    piece: AffinePieceRecord, slots: Sequence[EntitySlotRecord], hydro_codes: Sequence[int]
) -> NDArray[np.float64]:
    """One piece's `pi_varm` sub-array, extracted from its pool's raw coefficients.

    Storage positions are ascending in the entity manifest and, by construction
    of `decomp_hydro_codes.json`, positionally identical to `codigos_uhes`, so
    extraction is a direct slice. `hydro_code_for_position` is still consulted
    per position, so a plant absent from the code map raises `MappingError`
    (I9) rather than silently landing in the wrong `pi_varm` slot; this module
    adds no second policy on top of it.
    """
    positions = index_slots(slots).storage
    for position in positions:
        hydro_code_for_position(hydro_codes, position)
    coefficients = np.asarray(piece.coefficients, dtype=np.float64)
    return coefficients[np.asarray(positions, dtype=np.intp)]


def _validate_lengths(cut: CutInput, offsets: CutBlockOffsets) -> None:
    expected_varm = offsets.pi_gnl - offsets.pi_varm
    if len(cut.pi_varm) != expected_varm:
        raise LayoutError(
            f"pi_varm has {len(cut.pi_varm)} entries but the layout expects {expected_varm}"
        )
    expected_gnl = offsets.ncoef - offsets.pi_gnl
    if len(cut.pi_gnl) != expected_gnl:
        raise LayoutError(
            f"pi_gnl has {len(cut.pi_gnl)} entries but the layout expects {expected_gnl}"
        )


def _convert(cut: CutInput) -> tuple[float, NDArray[np.float64]]:
    """Apply premise P2 (÷1000) and P4 (GNL sign) exactly once, on the way in."""
    rhs = to_decomp_cost(cut.intercept)
    coefficients = to_decomp_costs(np.concatenate((cut.pi_varm, cut.pi_gnl)))
    gnl_start = len(cut.pi_varm)
    coefficients[gnl_start:] = negate_gnl_array(coefficients[gnl_start:])
    return rhs, coefficients


def _assert_finite(rhs: float, coefficients: NDArray[np.float64], pi_varm_width: int) -> None:
    if not np.isfinite(rhs):
        raise LayoutError(f"rhs is not finite: {rhs!r}")
    bad = np.flatnonzero(~np.isfinite(coefficients))
    if bad.size:
        position = int(bad[0])
        if position < pi_varm_width:
            raise LayoutError(
                f"pi_varm position {position} is not finite: {coefficients[position]!r}"
            )
        raise LayoutError(
            f"pi_gnl position {position - pi_varm_width} is not finite: {coefficients[position]!r}"
        )


def node_and_iteration(record_index: int, n_nodes: int) -> tuple[int, int]:
    """Which node and iteration own 0-based record `record_index` (J3).

    `node = (n_nodes - 1) - (i mod n_nodes)` and `iteration = i // n_nodes`, both
    0-based. Checks: the reference's node 0 head is 0-based 437 and
    `437 mod 6 == 5` gives node 0; its first record belongs to node 5. Within one
    iteration the records descend in node position, which is the order an SDDP
    backward pass produces cuts in.
    """
    if record_index < 0:
        raise LayoutError(f"record_index must be non-negative, got {record_index}")
    if n_nodes <= 0:
        raise LayoutError(f"n_nodes must be positive, got {n_nodes}")
    return (n_nodes - 1) - (record_index % n_nodes), record_index // n_nodes


def write_cortdeco(
    cuts: Sequence[Sequence[CutInput]],
    path: Path,
    *,
    numero_cortes: int,
    offsets: CutBlockOffsets,
) -> int:
    """Write a complete `cortdeco`, returning the record count.

    `cuts[node][iteration]` is one cut-building node's cut sequence, in
    node-position order, so `cuts[0]` is the node whose head is highest.
    `numero_cortes` is declared rather than derived from the sequence lengths
    (Requirement 8), and cross-checked against them, so a short cut list fails
    loudly instead of silently producing a shorter file.

    Built at a temporary path, checked against its own layout, and only then
    renamed, as `write_mapcut` does: a file that fails its invariant never
    appears at the destination.
    """
    n_nodes = len(cuts)
    if n_nodes == 0:
        raise LayoutError("no cut-building nodes were supplied")
    supplied = sum(len(node_cuts) for node_cuts in cuts)
    if supplied != numero_cortes:
        raise LayoutError(
            f"{supplied} cuts were supplied across {n_nodes} nodes but numero_cortes is "
            f"{numero_cortes}; the count is declared, not inferred, so the two must agree"
        )
    heads = cut_head_indices(numero_cortes, n_nodes, n_nodes)
    per_node = numero_cortes // n_nodes
    for node, node_cuts in enumerate(cuts):
        if len(node_cuts) != per_node:
            raise LayoutError(
                f"node {node} carries {len(node_cuts)} cuts but every node must carry "
                f"{per_node}: unequal chains cannot satisfy head(j) = numero_cortes - j"
            )

    expected = cortdeco_record_count(numero_cortes)
    records: list[bytes] = []
    for index in range(numero_cortes):
        node, iteration = node_and_iteration(index, n_nodes)
        pointer = chain_pointer(index, n_nodes)
        assert_pointer_in_range(pointer)
        records.append(serialize_cut(cuts[node][iteration], pointer, offsets))

    # The extra record duplicates the last node's last cut (premise P13) and its
    # pointer comes from the same rule as every other record, which lands on that
    # node's head (J4).
    extra_pointer = chain_pointer(numero_cortes, n_nodes)
    assert_pointer_in_range(extra_pointer)
    records.append(serialize_cut(cuts[n_nodes - 1][per_node - 1], extra_pointer, offsets))

    if len(records) != expected:
        raise LayoutError(
            f"assembled {len(records)} records but the layout invariant expects {expected}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_bytes(b"".join(records))
        assert_cortdeco_layout(temporary, numero_cortes, n_nodes)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    _logger.warning(
        "premise P13: the extra record beyond numero_cortes duplicates the last cut-building "
        "node's last cut. The reference holds the next backward-pass cut there (iteration %d), "
        "which a checkpoint with %d complete iterations cannot supply. A duplicate is "
        "mathematically inert, since a repeated hyperplane adds nothing to an FCF, while a "
        "zero-filled record would fabricate the constraint theta >= 0",
        per_node + 1,
        per_node,
    )
    _logger.info(
        "wrote cortdeco %s records=%d bytes=%d numero_cortes=%d nodes=%d ncoef=%d heads=%s",
        path,
        expected,
        expected * TAMANHO_CORTE,
        numero_cortes,
        n_nodes,
        offsets.ncoef,
        list(heads),
    )
    return expected


def serialize_cut(cut: CutInput, pointer: int, offsets: CutBlockOffsets) -> bytes:
    """One fixed `cortdeco` record: pointer, then `NCOEF` float64, zero-padded.

    Always exactly `TAMANHO_CORTE` bytes. Validation runs before any byte is
    produced: a length mismatch or a non-finite value raises and returns
    nothing, never a truncated or zero-extended record (Requirements 6, 7).
    """
    _validate_lengths(cut, offsets)
    rhs, coefficients = _convert(cut)
    _assert_finite(rhs, coefficients, len(cut.pi_varm))

    body = bytearray(TAMANHO_CORTE)
    pointer_view = np.frombuffer(body, dtype="<i4", count=1)
    pointer_view[0] = pointer
    coefficient_view = np.frombuffer(body, dtype="<f8", count=offsets.ncoef, offset=4)
    coefficient_view[offsets.rhs] = rhs
    coefficient_view[offsets.pi_varm :] = coefficients
    return bytes(body)
