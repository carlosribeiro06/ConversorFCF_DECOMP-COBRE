"""Native serializer for one DECOMP `cortdeco` cut record.

`idecomp.Cortdeco.cortes` has a no-op setter, so every byte here is written
natively, exactly as in `mapcut_writer`. One record is: an int32 chain pointer
at offset 0, then `NCOEF` float64 in block order `rhs | pi_varm | pi_gnl`, then
zero padding to `TAMANHO_CORTE`. `serialize_cut` owns that one record;
`write_cortdeco` owns the pointer values, the record order, the extra trailing
record and the whole file.

`CutInput.pi_varm` and `pi_gnl` are already placed in `CutBlockOffsets`' block
order, in Cobre's raw sign and units. `storage_coefficients` below extracts the
`pi_varm` sub-array, because a trunk pool's storage positions are ascending and
positionally identical to `codigos_uhes` (I3, I9).

`pi_gnl` carries **zeros** under premise P14, produced by `zeroed_gnl_block` and
enforced by `write_cortdeco` rather than left to the caller. Reducing Cobre's
anticipated-thermal ring positions to `pi_gnl`'s `(submarket, stage, block)`
address (I4) is unresolved and deferred to `ticket-016`: ring positions sharing
a delivery month carry different coefficients, so they are distinct state
variables rather than copies, and every candidate reduction writes a
structurally valid file that DESSEM reads without complaint. A wrong reduction
would therefore corrupt the cut's GNL slope invisibly, which is why this module
emits a declared zero instead of a guess. Consequently premises P4 (GNL sign)
and P5 (load-block weighting) are dormant: `_convert` still routes the block
through `negate_gnl_array`, but on zeros, so neither rule changes a byte.
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
    assert_gnl_span_is_zero,
    assert_pointer_in_range,
    chain_pointer,
    cortdeco_record_count,
    cut_head_indices,
    sync_directory,
    write_durably,
)
from conversor_fcf.logging_setup import get_logger
from conversor_fcf.mapping.rules import (
    MappingError,
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
    submarket-major GNL block; this dataclass does not know about submarkets,
    stages or load blocks, only about the flat vector `serialize_cut` writes.
    Under premise P14 that vector is `zeroed_gnl_block(offsets)`.
    """

    intercept: float
    pi_varm: NDArray[np.float64]
    pi_gnl: NDArray[np.float64]


def zeroed_gnl_block(offsets: CutBlockOffsets) -> NDArray[np.float64]:
    """The `pi_gnl` block every cut carries under premise P14: zeros, full width.

    The one named site that produces it, so the zeroing is a declared premise
    rather than a caller's accident, and so `ticket-016` has a single function to
    replace. The width comes from `offsets`, which is what keeps the block
    dimensioned by the `NCOEF` formula: P14 changes the values, never `NCOEF`,
    the record size or the file size.
    """
    return np.zeros(offsets.ncoef - offsets.pi_gnl, dtype=np.float64)


def storage_coefficients(
    piece: AffinePieceRecord, slots: Sequence[EntitySlotRecord], hydro_codes: Sequence[int]
) -> NDArray[np.float64]:
    """One piece's `pi_varm` sub-array, extracted from its pool's raw coefficients.

    Storage positions are ascending in the entity manifest and, by construction
    of `decomp_hydro_codes.json`, positionally identical to `codigos_uhes`, so
    extraction is a direct slice. Two checks guard that, because the alignment is
    what makes `pi_varm[i]` mean `codigos_uhes[i]`:

    - the storage positions must be exactly `range(len(hydro_codes))`, which is
      the alignment itself rather than a proxy for it;
    - `hydro_code_for_position` is still consulted per position (I9), so a plant
      absent from the code map raises `MappingError` by its own rule.

    The first subsumes the second on this deck, where storage occupies manifest
    positions 0-168. Both are kept: the second carries I9's own error message,
    and a strict superset is preferable to choosing between two readings of the
    same premise.
    """
    positions = index_slots(slots).storage
    # The invariant the alignment actually rests on, asserted rather than implied.
    # `pi_varm[i]` is only `codigos_uhes[i]`'s coefficient if the storage slots
    # occupy manifest positions 0..n_uhes-1 exactly, and consuming the code map
    # per position is a proxy for that which is neither necessary nor sufficient:
    # an interleaved manifest with a longer map satisfies the proxy while
    # producing a shorter pi_varm than the map declares. In this deck storage
    # occupies 0-168 so the two index spaces coincide; the exposure is at the
    # epic-06 seam, where a caller deriving n_uhes from the slots rather than
    # from the code map would place hydro coefficients against the wrong plants.
    if tuple(positions) != tuple(range(len(hydro_codes))):
        raise MappingError(
            f"storage occupies manifest positions {positions[:4]}... ({len(positions)} slots) but "
            f"the code map declares {len(hydro_codes)} plants at positions "
            f"0..{len(hydro_codes) - 1}; pi_varm[i] can only be codigos_uhes[i]'s coefficient "
            f"when the two agree exactly (I3, I9)"
        )
    for position in positions:
        hydro_code_for_position(hydro_codes, position)
    coefficients = np.asarray(piece.coefficients, dtype=np.float64)
    return coefficients[np.asarray(positions, dtype=np.intp)]


def _validate_lengths(cut: CutInput, offsets: CutBlockOffsets) -> None:
    # The record is a fixed TAMANHO_CORTE with the pointer at offset 0, so NCOEF
    # has a hard ceiling. Refused by name here for the same reason
    # assert_pointer_in_range exists: the alternative is numpy's unnamed "buffer
    # is smaller than requested size" from the frombuffer in serialize_cut.
    capacity = (TAMANHO_CORTE - 4) // 8
    if offsets.ncoef > capacity:
        raise LayoutError(
            f"NCOEF is {offsets.ncoef} but a {TAMANHO_CORTE}-byte record holds at most "
            f"{capacity} float64 coefficients after the 4-byte chain pointer"
        )
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


def _normalize_signed_zero(
    rhs: float, coefficients: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Turn `-0.0` into `+0.0`, and change nothing else.

    `negate_gnl_array` maps a zero coefficient to `-0.0`, whose byte pattern is
    `0x…80` rather than zero, so under premise P14 the whole GNL block would
    reach the disk as a pattern the reference deck holds nowhere: its 36 zero
    GNL slots are all true zeros. This is not a P14-only concern — finding F4
    records that 36 of the 42 slots are structurally zero even with a real
    reduction in place, so the negation would produce them anyway.

    Adding `0.0` is exact on every other finite value and returns it unchanged,
    which is why it can be applied to the whole vector rather than to the GNL
    block alone. Non-finite values never reach here: `_assert_finite` rejects
    them first.
    """
    return rhs + 0.0, coefficients + 0.0


def _assert_gnl_input_is_zero(cuts: Sequence[Sequence[CutInput]]) -> None:
    """Refuse a non-zero `pi_gnl` while premise P14 holds.

    The run manifest declares that every GNL coefficient is zero, so the writer
    makes that true of the file instead of trusting its caller: a manifest
    asserting a premise the artifact does not honour is the auditability defect
    this gate exists to prevent.

    `serialize_cut` deliberately does **not** carry the gate. It is the general
    record primitive `ticket-016` reuses once the reduction is settled, and it is
    still covered by tests that pass real non-zero GNL coefficients through the
    P2 and P4 conversions.
    """
    for node, node_cuts in enumerate(cuts):
        for iteration, cut in enumerate(node_cuts):
            nonzero = np.flatnonzero(np.asarray(cut.pi_gnl, dtype=np.float64) != 0.0)
            if nonzero.size:
                slot = int(nonzero[0])
                raise LayoutError(
                    f"node {node} iteration {iteration} carries a non-zero pi_gnl at slot "
                    f"{slot} ({cut.pi_gnl[slot]!r}), but premise P14 emits the whole GNL block "
                    f"as zeros; pass zeroed_gnl_block(offsets) until ticket-016 settles the "
                    f"ring-position reduction"
                )


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
    if per_node == 0:
        # Otherwise the extra record's `cuts[n_nodes - 1][per_node - 1]` indexes an
        # empty list at [-1] and raises a bare IndexError, while every other
        # rejected input in this function raises a named LayoutError.
        raise LayoutError(
            f"numero_cortes is {numero_cortes} over {n_nodes} nodes, so every chain would be "
            f"empty and there is no last cut for the extra record to duplicate (premise P13)"
        )
    for node, node_cuts in enumerate(cuts):
        if len(node_cuts) != per_node:
            raise LayoutError(
                f"node {node} carries {len(node_cuts)} cuts but every node must carry "
                f"{per_node}: unequal chains cannot satisfy head(j) = numero_cortes - j"
            )
    _assert_gnl_input_is_zero(cuts)

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
        write_durably(temporary, b"".join(records))
        assert_cortdeco_layout(temporary, numero_cortes, n_nodes)
        assert_gnl_span_is_zero(temporary, expected, offsets)
        os.replace(temporary, path)
        sync_directory(path.parent)
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
    _logger.warning(
        "premise P14: cortdeco's pi_gnl coefficients are emitted as zeros - %d slots at "
        "coefficient positions %d-%d of every one of the %d records. The block stays "
        "dimensioned, so NCOEF %d and the %d-byte file are exactly what a populated block would "
        "produce; only the values are zero. Reducing Cobre's anticipated-thermal ring positions "
        "to a (submarket, stage, block) address is unresolved and deferred to ticket-016, "
        "because every candidate reduction writes a file DESSEM reads without complaint and a "
        "wrong one would corrupt the GNL slope invisibly. Premises P4 (GNL sign) and P5 "
        "(load-block weighting) are therefore DORMANT: no real coefficient reaches either",
        offsets.ncoef - offsets.pi_gnl,
        offsets.pi_gnl,
        offsets.ncoef - 1,
        expected,
        offsets.ncoef,
        expected * TAMANHO_CORTE,
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
    rhs, coefficients = _normalize_signed_zero(rhs, coefficients)

    body = bytearray(TAMANHO_CORTE)
    pointer_view = np.frombuffer(body, dtype="<i4", count=1)
    pointer_view[0] = pointer
    coefficient_view = np.frombuffer(body, dtype="<f8", count=offsets.ncoef, offset=4)
    coefficient_view[offsets.rhs] = rhs
    coefficient_view[offsets.pi_varm :] = coefficients
    return bytes(body)
