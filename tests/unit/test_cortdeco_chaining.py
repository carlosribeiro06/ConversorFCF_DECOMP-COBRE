"""Unit tests for the `cortdeco` chain, head table and whole-file assembly.

Per the master plan's Numeric Path Testing Policy, every anchor is transcribed
from `ticket-010`'s findings: 439 records against the reference's `numero_cortes`
of 438, the heads `(438..433)`, `chain_pointer(437, 6) == 432`, and this
project's 289 records over 7,796,064 bytes.

The named deliberate mutations are **a 0-based chain pointer**, **`next_index = 0`
on the extra record** and **dropping the GNL term from `NCOEF`**. Each was
applied, observed red, reverted and confirmed byte-identical; the evidence is in
the plan's state entry. Every mutation guard below calls the function under
mutation (policy clause 4).

The last section covers premise P14, which emits `cortdeco`'s `pi_gnl`
coefficients as zeros while leaving the block dimensioned. Its anchors read the
written bytes rather than the input array, because an all-zeros `pi_gnl` is a
valid-looking input: a writer that dropped the block, mis-addressed it, or left
`-0.0` in it would satisfy any assertion made on what it was given.

Chains are always walked through the **on-disk** pointers, never recomputed from
`chain_pointer`: a check that regenerates the arithmetic it verifies proves only
that the formula equals itself.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.decomp import cortdeco_writer, layout
from conversor_fcf.decomp.cortdeco_writer import (
    CutInput,
    node_and_iteration,
    write_cortdeco,
    zeroed_gnl_block,
)
from conversor_fcf.decomp.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_cortdeco_layout,
    assert_gnl_span_is_zero,
    assert_pointer_in_range,
    chain_pointer,
    cortdeco_block_offsets,
    cortdeco_ncoef,
    cortdeco_record_count,
    cut_head_indices,
)

# The reference deck's own numbers (J2, J4).
REFERENCE_CORTES = 438
REFERENCE_NODES = 6
REFERENCE_NODES_TOTAL = 273
REFERENCE_HEADS = (438, 437, 436, 435, 434, 433)
REFERENCE_RECORDS = 439

# This project's own (J7).
PROJECT_CORTES = 288
PROJECT_HEADS = (288, 287, 286, 285, 284, 283)
PROJECT_RECORDS = 289
PROJECT_BYTES = 7_796_064

# The synthetic case this suite assembles: 3 nodes x 4 iterations.
NODES = 3
PER_NODE = 4
CORTES = NODES * PER_NODE
RECORDS = CORTES + 1

# This project's own cut geometry (I3, I4): NCOEF 212 with the 42-slot GNL block
# at positions 170-211. Zeroed under premise P14 but still dimensioned.
PROJECT_UHES = 169
PROJECT_NCOEF = 212
PROJECT_PI_GNL = 170
PROJECT_GNL_SLOTS = 42
PROJECT_GNL_LAST = 211
PROJECT_NODES = 6
PROJECT_PER_NODE = 48


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


def _offsets() -> CutBlockOffsets:
    return cortdeco_block_offsets(
        n_uhes=3, n_utv=0, max_lag=0, n_sbm_gnl=1, n_estagios=2, n_patamares=2
    )


def _cuts() -> list[list[CutInput]]:
    """A distinct cut per (node, iteration), so a misplaced record is visible."""
    return [
        [
            CutInput(
                intercept=float(1_000_000 * (node + 1) + 1000 * iteration),
                pi_varm=np.array([float(node), float(iteration), 0.0]),
                pi_gnl=np.zeros(4),
            )
            for iteration in range(PER_NODE)
        ]
        for node in range(NODES)
    ]


def _pointers(path: Path) -> list[int]:
    raw = path.read_bytes()
    return [
        int(np.frombuffer(raw[i * TAMANHO_CORTE : i * TAMANHO_CORTE + 4], dtype="<i4")[0])
        for i in range(len(raw) // TAMANHO_CORTE)
    ]


def _record(path: Path, index: int) -> bytes:
    raw = path.read_bytes()
    return raw[index * TAMANHO_CORTE : (index + 1) * TAMANHO_CORTE]


def _broken_pointer(path: Path, destination: Path, record: int, pointer: int) -> Path:
    """Write a copy of `path` at `destination` with one record's pointer overwritten."""
    raw = bytearray(path.read_bytes())
    np.frombuffer(raw, dtype="<i4", count=1, offset=record * TAMANHO_CORTE)[0] = pointer
    destination.write_bytes(bytes(raw))
    return destination


# --- the head table ---------------------------------------------------------


def test_head_indices_reproduce_the_reference_deck() -> None:
    """(438..433): the reference mapcut's own six non-zero heads."""
    heads = cut_head_indices(REFERENCE_CORTES, REFERENCE_NODES, REFERENCE_NODES_TOTAL)
    assert heads[:REFERENCE_NODES] == REFERENCE_HEADS
    assert set(heads[REFERENCE_NODES:]) == {0}
    assert len(heads) == REFERENCE_NODES_TOTAL


def test_head_indices_reproduce_this_project() -> None:
    """(288..283): exactly what ticket-007 already writes into mapcut reg 1."""
    heads = cut_head_indices(PROJECT_CORTES, 6, REFERENCE_NODES_TOTAL)
    assert heads[:6] == PROJECT_HEADS
    assert sum(heads[6:]) == 0


def test_head_indices_reject_an_indivisible_cut_count() -> None:
    with pytest.raises(LayoutError, match="not divisible"):
        cut_head_indices(437, 6, 273)


def test_head_indices_reject_more_cut_building_nodes_than_nodes() -> None:
    with pytest.raises(LayoutError, match="more cut-building nodes than nodes"):
        cut_head_indices(12, 7, 6)


@pytest.mark.parametrize("field", ["numero_cortes", "n_nodes", "n_nodes_total"])
def test_head_indices_reject_a_negative_scalar(field: str) -> None:
    kwargs = {"numero_cortes": 12, "n_nodes": 3, "n_nodes_total": 3}
    kwargs[field] = -1
    with pytest.raises(LayoutError, match=f"{field} must be non-negative"):
        cut_head_indices(**kwargs)


# --- the record count -------------------------------------------------------


def test_record_count_reproduces_the_reference_file() -> None:
    """439 is cortdeco.rv0's real record count: 11,842,464 / 26,976."""
    assert cortdeco_record_count(REFERENCE_CORTES) == REFERENCE_RECORDS
    assert REFERENCE_RECORDS * TAMANHO_CORTE == 11_842_464


def test_record_count_for_this_project() -> None:
    assert cortdeco_record_count(PROJECT_CORTES) == PROJECT_RECORDS
    assert PROJECT_RECORDS * TAMANHO_CORTE == PROJECT_BYTES


def test_record_count_rejects_a_negative_cut_count() -> None:
    with pytest.raises(LayoutError, match="numero_cortes must be non-negative"):
        cortdeco_record_count(-1)


# --- the chain pointer ------------------------------------------------------


def test_chain_pointer_reproduces_the_reference_record() -> None:
    """437 -> 432 is what cortdeco.rv0 actually holds at that record."""
    assert chain_pointer(437, REFERENCE_NODES) == 432


@pytest.mark.parametrize("index", list(range(REFERENCE_NODES)))
def test_the_first_record_of_each_node_terminates(index: int) -> None:
    """Exactly the first n_nodes records carry 0, one per node."""
    assert chain_pointer(index, REFERENCE_NODES) == 0
    assert chain_pointer(index + REFERENCE_NODES, REFERENCE_NODES) != 0


def test_the_extra_record_needs_no_special_pointer_rule() -> None:
    """chain_pointer at numero_cortes lands on the last node's head by itself (J4)."""
    heads = cut_head_indices(REFERENCE_CORTES, REFERENCE_NODES, REFERENCE_NODES)
    assert chain_pointer(REFERENCE_CORTES, REFERENCE_NODES) == 433 == heads[-1]
    project_heads = cut_head_indices(PROJECT_CORTES, 6, 6)
    assert chain_pointer(PROJECT_CORTES, 6) == 283 == project_heads[-1]


def test_a_zero_based_pointer_would_disagree_with_the_reference() -> None:
    """The first named mutation: 0-based would give 431, and the file holds 432."""
    one_based = chain_pointer(437, REFERENCE_NODES)
    assert one_based == 432
    assert one_based - 1 == 431, "the 0-based reading, which the reference contradicts"


@pytest.mark.parametrize(("index", "nodes"), [(-1, 6), (0, 0), (0, -3)])
def test_chain_pointer_rejects_invalid_input(index: int, nodes: int) -> None:
    with pytest.raises(LayoutError):
        chain_pointer(index, nodes)


@pytest.mark.parametrize("pointer", [-1, 2**31, 2**40])
def test_a_pointer_outside_int32_is_refused_by_name(pointer: int) -> None:
    """LayoutError, not the OverflowError numpy would raise on assignment."""
    with pytest.raises(LayoutError, match="outside the int32 range"):
        assert_pointer_in_range(pointer)


def test_the_largest_representable_pointer_is_accepted() -> None:
    assert_pointer_in_range(2**31 - 1)


# --- which record holds which cut ------------------------------------------


def test_the_first_record_belongs_to_the_last_node() -> None:
    """J3: node = (n_nodes - 1) - (i mod n_nodes), so record 0 is the last node's."""
    assert node_and_iteration(0, REFERENCE_NODES) == (5, 0)
    assert node_and_iteration(5, REFERENCE_NODES) == (0, 0)


def test_the_reference_head_record_belongs_to_node_zero() -> None:
    """0-based 437 is node 0's head, and 437 mod 6 == 5 gives node 0."""
    node, iteration = node_and_iteration(437, REFERENCE_NODES)
    assert node == 0
    assert iteration == 72, "the 73rd and last iteration, 0-based"


@pytest.mark.parametrize(("index", "nodes"), [(-1, 6), (0, 0)])
def test_node_and_iteration_rejects_invalid_input(index: int, nodes: int) -> None:
    with pytest.raises(LayoutError):
        node_and_iteration(index, nodes)


# --- whole-file assembly ----------------------------------------------------


def test_the_written_file_is_one_record_beyond_the_cut_count(tmp_path: Path) -> None:
    path = tmp_path / "cortdeco.rv0"
    count = write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    assert count == RECORDS == 13
    assert path.stat().st_size == RECORDS * TAMANHO_CORTE


def test_every_chain_is_walked_from_the_bytes_and_partitions_the_file(tmp_path: Path) -> None:
    """The central invariant, read from disk rather than recomputed."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    pointers = _pointers(path)
    heads = cut_head_indices(CORTES, NODES, NODES)

    visited: list[int] = []
    for head in heads:
        index = head - 1
        chain: list[int] = []
        while True:
            chain.append(index)
            pointer = pointers[index]
            if pointer == 0:
                break
            index = pointer - 1
        assert len(chain) == PER_NODE, f"head {head} walked {chain}"
        # Every record in one chain belongs to one node, so they share i mod n_nodes.
        assert len({i % NODES for i in chain}) == 1
        visited.extend(chain)

    assert sorted(visited) == list(range(CORTES)), "the chains partition the cut records"
    assert len(visited) == len(set(visited)), "no record is reached twice"


def test_the_extra_record_repeats_the_last_cut_but_not_its_pointer(tmp_path: Path) -> None:
    """Payload identical from byte 4; the pointer deliberately differs (J4)."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    heads = cut_head_indices(CORTES, NODES, NODES)

    last_head_record = heads[-1] - 1
    extra = _record(path, CORTES)
    source = _record(path, last_head_record)
    assert extra[4:] == source[4:], "the cut itself is duplicated"

    pointers = _pointers(path)
    assert pointers[CORTES] == heads[-1], "the extra record continues the last node's chain"
    assert pointers[last_head_record] != pointers[CORTES], (
        "a copied pointer would put the extra record in the wrong chain position"
    )


def test_the_layout_assertion_accepts_the_file_it_wrote(tmp_path: Path) -> None:
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    assert_cortdeco_layout(path, CORTES, NODES)


def test_a_corrupted_pointer_is_refused_naming_the_record(tmp_path: Path) -> None:
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    # Record 11 is node 0's head; terminate it immediately so its chain holds one
    # cut instead of PER_NODE. Pointing it at itself (pointer=12) would trip the
    # visited-record check instead, which the next test covers.
    broken = _broken_pointer(path, tmp_path / "broken.rv0", record=11, pointer=0)

    with pytest.raises(LayoutError) as excinfo:
        assert_cortdeco_layout(broken, CORTES, NODES)
    message = str(excinfo.value)
    assert "chain holds 1 cuts" in message
    assert "implies 4" in message


def test_a_zeroed_extra_record_pointer_is_refused(tmp_path: Path) -> None:
    """The second named mutation: a 0 there would read as a seventh chain head."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    broken = _broken_pointer(path, tmp_path / "broken.rv0", record=CORTES, pointer=0)

    with pytest.raises(LayoutError) as excinfo:
        assert_cortdeco_layout(broken, CORTES, NODES)
    message = str(excinfo.value)
    assert "extra record's pointer is 0" in message
    assert "continues that node's chain" in message


def test_a_pointer_leaving_the_cut_records_is_refused(tmp_path: Path) -> None:
    """A pointer into the extra record, or past it, is not a valid predecessor."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    # 13 is 1-based, so 0-based 12 — the extra record, outside the cut records.
    broken = _broken_pointer(path, tmp_path / "broken.rv0", record=11, pointer=13)

    with pytest.raises(LayoutError, match="reached record 12, outside the cut records 0..11"):
        assert_cortdeco_layout(broken, CORTES, NODES)


def test_a_negative_pointer_is_not_mistaken_for_a_terminator(tmp_path: Path) -> None:
    """Only `0` terminates a chain, and the check must be equality, not `<= 0`.

    Relaxing the terminator test to `pointer <= 0` survived the whole suite,
    because `_broken_pointer` was only ever exercised with 0 and with pointers
    past the end. A negative on-disk pointer would then read as a valid
    terminator, so a corrupted file whose chain length happened to come out right
    would be accepted.
    """
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    broken = _broken_pointer(path, tmp_path / "broken.rv0", record=11, pointer=-5)

    with pytest.raises(LayoutError, match="reached record -6, outside the cut records 0..11"):
        assert_cortdeco_layout(broken, CORTES, NODES)


def test_two_chains_reaching_one_record_are_refused(tmp_path: Path) -> None:
    """Merged chains keep their length, so only tracking visited records catches them."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    # Node 1's chain is 10 -> 7 -> 4 -> 1; redirect its first hop into node 0's
    # chain (11 -> 8 -> 5 -> 2) at the same depth, so both stay 4 records long.
    broken = _broken_pointer(path, tmp_path / "broken.rv0", record=10, pointer=9)

    with pytest.raises(LayoutError, match="record 8 is reached by more than one chain"):
        assert_cortdeco_layout(broken, CORTES, NODES)


def test_a_disagreement_between_the_invariant_and_the_assembly_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard exists so a wrong assembly loop cannot reach the destination."""
    monkeypatch.setattr(cortdeco_writer, "cortdeco_record_count", lambda _cortes: 99)
    path = tmp_path / "cortdeco.rv0"
    with pytest.raises(LayoutError, match="assembled 13 records but the layout invariant"):
        write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    assert not list(tmp_path.iterdir())


def test_a_truncated_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    short = tmp_path / "short.rv0"
    short.write_bytes(path.read_bytes()[: CORTES * TAMANHO_CORTE])
    with pytest.raises(LayoutError, match="holds 12 records"):
        assert_cortdeco_layout(short, CORTES, NODES)


def test_a_file_with_trailing_bytes_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    ragged = tmp_path / "ragged.rv0"
    ragged.write_bytes(path.read_bytes() + b"\x01")
    with pytest.raises(LayoutError, match="trailing bytes"):
        assert_cortdeco_layout(ragged, CORTES, NODES)


def test_a_non_positive_node_count_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    with pytest.raises(LayoutError, match="n_nodes must be positive"):
        assert_cortdeco_layout(path, CORTES, 0)


# --- the declared cut count is not inferred --------------------------------


def test_a_short_cut_list_is_refused_naming_both_counts(tmp_path: Path) -> None:
    cuts = _cuts()
    cuts[1] = cuts[1][:-1]
    path = tmp_path / "cortdeco.rv0"
    with pytest.raises(LayoutError, match="11 cuts were supplied.*numero_cortes is 12"):
        write_cortdeco(cuts, path, numero_cortes=CORTES, offsets=_offsets())
    assert not list(tmp_path.iterdir()), "no file, not even a partial one"


def test_unequal_chains_are_refused(tmp_path: Path) -> None:
    """Equal-length chains are what head(j) = numero_cortes - j presumes."""
    cuts = _cuts()
    cuts[0] = cuts[0] + [cuts[0][-1]]
    cuts[1] = cuts[1][:-1]
    path = tmp_path / "cortdeco.rv0"
    with pytest.raises(LayoutError, match="node 0 carries 5 cuts"):
        write_cortdeco(cuts, path, numero_cortes=CORTES, offsets=_offsets())


def test_a_zero_cut_count_is_refused_by_name(tmp_path: Path) -> None:
    """Not an IndexError: every other rejected input here raises LayoutError.

    With `per_node == 0` the extra record's `cuts[n_nodes - 1][per_node - 1]`
    indexes an empty list at `[-1]`, which used to escape as a bare
    `IndexError: list index out of range`.
    """
    with pytest.raises(LayoutError, match="every chain would be empty"):
        write_cortdeco([[], [], []], tmp_path / "cortdeco.rv0", numero_cortes=0, offsets=_offsets())
    assert not list(tmp_path.iterdir()), "nothing is written when the count is refused"


def test_an_ncoef_beyond_the_record_capacity_is_refused_by_name() -> None:
    """The record's other fixed-width limit, named like `assert_pointer_in_range`.

    A `TAMANHO_CORTE` record holds `(26976 - 4) // 8 = 3371` float64 after the
    4-byte pointer. Beyond that `np.frombuffer` raised an unnamed
    `ValueError: buffer is smaller than requested size`, mentioning neither
    `NCOEF` nor `TAMANHO_CORTE`. Unreachable for this deck at 212 against 3371.
    """
    capacity = (TAMANHO_CORTE - 4) // 8
    assert capacity == 3371
    offsets = CutBlockOffsets(rhs=0, pi_varm=1, pi_gnl=2, ncoef=capacity + 1)
    cut = CutInput(
        intercept=0.0, pi_varm=np.zeros(1), pi_gnl=np.zeros(offsets.ncoef - offsets.pi_gnl)
    )
    with pytest.raises(LayoutError, match=f"NCOEF is {capacity + 1}"):
        cortdeco_writer.serialize_cut(cut, pointer=0, offsets=offsets)


def test_the_project_ncoef_is_well_inside_the_record_capacity() -> None:
    """The guard above must not be able to refuse a real case."""
    assert _project_offsets().ncoef == PROJECT_NCOEF < (TAMANHO_CORTE - 4) // 8


def test_no_cut_building_nodes_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LayoutError, match="no cut-building nodes"):
        write_cortdeco([], tmp_path / "cortdeco.rv0", numero_cortes=0, offsets=_offsets())


# --- atomicity and the premise ---------------------------------------------


def test_the_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    assert path.is_file()
    assert not list(path.parent.glob("*.partial"))


def test_the_payload_is_synced_before_the_rename_and_the_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durability, in the order that makes it durability rather than decoration.

    `os.replace` is atomic for readers but not against a machine crash: the
    rename can reach the disk while the data blocks do not, leaving a full-size
    zero-filled file at the destination. In a zero-filled `cortdeco` every chain
    pointer reads 0, so every record looks like its own chain head and every cut
    like an all-zero hyperplane — `theta >= 0`, the fabrication premise P13
    refuses to write on purpose. It reads cleanly, which is why it must not be
    reachable.

    The two pre-rename invariant checks read the file back through the page
    cache, so they prove nothing about durability and cannot substitute for this.
    """
    events: list[str] = []
    # The originals come from `layout`, which defines them: mypy --strict forbids
    # reading a re-exported name back off the module that imported it.
    real_replace = os.replace
    real_write_durably = layout.write_durably
    real_sync_directory = layout.sync_directory

    def spy_write(target: Path, payload: bytes) -> None:
        real_write_durably(target, payload)
        events.append(f"fsync:{target.name}")

    def spy_replace(source: object, destination: object) -> None:
        events.append("replace")
        real_replace(source, destination)  # type: ignore[arg-type]

    def spy_sync_directory(target: Path) -> None:
        real_sync_directory(target)
        events.append("fsync:dir")

    monkeypatch.setattr(cortdeco_writer, "write_durably", spy_write)
    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(cortdeco_writer, "sync_directory", spy_sync_directory)

    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    assert events == ["fsync:cortdeco.rv0.partial", "replace", "fsync:dir"]
    assert path.stat().st_size == RECORDS * TAMANHO_CORTE, "and the file is still whole"


def test_the_layout_check_runs_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cortdeco.rv0"
    observed: dict[str, object] = {}

    def spy(path: Path, numero_cortes: int, n_nodes: int) -> None:
        observed["checked"] = path.name
        observed["destination_existed"] = destination.exists()
        raise LayoutError("simulated invariant failure")

    monkeypatch.setattr(cortdeco_writer, "assert_cortdeco_layout", spy)
    with pytest.raises(LayoutError, match="simulated invariant failure"):
        write_cortdeco(_cuts(), destination, numero_cortes=CORTES, offsets=_offsets())

    assert observed["checked"] == "cortdeco.rv0.partial"
    assert observed["destination_existed"] is False
    assert not destination.exists()
    assert not list(tmp_path.iterdir()), "the partial file must be cleaned up"


def test_premise_p13_is_logged_once_per_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        write_cortdeco(_cuts(), tmp_path / "cortdeco.rv0", numero_cortes=CORTES, offsets=_offsets())
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("premise P13:")]
    assert len(warnings) == 1, "one per run, not one per record"
    assert "duplicates" in warnings[0]
    assert "theta >= 0" in warnings[0], "the rejected alternative must be named"


# --- premise P14: the zeroed but dimensioned GNL block ----------------------
#
# Zeroing is easier to test wrongly than to test. Every anchor below reads the
# **written bytes**, never the input array: an all-zeros `pi_gnl` is a
# valid-looking input, so a writer that dropped the block entirely, or that laid
# it at the wrong offset, would satisfy any assertion made on its own input.


def _project_offsets() -> CutBlockOffsets:
    return cortdeco_block_offsets(
        n_uhes=PROJECT_UHES, n_utv=0, max_lag=0, n_sbm_gnl=2, n_estagios=7, n_patamares=3
    )


def _project_cuts() -> list[list[CutInput]]:
    """6 nodes x 48 iterations at this project's real coefficient geometry.

    Synthetic *values* at real *scalars*: the claim under test is dimensional -
    that the GNL block still occupies positions 170-211 of all 289 records - and
    real cut values are already covered byte-for-byte by
    `tests/integration/test_cortdeco_assembly.py`. Assembling this from the six
    real trunk pools would duplicate the pipeline `ticket-012` owns.
    """
    offsets = _project_offsets()
    return [
        [
            CutInput(
                intercept=float(1_000_000 * (node + 1) + 1000 * iteration),
                pi_varm=np.full(PROJECT_UHES, float(node + 1)),
                pi_gnl=zeroed_gnl_block(offsets),
            )
            for iteration in range(PROJECT_PER_NODE)
        ]
        for node in range(PROJECT_NODES)
    ]


def _gnl_span(record: bytes, offsets: CutBlockOffsets) -> bytes:
    start = 4 + 8 * offsets.pi_gnl
    return record[start : start + 8 * (offsets.ncoef - offsets.pi_gnl)]


def test_the_gnl_block_is_dimensioned_by_the_formula_not_by_its_contents() -> None:
    """The mutation guard for dropping the GNL term from NCOEF.

    Removing `n_sbm_gnl*n_estagios*n_patamares` from the formula takes NCOEF from
    212 to 170 and the GNL width from 42 to 0. The file would still be 289
    records of 26,976 bytes, because the record is fixed-size and zero-padded, so
    only the declared span moves - which is exactly why the span is asserted
    here and read from the bytes below rather than inferred from the file size.
    """
    offsets = _project_offsets()
    assert offsets.ncoef == PROJECT_NCOEF
    assert offsets.pi_gnl == PROJECT_PI_GNL
    assert offsets.ncoef - offsets.pi_gnl == PROJECT_GNL_SLOTS
    assert offsets.ncoef - 1 == PROJECT_GNL_LAST
    # The term itself, isolated: 2 submarkets x 7 stages x 3 blocks.
    assert (
        PROJECT_NCOEF
        - cortdeco_ncoef(
            n_uhes=PROJECT_UHES, n_utv=0, max_lag=0, n_sbm_gnl=0, n_estagios=7, n_patamares=3
        )
        == 2 * 7 * 3
    )


def test_zeroed_gnl_block_is_the_full_declared_width() -> None:
    offsets = _project_offsets()
    block = zeroed_gnl_block(offsets)
    assert len(block) == PROJECT_GNL_SLOTS
    assert block.dtype == np.float64
    assert not block.any()


def test_the_gnl_span_is_zero_bytes_in_every_record(tmp_path: Path) -> None:
    """Read from disk, not from the input array (Requirement 6, premise P14)."""
    offsets = _offsets()
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=offsets)

    width = offsets.ncoef - offsets.pi_gnl
    assert width > 0, "a zero-width block would make every assertion below vacuous"
    for index in range(RECORDS):
        assert _gnl_span(_record(path, index), offsets) == b"\x00" * (8 * width), (
            f"record {index} carries a non-zero byte in the pi_gnl span"
        )


def test_the_gnl_span_is_true_zero_not_negative_zero(tmp_path: Path) -> None:
    """The guard for `_normalize_signed_zero`, which P4's negation makes necessary.

    `-0.0` compares equal to zero, so a numeric assertion passes while the bytes
    read `0x…80` - a pattern the reference deck holds in none of its 36 zero GNL
    slots. Both readings are asserted so the difference between them stays
    visible.
    """
    offsets = _offsets()
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=offsets)

    span = _gnl_span(_record(path, 0), offsets)
    values = np.frombuffer(span, dtype="<f8")
    assert not values.any(), "numerically zero"
    assert np.signbit(values).sum() == 0, "and positively signed, so the bytes are zero"
    assert bytes.fromhex("0000000000000080") not in span


def test_the_gnl_span_sits_at_positions_170_to_211_of_this_projects_records(
    tmp_path: Path,
) -> None:
    """289 records, 7,796,064 bytes, NCOEF 212, and the 42 slots still addressed."""
    offsets = _project_offsets()
    path = tmp_path / "cortdeco.rv0"
    cortes = PROJECT_NODES * PROJECT_PER_NODE
    assert cortes == PROJECT_CORTES

    count = write_cortdeco(_project_cuts(), path, numero_cortes=cortes, offsets=offsets)
    assert count == PROJECT_RECORDS
    assert path.stat().st_size == PROJECT_BYTES

    for index in (0, PROJECT_CORTES // 2, PROJECT_CORTES):
        record = _record(path, index)
        coefficients = np.frombuffer(record, dtype="<f8", count=PROJECT_NCOEF, offset=4)
        # pi_varm is non-zero everywhere, so the first zero coefficient locates
        # the GNL block's start without being told where it is.
        assert coefficients[PROJECT_PI_GNL - 1] != 0.0
        assert int(np.flatnonzero(coefficients == 0.0)[0]) == PROJECT_PI_GNL
        assert not coefficients[PROJECT_PI_GNL:].any()
        assert len(coefficients[PROJECT_PI_GNL:]) == PROJECT_GNL_SLOTS
        assert _gnl_span(record, offsets) == b"\x00" * (8 * PROJECT_GNL_SLOTS)


def test_a_non_zero_gnl_coefficient_is_refused_naming_the_slot(tmp_path: Path) -> None:
    """P14 is enforced, not merely declared: the manifest cannot be made to lie."""
    offsets = _offsets()
    cuts = _cuts()
    cuts[1][2].pi_gnl[3] = -142_799.91
    with pytest.raises(LayoutError) as error:
        write_cortdeco(cuts, tmp_path / "cortdeco.rv0", numero_cortes=CORTES, offsets=offsets)

    message = str(error.value)
    assert "node 1 iteration 2" in message
    assert "slot 3" in message
    assert "ticket-016" in message, "the error must point at the follow-up"
    assert not list(tmp_path.iterdir()), "nothing is written when the gate refuses"


@pytest.mark.parametrize("dirty_record", [0, 7, RECORDS - 1])
def test_a_dirty_byte_in_the_gnl_span_is_refused_naming_the_record_and_slot(
    tmp_path: Path, dirty_record: int
) -> None:
    """Parametrized over the last record on purpose.

    `RECORDS - 1` is the P13 duplicate, appended by a `serialize_cut` call
    outside the `range(numero_cortes)` loop — so it is the one record whose P14
    verification a checker iterating `range(records - 1)` would skip, and that
    truncation survived the whole suite while only record 7 was covered.
    """
    offsets = _offsets()
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=offsets)

    raw = bytearray(path.read_bytes())
    dirty_slot = 2
    raw[dirty_record * TAMANHO_CORTE + 4 + 8 * (offsets.pi_gnl + dirty_slot)] = 1
    broken = tmp_path / "broken.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(LayoutError) as error:
        assert_gnl_span_is_zero(broken, RECORDS, offsets)
    message = str(error.value)
    assert f"record {dirty_record}" in message
    assert f"slot {dirty_slot}" in message
    assert f"position {offsets.pi_gnl + dirty_slot}" in message
    assert "P14" in message


def test_the_gnl_span_check_runs_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose GNL span is dirty must never appear at the destination."""
    destination = tmp_path / "cortdeco.rv0"
    observed: dict[str, object] = {}

    def spy(path: Path, records: int, offsets: CutBlockOffsets) -> None:
        observed["checked"] = path.name
        observed["records"] = records
        observed["destination_existed"] = destination.exists()
        raise LayoutError("simulated dirty GNL span")

    monkeypatch.setattr(cortdeco_writer, "assert_gnl_span_is_zero", spy)
    with pytest.raises(LayoutError, match="simulated dirty GNL span"):
        write_cortdeco(_cuts(), destination, numero_cortes=CORTES, offsets=_offsets())

    assert observed["checked"] == "cortdeco.rv0.partial"
    assert observed["records"] == RECORDS
    assert observed["destination_existed"] is False
    assert not list(tmp_path.iterdir()), "the partial file must be cleaned up"


def test_premise_p14_is_logged_once_per_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Once per call, which is the honest maximum until epic-06 gives the package
    a single call site (the caveat ticket-008 recorded as item C8)."""
    offsets = _project_offsets()
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        write_cortdeco(
            _project_cuts(),
            tmp_path / "cortdeco.rv0",
            numero_cortes=PROJECT_CORTES,
            offsets=offsets,
        )
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("premise P14:")]
    assert len(warnings) == 1, "one per run, not one per record"

    message = warnings[0]
    # The exact affected span, so an auditor need not consult the code.
    assert f"{PROJECT_GNL_SLOTS} slots" in message
    assert f"positions {PROJECT_PI_GNL}-{PROJECT_GNL_LAST}" in message
    assert f"{PROJECT_RECORDS} records" in message
    assert f"NCOEF {PROJECT_NCOEF}" in message
    assert f"{PROJECT_BYTES}-byte" in message
    assert "ticket-016" in message
    assert "DORMANT" in message and "P4" in message and "P5" in message


def test_the_audit_log_reports_the_head_table(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`heads` has exactly one consumer, so nothing else can pin it.

    The head table is what an auditor cross-checks against `mapcut` reg 1, and it
    reaches the record only through this INFO line. Replacing the
    `cut_head_indices` call with `tuple(range(n_nodes))` survived the whole
    suite, because the P13 and P14 warnings were pinned to exact substrings while
    the one field carrying real numbers was not.
    """
    with caplog.at_level(logging.INFO, logger="conversor_fcf"):
        write_cortdeco(
            _project_cuts(),
            tmp_path / "cortdeco.rv0",
            numero_cortes=PROJECT_CORTES,
            offsets=_project_offsets(),
        )
    written = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("wrote cortdeco")
    ]
    assert len(written) == 1

    message = written[0]
    assert f"heads={list(PROJECT_HEADS)}" in message, "the descending 1-based heads (J2)"
    assert f"records={PROJECT_RECORDS}" in message
    assert f"bytes={PROJECT_BYTES}" in message
    assert f"ncoef={PROJECT_NCOEF}" in message
