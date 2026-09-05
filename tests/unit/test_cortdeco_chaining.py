"""Unit tests for the `cortdeco` chain, head table and whole-file assembly.

Per the master plan's Numeric Path Testing Policy, every anchor is transcribed
from `ticket-010`'s findings: 439 records against the reference's `numero_cortes`
of 438, the heads `(438..433)`, `chain_pointer(437, 6) == 432`, and this
project's 289 records over 7,796,064 bytes.

The two named deliberate mutations are **a 0-based chain pointer** and
**`next_index = 0` on the extra record**. Both were applied, observed red,
reverted and confirmed byte-identical; the evidence is in the plan's state entry.
Every mutation guard below calls the function under mutation (policy clause 4).

Chains are always walked through the **on-disk** pointers, never recomputed from
`chain_pointer`: a check that regenerates the arithmetic it verifies proves only
that the formula equals itself.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.decomp import cortdeco_writer
from conversor_fcf.decomp.cortdeco_writer import (
    CutInput,
    node_and_iteration,
    write_cortdeco,
)
from conversor_fcf.decomp.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_cortdeco_layout,
    assert_pointer_in_range,
    chain_pointer,
    cortdeco_block_offsets,
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

    raw = bytearray(path.read_bytes())
    # Record 11 is node 0's head; point it at itself to break the walk length.
    np.frombuffer(raw, dtype="<i4", count=1, offset=11 * TAMANHO_CORTE)[0] = 0
    broken = tmp_path / "broken.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(LayoutError) as excinfo:
        assert_cortdeco_layout(broken, CORTES, NODES)
    message = str(excinfo.value)
    assert "chain holds 1 cuts" in message
    assert "implies 4" in message


def test_a_zeroed_extra_record_pointer_is_refused(tmp_path: Path) -> None:
    """The second named mutation: a 0 there would read as a seventh chain head."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    raw = bytearray(path.read_bytes())
    np.frombuffer(raw, dtype="<i4", count=1, offset=CORTES * TAMANHO_CORTE)[0] = 0
    broken = tmp_path / "broken.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(LayoutError) as excinfo:
        assert_cortdeco_layout(broken, CORTES, NODES)
    message = str(excinfo.value)
    assert "extra record's pointer is 0" in message
    assert "continues that node's chain" in message


def test_a_pointer_leaving_the_cut_records_is_refused(tmp_path: Path) -> None:
    """A pointer into the extra record, or past it, is not a valid predecessor."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    raw = bytearray(path.read_bytes())
    # 13 is 1-based, so 0-based 12 — the extra record, outside the cut records.
    np.frombuffer(raw, dtype="<i4", count=1, offset=11 * TAMANHO_CORTE)[0] = 13
    broken = tmp_path / "broken.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(LayoutError, match="reached record 12, outside the cut records 0..11"):
        assert_cortdeco_layout(broken, CORTES, NODES)


def test_two_chains_reaching_one_record_are_refused(tmp_path: Path) -> None:
    """Merged chains keep their length, so only a visit count catches them."""
    path = tmp_path / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())

    raw = bytearray(path.read_bytes())
    # Node 1's chain is 10 -> 7 -> 4 -> 1; redirect its first hop into node 0's
    # chain (11 -> 8 -> 5 -> 2) at the same depth, so both stay 4 records long.
    np.frombuffer(raw, dtype="<i4", count=1, offset=10 * TAMANHO_CORTE)[0] = 9
    broken = tmp_path / "broken.rv0"
    broken.write_bytes(bytes(raw))

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


def test_no_cut_building_nodes_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LayoutError, match="no cut-building nodes"):
        write_cortdeco([], tmp_path / "cortdeco.rv0", numero_cortes=0, offsets=_offsets())


# --- atomicity and the premise ---------------------------------------------


def test_the_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cortdeco.rv0"
    write_cortdeco(_cuts(), path, numero_cortes=CORTES, offsets=_offsets())
    assert path.is_file()
    assert not list(path.parent.glob("*.partial"))


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
