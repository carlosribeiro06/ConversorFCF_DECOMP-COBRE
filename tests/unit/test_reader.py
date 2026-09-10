"""Unit tests for the native DECOMP readers.

Anchored on the **reference** binaries, not on this project's own output. A
reader that only ever reads what this writer produced proves nothing: the two
would agree by construction, which is the failure this module exists to avoid.
CEPEL's real files are the independent witness, and both are on disk.

Every anchor below is transcribed from `/home/carlosribeiro/git/mapcut.rv0`
(356 records, 17,095,120 bytes) and `cortdeco.rv0` (439 records, 11,842,464
bytes), and cross-checks a fact `CLAUDE.md` already records.

The named deliberate mutation for the reader is **importing the writer's field
list instead of deriving positions from the bytes**, which must break a test that
reads the reference file.
"""

import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.decomp.layout import RECORD_SIZE, TAMANHO_CORTE, cortdeco_block_offsets
from conversor_fcf.decomp.reader import ReadError, read_cortdeco, read_mapcut

REFERENCE_MAPCUT = Path("/home/carlosribeiro/git/mapcut.rv0")
REFERENCE_CORTDECO = Path("/home/carlosribeiro/git/cortdeco.rv0")

# The reference deck's own numbers.
MAPCUT_RECORDS = 356
MAPCUT_BYTES = 17_095_120
CORTDECO_RECORDS = 439
CORTDECO_BYTES = 11_842_464
REFERENCE_NCOEF = 218
REFERENCE_PI_GNL = 176
REFERENCE_SCALARS = (73, 438, 5, 169, 273)
REFERENCE_HEADS = (438, 437, 436, 435, 434, 433)
REFERENCE_START = (25, 4, 2026)
REFERENCE_STAGE_SCALARS = (7, 6, 2, 3)
REFERENCE_NODES = 6

pytestmark = pytest.mark.skipif(
    not REFERENCE_MAPCUT.is_file() or not REFERENCE_CORTDECO.is_file(),
    reason="reference DECOMP binaries not present",
)


@pytest.fixture(autouse=True)
def propagating_package_logger() -> Iterator[None]:
    logger = logging.getLogger("conversor_fcf")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


@pytest.fixture(scope="module")
def reference_mapcut():  # type: ignore[no-untyped-def]
    return read_mapcut(REFERENCE_MAPCUT)


@pytest.fixture(scope="module")
def reference_cortdeco():  # type: ignore[no-untyped-def]
    offsets = cortdeco_block_offsets(
        n_uhes=169, n_utv=2, max_lag=3, n_sbm_gnl=2, n_estagios=7, n_patamares=3
    )
    assert offsets.ncoef == REFERENCE_NCOEF
    assert offsets.pi_gnl == REFERENCE_PI_GNL
    return read_cortdeco(REFERENCE_CORTDECO, offsets, REFERENCE_NODES)


# --- mapcut ----------------------------------------------------------------


def test_the_reference_record_count_is_recovered_from_the_header_alone(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """356 records, and the count is cross-checked against the header's own scalars."""
    assert reference_mapcut.record_count == MAPCUT_RECORDS
    assert MAPCUT_RECORDS * RECORD_SIZE == MAPCUT_BYTES


def test_the_reference_reg_one_scalars(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    scalars = reference_mapcut.scalars
    assert (
        scalars.numero_iteracoes,
        scalars.numero_cortes,
        scalars.numero_submercados,
        scalars.numero_uhes,
        scalars.numero_cenarios,
    ) == REFERENCE_SCALARS
    # numero_cortes = numero_iteracoes * n_cut_building_nodes.
    assert scalars.numero_cortes == scalars.numero_iteracoes * REFERENCE_NODES


def test_the_reference_cut_heads_descend_from_numero_cortes(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    heads = reference_mapcut.cut_heads
    assert heads[:REFERENCE_NODES] == REFERENCE_HEADS
    assert set(heads[REFERENCE_NODES:]) == {0}
    assert len(heads) == REFERENCE_SCALARS[4]


def test_the_reference_study_start_date(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """2026-04-25, and tamanho_corte is the format's fixed 26,976."""
    scalars = reference_mapcut.scalars
    assert (scalars.dia, scalars.mes, scalars.ano) == REFERENCE_START
    assert scalars.tamanho_corte == TAMANHO_CORTE


def test_the_reference_reg_six_scalars_and_arrays(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    scalars = reference_mapcut.scalars
    assert (
        scalars.numero_estagios,
        scalars.numero_semanas,
        scalars.n_utv,
        scalars.max_lag,
    ) == REFERENCE_STAGE_SCALARS
    assert reference_mapcut.indice_primeiro_no_estagio == (1, 2, 3, 4, 5, 6, 7)
    assert reference_mapcut.patamares_por_estagio == (3, 3, 3, 3, 3, 3, 3)
    # One lag per (plant, stage) = 2 x 7, uniformly 3 in the reference. Not the
    # same quantity as the 2 x 7 x 4 = 56 travel-time RECORDS.
    assert reference_mapcut.travel_time_lags == (3,) * 14
    assert scalars.n_utv * scalars.numero_estagios * (scalars.max_lag + 1) == 56


def test_the_reference_tree_has_a_self_parent_at_the_root(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """Finding F7's shape, read off the reference rather than recomputed."""
    tree = reference_mapcut.indice_no_arvore
    assert tree[0] == 1, "the root is its own parent, 1-based"
    assert len(tree) == REFERENCE_SCALARS[4]


def test_the_downstream_codes_are_int32_on_disk(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """The quirk `CLAUDE.md` records: idecomp reads reg 4 as float32, the disk holds int32.

    Read as int32 the values are a clean plant-code sequence; the float32 reading
    is what needs `.view(np.int32)` to recover.
    """
    downstream = reference_mapcut.codigos_uhes_jusante
    assert len(downstream) == 169
    assert downstream[:4] == (2, 3, 4, 5)
    assert all(isinstance(code, int) for code in downstream)


def test_the_physical_span_is_populated_in_the_reference(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """Premise P1 is a real divergence, not a convention: all 14 records carry data.

    This is the check that makes P1 evidenced rather than declared.
    """
    assert reference_mapcut.physical_span_nonzero_records == tuple(range(4, 18))


def test_the_reference_gnl_configuration(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """`codigos_submercados_gnl` is int32 on disk despite idecomp surfacing [1.0, 3.0]."""
    assert reference_mapcut.codigos_submercados_gnl == (1, 3)
    assert reference_mapcut.lag_meses_gnl == (2, 2)
    assert reference_mapcut.patamares_gnl == (3, 3)


def test_the_reg_nine_trailing_block_holds_one_average_month(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """`ngnl * npat` = 6 values, three distinct summing to 730.5 = 365.25*24/12.

    Premise P12 declares this block's **axis** unsettled, and this test records
    what the bytes hold without settling it: the three distinct values repeat per
    submarket, so the layout is block-major with the submarket varying fastest.
    That observation must not be acted on - laying a stage-indexed vector into a
    block-indexed slot produces a wrong file that still reads cleanly.
    """
    block = reference_mapcut.gnl_trailing_block
    assert len(block) == 6, "ngnl * npat, not ngnl * n_estagios"
    distinct = sorted({round(value, 5) for value in block})
    assert distinct == [67.71735, 280.8042, 381.97845]
    assert sum(distinct) == pytest.approx(730.5, abs=1e-9)
    assert sum(distinct) == pytest.approx(365.25 * 24 / 12, abs=1e-9)
    assert block[0] == block[1], "block-major: each value repeats across submarkets"


def test_reg_nine_repeats_per_node_and_reg_ten_per_stage(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """273 reg 9 against 7 reg 10, which is why the CSV emits reg 9 once."""
    assert len(reference_mapcut.gnl_record_indices) == 273
    assert len(reference_mapcut.cost_records) == 7
    # The interleave: seven 9/10 pairs from record 76, then reg 9 alone.
    assert reference_mapcut.gnl_record_indices[:3] == (76, 78, 80)
    assert reference_mapcut.gnl_record_indices[7] == 90


def test_the_reference_discount_series(reference_mapcut) -> None:  # type: ignore[no-untyped-def]
    """`taxa_desconto` is reg 10's first field, and its ratio is the 365.25 basis."""
    factors = tuple(record[0] for record in reference_mapcut.cost_records)
    assert factors[0] == 1.0
    assert factors[1] == pytest.approx(0.997830417741, abs=1e-11)
    assert all(later < earlier for earlier, later in itertools.pairwise(factors))


def test_the_reg_nine_identity_is_verified_not_assumed(tmp_path: Path) -> None:
    """The CSV emits one row group with a repeat count; that is only honest if checked."""
    raw = bytearray(REFERENCE_MAPCUT.read_bytes())
    # Record 78 is the second reg 9. Dirty one byte of its GNL configuration.
    raw[78 * RECORD_SIZE + 4] ^= 0xFF
    broken = tmp_path / "broken.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(ReadError, match="reg 9 record 78 differs from record 76"):
        read_mapcut(broken)


def test_a_truncated_mapcut_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "short.rv0"
    broken.write_bytes(REFERENCE_MAPCUT.read_bytes()[: MAPCUT_BYTES - 10])
    with pytest.raises(ReadError, match="trailing bytes"):
        read_mapcut(broken)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.rv0"
    empty.write_bytes(b"")
    with pytest.raises(ReadError, match="not a whole number"):
        read_mapcut(empty)


def test_a_header_disagreeing_with_the_record_count_is_refused(tmp_path: Path) -> None:
    """The check that would catch a writer emitting the wrong number of records."""
    raw = bytearray(REFERENCE_MAPCUT.read_bytes())
    # Claim one more stage than the file was built for.
    np.frombuffer(raw, dtype="<i4", count=1, offset=19 * RECORD_SIZE + 4)[0] = 8
    broken = tmp_path / "wrong_count.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(ReadError, match="its own header implies"):
        read_mapcut(broken)


def test_a_wrong_tamanho_corte_is_refused(tmp_path: Path) -> None:
    raw = bytearray(REFERENCE_MAPCUT.read_bytes())
    np.frombuffer(raw, dtype="<i4", count=1, offset=RECORD_SIZE)[0] = 12345
    broken = tmp_path / "wrong_size.rv0"
    broken.write_bytes(bytes(raw))

    with pytest.raises(ReadError, match="the format fixes it at"):
        read_mapcut(broken)


def _corrupt_int(source: Path, destination: Path, record: int, index: int, value: int) -> Path:
    """Overwrite one int32 of one record, so a defensive branch can be reached."""
    raw = bytearray(source.read_bytes())
    np.frombuffer(raw, dtype="<i4", count=1, offset=record * RECORD_SIZE + 4 * index)[0] = value
    destination.write_bytes(bytes(raw))
    return destination


@pytest.mark.parametrize(
    ("field_index", "field_name"),
    [(2, "numero_submercados"), (3, "numero_uhes"), (4, "numero_cenarios")],
)
def test_a_non_positive_reg_one_scalar_is_refused(
    tmp_path: Path, field_index: int, field_name: str
) -> None:
    """A zero count cannot be decoded, and saying which one is the whole value of the check."""
    broken = _corrupt_int(
        REFERENCE_MAPCUT, tmp_path / "zero.rv0", record=0, index=field_index, value=0
    )
    with pytest.raises(ReadError, match=f"{field_name} = 0"):
        read_mapcut(broken)


def test_a_non_positive_stage_count_is_refused(tmp_path: Path) -> None:
    broken = _corrupt_int(REFERENCE_MAPCUT, tmp_path / "no_stages.rv0", record=19, index=1, value=0)
    with pytest.raises(ReadError, match="numero_estagios = 0"):
        read_mapcut(broken)


def test_more_stages_than_nodes_is_refused(tmp_path: Path) -> None:
    """The reg 9 / reg 10 interleave is undefined then, so decoding must stop rather than guess."""
    broken = _corrupt_int(
        REFERENCE_MAPCUT, tmp_path / "too_many_stages.rv0", record=19, index=1, value=9999
    )
    with pytest.raises(ReadError, match="the reg 9 / reg 10 interleave is undefined"):
        read_mapcut(broken)


@pytest.mark.parametrize("ngnl", [0, 99])
def test_an_implausible_gnl_submarket_count_is_refused(tmp_path: Path, ngnl: int) -> None:
    """Checked against reg 1's own submarket count, not against a constant."""
    broken = _corrupt_int(
        REFERENCE_MAPCUT, tmp_path / f"ngnl_{ngnl}.rv0", record=76, index=0, value=ngnl
    )
    with pytest.raises(ReadError, match=f"reg 9 declares ngnl = {ngnl}"):
        read_mapcut(broken)


def test_a_reg9_header_block_beyond_the_record_capacity_is_refused(tmp_path: Path) -> None:
    """Two single-int32 corruptions, both individually plausible, that together
    overflow reg 9's int32 header block instead of the 48020-byte record.

    `ngnl <= n_submercados` is checked, but `n_submercados` itself is only
    checked `> 0` and never enters `mapcut_record_count`, so a corrupted reg 1
    can raise that bound far enough for a correspondingly large `ngnl` to pass
    it and still overflow the record - reproducing numpy's unnamed "buffer is
    smaller than requested size" this check replaces.
    """
    widened_submercados = _corrupt_int(
        REFERENCE_MAPCUT, tmp_path / "wide_submercados.rv0", record=0, index=2, value=6000
    )
    broken = _corrupt_int(
        widened_submercados, tmp_path / "ngnl_head_overflow.rv0", record=76, index=0, value=4500
    )
    with pytest.raises(ReadError, match=r"header block \(54004 bytes\) does not fit"):
        read_mapcut(broken)


def test_a_reg9_trailing_block_beyond_the_record_capacity_is_refused(tmp_path: Path) -> None:
    """One corrupted `patamares_por_estagio` entry inflates `n_patamares` (the
    widest stage) enough to overflow reg 9's trailing float64 block, the same
    unnamed-ValueError failure mode as the header-block case above.
    """
    broken = _corrupt_int(
        REFERENCE_MAPCUT,
        tmp_path / "huge_n_patamares.rv0",
        record=19,
        index=12,
        value=10_000_000,
    )
    with pytest.raises(ReadError, match="trailing block for ngnl=2 and n_patamares=10000000"):
        read_mapcut(broken)


@pytest.mark.parametrize(
    ("numero_uhes", "match"),
    [
        (10**9, r"numero_uhes = 1000000000"),
        # A value comfortably under the reference file's own total size, so
        # nothing else refuses it: this is the case that reads past record 2
        # into whatever follows with no error at all, and
        # write_mapcut_content_csv would happily emit 20 000 rows from it.
        (20_000, r"numero_uhes = 20000"),
    ],
)
def test_an_oversized_numero_uhes_is_refused_by_name(
    tmp_path: Path, numero_uhes: int, match: str
) -> None:
    """Unlike numero_submercados and numero_cenarios above, numero_uhes was
    never bounded against a record's own capacity: it never enters
    mapcut_record_count, so nothing else catches a corrupted value. A huge
    value already fails inside numpy with its unnamed "buffer is smaller than
    requested size"; a moderate one succeeds silently today. Both must now be
    refused by name.
    """
    broken = _corrupt_int(
        REFERENCE_MAPCUT,
        tmp_path / f"uhes_{numero_uhes}.rv0",
        record=0,
        index=3,
        value=numero_uhes,
    )
    with pytest.raises(ReadError, match=match):
        read_mapcut(broken)


def test_an_oversized_numero_cenarios_is_refused_by_name(tmp_path: Path) -> None:
    """numero_cenarios sizes reg 1's own cut_heads array (offset 20). A value
    this large would also fail the whole-file record count check further
    down, but with a far less specific message; this bound is checked first.
    """
    broken = _corrupt_int(
        REFERENCE_MAPCUT, tmp_path / "wide_cenarios.rv0", record=0, index=4, value=12_001
    )
    with pytest.raises(
        ReadError, match=r"numero_cenarios = 12001, whose 48024-byte cut_heads span"
    ):
        read_mapcut(broken)


def test_oversized_reg_six_arrays_are_refused_by_name(tmp_path: Path) -> None:
    """numero_estagios/n_utv size three reg 6 arrays past its five scalars.

    numero_cenarios is widened too, in the same corrupted file - otherwise the
    "reg 9 / reg 10 interleave is undefined" check above would fire first and
    mask this one, the same two-corruption technique the reg 9 header/trailing
    tests above use.
    """
    widened_cenarios = _corrupt_int(
        REFERENCE_MAPCUT, tmp_path / "wide_cenarios_for_stage.rv0", record=0, index=4, value=5000
    )
    broken = _corrupt_int(
        widened_cenarios, tmp_path / "wide_stage_arrays.rv0", record=19, index=1, value=4000
    )
    with pytest.raises(
        ReadError,
        match=r"numero_estagios = 4000 and n_utv = 2, whose 64020-byte combined scalar/array span",
    ):
        read_mapcut(broken)


# --- cortdeco --------------------------------------------------------------


def test_the_reference_cortdeco_shape(reference_cortdeco) -> None:  # type: ignore[no-untyped-def]
    """439 records, so numero_cortes is 438 - the count is recovered, not supplied."""
    assert reference_cortdeco.record_count == CORTDECO_RECORDS
    assert reference_cortdeco.numero_cortes == 438
    assert CORTDECO_RECORDS * TAMANHO_CORTE == CORTDECO_BYTES
    assert len(reference_cortdeco.cuts) == CORTDECO_RECORDS


def test_exactly_one_record_is_flagged_extra(reference_cortdeco) -> None:  # type: ignore[no-untyped-def]
    """Premise P13's record, which must be visible rather than look like a 439th cut."""
    extra = [cut for cut in reference_cortdeco.cuts if cut.is_extra_record]
    assert len(extra) == 1
    assert extra[0].record_index == 438


def test_the_reference_chain_pointers_read_as_the_format_declares(reference_cortdeco) -> None:  # type: ignore[no-untyped-def]
    """1-based, stride 6, 0 terminating - six terminators and no other value."""
    pointers = [cut.next_index for cut in reference_cortdeco.cuts]
    terminators = [index for index, pointer in enumerate(pointers) if pointer == 0]
    assert terminators == [0, 1, 2, 3, 4, 5]
    assert pointers[437] == 432
    assert pointers[438] == 433, "the extra record continues the last node's chain"
    strides = {index - (pointers[index] - 1) for index in range(6, 438)}
    assert strides == {REFERENCE_NODES}


def test_the_reference_gnl_block_populates_only_six_of_forty_two(reference_cortdeco) -> None:  # type: ignore[no-untyped-def]
    """Positions 176-178 and 197-199 relative to NCOEF - the fact P14 rests on.

    36 slots are structurally zero and none of them carries a negative zero,
    which is what made `_normalize_signed_zero` necessary.
    """
    cut = reference_cortdeco.cuts[0]
    assert len(cut.pi_gnl) == REFERENCE_NCOEF - REFERENCE_PI_GNL == 42
    populated = [index for index, value in enumerate(cut.pi_gnl) if value != 0.0]
    assert populated == [0, 1, 2, 21, 22, 23]
    assert [index + REFERENCE_PI_GNL for index in populated] == [176, 177, 178, 197, 198, 199]

    zeros = [value for value in cut.pi_gnl if value == 0.0]
    assert len(zeros) == 36
    assert not any(np.signbit(value) for value in zeros), "no negative zero in a real deck"


def test_the_reference_rhs_envelope(reference_cortdeco) -> None:  # type: ignore[no-untyped-def]
    """The deck's own rhs range, which the integration envelope is transcribed from."""
    values = [cut.rhs for cut in reference_cortdeco.cuts]
    assert min(values) == pytest.approx(-421176891.48, abs=0.5)
    assert max(values) == pytest.approx(8321718752.48, abs=0.5)


def test_a_non_divisible_cortdeco_is_refused(tmp_path: Path) -> None:
    offsets = cortdeco_block_offsets(
        n_uhes=169, n_utv=2, max_lag=3, n_sbm_gnl=2, n_estagios=7, n_patamares=3
    )
    truncated = tmp_path / "short.rv0"
    truncated.write_bytes(REFERENCE_CORTDECO.read_bytes()[: 100 * TAMANHO_CORTE])
    with pytest.raises(ReadError, match="not\n?.*divisible"):
        read_cortdeco(truncated, offsets, REFERENCE_NODES)


def test_a_non_positive_node_count_is_refused() -> None:
    offsets = cortdeco_block_offsets(
        n_uhes=169, n_utv=2, max_lag=3, n_sbm_gnl=2, n_estagios=7, n_patamares=3
    )
    with pytest.raises(ReadError, match="n_nodes must be positive"):
        read_cortdeco(REFERENCE_CORTDECO, offsets, 0)


def test_an_ncoef_beyond_the_record_capacity_is_refused() -> None:
    offsets = cortdeco_block_offsets(
        n_uhes=4000, n_utv=0, max_lag=0, n_sbm_gnl=1, n_estagios=1, n_patamares=1
    )
    with pytest.raises(ReadError, match="does not fit"):
        read_cortdeco(REFERENCE_CORTDECO, offsets, REFERENCE_NODES)


def test_the_reader_imports_nothing_from_either_writer() -> None:
    """The design property the whole module rests on, asserted rather than trusted.

    A reader that reused a writer's field list would agree with it by
    construction and could not witness its field order.
    """
    source = Path(__file__).resolve().parents[2] / "src/conversor_fcf/decomp/reader.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("mapcut_writer", "cortdeco_writer"):
        assert f"import {forbidden}" not in text
        assert f"from conversor_fcf.decomp.{forbidden}" not in text
