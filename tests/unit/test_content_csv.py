"""Unit tests for the content CSV emitters.

Driven by the **native reader** reading the **reference** binaries, which is the
whole design: the CSV is a witness only if the values in it came off a disk
rather than out of the array they were written from. That is the failure epic-02
was bitten by, and it is why these tests do not build a `MapcutContents` by hand
where they can read a real one instead.

The named deliberate mutation for the CSVs is **reg 9 emitted 273 times instead
of once**, which must break the row-count anchor.
"""

import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from conversor_fcf.decomp.layout import cortdeco_block_offsets
from conversor_fcf.decomp.reader import (
    CortdecoContents,
    CortdecoCut,
    MapcutContents,
    read_cortdeco,
    read_mapcut,
)
from conversor_fcf.reporting.content_csv import (
    COST_FIELDS,
    MAPCUT_COLUMNS,
    content_csv_path,
    cortdeco_columns,
    write_cortdeco_content_csv,
    write_mapcut_content_csv,
)

REFERENCE_MAPCUT = Path("/home/carlosribeiro/git/mapcut.rv0")
REFERENCE_CORTDECO = Path("/home/carlosribeiro/git/cortdeco.rv0")

REFERENCE_NODES = 6
REFERENCE_NCOEF = 218
REG_NINE_REPEATS = 273
REG_TEN_RECORDS = 7

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
def mapcut() -> MapcutContents:
    return read_mapcut(REFERENCE_MAPCUT)


@pytest.fixture(scope="module")
def cortdeco(mapcut: MapcutContents) -> CortdecoContents:
    scalars = mapcut.scalars
    offsets = cortdeco_block_offsets(
        n_uhes=scalars.numero_uhes,
        n_utv=scalars.n_utv,
        max_lag=scalars.max_lag,
        n_sbm_gnl=len(mapcut.codigos_submercados_gnl),
        n_estagios=scalars.numero_estagios,
        n_patamares=max(mapcut.patamares_por_estagio),
    )
    assert offsets.ncoef == REFERENCE_NCOEF
    return read_cortdeco(REFERENCE_CORTDECO, offsets, REFERENCE_NODES)


def _rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split(",") for line in lines]


def _write_reference_cortdeco(
    mapcut: MapcutContents, cortdeco: CortdecoContents, path: Path, n_patamares: int = 3
) -> int:
    """Mirror the reference `cortdeco` with the deck's own dimensions."""
    scalars = mapcut.scalars
    return write_cortdeco_content_csv(
        cortdeco,
        path,
        mapcut.codigos_uhes,
        mapcut.codigos_submercados_gnl,
        scalars.numero_estagios,
        n_patamares,
        scalars.n_utv,
        scalars.max_lag,
    )


# --- the mapcut mirror ------------------------------------------------------


def test_the_header_is_the_declared_long_format(mapcut: MapcutContents, tmp_path: Path) -> None:
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    assert _rows(path)[0] == list(MAPCUT_COLUMNS)


def test_the_row_count_is_returned_and_matches_the_file(
    mapcut: MapcutContents, tmp_path: Path
) -> None:
    path = tmp_path / "mapcut_content.csv"
    returned = write_mapcut_content_csv(mapcut, path)
    assert returned == len(_rows(path)) - 1, "the header is not a row"


def test_reg_nine_is_emitted_once_with_its_repeat_count(
    mapcut: MapcutContents, tmp_path: Path
) -> None:
    """The named mutation's anchor: 273 identical row groups would bury the 7 reg 10.

    `read_mapcut` has already verified the 273 records are byte-identical, so the
    compression is honest rather than assumed - and this test pins the count that
    makes it visible.
    """
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    rows = _rows(path)[1:]

    reg9 = [row for row in rows if row[1] == "reg9_gnl"]
    assert len({row[0] for row in reg9}) == 1, "one record index, not 273"
    assert {row[2] for row in reg9} == {str(REG_NINE_REPEATS)}
    # ngnl + 3 int arrays of 2 + the 6-value trailing block.
    assert len(reg9) == 1 + 3 * 2 + 6

    reg10 = [row for row in rows if row[1] == "reg10_cost"]
    assert len({row[0] for row in reg10}) == REG_TEN_RECORDS, "every reg 10 differs, so all appear"
    assert len(reg10) == REG_TEN_RECORDS * len(COST_FIELDS)


def test_premise_p1_is_visible_as_rows_not_as_an_absence(
    mapcut: MapcutContents, tmp_path: Path
) -> None:
    """All 14 span records appear, and the reference's are flagged non-zero.

    That flag is what makes P1 an evidenced divergence: in this project's own
    output every row reads 0, and against the reference every row reads 1.
    """
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    span = [row for row in _rows(path)[1:] if row[1] == "physical_p1"]

    assert [int(row[0]) for row in span] == list(range(4, 18))
    assert all(row[3] == "nonzero_bytes_found" for row in span)
    assert {row[5] for row in span} == {"1"}, "the reference populates all 14"


def test_the_reference_scalars_reach_the_csv_unaltered(
    mapcut: MapcutContents, tmp_path: Path
) -> None:
    """Spot-checked against the deck's own numbers rather than against the reader."""
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    rows = _rows(path)[1:]
    values = {(row[1], row[3], row[4]): row[5] for row in rows}

    # `position` is the field's index within its record, so the reg-1 scalars run
    # 0..4 in declaration order rather than all sitting at 0.
    assert values[("reg1_general", "numero_iteracoes", "0")] == "73"
    assert values[("reg1_general", "numero_cortes", "1")] == "438"
    assert values[("reg1_general", "numero_submercados", "2")] == "5"
    assert values[("reg1_general", "numero_uhes", "3")] == "169"
    assert values[("reg1_general", "numero_cenarios", "4")] == "273"
    assert values[("reg1_general", "registro_ultimo_corte_no", "0")] == "438"
    assert values[("reg1_general", "registro_ultimo_corte_no", "5")] == "433"
    assert values[("reg1_general", "registro_ultimo_corte_no", "6")] == "0"
    assert values[("reg2_case", "tamanho_corte", "0")] == "26976"
    assert values[("reg2_case", "dia", "1")] == "25"
    assert values[("reg2_case", "mes", "2")] == "4"
    assert values[("reg2_case", "ano", "3")] == "2026"
    assert values[("reg6_stage", "numero_estagios", "1")] == "7"
    assert values[("reg6_stage", "n_utv", "3")] == "2"
    assert values[("reg6_stage", "max_lag", "4")] == "3"
    assert values[("reg9_gnl", "codigos_submercados_gnl", "0")] == "1"
    assert values[("reg9_gnl", "codigos_submercados_gnl", "1")] == "3"


def test_the_discount_series_round_trips_exactly(mapcut: MapcutContents, tmp_path: Path) -> None:
    """`repr(float(...))` is load-bearing: six significant digits would lose the ratio."""
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    rows = _rows(path)[1:]
    written = [row[5] for row in rows if row[3] == "taxa_desconto"]
    recovered = [float(text) for text in written]

    expected = [record[0] for record in mapcut.cost_records]
    assert recovered == expected, "exact, not approximate"
    assert recovered[1] == pytest.approx(0.997830417741, abs=1e-11)
    assert "np.float64" not in path.read_text(encoding="utf-8")


def test_the_gnl_trailing_block_round_trips(mapcut: MapcutContents, tmp_path: Path) -> None:
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    rows = _rows(path)[1:]
    block = [float(row[5]) for row in rows if row[3] == "gnl_trailing_block"]
    assert block == list(mapcut.gnl_trailing_block)
    assert sum({round(value, 5) for value in block}) == pytest.approx(730.5, abs=1e-9)


# --- the cortdeco mirror ----------------------------------------------------


def test_the_column_names_mirror_idecomp_and_count_to_ncoef(
    mapcut: MapcutContents, cortdeco: CortdecoContents, tmp_path: Path
) -> None:
    """`NCOEF + 3`: the pointer, the extra-record flag and the record index."""
    scalars = mapcut.scalars
    path = tmp_path / "cortdeco_content.csv"
    rows = _write_reference_cortdeco(mapcut, cortdeco, path, max(mapcut.patamares_por_estagio))
    assert rows == cortdeco.record_count == 439

    header = _rows(path)[0]
    assert len(header) == REFERENCE_NCOEF + 3
    assert header[:4] == ["record_index", "next_index", "is_extra_record", "rhs"]
    assert header[4] == "pi_varm_uhe1", "named by DECOMP plant code, not by position"
    assert header[4 + scalars.numero_uhes] == "pi_qdefp_utv1_lag1"
    assert header[-1] == "pi_gnl_sbm3_pat3_lag7"


def test_the_travel_time_block_sits_between_varm_and_gnl(mapcut: MapcutContents) -> None:
    """The design error the reference caught: pi_varm is 175 wide here, not 169.

    This project holds `n_utv = 0` under premise P3, so its own artifacts carry no
    `pi_qdefp` columns at all - which is exactly why the span had to be tested
    against a foreign deck rather than against this converter's output.
    """
    scalars = mapcut.scalars
    with_travel = cortdeco_columns(
        mapcut.codigos_uhes,
        mapcut.codigos_submercados_gnl,
        scalars.numero_estagios,
        3,
        scalars.n_utv,
        scalars.max_lag,
    )
    without = cortdeco_columns(
        mapcut.codigos_uhes, mapcut.codigos_submercados_gnl, scalars.numero_estagios, 3
    )
    assert len(with_travel) - len(without) == scalars.n_utv * scalars.max_lag == 6
    assert not any(name.startswith("pi_qdefp") for name in without)


def test_exactly_one_row_is_flagged_as_the_extra_record(
    mapcut: MapcutContents, cortdeco: CortdecoContents, tmp_path: Path
) -> None:
    """Premise P13's record must not read as a 439th cut."""
    path = tmp_path / "cortdeco_content.csv"
    _write_reference_cortdeco(mapcut, cortdeco, path)
    rows = _rows(path)[1:]
    flagged = [row for row in rows if row[2] == "true"]
    assert len(flagged) == 1
    assert flagged[0][0] == "438"
    assert flagged[0][1] == "433", "and it continues the last node's chain"


def test_the_reference_gnl_columns_show_six_populated_of_forty_two(
    mapcut: MapcutContents, cortdeco: CortdecoContents, tmp_path: Path
) -> None:
    """Against the reference, so the CSV is checked where the block is populated.

    This project's own output would put zeros in all 42 under premise P14, which
    cannot distinguish a correctly-addressed block from a dropped one.
    """
    path = tmp_path / "cortdeco_content.csv"
    _write_reference_cortdeco(mapcut, cortdeco, path)
    header, first = _rows(path)[:2]
    named = dict(zip(header, first, strict=True))
    populated = [name for name in header if name.startswith("pi_gnl") and float(named[name]) != 0.0]
    assert populated == [
        "pi_gnl_sbm1_pat1_lag1",
        "pi_gnl_sbm1_pat2_lag1",
        "pi_gnl_sbm1_pat3_lag1",
        "pi_gnl_sbm3_pat1_lag1",
        "pi_gnl_sbm3_pat2_lag1",
        "pi_gnl_sbm3_pat3_lag1",
    ]


def test_the_rhs_round_trips_exactly(
    mapcut: MapcutContents, cortdeco: CortdecoContents, tmp_path: Path
) -> None:
    path = tmp_path / "cortdeco_content.csv"
    _write_reference_cortdeco(mapcut, cortdeco, path)
    rows = _rows(path)[1:]
    recovered = [float(row[3]) for row in rows]
    assert recovered == [cut.rhs for cut in cortdeco.cuts]
    assert "np.float64" not in path.read_text(encoding="utf-8")


def test_a_column_count_disagreeing_with_the_record_is_refused(
    mapcut: MapcutContents, cortdeco: CortdecoContents, tmp_path: Path
) -> None:
    """The check that catches a caller describing the wrong deck.

    Omitting `n_utv`/`max_lag` for a deck that has them is the exact mistake this
    module made before the reference caught it.
    """
    with pytest.raises(ValueError, match="column names for a record carrying"):
        write_cortdeco_content_csv(
            cortdeco,
            tmp_path / "wrong.csv",
            mapcut.codigos_uhes,
            mapcut.codigos_submercados_gnl,
            mapcut.scalars.numero_estagios,
            3,
        )


def test_a_zero_utv_deck_needs_no_travel_columns(tmp_path: Path) -> None:
    """This project's own shape: NCOEF 212, pi_varm 169, no pi_qdefp."""
    offsets = cortdeco_block_offsets(
        n_uhes=3, n_utv=0, max_lag=0, n_sbm_gnl=1, n_estagios=2, n_patamares=2
    )
    contents = CortdecoContents(
        record_count=2,
        numero_cortes=1,
        cuts=(
            CortdecoCut(
                record_index=0,
                next_index=0,
                rhs=1.5,
                pi_varm=(1.0, 2.0, 3.0),
                pi_gnl=(0.0, 0.0, 0.0, 0.0),
                is_extra_record=False,
            ),
            CortdecoCut(
                record_index=1,
                next_index=1,
                rhs=1.5,
                pi_varm=(1.0, 2.0, 3.0),
                pi_gnl=(0.0, 0.0, 0.0, 0.0),
                is_extra_record=True,
            ),
        ),
    )
    path = tmp_path / "small.csv"
    rows = write_cortdeco_content_csv(contents, path, (101, 102, 103), (1,), 2, 2)
    assert rows == 2
    header = _rows(path)[0]
    assert len(header) == offsets.ncoef + 3
    assert header[4:7] == ["pi_varm_uhe101", "pi_varm_uhe102", "pi_varm_uhe103"]
    assert not any(name.startswith("pi_qdefp") for name in header)


def test_the_path_helper_is_the_single_source_for_the_location(tmp_path: Path) -> None:
    assert content_csv_path(tmp_path, "content", "mapcut_content.csv") == (
        tmp_path / "content" / "mapcut_content.csv"
    )


def test_neither_csv_carries_windows_line_endings(
    mapcut: MapcutContents, cortdeco: CortdecoContents, tmp_path: Path
) -> None:
    """An official artifact read on a Linux server must not carry CRLF."""
    mapcut_path = tmp_path / "mapcut_content.csv"
    cortdeco_path = tmp_path / "cortdeco_content.csv"
    write_mapcut_content_csv(mapcut, mapcut_path)
    _write_reference_cortdeco(mapcut, cortdeco, cortdeco_path)
    for path in (mapcut_path, cortdeco_path):
        assert b"\r\n" not in path.read_bytes(), path.name


def test_the_mapcut_rows_are_grouped_by_ascending_record_index(
    mapcut: MapcutContents, tmp_path: Path
) -> None:
    """A reader filtering by record must not have to sort first."""
    path = tmp_path / "mapcut_content.csv"
    write_mapcut_content_csv(mapcut, path)
    indices = [int(row[0]) for row in _rows(path)[1:]]
    groups = [index for index, _ in itertools.groupby(indices)]
    assert groups == sorted(groups), "each record's rows are contiguous and ascending"
