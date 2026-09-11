"""Acceptance criteria for `ticket-013`: the `cortdeco` oracle leg.

`test_mapcut_assembly.py` already reads the emitted `mapcut` through
`idecomp.decomp.mapcut.Mapcut`, a code path built from nothing this project's
own writer or native reader shares. `cortdeco` has never been read by
anything but this project's own `decomp/reader.py`, which is a real
independence argument on its own (it imports nothing from either writer) but
is still this project's own understanding of the format. This module closes
that gap: `idecomp.decomp.cortdeco.Cortdeco` is the third opinion, and every
`Cortdeco.read` argument here is derived from the `Mapcut` object read back
from the same file, never from `decomp/reader.py`, so the oracle leg is
self-contained (Requirement 2).

Two `idecomp` traps are worked around rather than hit:

- `numero_total_cortes` must be cuts per node (48), not the study total
  (288); the total produces 1,729 rows, 1,440 of them all-zero `rhs`
  padding. Witnessed directly below rather than only avoided.
- `codigos_uhes_tempo_viagem` must be `[]`. Deriving it from
  `Mapcut.lag_tempo_viagem_por_uhe` would hit that property's own `[-0:]`
  slice bug (recorded in `CLAUDE.md`) and build 19 phantom `pi_qdefp`
  columns, mislabelling every GNL column after them.

Premise P14 holds throughout: `pi_gnl` is emitted as zeros, so nothing here
gates on a GNL **value** or on GNL magnitude agreement with the reference
deck (`CLAUDE.md` forbids the latter outright - the two artifacts come from
independent runs and span different orders of magnitude).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from idecomp.decomp.cortdeco import Cortdeco
from idecomp.decomp.mapcut import Mapcut

from conversor_fcf.decomp.layout import (
    CutBlockOffsets,
    assert_gnl_span_is_zero,
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
from conversor_fcf.reporting.content_csv import cortdeco_columns

EXPECTED_RECORDS = 289
EXPECTED_COLUMNS = 215
EXPECTED_ROW_COUNTS_BY_NODE = (48, 48, 48, 48, 48, 49)
EXPECTED_PADDED_ROWS = 1729
EXPECTED_PADDED_ZERO_RHS = 1440
EXPECTED_GNL_COLUMNS = 42
EXPECTED_GNL_VALUES = EXPECTED_RECORDS * EXPECTED_GNL_COLUMNS


def _read_oracle_cortdeco(cortdeco_path: Path, mapcut: Any, numero_total_cortes: int) -> Any:
    """`Cortdeco.read`, every argument derived from the `Mapcut` object alone.

    Typed as `Any` throughout, matching `test_mapcut_assembly.py`'s own
    idiom: `idecomp` declares its section properties as optional even where
    the format guarantees them, and fighting that in an oracle adds casts
    without adding safety.
    """
    return Cortdeco.read(
        str(cortdeco_path),
        tamanho_registro=int(mapcut.tamanho_corte),
        registro_ultimo_corte_no=mapcut.registro_ultimo_corte_no,
        numero_total_cortes=numero_total_cortes,
        numero_patamares_carga=int(mapcut.patamares_por_estagio[0]),
        numero_estagios=int(mapcut.numero_estagios),
        codigos_uhes=[int(v) for v in mapcut.codigos_uhes],
        codigos_uhes_tempo_viagem=[],
        codigos_submercados=[int(v) for v in mapcut.dados_gnl["codigo_submercado"]],
        lag_maximo_tempo_viagem=int(mapcut.maximo_lag_tempo_viagem),
    )


@pytest.fixture(scope="module")
def oracle_mapcut(converted_pair: OutputPaths) -> Any:
    """The emitted `mapcut`, read by `idecomp` - a different code path from
    both this project's writer and its native reader."""
    return Mapcut.read(str(converted_pair.mapcut))


@pytest.fixture(scope="module")
def n_nodes(oracle_mapcut: Any) -> int:
    """The cut-building node count. `numero_semanas` doubles as this in the
    project's own convention (`pipeline.cross_check_pair`)."""
    return int(oracle_mapcut.numero_semanas)


@pytest.fixture(scope="module")
def oracle_cortdeco(converted_pair: OutputPaths, oracle_mapcut: Any) -> Any:
    """`Cortdeco.read` with `numero_total_cortes` as cuts per node
    (Requirement 3) - the correct call. Its padding-producing alternative
    (the study total) gets its own test rather than sharing this fixture."""
    numero_total_cortes = int(oracle_mapcut.numero_cortes) // int(oracle_mapcut.numero_semanas)
    return _read_oracle_cortdeco(converted_pair.cortdeco, oracle_mapcut, numero_total_cortes)


@pytest.fixture(scope="module")
def oracle_frame(oracle_cortdeco: Any) -> Any:
    return oracle_cortdeco.cortes


@pytest.fixture(scope="module")
def native_mapcut(converted_pair: OutputPaths) -> MapcutContents:
    """The shipped native reader's own decode of the emitted `mapcut`,
    independent of `oracle_mapcut`: nothing here is derived from an
    `idecomp` object."""
    return read_mapcut(converted_pair.mapcut)


@pytest.fixture(scope="module")
def native_n_patamares(native_mapcut: MapcutContents) -> int:
    return assert_uniform_blocks(native_mapcut.patamares_por_estagio)


@pytest.fixture(scope="module")
def native_offsets(native_mapcut: MapcutContents, native_n_patamares: int) -> CutBlockOffsets:
    scalars = native_mapcut.scalars
    return cortdeco_block_offsets(
        n_uhes=scalars.numero_uhes,
        n_utv=scalars.n_utv,
        max_lag=scalars.max_lag,
        n_sbm_gnl=len(native_mapcut.codigos_submercados_gnl),
        n_estagios=scalars.numero_estagios,
        n_patamares=native_n_patamares,
    )


@pytest.fixture(scope="module")
def native_cortdeco(
    converted_pair: OutputPaths, native_mapcut: MapcutContents, native_offsets: CutBlockOffsets
) -> CortdecoContents:
    """The shipped native reader's own decode of the emitted `cortdeco`,
    built entirely from `decomp/reader.py` and `decomp/layout.py` - the
    comparand `test_the_oracle_and_the_native_reader_agree_...` needs, and
    never itself derived from `oracle_mapcut`."""
    return read_cortdeco(
        converted_pair.cortdeco, native_offsets, native_mapcut.scalars.numero_semanas
    )


def test_the_oracle_frame_has_the_shape_and_per_node_row_counts_cuts_per_node_implies(
    oracle_frame: Any,
) -> None:
    assert oracle_frame.shape == (EXPECTED_RECORDS, EXPECTED_COLUMNS)
    counts = oracle_frame.groupby("no").size().sort_index()
    assert tuple(int(v) for v in counts) == EXPECTED_ROW_COUNTS_BY_NODE


def test_passing_the_study_total_instead_of_cuts_per_node_produces_padding_rows(
    converted_pair: OutputPaths, oracle_mapcut: Any
) -> None:
    """The trap `CLAUDE.md` already records, witnessed by an executed read
    rather than only avoided by the fixture above."""
    padded = _read_oracle_cortdeco(
        converted_pair.cortdeco, oracle_mapcut, numero_total_cortes=int(oracle_mapcut.numero_cortes)
    )
    frame = padded.cortes
    assert frame.shape[0] == EXPECTED_PADDED_ROWS
    assert int((frame["rhs"] == 0.0).sum()) == EXPECTED_PADDED_ZERO_RHS


def test_every_oracle_row_maps_to_a_distinct_record_index_matching_the_native_reader(
    oracle_frame: Any, native_cortdeco: CortdecoContents, n_nodes: int
) -> None:
    """`record_index = n_nodes * indice_corte - no` must cover `range(289)`
    exactly, and at each mapped index the oracle's `rhs` and 169
    `pi_varm_uhe<code>` values must decode to exactly the same bytes
    `read_cortdeco` did - both readers decode the same IEEE-754 bytes, so
    `==` rather than `pytest.approx` is the correct comparison."""
    varm_columns = [str(c) for c in oracle_frame.columns if str(c).startswith("pi_varm_uhe")]
    by_record_index = {cut.record_index: cut for cut in native_cortdeco.cuts}

    mapped_indices: list[int] = []
    for _, row in oracle_frame.iterrows():
        record_index = int(n_nodes * row["indice_corte"] - row["no"])
        mapped_indices.append(record_index)
        native_cut = by_record_index[record_index]
        assert float(row["rhs"]) == native_cut.rhs, f"record {record_index}: rhs disagrees"
        oracle_varm = tuple(float(row[column]) for column in varm_columns)
        assert oracle_varm == native_cut.pi_varm, f"record {record_index}: pi_varm disagrees"

    assert sorted(mapped_indices) == list(range(EXPECTED_RECORDS))


def test_the_pi_gnl_block_is_zero_through_the_oracle_and_the_byte_level_check(
    converted_pair: OutputPaths,
    oracle_frame: Any,
    native_cortdeco: CortdecoContents,
    native_offsets: CutBlockOffsets,
) -> None:
    """Premise P14, both ways: the oracle's own 42 columns compare `== 0.0`
    over all 12,138 values, and `assert_gnl_span_is_zero`'s byte-level check
    on the same file additionally excludes `-0.0`, which `== 0.0` cannot."""
    gnl_columns = [str(c) for c in oracle_frame.columns if str(c).startswith("pi_gnl_")]
    assert len(gnl_columns) == EXPECTED_GNL_COLUMNS
    block = oracle_frame[gnl_columns]
    assert int(block.size) == EXPECTED_GNL_VALUES
    assert bool((block == 0.0).to_numpy().all())

    assert_gnl_span_is_zero(converted_pair.cortdeco, native_cortdeco.record_count, native_offsets)


def test_cortdeco_columns_matches_the_oracle_frames_column_names(
    oracle_frame: Any, native_mapcut: MapcutContents, native_n_patamares: int
) -> None:
    columns = cortdeco_columns(
        native_mapcut.codigos_uhes,
        native_mapcut.codigos_submercados_gnl,
        native_mapcut.scalars.numero_estagios,
        native_n_patamares,
        native_mapcut.scalars.n_utv,
        native_mapcut.scalars.max_lag,
    )
    oracle_columns = tuple(str(c) for c in oracle_frame.columns)
    assert columns[-212:] == oracle_columns[3:]
