"""Unit tests for `pipeline`'s M6 machinery: `cross_check_pair`'s disagreement
branches and `_reject_pair`'s rename-away (Fix 1 / C6).

`cross_check_pair`'s two branches need a `mapcut`/`cortdeco` pair whose bytes
genuinely disagree, which only a hand-built pair below the full pipeline can
produce: a real `run_conversion` computes `numero_cortes` once and feeds the
same value to both writers by construction (M6's first line of defence), so
its own end-to-end run can never exercise the disagreement this check exists
to catch. That is exactly why this needs a synthetic pair rather than the
reference case. The full-pipeline reconstruction of a real disagreement (both
writers' own invariants satisfied, both files fully published before the
cross-check runs) lives in `tests/integration/test_pipeline_end_to_end.py`,
which is also where `_reject_pair`'s successful rename is proven end-to-end;
this file covers only its failure-to-rename branch, which needs no full run.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from conversor_fcf.decomp.cortdeco_writer import CutInput, write_cortdeco, zeroed_gnl_block
from conversor_fcf.decomp.layout import (
    CutBlockOffsets,
    LayoutError,
    cortdeco_block_offsets,
    cut_head_indices,
)
from conversor_fcf.decomp.mapcut_writer import MapcutHeader, write_mapcut
from conversor_fcf.paths import OutputPaths
from conversor_fcf.pipeline import _reject_pair, cross_check_pair

_N_UHES = 3
_N_SBM_GNL = 1
_N_ESTAGIOS = 1
_N_PATAMARES = 1


def _header(**overrides: Any) -> MapcutHeader:
    defaults: dict[str, Any] = {
        "numero_iteracoes": 2,
        "numero_cortes": 6,
        "numero_submercados": 5,
        "numero_uhes": _N_UHES,
        "numero_cenarios": 3,
        "numero_estagios": _N_ESTAGIOS,
        "numero_semanas": 3,
        "n_utv": 0,
        "dia": 1,
        "mes": 1,
        "ano": 2026,
        "codigos_uhes": (1, 2, 3),
        "codigos_uhes_jusante": (2, 3, 0),
        "indice_no_arvore": (1, 1, 2),
        "indice_primeiro_no_estagio": (1,),
        "patamares_por_estagio": (_N_PATAMARES,),
        "registro_ultimo_corte_no": cut_head_indices(6, 3, 3),
        "codigos_submercados_gnl": (1,),
        "lag_meses_gnl": (2,),
        "patamares_gnl": (1,),
        "taxa_desconto": (1.0,),
    }
    defaults.update(overrides)
    return MapcutHeader(**defaults)


def _offsets() -> CutBlockOffsets:
    return cortdeco_block_offsets(
        n_uhes=_N_UHES,
        n_utv=0,
        max_lag=0,
        n_sbm_gnl=_N_SBM_GNL,
        n_estagios=_N_ESTAGIOS,
        n_patamares=_N_PATAMARES,
    )


def _cuts(n_nodes: int, per_node: int, offsets: CutBlockOffsets) -> list[list[CutInput]]:
    pi_gnl: NDArray[np.float64] = zeroed_gnl_block(offsets)
    return [
        [
            CutInput(
                intercept=float(node * 1000 + iteration),
                pi_varm=np.array([float(node), float(iteration), 0.0]),
                pi_gnl=pi_gnl,
            )
            for iteration in range(per_node)
        ]
        for node in range(n_nodes)
    ]


def test_a_numero_cortes_disagreement_is_refused_naming_both_values(tmp_path: Path) -> None:
    """The wiring defect M6 exists to catch: mapcut's header says 6, but
    cortdeco was written with a different `numero_cortes` (3)."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    write_mapcut(_header(), mapcut_path)

    offsets = _offsets()
    write_cortdeco(_cuts(3, 1, offsets), cortdeco_path, numero_cortes=3, offsets=offsets)

    with pytest.raises(LayoutError, match=r"numero_cortes=6.*numero_cortes=3"):
        cross_check_pair(mapcut_path, cortdeco_path)


def test_a_head_table_disagreement_is_refused(tmp_path: Path) -> None:
    """`numero_cortes` agrees, but mapcut's own declared heads do not match
    `cut_head_indices`' formula - the second, independent half of M6."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    write_mapcut(_header(registro_ultimo_corte_no=(5, 5, 5)), mapcut_path)

    offsets = _offsets()
    write_cortdeco(_cuts(3, 2, offsets), cortdeco_path, numero_cortes=6, offsets=offsets)

    with pytest.raises(LayoutError, match="disagree with"):
        cross_check_pair(mapcut_path, cortdeco_path)


def test_an_agreeing_pair_returns_both_contents(tmp_path: Path) -> None:
    """The success path: no exception, and both decoded contents come back."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    write_mapcut(_header(), mapcut_path)

    offsets = _offsets()
    write_cortdeco(_cuts(3, 2, offsets), cortdeco_path, numero_cortes=6, offsets=offsets)

    mapcut_contents, cortdeco_contents = cross_check_pair(mapcut_path, cortdeco_path)
    assert mapcut_contents.scalars.numero_cortes == 6
    assert cortdeco_contents.numero_cortes == 6


# --- the GNL span check on the re-derived offsets (ticket-018) -------------
#
# `cross_check_pair` re-derives `offsets` from the read-back mapcut and never
# reuses `write_cortdeco`'s own; these three cases poke the published
# `cortdeco` bytes directly, after a normal write, so the corruption is
# something neither writer's own gate could have seen.


def test_a_dirty_gnl_byte_is_refused_naming_the_record_slot_and_position(
    tmp_path: Path,
) -> None:
    """Byte 36 of record 0 is `pi_gnl` slot 0's first byte (coefficient
    position 4, from `_offsets`' `n_uhes=3`): overwriting it with `1` must be
    caught by `cross_check_pair`'s own span check, not just `write_cortdeco`'s."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    write_mapcut(_header(), mapcut_path)

    offsets = _offsets()
    write_cortdeco(_cuts(3, 2, offsets), cortdeco_path, numero_cortes=6, offsets=offsets)

    raw = bytearray(cortdeco_path.read_bytes())
    raw[36] = 1
    cortdeco_path.write_bytes(bytes(raw))

    with pytest.raises(LayoutError, match="record 0") as error:
        cross_check_pair(mapcut_path, cortdeco_path)
    message = str(error.value)
    assert "slot 0" in message
    assert "coefficient position 4" in message


def test_a_negative_zero_gnl_byte_is_refused_naming_byte_seven(tmp_path: Path) -> None:
    """`-0.0`'s little-endian sign byte (`0x80`) at the slot's last byte must
    be caught even though it compares numerically equal to zero."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    write_mapcut(_header(), mapcut_path)

    offsets = _offsets()
    write_cortdeco(_cuts(3, 2, offsets), cortdeco_path, numero_cortes=6, offsets=offsets)

    raw = bytearray(cortdeco_path.read_bytes())
    raw[36:44] = bytes.fromhex("0000000000000080")
    cortdeco_path.write_bytes(bytes(raw))

    with pytest.raises(LayoutError, match="record 0") as error:
        cross_check_pair(mapcut_path, cortdeco_path)
    message = str(error.value)
    assert "byte 7" in message


def test_a_call_site_offset_disagreement_mislabels_a_column_and_is_refused(
    tmp_path: Path,
) -> None:
    """Reconstructs the epic-06 finding this ticket closes: `cortdeco` is
    written with `n_uhes=3` offsets (`pi_gnl` 4, `NCOEF` 5), but paired with a
    `mapcut` declaring `numero_uhes=2`, so `cross_check_pair` re-derives
    `pi_gnl` 3, `NCOEF` 4 - one slot into the real `pi_varm` block. Its third
    value must be non-zero (7.0, never `_cuts`' 0.0) or the mislabelled read
    would silently agree with zero too."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    write_mapcut(
        _header(numero_uhes=2, codigos_uhes=(1, 2), codigos_uhes_jusante=(2, 0)), mapcut_path
    )

    written_offsets = _offsets()
    assert written_offsets.pi_gnl == 4
    assert written_offsets.ncoef == 5
    pi_gnl = zeroed_gnl_block(written_offsets)
    cuts = [
        [
            CutInput(
                intercept=float(node * 1000 + iteration),
                pi_varm=np.array([float(node), float(iteration), 7.0]),
                pi_gnl=pi_gnl,
            )
            for iteration in range(2)
        ]
        for node in range(3)
    ]
    write_cortdeco(cuts, cortdeco_path, numero_cortes=6, offsets=written_offsets)

    with pytest.raises(LayoutError, match="coefficient position 3"):
        cross_check_pair(mapcut_path, cortdeco_path)


# --- _reject_pair (Fix 1 / C6): the rename-failure branch -------------------
#
# The successful rename - both files present as `.rejected`, neither at its
# published name - is proven end-to-end in
# tests/integration/test_pipeline_end_to_end.py, through a real disagreement
# reconstructed via run_conversion. What only needs a unit test is the branch
# that must never fire in that scenario: a rename itself failing.


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


def test_a_rename_failure_is_logged_not_raised_and_the_second_file_is_still_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A rename failure must never mask the cross-check's own `LayoutError` -
    `_reject_pair` runs entirely inside the caller's `except` block and must
    not itself raise - and both binaries must be attempted even though the
    first one fails."""
    mapcut_path = tmp_path / "mapcut.rv0"
    cortdeco_path = tmp_path / "cortdeco.rv0"
    mapcut_path.write_bytes(b"mapcut-bytes")
    cortdeco_path.write_bytes(b"cortdeco-bytes")
    paths = OutputPaths(
        root=tmp_path,
        mapcut=mapcut_path,
        cortdeco=cortdeco_path,
        run_manifest=tmp_path / "run_manifest.json",
        eco_dir=tmp_path / "eco",
        content_dir=tmp_path / "content",
        log=tmp_path / "logs" / "conversor-fcf.log",
    )

    real_replace = os.replace

    def failing_for_mapcut_only(source: Any, destination: Any) -> None:
        if Path(source) == mapcut_path:
            raise OSError("simulated rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", failing_for_mapcut_only)

    with caplog.at_level(logging.ERROR, logger="conversor_fcf"):
        _reject_pair(paths)  # must not raise

    assert mapcut_path.exists(), "the failed rename must leave the original file in place"
    assert not cortdeco_path.exists(), "the second file's rename must still be attempted"
    assert (tmp_path / "cortdeco.rv0.rejected").exists()
    assert any("failed to rename" in record.getMessage() for record in caplog.records)
