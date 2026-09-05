"""Unit tests for the mapcut writer.

Per the master plan's Numeric Path Testing Policy, the named deliberate mutation
for this ticket is **swapping reg 9 and reg 10 in the interleaved span**. The
demonstration is `test_swapping_reg9_and_reg10_changes_the_bytes`, which pins
the two records' distinguishing property directly, plus the read-back assertions
in the integration suite, which fail outright under the swap.

The layout invariant itself — the record count, the travel-time guard and the
file-level checks — lives in `test_mapcut_invariants.py`, which owns `layout.py`.
"""

import hashlib
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.decomp import mapcut_writer
from conversor_fcf.decomp.layout import (
    PHYSICAL_RECORD_COUNT,
    RECORD_SIZE,
    TAMANHO_CORTE,
    LayoutError,
    mapcut_record_count,
)
from conversor_fcf.decomp.mapcut_writer import MapcutHeader, write_mapcut

HeaderFactory = Callable[..., MapcutHeader]


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


def _records(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    return [raw[i * RECORD_SIZE : (i + 1) * RECORD_SIZE] for i in range(len(raw) // RECORD_SIZE)]


def test_every_record_is_exactly_one_record_long(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    path = tmp_path / "mapcut.rv0"
    count = write_mapcut(mapcut_header(), path)
    assert count == mapcut_record_count(0, 3, 0, 5)
    assert path.stat().st_size == count * RECORD_SIZE
    assert all(len(record) == RECORD_SIZE for record in _records(path))


def test_an_oversized_record_is_refused(tmp_path: Path, mapcut_header: HeaderFactory) -> None:
    """reg 1 packs 5 + n_cenarios int32, so a huge node count overflows it."""
    too_many = RECORD_SIZE // 4
    header = mapcut_header(
        numero_cenarios=too_many,
        indice_no_arvore=tuple(1 for _ in range(too_many)),
        registro_ultimo_corte_no=tuple(0 for _ in range(too_many)),
    )
    with pytest.raises(LayoutError, match="more than the 48020-byte record"):
        write_mapcut(header, tmp_path / "mapcut.rv0")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"codigos_uhes": (1, 2)}, "codigos_uhes has 2 entries but numero_uhes is 4"),
        ({"codigos_uhes_jusante": (1,)}, "codigos_uhes_jusante has 1 entries"),
        ({"indice_no_arvore": (1,)}, "indice_no_arvore has 1 entries"),
        ({"registro_ultimo_corte_no": (1,)}, "registro_ultimo_corte_no has 1 entries"),
        ({"indice_primeiro_no_estagio": (1,)}, "indice_primeiro_no_estagio has 1 entries"),
        ({"patamares_por_estagio": (3,)}, "patamares_por_estagio has 1 entries"),
        ({"taxa_desconto": (1.0,)}, "taxa_desconto has 1 entries"),
        ({"lag_meses_gnl": (2,)}, "lag_meses_gnl has 1 entries"),
        ({"patamares_gnl": (3,)}, "patamares_gnl has 1 entries"),
    ],
)
def test_an_inconsistent_header_is_refused_before_any_byte(
    tmp_path: Path,
    mapcut_header: HeaderFactory,
    override: dict[str, object],
    match: str,
) -> None:
    path = tmp_path / "mapcut.rv0"
    with pytest.raises(LayoutError, match=match):
        write_mapcut(mapcut_header(**override), path)
    assert not path.exists(), "no file may appear when the header is rejected"
    assert not list(tmp_path.iterdir()), "not even a partial file"


def test_a_header_with_travel_time_is_refused(tmp_path: Path, mapcut_header: HeaderFactory) -> None:
    with pytest.raises(LayoutError, match="regs 7/8 are unimplemented"):
        write_mapcut(mapcut_header(n_utv=2), tmp_path / "mapcut.rv0")


def test_the_physical_span_is_entirely_zero(tmp_path: Path, mapcut_header: HeaderFactory) -> None:
    path = tmp_path / "mapcut.rv0"
    write_mapcut(mapcut_header(), path)
    records = _records(path)
    for index in range(4, 4 + PHYSICAL_RECORD_COUNT):
        assert records[index] == b"\x00" * RECORD_SIZE, f"record {index}"


def test_every_divergence_warning_fires_exactly_once(
    tmp_path: Path, mapcut_header: HeaderFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """Every record this writer zeroes but the reference populates must announce itself."""
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        write_mapcut(mapcut_header(), tmp_path / "mapcut.rv0")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    # The colon matters: "premise P1" is a substring of both "P11" and "P12".
    p1 = [m for m in warnings if m.startswith("premise P1:")]
    p11 = [m for m in warnings if m.startswith("premise P11:")]
    p12 = [m for m in warnings if m.startswith("premise P12:")]
    assert len(p1) == 1, "one WARNING per run, not one per zero-filled record"
    assert len(p11) == 1
    assert len(p12) == 1, "one per run, not one per node — reg 9 is written 5 times here"
    assert len(warnings) == 3, f"exactly three declared divergences, got {warnings}"
    assert "ticket-015" in p1[0]
    assert "diverges" in p1[0], "the divergence from the reference must be stated"
    for field in (
        "parcela_custo_geracao_termica_minima",
        "geracao_termica_minima_gerada_gnl",
    ):
        assert field in p11[0], f"the P11 warning must name {field}"
    assert "730.5" in p12[0], "the reference's own value anchors the divergence"


def test_reg9_records_are_identical_and_reg10_records_differ(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    """Reg 9 carries configuration, so it repeats; reg 10 is per-stage, so it varies."""
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    write_mapcut(header, path)
    records = _records(path)

    first = 4 + PHYSICAL_RECORD_COUNT + 2
    stages = header.numero_estagios
    reg9 = [records[first + 2 * k] for k in range(stages)]
    reg10 = [records[first + 2 * k + 1] for k in range(stages)]
    trailing = records[first + 2 * stages :]

    def digest(body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()

    assert len({digest(r) for r in reg9}) == 1, "every reg 9 is the same payload"
    assert len({digest(r) for r in reg10}) == stages, "every reg 10 differs"
    assert {digest(r) for r in trailing} == {digest(reg9[0])}
    assert len(trailing) == header.numero_cenarios - stages


def test_swapping_reg9_and_reg10_changes_the_bytes(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    """The named deliberate mutation, pinned by what distinguishes the two records.

    Reg 9 opens with an int32 count of GNL submarkets; reg 10 opens with a float64
    discount factor. Swapping them makes the first record of each pair
    unrecognisable, which is what the read-back assertions detect.
    """
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    write_mapcut(header, path)
    records = _records(path)
    first = 4 + PHYSICAL_RECORD_COUNT + 2

    gnl_head = int(np.frombuffer(records[first][:4], dtype="<i4")[0])
    cost_head = float(np.frombuffer(records[first + 1][:8], dtype="<f8")[0])
    assert gnl_head == len(header.codigos_submercados_gnl)
    assert cost_head == header.taxa_desconto[0]

    # Under a swap the first record of the pair would open with the discount
    # factor's bytes, which do not read as the GNL count.
    swapped_head = int(np.frombuffer(records[first + 1][:4], dtype="<i4")[0])
    assert swapped_head != gnl_head


def test_the_write_is_atomic(tmp_path: Path, mapcut_header: HeaderFactory) -> None:
    path = tmp_path / "nested" / "mapcut.rv0"
    write_mapcut(mapcut_header(), path)
    assert path.is_file()
    assert not list(path.parent.glob("*.partial")), "no temporary file may survive"


def test_tamanho_corte_is_the_fixed_decomp_maximum(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    path = tmp_path / "mapcut.rv0"
    write_mapcut(mapcut_header(), path)
    case_record = _records(path)[1]
    fields = np.frombuffer(case_record[:16], dtype="<i4")
    assert int(fields[0]) == TAMANHO_CORTE == 26976
    assert [int(v) for v in fields[1:]] == [25, 4, 2026]


def test_a_disagreement_between_the_invariant_and_the_assembly_is_refused(
    tmp_path: Path, mapcut_header: HeaderFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard exists so a wrong assembly loop can never reach the destination."""
    monkeypatch.setattr(mapcut_writer, "mapcut_record_count", lambda *_args, **_kwargs: 999)
    path = tmp_path / "mapcut.rv0"
    # 4 + 14 + 1 + 1 + 0 + 5 nodes + 3 stages = 28 for this small header.
    with pytest.raises(LayoutError, match="assembled 28 records but the layout invariant"):
        write_mapcut(mapcut_header(), path)
    assert not list(tmp_path.iterdir())


def test_a_short_write_is_refused(
    tmp_path: Path, mapcut_header: HeaderFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer distrusts the on-disk size rather than assuming the write was whole.

    The write is truncated for real rather than the checker mocked, so the file
    reaching `assert_mapcut_layout` is genuinely one record short.
    """
    original = Path.write_bytes

    def truncating(self: Path, data: bytes) -> int:
        if self.name.endswith(".partial"):
            return original(self, data[:-RECORD_SIZE])
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", truncating)
    path = tmp_path / "mapcut.rv0"
    with pytest.raises(LayoutError, match="holds 27 records but its header declares 28"):
        write_mapcut(mapcut_header(), path)
    assert not path.exists()
    assert not list(tmp_path.iterdir()), "the partial file must be cleaned up"


def test_a_failed_write_leaves_nothing_behind(
    tmp_path: Path, mapcut_header: HeaderFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.write_bytes

    def failing(self: Path, data: bytes) -> int:
        if self.name.endswith(".partial"):
            original(self, data[: RECORD_SIZE // 2])
            raise OSError("no space left on device")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", failing)
    path = tmp_path / "mapcut.rv0"
    with pytest.raises(OSError, match="no space left"):
        write_mapcut(mapcut_header(), path)
    assert not path.exists()
    assert not list(tmp_path.iterdir()), "the partial file must be cleaned up"
