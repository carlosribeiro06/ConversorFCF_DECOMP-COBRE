"""Acceptance criteria for ticket-007 and ticket-008 against the real Cobre case.

Every anchor is transcribed from the reference `mapcut.rv0`, and every read-back
goes through `idecomp`, a different code path from the writer, so no assertion
compares the output against the value that produced it.

Two `idecomp` limitations are worked around rather than trusted:

- `lag_tempo_viagem_por_uhe` slices `[-(n_utv * n_estagios):]`, and `[-0:]`
  returns the whole underlying list, so an empty travel-time block cannot be
  asserted through it. Emptiness is asserted from the record count instead.
- `dados_gnl` walks its own buffer with a stride that does not match what the
  reader appended, so it surfaces only the first stage. Reg 9 is therefore
  asserted at the byte level.
"""

import hashlib
import logging
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from idecomp.decomp.mapcut import Mapcut

from conversor_fcf.cobre.inputs_reader import (
    CaseInputs,
    anticipated_thermals,
    read_case_inputs,
)
from conversor_fcf.cobre.policy_reader import (
    EntitySlotRecord,
    PolicyManifest,
    nodes_by_pool,
    read_policy_manifest,
    read_stage_cuts,
)
from conversor_fcf.decomp.layout import (
    PHYSICAL_RECORD_COUNT,
    RECORD_SIZE,
    LayoutError,
    assert_mapcut_layout,
    assert_no_travel_time,
    derive_n_utv,
    mapcut_record_count,
)
from conversor_fcf.decomp.mapcut_writer import MapcutHeader, write_mapcut
from conversor_fcf.mapping.rules import (
    discount_factors,
    load_hydro_codes,
    submarket_for_bus,
    tree_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
POLICY_DIR = REFERENCE_CASE / "output" / "policy"
REFERENCE_MAPCUT = Path("/home/carlosribeiro/git/mapcut.rv0")

EXPECTED_RECORDS = 300
EXPECTED_BYTES = 14_406_000
REFERENCE_DISCOUNTS = (
    1.0,
    0.997830417741,
    0.995665542570,
    0.993505364273,
    0.991349872661,
    0.989199057565,
    0.987052908840,
)
# Transcribed from the reference mapcut.rv0.
REFERENCE_TREE_PREFIX = (1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6)
REFERENCE_FIRST_NODES = (1, 2, 3, 4, 5, 6, 7)
REFERENCE_BLOCKS_PER_STAGE = (3, 3, 3, 3, 3, 3, 3)
REFERENCE_GNL_SUBMARKETS = (1, 3)

pytestmark = pytest.mark.skipif(
    not POLICY_DIR.is_dir(),
    reason=f"Cobre reference case not present at {REFERENCE_CASE}",
)

# Claims about the reference DECOMP deck skip explicitly when it is absent, rather
# than passing while asserting nothing.
needs_reference_deck = pytest.mark.skipif(
    not REFERENCE_MAPCUT.is_file(),
    reason=f"reference DECOMP mapcut not present at {REFERENCE_MAPCUT}",
)


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


@pytest.fixture(scope="module")
def inputs() -> CaseInputs:
    return read_case_inputs(REFERENCE_CASE)


@pytest.fixture(scope="module")
def manifest() -> PolicyManifest:
    return read_policy_manifest(POLICY_DIR / "manifest.bin")


@pytest.fixture(scope="module")
def trunk_slots() -> Sequence[EntitySlotRecord]:
    """One trunk pool's entity manifest; they are structurally identical."""
    return read_stage_cuts(POLICY_DIR / "cuts" / "000.bin").slots


@pytest.fixture(scope="module")
def header(
    inputs: CaseInputs, manifest: PolicyManifest, trunk_slots: Sequence[EntitySlotRecord]
) -> MapcutHeader:
    """Assemble a header from the real case.

    Building this from a Cobre case is `ticket-012`'s job; this local assembly
    exists only so `ticket-007` can be verified against real values.
    """
    pool_ids = sorted(nodes_by_pool(manifest))
    trunk = pool_ids[:-1]
    total_cuts = len(trunk) * manifest.completed_iterations
    node_count = len(manifest.nodes)
    heads = tuple(
        total_cuts - position if position < len(trunk) else 0 for position in range(node_count)
    )

    first_node_by_stage: dict[int, int] = {}
    for node in manifest.nodes:
        first_node_by_stage.setdefault(node.stage_id, node.id + 1)

    gnl = anticipated_thermals(inputs)
    start = date.fromisoformat(inputs.stages[0].start_date)
    return MapcutHeader(
        numero_iteracoes=manifest.completed_iterations,
        numero_cortes=total_cuts,
        numero_submercados=5,
        numero_uhes=len(inputs.hydros),
        numero_cenarios=node_count,
        numero_estagios=manifest.num_stages,
        numero_semanas=len(trunk),
        # Derived from the case, never assumed: premise P3 is a claim about the
        # input. max_lag is left to its default, since there is no axis to bound.
        n_utv=derive_n_utv(trunk_slots),
        dia=start.day,
        mes=start.month,
        ano=start.year,
        codigos_uhes=load_hydro_codes(REPO_ROOT / "decomp_hydro_codes.json"),
        codigos_uhes_jusante=tuple(
            (hydro.downstream_id + 1) if hydro.downstream_id is not None else 0
            for hydro in inputs.hydros
        ),
        indice_no_arvore=tree_indices(manifest),
        indice_primeiro_no_estagio=tuple(first_node_by_stage.values()),
        patamares_por_estagio=tuple(len(stage.blocks) for stage in inputs.stages),
        registro_ultimo_corte_no=heads,
        codigos_submercados_gnl=tuple(submarket_for_bus(t.bus_id) for t in gnl),
        lag_meses_gnl=tuple(2 for _ in gnl),
        patamares_gnl=tuple(len(inputs.stages[0].blocks) for _ in gnl),
        taxa_desconto=discount_factors(inputs.stages, inputs.annual_discount_rate),
    )


@pytest.fixture(scope="module")
def written(header: MapcutHeader, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mapcut") / "mapcut.rv0"
    assert write_mapcut(header, path) == EXPECTED_RECORDS
    return path


@pytest.fixture(scope="module")
def readback(written: Path) -> Any:
    """Typed as Any deliberately: idecomp declares its section properties as
    optional even where the format guarantees them, and fighting that in a
    read-back oracle would add casts without adding safety."""
    return Mapcut.read(str(written))


def test_the_case_carries_no_travel_time_state(
    trunk_slots: Sequence[EntitySlotRecord],
    header: MapcutHeader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Premise P3 over the real manifest, where one INFO per slot would mean 183."""
    with caplog.at_level(logging.INFO, logger="conversor_fcf"):
        assert_no_travel_time(trunk_slots)
    premise = [r.getMessage() for r in caplog.records if "premise P3 holds" in r.getMessage()]
    assert len(premise) == 1, f"one INFO for the whole manifest, not one per {len(trunk_slots)}"
    assert "regs 7/8" in premise[0]
    assert "NCOEF" in premise[0]

    assert derive_n_utv(trunk_slots) == 0
    assert header.n_utv == 0, "derived from the case, not a literal"
    assert header.max_lag == 0, "the documented default, not a declared lag bound"


def test_the_written_file_satisfies_the_layout_its_header_declares(
    written: Path, header: MapcutHeader
) -> None:
    assert_mapcut_layout(written, header)
    size = written.stat().st_size
    assert size == EXPECTED_BYTES == EXPECTED_RECORDS * RECORD_SIZE


def test_a_reference_scale_file_missing_one_record_is_refused(
    written: Path, header: MapcutHeader, tmp_path: Path
) -> None:
    """The named truncation mutation at full scale: 299 records against 300."""
    truncated = tmp_path / "truncated.rv0"
    truncated.write_bytes(written.read_bytes()[: (EXPECTED_RECORDS - 1) * RECORD_SIZE])
    with pytest.raises(LayoutError) as excinfo:
        assert_mapcut_layout(truncated, header)
    message = str(excinfo.value)
    assert f"holds {EXPECTED_RECORDS - 1} records" in message
    assert f"declares {EXPECTED_RECORDS}" in message


def test_the_file_is_exactly_three_hundred_whole_records(written: Path) -> None:
    size = written.stat().st_size
    assert size == EXPECTED_BYTES
    assert size % RECORD_SIZE == 0
    assert size // RECORD_SIZE == EXPECTED_RECORDS
    assert mapcut_record_count(0, 7, 0, 273) == EXPECTED_RECORDS


@needs_reference_deck
def test_the_reference_dimensions_reproduce_its_own_record_count() -> None:
    """356 is the reference file's real count, and 17,095,120 its real size."""
    assert mapcut_record_count(n_utv=2, n_estagios=7, max_lag=3, n_cenarios=273) == 356
    assert REFERENCE_MAPCUT.stat().st_size == 356 * RECORD_SIZE == 17_095_120


def test_scalars_read_back_as_the_case_declares(readback: Any) -> None:
    assert int(readback.numero_estagios) == 7
    assert int(readback.numero_uhes) == 169
    assert int(readback.numero_submercados) == 5
    assert int(readback.numero_cenarios) == 273
    assert int(readback.numero_semanas) == 6
    assert int(readback.tamanho_corte) == 26976
    assert int(readback.numero_iteracoes) == 48
    assert int(readback.numero_cortes) == 288
    assert readback.data_inicio.date() == date(2026, 4, 25)


def test_travel_time_is_absent_by_record_count_not_by_the_oracle(
    readback: Any, written: Path
) -> None:
    """idecomp's lag slice degenerates to [-0:], so emptiness is structural here."""
    assert int(readback.numero_uhes_tempo_viagem) == 0
    assert int(readback.maximo_lag_tempo_viagem) == 0
    assert written.stat().st_size // RECORD_SIZE == mapcut_record_count(0, 7, 0, 273)


def test_cut_heads_descend_across_the_six_cut_building_nodes(readback: Any) -> None:
    frame = readback.registro_ultimo_corte_no
    non_zero = frame.loc[frame["indice_ultimo_corte"] != 0, "indice_ultimo_corte"].tolist()
    assert [int(v) for v in non_zero] == [288, 287, 286, 285, 284, 283]
    assert int((frame["indice_ultimo_corte"] == 0).sum()) == 273 - 6


def test_tree_indices_read_back_as_the_reference_prefix(readback: Any) -> None:
    assert tuple(int(v) for v in readback.indice_no_arvore[:14]) == REFERENCE_TREE_PREFIX
    assert len(readback.indice_no_arvore) == 273


def test_stage_arrays_read_back_as_the_reference_holds_them(readback: Any) -> None:
    assert tuple(int(v) for v in readback.indice_primeiro_no_estagio) == REFERENCE_FIRST_NODES
    assert tuple(int(v) for v in readback.patamares_por_estagio) == REFERENCE_BLOCKS_PER_STAGE


def test_hydro_codes_read_back_in_positional_order(readback: Any) -> None:
    codes = [int(v) for v in readback.codigos_uhes]
    assert len(codes) == 169
    assert codes[:5] == [1, 2, 4, 6, 7]
    assert codes[168] == 315


def test_discount_series_reads_back_as_the_reference_and_the_rest_is_zero(
    readback: Any,
) -> None:
    frame = readback.dados_custos
    for stage, reference in enumerate(REFERENCE_DISCOUNTS):
        assert abs(float(frame["taxa_desconto"][stage]) - reference) < 1e-12, f"stage {stage}"
    others = frame.drop(columns=["estagio", "taxa_desconto"])
    assert bool((others == 0.0).all().all()), "premise P11 emits these as zeros"


def test_gnl_submarkets_match_bus_id_plus_one(header: MapcutHeader) -> None:
    """The reference holds [1.0, 3.0]; buses 0 (SE) and 2 (NE) map to 1 and 3."""
    assert header.codigos_submercados_gnl == REFERENCE_GNL_SUBMARKETS


def test_the_physical_span_of_our_file_is_zero(written: Path) -> None:
    raw = written.read_bytes()
    for index in range(4, 4 + PHYSICAL_RECORD_COUNT):
        assert raw[index * RECORD_SIZE : (index + 1) * RECORD_SIZE] == b"\x00" * RECORD_SIZE


@needs_reference_deck
def test_the_reference_populates_reg_nine_where_premise_p12_emits_zeros(written: Path) -> None:
    """P12's evidence, read from both files rather than asserted in prose.

    The reference holds the GNL lag month's hours per load block: three values per
    submarket summing to 730.5 = 365.25*24/12, on the same day-count basis as the
    discount series. Our reg 9 carries the identical int32 head and a zero tail.
    """
    reference_reg9 = REFERENCE_MAPCUT.read_bytes()[76 * RECORD_SIZE : 77 * RECORD_SIZE]
    tail = np.frombuffer(reference_reg9[28 : 28 + 8 * 14], dtype="<f8")
    ngnl = len(REFERENCE_GNL_SUBMARKETS)
    populated = int(np.nonzero(tail == 0)[0][0])
    assert populated == ngnl * 3, "the populated width follows the block axis, not the stage axis"
    for submarket in range(ngnl):
        assert float(tail[submarket::ngnl][:3].sum()) == pytest.approx(730.5, abs=1e-9)

    ours = written.read_bytes()
    first_reg9 = (4 + PHYSICAL_RECORD_COUNT + 2) * RECORD_SIZE
    our_tail = np.frombuffer(ours[first_reg9 + 28 : first_reg9 + 28 + 8 * 14], dtype="<f8")
    assert not our_tail.any(), "premise P12: we emit zeros here"


@needs_reference_deck
def test_the_reference_populates_every_record_our_file_zeroes() -> None:
    """Premise P1 is a real divergence, and this is the assertion that proves it."""
    body = REFERENCE_MAPCUT.read_bytes()
    populated = sum(
        1
        for index in range(4, 4 + PHYSICAL_RECORD_COUNT)
        if any(body[index * RECORD_SIZE : (index + 1) * RECORD_SIZE])
    )
    assert populated == PHYSICAL_RECORD_COUNT, "all 14 are populated in the reference"


def test_a_reference_scale_dirty_byte_in_the_physical_span_is_refused(
    written: Path, header: MapcutHeader, tmp_path: Path
) -> None:
    """The span sits at fixed records 4-17, so scale cannot change the mechanism.

    Tested here anyway: the truncation mutation got a reference-scale case and this
    one did not, and an asymmetry in the evidence is not an argument.
    """
    raw = bytearray(written.read_bytes())
    raw[9 * RECORD_SIZE + 17] = 1
    dirty = tmp_path / "dirty.rv0"
    dirty.write_bytes(bytes(raw))

    with pytest.raises(LayoutError) as excinfo:
        assert_mapcut_layout(dirty, header)
    message = str(excinfo.value)
    assert "record 9" in message
    assert "byte 17" in message


def test_reg9_repeats_per_node_and_reg10_varies_per_stage(written: Path) -> None:
    raw = written.read_bytes()
    first = 4 + PHYSICAL_RECORD_COUNT + 2

    def digest(index: int) -> str:
        return hashlib.sha256(raw[index * RECORD_SIZE : (index + 1) * RECORD_SIZE]).hexdigest()

    reg9 = {digest(first + 2 * k) for k in range(7)}
    reg10 = {digest(first + 2 * k + 1) for k in range(7)}
    trailing = {digest(index) for index in range(first + 14, EXPECTED_RECORDS)}

    assert len(reg9) == 1
    assert len(reg10) == 7
    assert trailing == reg9, "the trailing reg-9 records repeat the same payload"
    assert len(range(first + 14, EXPECTED_RECORDS)) == 273 - 7
