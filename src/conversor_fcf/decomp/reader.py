"""Native readers for the two DECOMP binaries, for the content CSVs to witness.

This module exists because two constraints meet: the content CSVs must be
produced by **reading the written binary back** rather than by re-serializing the
records in memory, and `idecomp` must never become a runtime dependency. An
artifact generated from the values it was written from stays self-consistent
under any serialization bug, which is precisely how epic-02's decode corruption
stayed invisible to a fully covered suite.

**This module imports nothing from either writer, deliberately.** A reader that
reuses a writer's field list cannot witness that writer's field order; it would
agree with it by construction. Every count here is recovered from the file's own
header records, and only `layout.py` — which writes no bytes — is shared.

The record map is `idecomp`'s section definitions, re-expressed rather than
imported, and every position below was re-read from the reference
`mapcut.rv0` (356 records) before this module was written:

| Record | Content |
|--------|---------|
| 0 | reg 1: five int32 scalars, then `numero_cenarios` cut heads |
| 1 | reg 2: `[tamanho_corte, dia, mes, ano]` |
| 2 | reg 3: `codigos_uhes` |
| 3 | reg 4: `codigos_uhes_jusante`, int32 on disk |
| 4-17 | the undocumented per-plant physical span (premise P1) |
| 18 | reg 5: `indice_no_arvore` |
| 19 | reg 6: five int32 scalars, then two `n_estagios` arrays, then the lags |
| 20.. | `n_utv * n_estagios * (max_lag + 1)` travel-time records |
| .. | reg 9 per node and reg 10 per stage, interleaved across the first
       `n_estagios` pairs, then reg 9 alone |

Classification is **positional**, never by sniffing content: the interleave is a
deterministic function of `n_cenarios` and `n_estagios`, and a reader that
guessed a record's kind from its first int32 could misclassify one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conversor_fcf.decomp.layout import (
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    TAMANHO_CORTE,
    CutBlockOffsets,
    mapcut_record_count,
)
from conversor_fcf.logging_setup import get_logger

_logger = get_logger("reader")

_REG_GENERAL = 0
_REG_CASE = 1
_REG_HYDRO_CODES = 2
_REG_DOWNSTREAM = 3
_REG_TREE = PHYSICAL_RECORD_LAST + 1
_REG_STAGE = _REG_TREE + 1
_FIRST_TRAVEL_RECORD = _REG_STAGE + 1

_COST_FIELDS_PER_STAGE = 6
_GNL_ARRAYS = 3


class ReadError(Exception):
    """Raised when a DECOMP binary cannot be decoded as its own header describes."""


@dataclass(frozen=True)
class MapcutScalars:
    """Every scalar recovered from the file itself, not supplied by a caller."""

    numero_iteracoes: int
    numero_cortes: int
    numero_submercados: int
    numero_uhes: int
    numero_cenarios: int
    tamanho_corte: int
    dia: int
    mes: int
    ano: int
    numero_estagios: int
    numero_semanas: int
    n_utv: int
    max_lag: int


@dataclass(frozen=True)
class MapcutContents:
    """A decoded `mapcut`, as the bytes on disk describe it."""

    record_count: int
    scalars: MapcutScalars
    cut_heads: tuple[int, ...]
    codigos_uhes: tuple[int, ...]
    codigos_uhes_jusante: tuple[int, ...]
    indice_no_arvore: tuple[int, ...]
    indice_primeiro_no_estagio: tuple[int, ...]
    patamares_por_estagio: tuple[int, ...]
    travel_time_lags: tuple[int, ...]
    physical_span_nonzero_records: tuple[int, ...]
    codigos_submercados_gnl: tuple[int, ...]
    lag_meses_gnl: tuple[int, ...]
    patamares_gnl: tuple[int, ...]
    gnl_trailing_block: tuple[float, ...]
    gnl_record_indices: tuple[int, ...]
    cost_records: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CortdecoCut:
    """One decoded cut record."""

    record_index: int
    next_index: int
    rhs: float
    pi_varm: tuple[float, ...]
    pi_gnl: tuple[float, ...]
    is_extra_record: bool


@dataclass(frozen=True)
class CortdecoContents:
    """A decoded `cortdeco`."""

    record_count: int
    numero_cortes: int
    cuts: tuple[CortdecoCut, ...]


def _record(raw: bytes, index: int) -> bytes:
    return raw[index * RECORD_SIZE : (index + 1) * RECORD_SIZE]


def _ints(raw: bytes, index: int, count: int, offset: int = 0) -> tuple[int, ...]:
    start = index * RECORD_SIZE + offset
    return tuple(int(v) for v in np.frombuffer(raw, dtype="<i4", count=count, offset=start))


def _floats(raw: bytes, index: int, count: int, offset: int = 0) -> tuple[float, ...]:
    start = index * RECORD_SIZE + offset
    return tuple(float(v) for v in np.frombuffer(raw, dtype="<f8", count=count, offset=start))


def _read_whole(path: Path, record_size: int, label: str) -> tuple[bytes, int]:
    raw = path.read_bytes()
    records, remainder = divmod(len(raw), record_size)
    if remainder or records == 0:
        raise ReadError(
            f"{path} is {len(raw)} bytes, not a whole number of {record_size}-byte {label} "
            f"records: {records} records plus {remainder} trailing bytes"
        )
    return raw, records


def read_mapcut(path: Path) -> MapcutContents:
    """Decode a `mapcut` using only its own header records.

    The caller supplies no scalars, which is the point: the recovered values are
    what a test or a content CSV compares *against* the values the writer was
    given.
    """
    raw, records = _read_whole(path, RECORD_SIZE, "mapcut")

    n_iteracoes, n_cortes, n_submercados, n_uhes, n_cenarios = _ints(raw, _REG_GENERAL, 5)
    for name, value in (
        ("numero_uhes", n_uhes),
        ("numero_cenarios", n_cenarios),
        ("numero_submercados", n_submercados),
    ):
        if value <= 0:
            raise ReadError(f"{path}: reg 1 declares {name} = {value}, which cannot be decoded")

    # n_uhes and n_cenarios, like n_submercados above, are only checked > 0:
    # neither enters mapcut_record_count, so a corrupted reg 1 can still send
    # a field sized by one of them past its own record's capacity, well before
    # the whole-file record count is ever checked. Named here for both an
    # oversized value (which already fails inside numpy with its unnamed
    # "buffer is smaller than requested size") and a moderate one (comfortably
    # under the file's own total size, so it reads past the one record into
    # whatever follows with no error at all) - the same standard
    # `cortdeco_writer._validate_lengths` holds its own record capacity to.
    uhes_span = n_uhes * 4
    if uhes_span > RECORD_SIZE:
        raise ReadError(
            f"{path}: reg 1 declares numero_uhes = {n_uhes}, whose {uhes_span}-byte "
            f"codigos_uhes/codigos_uhes_jusante span does not fit the {RECORD_SIZE}-byte record"
        )
    # n_cenarios sizes reg 1's own cut_heads (offset 20) and reg 5's whole
    # indice_no_arvore record; reg 1's is the tighter bound for the same
    # n_cenarios because of its +20 offset, so bounding it here covers both.
    cenarios_span = 20 + 4 * n_cenarios
    if cenarios_span > RECORD_SIZE:
        raise ReadError(
            f"{path}: reg 1 declares numero_cenarios = {n_cenarios}, whose {cenarios_span}-byte "
            f"cut_heads span (offset 20) does not fit the {RECORD_SIZE}-byte record"
        )

    tamanho_corte, dia, mes, ano = _ints(raw, _REG_CASE, 4)
    if tamanho_corte != TAMANHO_CORTE:
        raise ReadError(
            f"{path}: reg 2 declares tamanho_corte {tamanho_corte} but the format fixes it at "
            f"{TAMANHO_CORTE}"
        )

    _, n_estagios, n_semanas, n_utv, max_lag = _ints(raw, _REG_STAGE, 5)
    if n_estagios <= 0:
        raise ReadError(f"{path}: reg 6 declares numero_estagios = {n_estagios}")
    if n_estagios > n_cenarios:
        raise ReadError(
            f"{path}: reg 6 declares {n_estagios} stages but reg 1 declares only {n_cenarios} "
            f"nodes, so the reg 9 / reg 10 interleave is undefined"
        )
    # n_estagios/n_utv size three reg 6 arrays past its five leading scalars:
    # indice_primeiro_no_estagio, patamares_por_estagio and travel_time_lags.
    # travel_time_lags has the largest offset and span of the three, so
    # bounding it covers the other two as well - same reasoning as n_cenarios
    # above, checked here rather than after the whole-file record count for
    # the same reason: neither n_estagios nor n_utv alone decides that count.
    stage_arrays_span = (5 + 2 * n_estagios + n_utv * n_estagios) * 4
    if stage_arrays_span > RECORD_SIZE:
        raise ReadError(
            f"{path}: reg 6 declares numero_estagios = {n_estagios} and n_utv = {n_utv}, whose "
            f"{stage_arrays_span}-byte combined scalar/array span does not fit the "
            f"{RECORD_SIZE}-byte record"
        )

    scalars = MapcutScalars(
        numero_iteracoes=n_iteracoes,
        numero_cortes=n_cortes,
        numero_submercados=n_submercados,
        numero_uhes=n_uhes,
        numero_cenarios=n_cenarios,
        tamanho_corte=tamanho_corte,
        dia=dia,
        mes=mes,
        ano=ano,
        numero_estagios=n_estagios,
        numero_semanas=n_semanas,
        n_utv=n_utv,
        max_lag=max_lag,
    )

    expected = mapcut_record_count(n_utv, n_estagios, max_lag, n_cenarios)
    if records != expected:
        raise ReadError(
            f"{path} holds {records} records but its own header implies {expected} "
            f"(n_utv={n_utv}, n_estagios={n_estagios}, max_lag={max_lag}, n_cenarios={n_cenarios})"
        )

    # Two different quantities, easy to conflate: reg 6 carries one lag per
    # (plant, stage), while regs 7/8 occupy one RECORD per (plant, stage, lag).
    lag_count = n_utv * n_estagios
    travel_records = lag_count * (max_lag + 1)
    first_gnl = _FIRST_TRAVEL_RECORD + travel_records
    gnl_indices = tuple(
        [first_gnl + 2 * stage for stage in range(n_estagios)]
        + [first_gnl + 2 * n_estagios + offset for offset in range(n_cenarios - n_estagios)]
    )
    cost_indices = tuple(first_gnl + 2 * stage + 1 for stage in range(n_estagios))

    ngnl = _ints(raw, gnl_indices[0], 1)[0]
    if not 0 < ngnl <= n_submercados:
        raise ReadError(
            f"{path}: reg 9 declares ngnl = {ngnl}, which is not a plausible submarket count "
            f"against reg 1's {n_submercados}"
        )
    # ngnl is bounded above by n_submercados, but n_submercados itself is only
    # checked > 0 (never entering mapcut_record_count), so a corrupted reg 1
    # can still let ngnl's own int32 header block overflow the record. Named
    # here rather than left to numpy's unnamed "buffer is smaller than
    # requested size", the same standard `cortdeco_writer._validate_lengths`
    # holds its own record capacity to.
    head_count = 1 + _GNL_ARRAYS * ngnl
    head_span = head_count * 4
    if head_span > RECORD_SIZE:
        raise ReadError(
            f"{path}: reg 9 declares ngnl = {ngnl}, whose {head_count}-int32 header block "
            f"({head_span} bytes) does not fit the {RECORD_SIZE}-byte record"
        )
    gnl_head = _ints(raw, gnl_indices[0], head_count)
    # The widest stage, so a ragged deck is still decoded rather than refused: this
    # is a reader, and `assert_uniform_blocks` is where a ragged deck is rejected
    # for writing. Uniformly 3 in the reference.
    n_patamares = max(_ints(raw, _REG_STAGE, n_estagios, offset=(5 + n_estagios) * 4))
    trailing_offset = head_span
    trailing_count = ngnl * n_patamares
    trailing_span = trailing_count * 8
    if trailing_offset + trailing_span > RECORD_SIZE:
        raise ReadError(
            f"{path}: reg 9's trailing block for ngnl={ngnl} and n_patamares={n_patamares} "
            f"needs {trailing_count} float64 ({trailing_span} bytes) at offset {trailing_offset}, "
            f"which does not fit the {RECORD_SIZE}-byte record"
        )
    gnl_trailing = _floats(raw, gnl_indices[0], trailing_count, offset=trailing_offset)

    _assert_gnl_records_identical(raw, gnl_indices, path)

    return MapcutContents(
        record_count=records,
        scalars=scalars,
        cut_heads=_ints(raw, _REG_GENERAL, n_cenarios, offset=5 * 4),
        codigos_uhes=_ints(raw, _REG_HYDRO_CODES, n_uhes),
        codigos_uhes_jusante=_ints(raw, _REG_DOWNSTREAM, n_uhes),
        indice_no_arvore=_ints(raw, _REG_TREE, n_cenarios),
        indice_primeiro_no_estagio=_ints(raw, _REG_STAGE, n_estagios, offset=5 * 4),
        patamares_por_estagio=_ints(raw, _REG_STAGE, n_estagios, offset=(5 + n_estagios) * 4),
        travel_time_lags=_ints(raw, _REG_STAGE, lag_count, offset=(5 + 2 * n_estagios) * 4),
        physical_span_nonzero_records=_physical_span_nonzero(raw),
        codigos_submercados_gnl=gnl_head[1 : 1 + ngnl],
        lag_meses_gnl=gnl_head[1 + ngnl : 1 + 2 * ngnl],
        patamares_gnl=gnl_head[1 + 2 * ngnl : 1 + 3 * ngnl],
        gnl_trailing_block=gnl_trailing,
        gnl_record_indices=gnl_indices,
        cost_records=tuple(_floats(raw, index, _COST_FIELDS_PER_STAGE) for index in cost_indices),
    )


def _physical_span_nonzero(raw: bytes) -> tuple[int, ...]:
    """Which of records 4-17 carry a non-zero byte (premise P1 is a claim about these)."""
    return tuple(
        index
        for index in range(PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST + 1)
        if _record(raw, index).strip(b"\x00")
    )


def _assert_gnl_records_identical(raw: bytes, indices: tuple[int, ...], path: Path) -> None:
    """Verify the reg-9 records really are identical rather than assuming it.

    The content CSV emits one row group with a repeat count instead of 273
    near-identical ones, and that compression is only honest if the identity
    holds. It does in both the reference deck and this project's output, but a
    reader that assumed it would hide the one case that matters.
    """
    first = _record(raw, indices[0])
    for index in indices[1:]:
        if _record(raw, index) != first:
            raise ReadError(
                f"{path}: reg 9 record {index} differs from record {indices[0]}, but every reg 9 "
                f"record carries the same GNL configuration and must be byte-identical"
            )


def read_cortdeco(path: Path, offsets: CutBlockOffsets, n_nodes: int) -> CortdecoContents:
    """Decode a `cortdeco`. `NCOEF` is not in the file, so `offsets` must be supplied.

    `numero_cortes` **is** derivable — it is the record count less CEPEL's one
    extra record (premise P13) — so it is recovered rather than taken.
    """
    if n_nodes <= 0:
        raise ReadError(f"n_nodes must be positive, got {n_nodes}")
    raw, records = _read_whole(path, TAMANHO_CORTE, "cortdeco")

    numero_cortes = records - 1
    if numero_cortes % n_nodes:
        raise ReadError(
            f"{path} holds {records} records, implying numero_cortes {numero_cortes}, which is not "
            f"divisible by {n_nodes} cut-building nodes"
        )

    width_varm = offsets.pi_gnl - offsets.pi_varm
    width_gnl = offsets.ncoef - offsets.pi_gnl
    if 4 + 8 * offsets.ncoef > TAMANHO_CORTE:
        raise ReadError(f"NCOEF {offsets.ncoef} does not fit a {TAMANHO_CORTE}-byte record")

    cuts = tuple(
        CortdecoCut(
            record_index=index,
            next_index=int(
                np.frombuffer(raw, dtype="<i4", count=1, offset=index * TAMANHO_CORTE)[0]
            ),
            rhs=float(
                np.frombuffer(
                    raw, dtype="<f8", count=1, offset=index * TAMANHO_CORTE + 4 + 8 * offsets.rhs
                )[0]
            ),
            pi_varm=tuple(
                float(v)
                for v in np.frombuffer(
                    raw,
                    dtype="<f8",
                    count=width_varm,
                    offset=index * TAMANHO_CORTE + 4 + 8 * offsets.pi_varm,
                )
            ),
            pi_gnl=tuple(
                float(v)
                for v in np.frombuffer(
                    raw,
                    dtype="<f8",
                    count=width_gnl,
                    offset=index * TAMANHO_CORTE + 4 + 8 * offsets.pi_gnl,
                )
            ),
            is_extra_record=index == numero_cortes,
        )
        for index in range(records)
    )

    _logger.info(
        "read cortdeco %s records=%d numero_cortes=%d ncoef=%d",
        path,
        records,
        numero_cortes,
        offsets.ncoef,
    )
    return CortdecoContents(record_count=records, numero_cortes=numero_cortes, cuts=cuts)
