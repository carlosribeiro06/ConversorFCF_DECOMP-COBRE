"""Native serializer for the DECOMP `mapcut` binary.

`idecomp.Mapcut.write` raises `NotImplementedError`, so every byte here is
written natively and `idecomp` is only ever a read-back oracle.

Field order within each record follows `idecomp`'s own section definitions,
which are the authority on what a readable file looks like:

- reg 1: `[numero_iteracoes, numero_cortes, numero_submercados, numero_uhes,
  numero_cenarios]` then `numero_cenarios` cut-head indices, all int32.
- reg 2: `[tamanho_corte, dia, mes, ano]`, int32.
- reg 3: `codigos_uhes`, int32.
- reg 4: `codigos_uhes_jusante`, int32 **on disk** even though `idecomp` reads
  it back as float32; a read-back oracle needs `.view(np.int32)`.
- reg 5: `indice_no_arvore`, int32.
- reg 6: `[flag, n_estagios, n_semanas, n_utv, max_lag]`, then
  `indice_primeiro_no_estagio`, then `patamares_por_estagio`, then the
  `n_utv * n_estagios` travel-time lags, all int32.
- reg 9: `[ngnl]`, then three `ngnl`-length int32 arrays (submarket, lag index,
  block count), then a trailing float64 block whose **axis is unsettled**. The
  reference populates exactly `ngnl * npat` values — the GNL lag month's hours
  per load block, three per submarket summing to 730.5 — and its first zero sits
  at index `ngnl * npat`, which argues for a block axis. `idecomp` is internally
  inconsistent here: `dados_gnl` strides by `sum(patamares)` while
  `__le_nono_registro` reads `n_estagios * ngnl`. This writer emits zeros under
  premise P12, so the width it declares is byte-neutral today; whoever populates
  the block must settle the axis first, because laying a stage-indexed vector
  into a block-indexed slot produces a wrong file that still reads.
- reg 10: six float64, `taxa_desconto` first.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from conversor_fcf.decomp.layout import (
    PHYSICAL_RECORD_COUNT,
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    TAMANHO_CORTE,
    LayoutError,
    assert_mapcut_layout,
    mapcut_record_count,
)
from conversor_fcf.logging_setup import get_logger

# Reg 6's first field is 1 in the reference file and its meaning is not
# documented anywhere available; it is reproduced verbatim rather than guessed at.
STAGE_RECORD_LEADING_FLAG = 1

_COST_FIELDS_PER_STAGE = 6

_logger = get_logger("mapcut_writer")

_Segment = tuple[Literal["<i4", "<f8"], Sequence[int] | Sequence[float]]


@dataclass(frozen=True)
class MapcutHeader:
    """Every value the `mapcut` records need, so the writer is a pure function.

    `stage_id` semantics deliberately do not appear here: this is DECOMP's own
    vocabulary, assembled by the pipeline from the Cobre side.
    """

    numero_iteracoes: int
    numero_cortes: int
    numero_submercados: int
    numero_uhes: int
    numero_cenarios: int
    numero_estagios: int
    numero_semanas: int
    n_utv: int
    dia: int
    mes: int
    ano: int
    codigos_uhes: tuple[int, ...]
    codigos_uhes_jusante: tuple[int, ...]
    indice_no_arvore: tuple[int, ...]
    indice_primeiro_no_estagio: tuple[int, ...]
    patamares_por_estagio: tuple[int, ...]
    registro_ultimo_corte_no: tuple[int, ...]
    codigos_submercados_gnl: tuple[int, ...]
    lag_meses_gnl: tuple[int, ...]
    patamares_gnl: tuple[int, ...]
    taxa_desconto: tuple[float, ...]

    # Last and defaulted because it only ever multiplies `n_utv` in the record
    # count, so a case with no travel-time axis has no lag bound to declare. It
    # is still written into reg 6, hence the non-negative check.
    max_lag: int = 0


def _validate(header: MapcutHeader) -> None:
    checks = (
        ("codigos_uhes", len(header.codigos_uhes), "numero_uhes", header.numero_uhes),
        (
            "codigos_uhes_jusante",
            len(header.codigos_uhes_jusante),
            "numero_uhes",
            header.numero_uhes,
        ),
        (
            "indice_no_arvore",
            len(header.indice_no_arvore),
            "numero_cenarios",
            header.numero_cenarios,
        ),
        (
            "registro_ultimo_corte_no",
            len(header.registro_ultimo_corte_no),
            "numero_cenarios",
            header.numero_cenarios,
        ),
        (
            "indice_primeiro_no_estagio",
            len(header.indice_primeiro_no_estagio),
            "numero_estagios",
            header.numero_estagios,
        ),
        (
            "patamares_por_estagio",
            len(header.patamares_por_estagio),
            "numero_estagios",
            header.numero_estagios,
        ),
        (
            "taxa_desconto",
            len(header.taxa_desconto),
            "numero_estagios",
            header.numero_estagios,
        ),
        (
            "lag_meses_gnl",
            len(header.lag_meses_gnl),
            "codigos_submercados_gnl",
            len(header.codigos_submercados_gnl),
        ),
        (
            "patamares_gnl",
            len(header.patamares_gnl),
            "codigos_submercados_gnl",
            len(header.codigos_submercados_gnl),
        ),
    )
    for field, actual, against, expected in checks:
        if actual != expected:
            raise LayoutError(
                f"{field} has {actual} entries but {against} is {expected}; the header is "
                f"internally inconsistent and no byte was written"
            )
    if header.n_utv:
        raise LayoutError(
            f"n_utv is {header.n_utv}; mapcut regs 7/8 are unimplemented, so a case with a "
            f"travel-time axis cannot be written (premise P3)"
        )
    if header.max_lag < 0:
        raise LayoutError(
            f"max_lag is {header.max_lag}; reg 6 declares it as the travel-time lag bound, "
            f"which cannot be negative"
        )


def _record(index: int, *segments: _Segment) -> bytes:
    """One fixed-size record, zero-padded. Every record goes through here."""
    body = b"".join(np.asarray(values, dtype=dtype).tobytes() for dtype, values in segments)
    if len(body) > RECORD_SIZE:
        raise LayoutError(
            f"record {index} packs {len(body)} bytes, more than the {RECORD_SIZE}-byte record"
        )
    return body.ljust(RECORD_SIZE, b"\x00")


def _general_record(header: MapcutHeader) -> bytes:
    return _record(
        0,
        (
            "<i4",
            (
                header.numero_iteracoes,
                header.numero_cortes,
                header.numero_submercados,
                header.numero_uhes,
                header.numero_cenarios,
                *header.registro_ultimo_corte_no,
            ),
        ),
    )


def _case_record(header: MapcutHeader) -> bytes:
    return _record(1, ("<i4", (TAMANHO_CORTE, header.dia, header.mes, header.ano)))


def _stage_record(header: MapcutHeader) -> bytes:
    return _record(
        19,
        (
            "<i4",
            (
                STAGE_RECORD_LEADING_FLAG,
                header.numero_estagios,
                header.numero_semanas,
                header.n_utv,
                header.max_lag,
                *header.indice_primeiro_no_estagio,
                *header.patamares_por_estagio,
            ),
        ),
    )


def _gnl_record(header: MapcutHeader, index: int) -> bytes:
    """Reg 9. Identical for every node, because it carries configuration only.

    The trailing float64 block is zeros under premise P12. Its width follows the
    stage axis only because that is the wider of the two candidate readings, so
    the zeros cover either; see the module docstring.
    """
    ngnl = len(header.codigos_submercados_gnl)
    return _record(
        index,
        (
            "<i4",
            (
                ngnl,
                *header.codigos_submercados_gnl,
                *header.lag_meses_gnl,
                *header.patamares_gnl,
            ),
        ),
        ("<f8", [0.0] * (header.numero_estagios * ngnl)),
    )


def _cost_record(header: MapcutHeader, stage: int, index: int) -> bytes:
    """Reg 10. Only `taxa_desconto` is derivable from Cobre (premise P11)."""
    fields = [header.taxa_desconto[stage]] + [0.0] * (_COST_FIELDS_PER_STAGE - 1)
    return _record(index, ("<f8", fields))


def write_mapcut(header: MapcutHeader, path: Path) -> int:
    """Write a complete `mapcut`, returning the record count.

    The file is built at a temporary path in the destination directory, checked
    against its own layout invariant, and only then renamed into place, so an
    interrupted or invalid run cannot leave something that looks complete.
    """
    _validate(header)
    expected = mapcut_record_count(
        header.n_utv, header.numero_estagios, header.max_lag, header.numero_cenarios
    )

    records: list[bytes] = [
        _general_record(header),
        _case_record(header),
        _record(2, ("<i4", header.codigos_uhes)),
        # int32 on disk despite idecomp reading it back as float32.
        _record(3, ("<i4", header.codigos_uhes_jusante)),
    ]
    records.extend(b"\x00" * RECORD_SIZE for _ in range(PHYSICAL_RECORD_COUNT))
    records.append(_record(18, ("<i4", header.indice_no_arvore)))
    records.append(_stage_record(header))

    # Reg 9 and reg 10 interleave across the first n_estagios pairs, then reg 9
    # continues alone to one record per node.
    for stage in range(header.numero_estagios):
        records.append(_gnl_record(header, len(records)))
        records.append(_cost_record(header, stage, len(records)))
    for _ in range(header.numero_cenarios - header.numero_estagios):
        records.append(_gnl_record(header, len(records)))

    if len(records) != expected:
        raise LayoutError(
            f"assembled {len(records)} records but the layout invariant expects {expected}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_bytes(b"".join(records))
        assert_mapcut_layout(temporary, header)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    _logger.warning(
        "premise P1: mapcut records %d-%d emitted as zeros. The reference deck holds populated "
        "per-plant physical data there, so this file genuinely diverges from it; populating them "
        "is deferred to ticket-015",
        PHYSICAL_RECORD_FIRST,
        PHYSICAL_RECORD_LAST,
    )
    _logger.warning(
        "premise P11: reg 10 emits zeros for parcela_custo_geracao_termica_minima, "
        "parcela_custo_contrato_importacao_minimo, parcela_custo_contrato_exportacao_minimo, "
        "geracao_termica_minima_sinalizada_gnl and geracao_termica_minima_gerada_gnl. "
        "They are DECOMP operational quantities from the deck's own data and are not derivable "
        "from a Cobre policy checkpoint, whose cut intercept already embeds the constant term"
    )
    _logger.warning(
        "premise P12: reg 9's trailing float64 block emitted as zeros. The reference populates it "
        "with the GNL lag month's hours per load block, three values per submarket summing to "
        "730.5 = 365.25*24/12. Those values are not derivable from a Cobre case: the split is a "
        "monthly structure from the DECOMP deck, while Cobre supplies weekly stage blocks, and no "
        "aggregation of this case's blocks reproduces it. The block's axis is unsettled too "
        "(ngnl*npat on disk against idecomp's ngnl*n_estagios), so a wrongly-shaped hour vector "
        "would be worse than none"
    )
    _logger.info(
        "wrote mapcut %s records=%d bytes=%d numero_cortes=%d numero_cenarios=%d",
        path,
        expected,
        expected * RECORD_SIZE,
        header.numero_cortes,
        header.numero_cenarios,
    )
    return expected
