"""CSV mirrors of what the two DECOMP binaries actually contain.

These are the post-transformation audit artifacts, and their defining property is
that they are produced by **reading the written binary back** through
`decomp.reader`, never by re-serializing the records in memory. An artifact
generated from the values it was written from is self-consistent under any
serialization bug, which is exactly how epic-02's decode corruption stayed
invisible to a fully covered suite.

The values are the binary's own, **post-division** (premise P2). The ÷1000 stays
visible by comparing against `ticket-005`'s ECO CSVs, which hold Cobre's raw
values: the two artifacts together exhibit the transformation without either one
duplicating it. A pre-division column reconstructed here would not be an
independent witness - it is the same number multiplied back.

Floats are written through `repr(float(value))` for the reason `eco_csv.py`
records: `repr` of a `numpy.float64` yields `np.float64(...)` rather than a bare
literal, and `pandas.to_csv` truncates to six significant digits, either of which
would break the exact round trip these artifacts exist to provide.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path

from conversor_fcf.decomp.layout import PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST
from conversor_fcf.decomp.reader import CortdecoContents, MapcutContents
from conversor_fcf.logging_setup import get_logger

_logger = get_logger("content_csv")

MAPCUT_COLUMNS = ("record_index", "record_kind", "repeats", "field", "position", "value")

# Reg 10's six float64, in order. Only the first is derivable from a Cobre case;
# the other five are DECOMP operational quantities emitted as zeros under P11.
COST_FIELDS = (
    "taxa_desconto",
    "parcela_custo_geracao_termica_minima",
    "parcela_custo_contrato_importacao_minimo",
    "parcela_custo_contrato_exportacao_minimo",
    "geracao_termica_minima_sinalizada_gnl",
    "geracao_termica_minima_gerada_gnl",
)

_REG_ONE_SCALARS = (
    "numero_iteracoes",
    "numero_cortes",
    "numero_submercados",
    "numero_uhes",
    "numero_cenarios",
)
_REG_TWO_SCALARS = ("tamanho_corte", "dia", "mes", "ano")
_REG_SIX_SCALARS = ("flag", "numero_estagios", "numero_semanas", "n_utv", "max_lag")


def content_csv_path(output_dir: Path, content_subdir: str, name: str) -> Path:
    """Where a content CSV lives, so the pipeline and the tests cannot disagree."""
    return output_dir / content_subdir / name


def _text(value: object) -> str:
    return repr(float(value)) if isinstance(value, float) else str(value)


def _mapcut_rows(contents: MapcutContents) -> Iterator[tuple[object, ...]]:
    scalars = contents.scalars

    for position, name in enumerate(_REG_ONE_SCALARS):
        yield (0, "reg1_general", 1, name, position, getattr(scalars, name))
    for position, head in enumerate(contents.cut_heads):
        yield (0, "reg1_general", 1, "registro_ultimo_corte_no", position, head)

    for position, name in enumerate(_REG_TWO_SCALARS):
        yield (1, "reg2_case", 1, name, position, getattr(scalars, name))

    for position, code in enumerate(contents.codigos_uhes):
        yield (2, "reg3_hydro_codes", 1, "codigos_uhes", position, code)
    for position, code in enumerate(contents.codigos_uhes_jusante):
        yield (3, "reg4_downstream", 1, "codigos_uhes_jusante", position, code)

    # Premise P1's span appears as rows rather than as an absence, so the
    # divergence is visible in the artifact instead of having to be inferred.
    nonzero = set(contents.physical_span_nonzero_records)
    for index in range(PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST + 1):
        yield (index, "physical_p1", 1, "nonzero_bytes_found", 0, 1 if index in nonzero else 0)

    tree_record = 18
    for position, parent in enumerate(contents.indice_no_arvore):
        yield (tree_record, "reg5_tree", 1, "indice_no_arvore", position, parent)

    stage_record = 19
    reg_six_values = (
        1,
        scalars.numero_estagios,
        scalars.numero_semanas,
        scalars.n_utv,
        scalars.max_lag,
    )
    reg_six = zip(_REG_SIX_SCALARS, reg_six_values, strict=True)
    for position, (name, scalar) in enumerate(reg_six):
        yield (stage_record, "reg6_stage", 1, name, position, scalar)
    for position, first_node in enumerate(contents.indice_primeiro_no_estagio):
        yield (stage_record, "reg6_stage", 1, "indice_primeiro_no_estagio", position, first_node)
    for position, blocks in enumerate(contents.patamares_por_estagio):
        yield (stage_record, "reg6_stage", 1, "patamares_por_estagio", position, blocks)
    for position, lag in enumerate(contents.travel_time_lags):
        yield (stage_record, "reg6_stage", 1, "lag_tempo_viagem", position, lag)

    # Reg 9 once, with its repeat count. Every reg-9 record carries the same GNL
    # configuration and `read_mapcut` has already verified they are byte-identical,
    # so 273 near-identical row groups would bury the seven reg-10 records that
    # actually differ.
    gnl_record = contents.gnl_record_indices[0]
    repeats = len(contents.gnl_record_indices)
    yield (gnl_record, "reg9_gnl", repeats, "ngnl", 0, len(contents.codigos_submercados_gnl))
    gnl_blocks: tuple[tuple[str, Sequence[float]], ...] = (
        ("codigos_submercados_gnl", contents.codigos_submercados_gnl),
        ("lag_meses_gnl", contents.lag_meses_gnl),
        ("patamares_gnl", contents.patamares_gnl),
        ("gnl_trailing_block", contents.gnl_trailing_block),
    )
    for field, values in gnl_blocks:
        for position, entry in enumerate(values):
            yield (gnl_record, "reg9_gnl", repeats, field, position, entry)

    for stage, record in enumerate(contents.cost_records):
        record_index = contents.gnl_record_indices[stage] + 1
        for position, (name, amount) in enumerate(zip(COST_FIELDS, record, strict=True)):
            yield (record_index, "reg10_cost", 1, name, position, amount)


def write_mapcut_content_csv(contents: MapcutContents, path: Path) -> int:
    """Write the long-format `mapcut` mirror, returning the row count.

    Long format rather than one CSV per record family: fourteen near-empty files
    for the zero span alone would be worse than one table a reader can filter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MAPCUT_COLUMNS)
        for row in _mapcut_rows(contents):
            writer.writerow([_text(value) for value in row])
            rows += 1
    _logger.info("wrote %s rows=%d columns=%d", path, rows, len(MAPCUT_COLUMNS))
    return rows


def cortdeco_columns(
    hydro_codes: Sequence[int],
    submarket_codes: Sequence[int],
    n_estagios: int,
    n_patamares: int,
    n_utv: int = 0,
    max_lag: int = 0,
) -> tuple[str, ...]:
    """Column names mirroring `idecomp`'s own, so the CSV is directly comparable.

    `pi_varm` columns are named by **DECOMP plant code**, not by position, which
    is what `decomp_hydro_codes.json` exists for. The `pi_gnl` order follows the
    block's submarket-major layout: submarket, then stage, then load block.

    `n_utv` and `max_lag` default to zero because premise P3 holds for every case
    this converter accepts — `assert_no_travel_time` refuses one that carries
    `HydroTransitBucket` state — so the `pi_qdefp` block is absent from its own
    artifacts. They are parameters rather than constants so a **foreign** deck can
    still be mirrored, which is what makes the reference deck a usable comparison:
    it declares `n_utv = 2, max_lag = 3`, giving a 175-wide `pi_varm` span where
    this project's is 169. Those columns are named positionally, since which
    plants own the travel-time axes is not recoverable from `cortdeco` alone.
    """
    varm = tuple(f"pi_varm_uhe{code}" for code in hydro_codes)
    qdefp = tuple(
        f"pi_qdefp_utv{index + 1}_lag{lag + 1}" for index in range(n_utv) for lag in range(max_lag)
    )
    gnl = tuple(
        f"pi_gnl_sbm{code}_pat{block + 1}_lag{stage + 1}"
        for code in submarket_codes
        for stage in range(n_estagios)
        for block in range(n_patamares)
    )
    return ("record_index", "next_index", "is_extra_record", "rhs", *varm, *qdefp, *gnl)


def write_cortdeco_content_csv(
    contents: CortdecoContents,
    path: Path,
    hydro_codes: Sequence[int],
    submarket_codes: Sequence[int],
    n_estagios: int,
    n_patamares: int,
    n_utv: int = 0,
    max_lag: int = 0,
) -> int:
    """Write the wide `cortdeco` mirror, returning the row count."""
    columns = cortdeco_columns(
        hydro_codes, submarket_codes, n_estagios, n_patamares, n_utv, max_lag
    )
    first = contents.cuts[0]
    width = 4 + len(first.pi_varm) + len(first.pi_gnl)
    if len(columns) != width:
        raise ValueError(
            f"{len(columns)} column names for a record carrying {width} fields: "
            f"{len(hydro_codes)} hydro codes and {len(submarket_codes)} GNL submarkets do not "
            f"describe {len(first.pi_varm)} pi_varm and {len(first.pi_gnl)} pi_gnl values"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for cut in contents.cuts:
            writer.writerow(
                [
                    str(cut.record_index),
                    str(cut.next_index),
                    "true" if cut.is_extra_record else "false",
                    _text(cut.rhs),
                    *(_text(value) for value in cut.pi_varm),
                    *(_text(value) for value in cut.pi_gnl),
                ]
            )
    rows = len(contents.cuts)
    _logger.info("wrote %s rows=%d columns=%d", path, rows, len(columns))
    return rows
