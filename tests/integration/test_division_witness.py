"""Witness premise P2's division across the two published CSV families.

`reporting/content_csv.py` promises its post-division values are checked against
`reporting/eco_csv.py`'s pre-division ones. Nothing asserted that until now.
`ticket-017` proved the reference deck's magnitude envelopes cannot substitute
for this comparison, so this module is the whole-scale, exact-equality witness
instead of a band.

It reads only the two published CSV families a real run leaves on disk, through
`converted_pair`. It imports no writer, no mapping rule, and none of
`decomp/reader.py`'s cut decoding, so the comparison stays independent of every
code path that produced either file.

The comparison is exact and one-directional: `content == eco / 1000.0` holds for
every measured `rhs` and `pi_varm` value, while `content * 1000.0 == eco` does
not (a float64 division is not exactly invertible). Write the comparison the
first way; `test_the_inverted_direction_...` below exists so the second way is
never rediscovered as a bug.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from conversor_fcf.paths import OutputPaths

REPO_ROOT = Path(__file__).resolve().parents[2]
HYDRO_CODES_PATH = REPO_ROOT / "decomp_hydro_codes.json"

MAPCUT_CONTENT_CSV_NAME = "mapcut_content.csv"
CORTDECO_CONTENT_CSV_NAME = "cortdeco_content.csv"

_POOL_FILE_PATTERN = re.compile(r"^eco_cuts_pool_(\d+)\.csv$")
_NON_RHS_ONWARD_COLUMNS = ("record_index", "next_index", "is_extra_record")

# Every number below was measured against one real run of the reference Cobre
# case during ticket-020's refinement.
EXPECTED_MATCHED_RECORDS = 288
EXPECTED_PI_VARM_COMPARISONS = 48_672
UNMATCHED_RECORD_INDEX = 288
REPEATED_RECORD_INDEX = 282
MISSED_DIVISION_MINIMUM_PI_VARM_FAILURES = 48_000
INVERTED_DIRECTION_PI_VARM_MISMATCHES = 643


@dataclass(frozen=True)
class MappedRow:
    """One ECO row, located at the content `record_index` it must equal (÷1000)."""

    record_index: int
    pool_id: int
    rank: int
    eco_row: dict[str, str]


def _read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Every data row, keyed by header name, values held as strings.

    Both writers emit `repr(float(value))`, so the round trip through `float()`
    is exact; nothing here parses a value early.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return tuple(dict(zip(header, row, strict=True)) for row in reader)


@pytest.fixture(scope="module")
def eco_pool_files(converted_pair: OutputPaths) -> tuple[tuple[int, Path], ...]:
    """Every `eco_cuts_pool_<id>.csv` the run published, ascending by pool id."""
    eco_dir = converted_pair.eco_dir
    found: list[tuple[int, Path]] = []
    for path in eco_dir.glob("eco_cuts_pool_*.csv"):
        match = _POOL_FILE_PATTERN.match(path.name)
        if match is None:
            raise AssertionError(f"{path} does not match the eco_cuts_pool_<id>.csv naming")
        found.append((int(match.group(1)), path))
    if not found:
        raise AssertionError(f"no eco_cuts_pool_*.csv files found under {eco_dir}")
    return tuple(sorted(found, key=lambda item: item[0]))


@pytest.fixture(scope="module")
def mapcut_content_rows(converted_pair: OutputPaths) -> tuple[dict[str, str], ...]:
    return _read_csv_rows(converted_pair.content_dir / MAPCUT_CONTENT_CSV_NAME)


@pytest.fixture(scope="module")
def n_nodes(mapcut_content_rows: tuple[dict[str, str], ...]) -> int:
    """`numero_semanas` from the published `mapcut`, which doubles as the
    cut-building-node count (this project's own convention, recorded in
    `CLAUDE.md`)."""
    matches = [
        row["value"]
        for row in mapcut_content_rows
        if row["record_kind"] == "reg6_stage" and row["field"] == "numero_semanas"
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one reg6_stage/numero_semanas row in "
            f"{MAPCUT_CONTENT_CSV_NAME}, found {len(matches)}"
        )
    return int(matches[0])


@pytest.fixture(scope="module")
def content_rows(converted_pair: OutputPaths) -> dict[int, dict[str, str]]:
    rows = _read_csv_rows(converted_pair.content_dir / CORTDECO_CONTENT_CSV_NAME)
    return {int(row["record_index"]): row for row in rows}


@pytest.fixture(scope="module")
def mapped_rows(
    eco_pool_files: tuple[tuple[int, Path], ...], n_nodes: int
) -> tuple[MappedRow, ...]:
    """Every ECO row, at the content `record_index` it must equal, divided by 1000
    (Requirement 2).

    `node_index` is the position of a file's pool id among the ascending pool
    ids found, not the pool id itself: `assemble_cut_inputs` orders nodes by
    that position, which coincides with the pool id on this deck and need not
    on another. `rank` is the 0-based position of a row among its own file's
    rows sorted ascending by `iteration`, not the `iteration` value, which runs
    1-48 in the reference case, not 0-47.
    """
    mapped: list[MappedRow] = []
    for node_index, (pool_id, path) in enumerate(eco_pool_files):
        ranked = sorted(_read_csv_rows(path), key=lambda row: int(row["iteration"]))
        for rank, row in enumerate(ranked):
            record_index = n_nodes * rank + (n_nodes - 1) - node_index
            mapped.append(MappedRow(record_index, pool_id, rank, row))
    return tuple(mapped)


@pytest.fixture(scope="module")
def hydro_codes() -> tuple[int, ...]:
    """`decomp_hydro_codes.json`'s 169 codes, in index order."""
    payload = json.loads(HYDRO_CODES_PATH.read_text(encoding="utf-8"))
    return tuple(int(payload[str(index)]) for index in range(len(payload)))


# --- Requirements 2/3: the bijection, and its one deliberate exception ------


def test_the_bijection_maps_288_eco_rows_onto_content_records_0_through_287(
    mapped_rows: tuple[MappedRow, ...],
    content_rows: dict[int, dict[str, str]],
    eco_pool_files: tuple[tuple[int, Path], ...],
    n_nodes: int,
) -> None:
    """Checked before any number is compared (Requirement 2): a mapping defect
    and an arithmetic defect would otherwise produce the same failure, and the
    bijection is the cheaper one to diagnose.
    """
    assert n_nodes == len(eco_pool_files)
    assert len(mapped_rows) == EXPECTED_MATCHED_RECORDS

    indices = sorted(row.record_index for row in mapped_rows)
    assert indices == list(range(EXPECTED_MATCHED_RECORDS)), "the mapping is not a bijection"

    mapped_indices = {row.record_index for row in mapped_rows}
    unmatched = sorted(set(content_rows) - mapped_indices)
    assert unmatched == [UNMATCHED_RECORD_INDEX]

    extra = content_rows[UNMATCHED_RECORD_INDEX]
    repeated = content_rows[REPEATED_RECORD_INDEX]
    for column in extra:
        if column in _NON_RHS_ONWARD_COLUMNS:
            continue
        assert extra[column] == repeated[column], (
            f"record {UNMATCHED_RECORD_INDEX} column {column!r} = {extra[column]!r} disagrees "
            f"with record {REPEATED_RECORD_INDEX}'s {repeated[column]!r}"
        )


# --- Requirement 4's exact direction: content == eco / 1000.0 --------------


def test_every_rhs_equals_the_matching_eco_intercept_divided_by_one_thousand(
    mapped_rows: tuple[MappedRow, ...], content_rows: dict[int, dict[str, str]]
) -> None:
    for row in mapped_rows:
        content = content_rows[row.record_index]
        expected = float(row.eco_row["intercept"]) / 1000.0
        actual = float(content["rhs"])
        assert actual == expected, (
            f"record_index={row.record_index} pool_id={row.pool_id} rank={row.rank} "
            f"column=rhs: content {actual!r} != eco intercept/1000 {expected!r}"
        )


def test_every_pi_varm_equals_the_matching_eco_storage_slot_divided_by_one_thousand(
    mapped_rows: tuple[MappedRow, ...],
    content_rows: dict[int, dict[str, str]],
    hydro_codes: tuple[int, ...],
) -> None:
    compared = 0
    for row in mapped_rows:
        content = content_rows[row.record_index]
        for slot, code in enumerate(hydro_codes):
            column = f"pi_varm_uhe{code}"
            expected = float(row.eco_row[f"storage_h{slot}"]) / 1000.0
            actual = float(content[column])
            assert actual == expected, (
                f"record_index={row.record_index} pool_id={row.pool_id} rank={row.rank} "
                f"column={column}: content {actual!r} != eco storage_h{slot}/1000 {expected!r}"
            )
            compared += 1
    assert compared == EXPECTED_PI_VARM_COMPARISONS


# --- Requirement 5: the missed-division mutation is not vacuous ------------


def test_a_missed_division_mutation_fails_every_rhs_and_most_pi_varm_comparisons(
    mapped_rows: tuple[MappedRow, ...],
    content_rows: dict[int, dict[str, str]],
    hydro_codes: tuple[int, ...],
) -> None:
    """Multiplies parsed `rhs`/`pi_varm` values by 1000 in memory, simulating a
    ÷1000 that never ran, then re-runs the same `content == eco / 1000.0`
    comparison. Only the floats already parsed into `mapped_rows`/`content_rows`
    are scaled; nothing on disk is touched.
    """
    rhs_failures = 0
    pi_varm_failures = 0
    pi_varm_total = 0
    for row in mapped_rows:
        content = content_rows[row.record_index]
        mutated_rhs = float(content["rhs"]) * 1000.0
        expected_rhs = float(row.eco_row["intercept"]) / 1000.0
        if mutated_rhs != expected_rhs:
            rhs_failures += 1
        for slot, code in enumerate(hydro_codes):
            mutated_pi_varm = float(content[f"pi_varm_uhe{code}"]) * 1000.0
            expected_pi_varm = float(row.eco_row[f"storage_h{slot}"]) / 1000.0
            pi_varm_total += 1
            if mutated_pi_varm != expected_pi_varm:
                pi_varm_failures += 1

    assert rhs_failures == EXPECTED_MATCHED_RECORDS
    assert pi_varm_total == EXPECTED_PI_VARM_COMPARISONS
    assert pi_varm_failures >= MISSED_DIVISION_MINIMUM_PI_VARM_FAILURES


# --- Requirement 4's forbidden direction: content * 1000.0 == eco ----------


def test_the_inverted_direction_content_times_1000_is_not_exact_so_compare_as_eco_over_1000(
    mapped_rows: tuple[MappedRow, ...],
    content_rows: dict[int, dict[str, str]],
    hydro_codes: tuple[int, ...],
) -> None:
    """`content * 1000.0 == eco` disagrees on 643 of 48,672 `pi_varm` pairs in
    this checkpoint, because a float64 division is not exactly invertible.
    Every comparison in this module therefore reads `content == eco / 1000.0`;
    the two `test_every_*_equals_the_matching_eco_*` tests above do exactly
    that, and never this inverted form.
    """
    mismatches = 0
    for row in mapped_rows:
        content = content_rows[row.record_index]
        for slot, code in enumerate(hydro_codes):
            content_value = float(content[f"pi_varm_uhe{code}"])
            eco_value = float(row.eco_row[f"storage_h{slot}"])
            if content_value * 1000.0 != eco_value:
                mismatches += 1
    assert mismatches == INVERTED_DIRECTION_PI_VARM_MISMATCHES
