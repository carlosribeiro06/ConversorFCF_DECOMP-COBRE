import gzip
import logging
from pathlib import Path

import numpy as np
import pytest

from conversor_fcf.cobre.policy_reader import (
    AffinePieceRecord,
    EntitySlotRecord,
    StageCutPool,
)
from conversor_fcf.reporting.eco_csv import (
    FIXED_COLUMNS,
    eco_csv_gzip_path,
    eco_csv_path,
    emit_eco_csvs,
    select_eco_pools,
    write_eco_csv,
    write_eco_csv_gzip,
)

# Real magnitudes from the reference deck, so the round-trip assertion is not
# exercised on conveniently short decimals.
INTERCEPT = 2312687234790.0674
COEFFICIENTS = (-5.028946e7, 3.181296e5)


def _piece(piece_id: int, coefficient_count: int = 2, is_active: bool = True) -> AffinePieceRecord:
    return AffinePieceRecord(
        piece_id=piece_id,
        slot_index=piece_id,
        iteration=3,
        forward_pass_index=0,
        intercept=INTERCEPT,
        coefficients=np.array(COEFFICIENTS[:coefficient_count], dtype=np.float64),
        is_active=is_active,
    )


def _pool(
    pieces: tuple[AffinePieceRecord, ...] | None = None,
    active_cut_indices: tuple[int, ...] = (0,),
) -> StageCutPool:
    slots = (
        EntitySlotRecord(
            entity_type=0, entity_id=0, subindex=0, was_active=True, delivery_date=-2147483648
        ),
        EntitySlotRecord(
            entity_type=2, entity_id=112, subindex=1, was_active=True, delivery_date=20260425
        ),
    )
    return StageCutPool(
        stage_id=4,
        node_id=4,
        graph_stage_id=4,
        state_dimension=2211,
        capacity=100,
        warm_start_count=0,
        populated_count=2,
        cost_scale_factor=1000000.0,
        slots=slots,
        pieces=(_piece(0), _piece(1, is_active=False)) if pieces is None else pieces,
        active_cut_indices=active_cut_indices,
    )


def _rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text, "line endings must be bare newlines"
    return [line.split(",") for line in text.splitlines()]


def test_header_is_the_nine_fixed_columns_plus_one_per_slot(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(_pool(), path)
    header = _rows(path)[0]
    assert len(header) == len(FIXED_COLUMNS) + 2
    assert header[: len(FIXED_COLUMNS)] == list(FIXED_COLUMNS)
    assert header[9:] == ["storage_h0", "anticipated_thermal_t112_s1"]


def test_row_count_is_returned_and_matches_the_body(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    assert write_eco_csv(_pool(), path) == 2
    assert len(_rows(path)) == 3


def test_floats_round_trip_exactly(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(_pool(), path)
    header, first, _ = _rows(path)
    intercept_column = header.index("intercept")
    assert first[intercept_column] == repr(INTERCEPT)
    assert float(first[intercept_column]) == INTERCEPT
    assert [float(value) for value in first[intercept_column + 1 :]] == list(COEFFICIENTS)


def test_coefficients_are_not_rendered_as_numpy_repr(tmp_path: Path) -> None:
    """`repr` of a numpy scalar would embed `np.float64(...)` into the CSV."""
    path = tmp_path / "eco.csv"
    write_eco_csv(_pool(), path)
    assert "np.float64" not in path.read_text(encoding="utf-8")


def test_values_are_written_verbatim_with_no_scaling(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(_pool(), path)
    first = _rows(path)[1]
    # cost_scale_factor is 1e6 and premise P2's divisor is 1000; neither may appear here.
    assert float(first[8]) == INTERCEPT
    assert float(first[9]) == COEFFICIENTS[0]


def test_active_flags_are_recorded_and_never_filter_rows(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    rows_written = write_eco_csv(_pool(active_cut_indices=(0,)), path)
    header, first, second = _rows(path)
    assert rows_written == 2, "premise P10: every populated piece is emitted"
    assert header[6:8] == ["is_active", "in_active_cut_indices"]
    assert (first[6], first[7]) == ("True", "True")
    assert (second[6], second[7]) == ("False", "False")


def test_in_active_cut_indices_reflects_the_positional_index(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(_pool(active_cut_indices=(1,)), path)
    _, first, second = _rows(path)
    assert first[7] == "False"
    assert second[7] == "True"


def test_coefficient_length_mismatch_names_the_piece_and_both_lengths(tmp_path: Path) -> None:
    pool = _pool(pieces=(_piece(0), _piece(1, coefficient_count=1)))
    with pytest.raises(ValueError) as excinfo:
        write_eco_csv(pool, tmp_path / "eco.csv")
    message = str(excinfo.value)
    assert "piece index 1" in message
    assert "1 coefficients" in message
    assert "2 slots" in message


def test_gzip_and_plain_writers_agree(tmp_path: Path) -> None:
    pool = _pool()
    plain = tmp_path / "eco.csv"
    compressed = tmp_path / "eco.csv.gz"
    assert write_eco_csv(pool, plain) == write_eco_csv_gzip(pool, compressed)
    with gzip.open(compressed, "rt", newline="", encoding="utf-8") as handle:
        assert handle.read() == plain.read_text(encoding="utf-8")


def test_parent_directories_are_created(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "eco.csv"
    write_eco_csv(_pool(), path)
    assert path.is_file()


def test_path_helpers_zero_pad_the_pool_id_and_add_the_gzip_suffix(tmp_path: Path) -> None:
    plain = eco_csv_path(tmp_path, "eco", 6)
    assert plain == tmp_path / "eco" / "eco_cuts_pool_006.csv"
    assert eco_csv_gzip_path(tmp_path, "eco", 6) == tmp_path / "eco" / "eco_cuts_pool_006.csv.gz"


@pytest.mark.parametrize(
    ("include_terminal", "expected"),
    [(False, (0, 1, 2, 3, 4, 5)), (True, (0, 1, 2, 3, 4, 5, 6))],
)
def test_select_eco_pools_honours_the_terminal_flag(
    include_terminal: bool, expected: tuple[int, ...]
) -> None:
    assert select_eco_pools((0, 1, 2, 3, 4, 5, 6), 6, include_terminal) == expected


def test_select_eco_pools_sorts_ascending() -> None:
    assert select_eco_pools((5, 0, 3), 6, False) == (0, 3, 5)


def test_select_eco_pools_rejects_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        select_eco_pools((), 6, False)


def test_emit_writes_plain_csvs_and_skips_the_terminal_pool(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pools = {pool_id: _pool() for pool_id in range(6)}
    with caplog.at_level(logging.INFO, logger="conversor_fcf"):
        written = emit_eco_csvs(
            pools, range(7), tmp_path, "eco", terminal_pool_id=6, include_terminal=False
        )

    assert sorted(written) == [0, 1, 2, 3, 4, 5]
    assert all(not str(path).endswith(".gz") for path in written.values())
    assert all(path.is_file() for path in written.values())
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "skipping terminal pool 6" in messages


def test_emit_selects_before_indexing_so_an_unloaded_terminal_pool_is_fine(
    tmp_path: Path,
) -> None:
    """A caller that never read pool 6 is valid when the terminal pool is excluded."""
    pools = {pool_id: _pool() for pool_id in range(6)}
    written = emit_eco_csvs(
        pools, range(7), tmp_path, "eco", terminal_pool_id=6, include_terminal=False
    )
    assert 6 not in written


def test_emit_gzips_only_the_terminal_pool(tmp_path: Path) -> None:
    pools = {0: _pool(), 6: _pool()}
    written = emit_eco_csvs(
        pools, (0, 6), tmp_path, "eco", terminal_pool_id=6, include_terminal=True
    )
    assert written[0].suffix == ".csv"
    assert written[6].suffix == ".gz"
    assert written[6].is_file()


def test_emit_raises_key_error_when_a_declared_trunk_pool_is_missing(tmp_path: Path) -> None:
    """Silent under-delivery is the failure mode that matters for an audit artifact."""
    pools = {0: _pool(), 1: _pool(), 3: _pool()}
    with pytest.raises(KeyError):
        emit_eco_csvs(pools, range(4), tmp_path, "eco", terminal_pool_id=6, include_terminal=False)


def test_emit_writes_every_declared_trunk_pool(tmp_path: Path) -> None:
    pools = {pool_id: _pool() for pool_id in range(4)}
    written = emit_eco_csvs(
        pools, range(4), tmp_path, "eco", terminal_pool_id=6, include_terminal=False
    )
    assert sorted(written) == [0, 1, 2, 3]
