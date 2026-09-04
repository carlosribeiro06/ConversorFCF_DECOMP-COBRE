"""Acceptance criteria for ticket-005 against the real Cobre reference pools.

The reference pool-6 artifact is roughly 0.5 GB and is deliberately never
written here; the gzip path is proven on a synthetic pool in the unit suite.
"""

import logging
from pathlib import Path

import pytest

from conversor_fcf.cobre.policy_reader import StageCutPool, read_stage_cuts
from conversor_fcf.reporting.eco_csv import emit_eco_csvs, write_eco_csv

REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
CUTS_DIR = REFERENCE_CASE / "output" / "policy" / "cuts"
TRUNK_POOL_IDS = (0, 1, 2, 3, 4, 5)
TERMINAL_POOL_ID = 6

pytestmark = pytest.mark.skipif(
    not CUTS_DIR.is_dir(),
    reason=f"Cobre reference case not present at {REFERENCE_CASE}",
)


@pytest.fixture(scope="module")
def pool_zero() -> StageCutPool:
    return read_stage_cuts(CUTS_DIR / "000.bin")


@pytest.fixture(scope="module")
def trunk_pools() -> dict[int, StageCutPool]:
    return {pool_id: read_stage_cuts(CUTS_DIR / f"{pool_id:03d}.bin") for pool_id in TRUNK_POOL_IDS}


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_pool_zero_writes_48_rows_and_192_columns(pool_zero: StageCutPool, tmp_path: Path) -> None:
    path = tmp_path / "eco_cuts_pool_000.csv"
    assert write_eco_csv(pool_zero, path) == 48

    header = _lines(path)[0].split(",")
    assert len(header) == 192
    # 1-indexed field positions from the acceptance criterion.
    assert header[9] == "storage_h0"
    assert header[177] == "storage_h168"
    assert header[178] == "anticipated_thermal_t112_s0"


def test_intercept_survives_verbatim_at_full_precision(
    pool_zero: StageCutPool, tmp_path: Path
) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(pool_zero, path)
    lines = _lines(path)
    header, first = lines[0].split(","), lines[1].split(",")
    intercept = first[header.index("intercept")]
    assert intercept == "2312687234790.0674"
    assert float(intercept) == 2312687234790.0674
    assert float(intercept) == pool_zero.pieces[0].intercept


def test_every_coefficient_survives_the_round_trip(pool_zero: StageCutPool, tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(pool_zero, path)
    first = _lines(path)[1].split(",")
    written = [float(value) for value in first[9:]]
    assert written == [float(value) for value in pool_zero.pieces[0].coefficients]


def test_active_columns_are_present_and_filter_nothing(
    pool_zero: StageCutPool, tmp_path: Path
) -> None:
    path = tmp_path / "eco.csv"
    write_eco_csv(pool_zero, path)
    lines = _lines(path)
    header = lines[0].split(",")
    assert "is_active" in header
    assert "in_active_cut_indices" in header
    assert len(lines) - 1 == 48 == len(pool_zero.pieces)


def test_emit_writes_six_plain_csvs_and_reports_the_skipped_terminal_pool(
    trunk_pools: dict[int, StageCutPool], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="conversor_fcf"):
        written = emit_eco_csvs(
            trunk_pools,
            range(7),
            tmp_path,
            "eco",
            terminal_pool_id=TERMINAL_POOL_ID,
            include_terminal=False,
        )

    assert sorted(written) == list(TRUNK_POOL_IDS)
    assert all(not str(path).endswith(".gz") for path in written.values())
    assert all(path.is_file() for path in written.values())
    assert not list((tmp_path / "eco").glob("*006*"))

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "terminal pool 6" in messages
    assert "builds no cuts" in messages


def test_every_trunk_pool_writes_one_row_per_piece(
    trunk_pools: dict[int, StageCutPool], tmp_path: Path
) -> None:
    written = emit_eco_csvs(
        trunk_pools,
        range(7),
        tmp_path,
        "eco",
        terminal_pool_id=TERMINAL_POOL_ID,
        include_terminal=False,
    )
    for pool_id, path in written.items():
        pool = trunk_pools[pool_id]
        lines = _lines(path)
        assert len(lines) - 1 == len(pool.pieces)
        assert len(lines[0].split(",")) == 9 + len(pool.slots)
