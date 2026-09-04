"""Acceptance criteria for ticket-003 against the real Cobre reference case."""

import hashlib
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest

from conversor_fcf.cobre.entities import (
    DELIVERY_DATE_SENTINEL,
    has_delivery_date,
    index_slots,
    slot_label,
)
from conversor_fcf.cobre.policy_reader import (
    StageCutPool,
    nodes_by_pool,
    read_policy_manifest,
    read_stage_cuts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
POLICY_DIR = REFERENCE_CASE / "output" / "policy"
CUTS_DIR_006 = POLICY_DIR / "cuts" / "006.bin"

pytestmark = pytest.mark.skipif(
    not POLICY_DIR.is_dir(),
    reason=f"Cobre reference case not present at {REFERENCE_CASE}",
)


@pytest.fixture(scope="module")
def pool_zero() -> StageCutPool:
    return read_stage_cuts(POLICY_DIR / "cuts" / "000.bin")


@pytest.fixture
def propagating_package_logger() -> Iterator[None]:
    """Let caplog see package records regardless of what configure_logging left behind."""
    logger = logging.getLogger("conversor_fcf")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


def test_vendored_modules_are_byte_identical_to_the_source_case() -> None:
    vendored = REPO_ROOT / "src" / "Cobre"
    modules = sorted(vendored.rglob("*.py"))
    assert len(modules) == 12

    for module in modules:
        relative = module.relative_to(vendored)
        source = REFERENCE_CASE / "Cobre" / relative
        assert source.is_file(), f"no counterpart in the source case for {relative}"
        assert (
            hashlib.sha256(module.read_bytes()).hexdigest()
            == hashlib.sha256(source.read_bytes()).hexdigest()
        ), f"{relative} drifted from the source case"


def test_no_bytecode_is_tracked_and_no_root_copy_remains() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "src/Cobre"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert len([path for path in tracked if path.endswith(".py")]) == 12
    assert [path for path in tracked if path.endswith(".pyc")] == []
    assert [path for path in tracked if "__pycache__" in path] == []
    assert not (REPO_ROOT / "Cobre").exists()


def test_provenance_records_the_schema_namespace_identifier_and_digests() -> None:
    provenance = (REPO_ROOT / "src" / "Cobre" / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "crates/cobre-io/schemas/policy.fbs" in provenance
    assert "Cobre.IO.Policy" in provenance
    assert "CBVF" in provenance
    assert str(REFERENCE_CASE) in provenance
    assert provenance.count("`") >= 12

    vendored = REPO_ROOT / "src" / "Cobre"
    for module in sorted(vendored.rglob("*.py")):
        assert hashlib.sha256(module.read_bytes()).hexdigest() in provenance


def test_manifest_matches_the_reference_study() -> None:
    manifest = read_policy_manifest(POLICY_DIR / "manifest.bin")
    assert manifest.format_version == 1
    assert manifest.num_stages == 7
    assert manifest.n_pools == 7
    assert manifest.completed_iterations == 48
    assert manifest.cost_scale_factor == 1000000.0
    assert len(manifest.nodes) == 273
    assert len(manifest.edges) == 272


def test_nodes_group_one_per_trunk_pool_and_267_in_the_terminal_fan() -> None:
    manifest = read_policy_manifest(POLICY_DIR / "manifest.bin")
    grouped = nodes_by_pool(manifest)
    assert tuple(grouped) == (0, 1, 2, 3, 4, 5, 6)
    assert tuple(len(ids) for ids in grouped.values()) == (1, 1, 1, 1, 1, 1, 267)


def test_trunk_pool_is_sized_by_its_manifest_not_by_state_dimension() -> None:
    pool = read_stage_cuts(POLICY_DIR / "cuts" / "000.bin")
    assert len(pool.slots) == 183
    assert len(pool.pieces) == 48
    assert {len(piece.coefficients) for piece in pool.pieces} == {183}
    assert pool.state_dimension == 2211
    assert pool.pieces[0].intercept == 2312687234790.0674


def test_mismatched_state_dimension_is_reported_as_a_warning(
    caplog: pytest.LogCaptureFixture, propagating_package_logger: None
) -> None:
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        read_stage_cuts(POLICY_DIR / "cuts" / "000.bin")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("state_dimension" in message for message in warnings)


def test_slot_axes_and_labels_follow_entity_manifest_order() -> None:
    pool = read_stage_cuts(POLICY_DIR / "cuts" / "000.bin")
    index = index_slots(pool.slots)
    assert len(index.storage) == 169
    assert len(index.anticipated_thermal) == 14
    assert index.inflow_lag == ()
    assert index.transit_bucket == ()
    # subindex is 0-based: position 171 is thermal 112 at ring position 1.
    assert slot_label(pool.slots[171]) == "anticipated_thermal_t112_s1"
    assert slot_label(pool.slots[0]) == "storage_h0"


@pytest.mark.slow
def test_terminal_pool_carries_the_full_state_and_only_warm_start_cuts() -> None:
    pool = read_stage_cuts(POLICY_DIR / "cuts" / "006.bin")
    assert pool.node_id == -1
    assert pool.warm_start_count == 10000
    assert len(pool.slots) == 2211
    assert len(pool.pieces) == 10000


def test_coefficient_values_are_decoded_exactly(pool_zero: StageCutPool) -> None:
    """Absolute anchors, because lengths alone leave a wrong dtype or axis invisible.

    A float32 dtype or a reversed coefficient array both leave every length-only
    assertion green, so the decode is pinned by value here. Index 168 and 169
    straddle the storage-to-thermal axis boundary, which pins the axis order too.
    """
    coefficients = pool_zero.pieces[0].coefficients
    assert coefficients.dtype == numpy.float64
    assert float(coefficients[0]) == -2488099.107139592
    assert float(coefficients[1]) == -2464022.495712601
    assert float(coefficients[168]) == -1063430.391132536
    assert float(coefficients[169]) == -165184.80464964395
    assert float(coefficients[182]) == -542599.0116637845
    assert float(coefficients.sum()) == -150222083.9761167
    assert float(pool_zero.pieces[47].coefficients[0]) == -2502371.987929096


def test_delivery_dates_and_active_flags_are_decoded_from_the_deck(
    pool_zero: StageCutPool,
) -> None:
    """No test pinned a real delivery_date, yet the GNL stage axis depends on it."""
    index = index_slots(pool_zero.slots)
    dates = [pool_zero.slots[position].delivery_date for position in index.anticipated_thermal]
    assert dates == [
        20260401,
        20260401,
        20260501,
        20260501,
        20260501,
        20260501,
        20260501,
        20260501,
        20260501,
        20260501,
        20260501,
        20260501,
        20260601,
        20260601,
    ]
    assert all(pool_zero.slots[position].was_active for position in index.anticipated_thermal)
    assert all(
        has_delivery_date(pool_zero.slots[position]) for position in index.anticipated_thermal
    )

    storage_dates = {pool_zero.slots[position].delivery_date for position in index.storage}
    assert storage_dates == {DELIVERY_DATE_SENTINEL}
    assert not any(has_delivery_date(pool_zero.slots[position]) for position in index.storage)


def test_manifest_globals_agree_with_the_node_set() -> None:
    manifest = read_policy_manifest(POLICY_DIR / "manifest.bin")
    assert manifest.n_pools == len(nodes_by_pool(manifest))
    assert manifest.num_stages == len({node.stage_id for node in manifest.nodes})


@pytest.mark.slow
def test_terminal_pool_is_reported_as_shared(
    caplog: pytest.LogCaptureFixture, propagating_package_logger: None
) -> None:
    """node_id == -1 is the same class of trap as state_dimension and is real here."""
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        pool = read_stage_cuts(CUTS_DIR_006)
    assert pool.node_id == -1
    assert any("shared pool" in record.getMessage() for record in caplog.records)
