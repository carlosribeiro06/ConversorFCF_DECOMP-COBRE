"""Acceptance criteria for `ticket-012`'s wired end-to-end pipeline.

The full run goes through `pipeline.run_conversion` directly, except for the one
test that exercises the installed console script as a subprocess - the
acceptance criterion that needs a real process boundary rather than an in-process
call. Every numeric anchor is one already established against this project's own
scale in `ticket-007`..`ticket-010`: 300 mapcut records (14,406,000 bytes) and 289
cortdeco records over 7,796,064 bytes.

`run_result` runs the pipeline exactly once (module-scoped) and captures every
log record it emits through a plain collector handler, rather than through
`caplog` (function-scoped, so it cannot see a fixture that runs once for the
whole module) or through `configure_logging` (which would need a real log file
and is `cli.main`'s job, not this test's).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from conversor_fcf import pipeline as pipeline_module
from conversor_fcf.cobre.inputs_reader import InputReadError
from conversor_fcf.cobre.policy_reader import StageCutPool, nodes_by_pool, read_policy_manifest
from conversor_fcf.config import Settings, load_settings
from conversor_fcf.decomp.layout import RECORD_SIZE, TAMANHO_CORTE, LayoutError
from conversor_fcf.decomp.mapcut_writer import MapcutHeader
from conversor_fcf.decomp.mapcut_writer import write_mapcut as real_write_mapcut
from conversor_fcf.decomp.reader import read_mapcut
from conversor_fcf.mapping.rules import MappingError
from conversor_fcf.paths import POLICY_MANIFEST_RELATIVE, OutputPaths, resolve_output_paths
from conversor_fcf.pipeline import load_pools, run_conversion

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
TRACKED_SETTINGS = REPO_ROOT / "settings.json"

MAPCUT_RECORDS = 300
MAPCUT_BYTES = 14_406_000
CORTDECO_RECORDS = 289
CORTDECO_BYTES = 7_796_064
TRUNK_POOL_IDS = (0, 1, 2, 3, 4, 5)

pytestmark = pytest.mark.skipif(
    not REFERENCE_CASE.is_dir(), reason=f"Cobre reference case not present at {REFERENCE_CASE}"
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


def _settings_with_absolute_hydro_codes(directory: Path) -> Path:
    """A settings copy whose `hydro_codes_path` does not depend on the CWD a
    test happens to run from: `conversion.hydro_codes_path` is resolved exactly
    as given, the same way `--settings settings.json`'s own default resolves
    against the working directory, so a test must supply an absolute one."""
    written = directory / "settings.json"
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    payload["conversion"]["hydro_codes_path"] = str(REPO_ROOT / "decomp_hydro_codes.json")
    written.write_text(json.dumps(payload), encoding="utf-8")
    return written


class _RecordCollector(logging.Handler):
    """Collects every record, independent of `caplog`'s per-test lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _messages(records: list[logging.LogRecord], prefix: str) -> list[str]:
    return [record.getMessage() for record in records if record.getMessage().startswith(prefix)]


@pytest.fixture(scope="module")
def manifest_pool_ids() -> tuple[int, ...]:
    manifest = read_policy_manifest(REFERENCE_CASE / POLICY_MANIFEST_RELATIVE)
    return tuple(sorted(nodes_by_pool(manifest)))


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings_path = _settings_with_absolute_hydro_codes(tmp_path_factory.mktemp("settings"))
    return load_settings(settings_path)


@pytest.fixture(scope="module")
def paths(tmp_path_factory: pytest.TempPathFactory, settings: Settings) -> OutputPaths:
    output_root = tmp_path_factory.mktemp("output") / "decomp_fcf"
    return resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)


@pytest.fixture(scope="module")
def run_result(paths: OutputPaths, settings: Settings) -> tuple[int, list[logging.LogRecord]]:
    """Run the real pipeline exactly once, capturing every log record it emits."""
    collector = _RecordCollector()
    logger = logging.getLogger("conversor_fcf")
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        code = run_conversion(REFERENCE_CASE, "rv0", paths, settings, include_terminal=False)
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    return code, collector.records


# --- the full run, every declared artifact ----------------------------------


def test_the_full_run_exits_ok(run_result: tuple[int, list[logging.LogRecord]]) -> None:
    code, _ = run_result
    assert code == 0


def test_mapcut_holds_the_reference_scale_anchor(
    run_result: tuple[int, list[logging.LogRecord]], paths: OutputPaths
) -> None:
    _ = run_result
    assert paths.mapcut.stat().st_size == MAPCUT_BYTES == MAPCUT_RECORDS * RECORD_SIZE


def test_cortdeco_holds_the_reference_scale_anchor(
    run_result: tuple[int, list[logging.LogRecord]], paths: OutputPaths
) -> None:
    _ = run_result
    assert paths.cortdeco.stat().st_size == CORTDECO_BYTES == CORTDECO_RECORDS * TAMANHO_CORTE


def test_the_eco_csvs_cover_the_six_trunk_pools_and_exclude_the_terminal_one(
    run_result: tuple[int, list[logging.LogRecord]], paths: OutputPaths
) -> None:
    _ = run_result
    names = sorted(path.name for path in paths.eco_dir.glob("eco_cuts_pool_*.csv"))
    assert names == [f"eco_cuts_pool_{pool_id:03d}.csv" for pool_id in TRUNK_POOL_IDS]
    assert not list(paths.eco_dir.glob("*006*")), "include_terminal=False excludes pool 6"


def test_the_content_csvs_exist_with_the_expected_row_counts(
    run_result: tuple[int, list[logging.LogRecord]], paths: OutputPaths
) -> None:
    _ = run_result
    mapcut_content = paths.content_dir / "mapcut_content.csv"
    cortdeco_content = paths.content_dir / "cortdeco_content.csv"
    assert mapcut_content.is_file()
    assert cortdeco_content.is_file()
    assert len(mapcut_content.read_text(encoding="utf-8").splitlines()) > 1
    # Header plus one row per cut record (M2's long format), never inflated by
    # a per-node repeat of reg 9 (M3, this ticket's named mutation 3).
    assert len(cortdeco_content.read_text(encoding="utf-8").splitlines()) == CORTDECO_RECORDS + 1


def test_the_run_manifest_reports_success_with_no_failed_step(
    run_result: tuple[int, list[logging.LogRecord]], paths: OutputPaths
) -> None:
    _ = run_result
    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["failed_step"] is None
    assert manifest["outputs"]["mapcut"] == str(paths.mapcut)
    assert manifest["outputs"]["log"] == str(paths.log)


# --- once-per-run premises (Requirement 7, closing ticket-008 item C8) ------


def test_premises_thirteen_and_fourteen_are_each_logged_exactly_once(
    run_result: tuple[int, list[logging.LogRecord]],
) -> None:
    """The single call site this ticket wires in makes this a real per-run
    guarantee, not the per-call one `ticket-008` had to settle for."""
    _, records = run_result
    assert len(_messages(records, "premise P13:")) == 1
    assert len(_messages(records, "premise P14:")) == 1


def test_the_inflow_lag_audit_is_reported_from_a_real_run(
    run_result: tuple[int, list[logging.LogRecord]],
) -> None:
    """Requirement 8: P8 is reachable only from tests until a real run emits
    it. The reference deck's trunk pools carry no inflow-lag slots, so the
    counted audit reports zero rather than a silent truncation."""
    _, records = run_result
    audited = [record for record in records if "premise P8" in record.getMessage()]
    assert audited, "the inflow-lag audit must reach the log from a real run"
    assert any("drops no inflow-lag coefficients" in record.getMessage() for record in audited)


def test_a_successful_default_run_emits_no_cut_building_pools_warning(
    run_result: tuple[int, list[logging.LogRecord]],
) -> None:
    """Fix 2's regression test: the guardian reproduced this WARNING on an
    unmodified, successful, default (`include_terminal=False`) run, because
    `cut_building_pools` used to infer the expected count from `len(pools)`,
    which undercounts once the terminal pool is deliberately not loaded. A
    WARNING that always fires on an ordinary run trains an auditor to ignore
    WARNINGs, which defeats the audit log's purpose."""
    _, records = run_result
    spurious = [
        record
        for record in records
        if record.levelno == logging.WARNING and "cut-building pools" in record.getMessage()
    ]
    assert not spurious, [record.getMessage() for record in spurious]


# --- include_terminal=True (opt-in ECO inspection of the terminal pool) ----


@pytest.mark.slow
def test_load_pools_reads_the_terminal_pool_only_when_asked(
    manifest_pool_ids: tuple[int, ...],
) -> None:
    """`load_pools`'s own `include_terminal=True` branch, in isolation.

    Deliberately calling `load_pools` rather than the full `run_conversion`:
    reading `006.bin` (177 MB) costs well under a second, but writing its ECO
    CSV afterwards - which a full `include_terminal=True` run would also do -
    does not, and `test_eco_csv_reference.py` already declines to pay that
    cost for the same reason ("deliberately never written here"). Still marked
    slow, since it reads the 177 MB terminal-pool checkpoint.
    """
    terminal_pool_id = manifest_pool_ids[-1]
    without_terminal = load_pools(REFERENCE_CASE, manifest_pool_ids, include_terminal=False)
    assert terminal_pool_id not in without_terminal
    assert set(without_terminal) == set(TRUNK_POOL_IDS)

    with_terminal = load_pools(REFERENCE_CASE, manifest_pool_ids, include_terminal=True)
    assert terminal_pool_id in with_terminal
    assert len(with_terminal[terminal_pool_id].pieces) == 10_000
    assert set(with_terminal) == {*TRUNK_POOL_IDS, terminal_pool_id}


# --- a failing cross-check unpublishes the pair (Fix 1, C6) -----------------


def test_a_cross_check_failure_unpublishes_both_artifacts_and_names_both_values(
    tmp_path_factory: pytest.TempPathFactory, settings: Settings
) -> None:
    """Constructed the way the guardian proved the defect: tamper the header
    `write_mapcut` receives (`numero_cortes - 1`) rather than either writer's
    own arguments, so each writer's *own* internal invariants stay satisfied
    - `write_mapcut` never validates `numero_cortes` against anything, and
    `write_cortdeco` gets the untouched, self-consistent value it was always
    going to get. Both binaries are therefore fully written and published
    before the cross-check ever runs, which is exactly the scenario C6 needs:
    a real, structurally valid disagreement discovered only by reading both
    back, not a short write or a supplied-count mismatch either writer would
    already refuse on its own.
    """
    output_root = tmp_path_factory.mktemp("rejected") / "decomp_fcf"
    paths_here = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)

    def tampered_write_mapcut(header: MapcutHeader, path: Path) -> int:
        return real_write_mapcut(replace(header, numero_cortes=header.numero_cortes - 1), path)

    with (
        mock.patch.object(pipeline_module, "write_mapcut", side_effect=tampered_write_mapcut),
        pytest.raises(LayoutError, match=r"numero_cortes=\d+.*numero_cortes=\d+"),
    ):
        run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=False)

    assert not paths_here.mapcut.exists(), "a rejected mapcut must not remain at its published name"
    assert not paths_here.cortdeco.exists(), (
        "a rejected cortdeco must not remain at its published name"
    )

    mapcut_rejected = paths_here.mapcut.with_name(paths_here.mapcut.name + ".rejected")
    cortdeco_rejected = paths_here.cortdeco.with_name(paths_here.cortdeco.name + ".rejected")
    # Only the header field is tampered; mapcut's record count never depends on
    # numero_cortes (mapcut_record_count doesn't take it), and cortdeco was
    # never touched at all, so both keep their full, untampered sizes.
    assert mapcut_rejected.stat().st_size == MAPCUT_BYTES
    assert cortdeco_rejected.stat().st_size == CORTDECO_BYTES

    manifest = json.loads(paths_here.run_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_step"] == "cross-check the pair"


def test_a_force_rerun_failing_after_the_mapcut_write_leaves_neither_published_name(
    tmp_path_factory: pytest.TempPathFactory, settings: Settings
) -> None:
    """The scenario Fix 1 exists for, reproduced exactly: run 1 publishes both
    binaries; a `--force` re-run fails at `write cortdeco` (mocked here, but
    `write_cortdeco`'s own guards, a `MappingError` from `storage_coefficients`,
    ENOSPC on the 7.8 MB write, or a `KeyboardInterrupt` would all land at the
    same place). `write_mapcut` has already replaced the published mapcut by
    the time this fails, so a condition scoped to the cross-check step alone
    (the first version of this fix) left that fresh mapcut sitting beside
    run 1's *stale* cortdeco - an uncross-checked mixed pair, at both
    published names, that no writer's own guard can see. Neither published
    name may survive the second run.
    """
    output_root = tmp_path_factory.mktemp("force_rerun") / "decomp_fcf"
    paths_here = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)

    # Run 1: real, successful, publishes both binaries.
    assert run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=False) == 0
    run1_cortdeco = paths_here.cortdeco.read_bytes()
    assert paths_here.mapcut.is_file()
    assert paths_here.cortdeco.is_file()

    # Run 2 (what a --force re-run permits `run_conversion` to attempt): the
    # mapcut write succeeds and replaces the published one; write_cortdeco
    # then fails before touching its own file at all, so run 1's cortdeco is
    # still sitting at the published name, stale and never cross-checked
    # against run 2's mapcut.
    def failing_write_cortdeco(*_args: object, **_kwargs: object) -> int:
        raise LayoutError("simulated write_cortdeco failure")

    with (
        mock.patch.object(pipeline_module, "write_cortdeco", side_effect=failing_write_cortdeco),
        pytest.raises(LayoutError, match="simulated write_cortdeco failure"),
    ):
        run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=False)

    assert not paths_here.mapcut.exists(), "run 2's fresh mapcut must not remain published"
    assert not paths_here.cortdeco.exists(), "run 1's stale cortdeco must not remain published"

    mapcut_rejected = paths_here.mapcut.with_name(paths_here.mapcut.name + ".rejected")
    cortdeco_rejected = paths_here.cortdeco.with_name(paths_here.cortdeco.name + ".rejected")
    assert mapcut_rejected.is_file()
    assert cortdeco_rejected.is_file()
    # write_cortdeco was mocked to raise before writing anything, so the
    # rejected cortdeco is exactly run 1's untouched, stale file.
    assert cortdeco_rejected.read_bytes() == run1_cortdeco

    manifest = json.loads(paths_here.run_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_step"] == "write cortdeco"


def test_a_first_run_failing_in_the_validation_window_logs_no_error_about_the_missing_cortdeco(
    tmp_path_factory: pytest.TempPathFactory, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """The defect Fix 1 introduced, on what is now the *normal* failure shape:
    a first run (clean output directory) failing between the mapcut write and
    the cortdeco write has nothing at `paths.cortdeco` yet, so `_reject_pair`'s
    `os.replace` on it raises `FileNotFoundError`, not a rename defect. That
    must be logged at DEBUG, naming it as expected, never at ERROR - an ERROR
    here trains an auditor to ignore alarms, exactly what `cut_building_pools`'
    own WARNING was already fixed for. The mapcut, which really was written and
    really does need rejecting, must still be moved aside.
    """
    output_root = tmp_path_factory.mktemp("first_run_validation_window") / "decomp_fcf"
    paths_here = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)
    assert not paths_here.mapcut.exists(), "must be a first run: nothing published yet"
    assert not paths_here.cortdeco.exists(), "must be a first run: nothing published yet"

    def failing_write_cortdeco(*_args: object, **_kwargs: object) -> int:
        raise LayoutError("simulated write_cortdeco failure")

    with (
        mock.patch.object(pipeline_module, "write_cortdeco", side_effect=failing_write_cortdeco),
        caplog.at_level(logging.DEBUG, logger="conversor_fcf"),
        pytest.raises(LayoutError, match="simulated write_cortdeco failure"),
    ):
        run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=False)

    # log_step's own "fail write cortdeco" ERROR is the genuine failure and is
    # expected; what must not appear is an ERROR about _reject_pair failing to
    # rename a cortdeco that was never written in the first place.
    assert not any(
        record.levelno >= logging.ERROR and "rename" in record.getMessage()
        for record in caplog.records
    ), "a first run failing before cortdeco exists must log no ERROR about rejecting it"

    assert not paths_here.mapcut.exists(), "the mapcut that was written must not remain published"
    assert paths_here.mapcut.with_name(paths_here.mapcut.name + ".rejected").is_file()

    assert not paths_here.cortdeco.exists(), "cortdeco was never written"
    assert not paths_here.cortdeco.with_name(paths_here.cortdeco.name + ".rejected").exists(), (
        "nothing to reject: cortdeco never existed"
    )

    assert any(
        record.levelno == logging.DEBUG and "nothing to reject" in record.getMessage()
        for record in caplog.records
    )


def test_a_manifest_write_failure_on_the_failure_path_does_not_mask_the_original_error(
    tmp_path_factory: pytest.TempPathFactory, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Fix 2: an `OSError` from `write_run_manifest` on the failure path
    (ENOSPC is plausible exactly when the conversion itself just failed on a
    full disk) must not replace the original error - it is logged, and the
    original exception still propagates, so `cli.main`'s classified exit code
    and audit-log `ERROR` line are still reachable.
    """
    output_root = tmp_path_factory.mktemp("manifest_write_failure") / "decomp_fcf"
    paths_here = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)

    def failing_write_cortdeco(*_args: object, **_kwargs: object) -> int:
        raise LayoutError("simulated write_cortdeco failure")

    def failing_write_run_manifest(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated ENOSPC")

    with (
        mock.patch.object(pipeline_module, "write_cortdeco", side_effect=failing_write_cortdeco),
        mock.patch.object(
            pipeline_module, "write_run_manifest", side_effect=failing_write_run_manifest
        ),
        caplog.at_level(logging.ERROR, logger="conversor_fcf"),
        pytest.raises(LayoutError, match="simulated write_cortdeco failure"),
    ):
        run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=False)

    assert any(
        "failed to write the failure manifest" in record.getMessage() for record in caplog.records
    )


# --- an empty trunk_pool_ids is refused by name (Fix 5) ---------------------


def test_an_empty_trunk_pool_ids_is_refused_by_name(
    tmp_path_factory: pytest.TempPathFactory, settings: Settings
) -> None:
    """`cut_building_pools` returning `()` is its own documented outcome, and
    a checkpoint carrying only seeded or warm-start cuts - an early or aborted
    Cobre policy run - produces exactly that. The pipeline must refuse by name
    rather than crash on `trunk_pool_ids[0]` with a bare `IndexError`.
    """
    output_root = tmp_path_factory.mktemp("empty_trunk") / "decomp_fcf"
    paths_here = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)

    with (
        mock.patch.object(pipeline_module, "cut_building_pools", return_value=()),
        pytest.raises(MappingError, match="no pool built cuts"),
    ):
        run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=False)

    assert not paths_here.mapcut.exists()
    assert not paths_here.cortdeco.exists()


# --- --include-terminal-pool cannot reach the binaries (Fix 6) -------------


def test_include_terminal_pool_cannot_change_which_pools_build_cuts(
    settings: Settings, tmp_path_factory: pytest.TempPathFactory, manifest_pool_ids: tuple[int, ...]
) -> None:
    """The terminal pool must never reach `cut_building_pools`' candidate set,
    regardless of its own populated/warm-start counts - otherwise
    `--include-terminal-pool` could change `trunk_pool_ids`, against premise
    P7 and the README's own guarantee that the flag only affects the ECO
    CSVs. Inert on the real reference deck only because pool 006 happens to
    have `populated_count == warm_start_count`; this test makes it look
    cut-building instead, without paying for its 177 MB real read.
    """
    terminal_pool_id = manifest_pool_ids[-1]
    real_load_pools = pipeline_module.load_pools

    def tampered_load_pools(
        case_dir: Path, pool_ids: Sequence[int], include_terminal: bool
    ) -> dict[int, StageCutPool]:
        pools = real_load_pools(case_dir, pool_ids, False)
        pools[terminal_pool_id] = StageCutPool(
            stage_id=terminal_pool_id,
            node_id=-1,
            graph_stage_id=terminal_pool_id,
            state_dimension=0,
            capacity=0,
            warm_start_count=0,
            populated_count=1,
            cost_scale_factor=1.0,
            slots=(),
            pieces=(),
            active_cut_indices=(),
        )
        return pools

    output_root = tmp_path_factory.mktemp("fix6") / "decomp_fcf"
    paths_here = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)

    with mock.patch.object(pipeline_module, "load_pools", side_effect=tampered_load_pools):
        assert (
            run_conversion(REFERENCE_CASE, "rv0", paths_here, settings, include_terminal=True) == 0
        )

    contents = read_mapcut(paths_here.mapcut)
    assert contents.scalars.numero_semanas == len(TRUNK_POOL_IDS)


# --- the failed-run manifest (M7, named mutation 2) -------------------------


@pytest.fixture
def broken_case(tmp_path: Path) -> Path:
    """A case that passes `assert_case_readable` but carries no `stages.json`,
    so the pipeline fails at its very first step."""
    case = tmp_path / "case" / "DEC_ONS_052026_RV0_VE_CONVERTIDO"
    manifest = case / "output" / "policy" / "manifest.bin"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"")
    return case


def test_a_failed_run_still_writes_a_manifest_naming_the_failing_step(
    broken_case: Path, tmp_path: Path
) -> None:
    settings_path = _settings_with_absolute_hydro_codes(tmp_path)
    settings_for_run = load_settings(settings_path)
    output_paths = resolve_output_paths(
        broken_case, "rv0", settings_for_run, output_override=tmp_path / "out"
    )

    with pytest.raises(InputReadError):
        run_conversion(broken_case, "rv0", output_paths, settings_for_run, include_terminal=False)

    manifest = json.loads(output_paths.run_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_step"] == "read settings-derived inputs"
    assert not output_paths.mapcut.exists(), "a failed run must not leave a mapcut behind"
    assert not output_paths.cortdeco.exists()


# --- the console script as a subprocess -------------------------------------


def test_the_console_script_runs_end_to_end_as_a_subprocess(tmp_path: Path) -> None:
    settings_path = _settings_with_absolute_hydro_codes(tmp_path)
    output_dir = tmp_path / "out"
    console_script = Path(sys.executable).with_name("conversor-fcf")
    assert console_script.is_file(), "the package must be installed into this venv (pip install -e)"

    result = subprocess.run(
        [
            str(console_script),
            str(REFERENCE_CASE),
            "--settings",
            str(settings_path),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert (output_dir / "mapcut.rv0").stat().st_size == MAPCUT_BYTES
    assert (output_dir / "cortdeco.rv0").stat().st_size == CORTDECO_BYTES
    assert (output_dir / "content" / "mapcut_content.csv").is_file()
    assert (output_dir / "content" / "cortdeco_content.csv").is_file()
    for pool_id in TRUNK_POOL_IDS:
        assert (output_dir / "eco" / f"eco_cuts_pool_{pool_id:03d}.csv").is_file()

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
