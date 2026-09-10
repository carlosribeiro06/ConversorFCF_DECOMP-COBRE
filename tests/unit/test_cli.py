"""Unit tests for the CLI entry point.

`main` is exercised through `argv` rather than through `subprocess`, so a failure
surfaces as an assertion rather than as an opaque exit code. The one
console-script subprocess test belongs to `ticket-012`'s end-to-end test, where
there is something to run.

The named deliberate mutation for this ticket is **`--include-terminal-pool`
defaulting to `False` instead of `None`**, which would make an unset flag
override a settings file saying true - the one defect here that would silently
change what a run converts rather than failing.
"""

import json
import logging
import runpy
from collections.abc import Iterator
from pathlib import Path

import pytest

from conversor_fcf import cli
from conversor_fcf.cobre.inputs_reader import InputReadError
from conversor_fcf.cobre.policy_reader import PolicyFormatError
from conversor_fcf.config import Settings
from conversor_fcf.decomp.layout import LayoutError
from conversor_fcf.decomp.reader import ReadError
from conversor_fcf.mapping.rules import MappingError
from conversor_fcf.paths import OutputPaths

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_SETTINGS = REPO_ROOT / "settings.json"
CASE_NAME = "DEC_ONS_052026_RV0_VE_CONVERTIDO"


@pytest.fixture(autouse=True)
def isolated_package_logger() -> Iterator[None]:
    """Restore the package logger, because these tests really call `configure_logging`.

    `configure_logging` mutates process-global state: it removes whatever handlers
    the logger had, attaches a `RichHandler` that writes to **stdout** and a
    `RotatingFileHandler` holding an open file, and sets `propagate = False`.
    Without this fixture those handlers leak into later tests, where a logged
    error reappears on stdout and breaks an unrelated `capsys` assertion - which
    is exactly how this fixture came to be written. Closing the file handler also
    releases the log inside `tmp_path` before pytest removes it.
    """
    logger = logging.getLogger("conversor_fcf")
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture
def case(tmp_path: Path) -> Path:
    """A directory that passes the case check: the policy manifest is what is read first."""
    directory = tmp_path / CASE_NAME
    manifest = directory / "output" / "policy" / "manifest.bin"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"")
    return directory


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    """The tracked settings, copied so a test can edit it without touching the repo."""
    written = tmp_path / "settings.json"
    written.write_text(TRACKED_SETTINGS.read_text(encoding="utf-8"), encoding="utf-8")
    return written


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace the conversion seam with a recorder, so main is tested in isolation."""
    calls: list[dict[str, object]] = []

    def spy(
        case_dir: Path,
        revision: str,
        paths: OutputPaths,
        settings: Settings,
        include_terminal: bool,
    ) -> int:
        calls.append(
            {
                "case_dir": case_dir,
                "revision": revision,
                "paths": paths,
                "include_terminal": include_terminal,
            }
        )
        return 0

    monkeypatch.setattr(cli, "run_conversion", spy)
    return calls


def test_a_valid_invocation_returns_zero_and_calls_the_seam_once(
    case: Path, settings_file: Path, recorded: list[dict[str, object]]
) -> None:
    assert cli.main([str(case), "--settings", str(settings_file)]) == cli.EXIT_OK
    assert len(recorded) == 1
    assert recorded[0]["revision"] == "rv0", "derived from the case directory name"
    assert recorded[0]["case_dir"] == case.resolve()

    paths = recorded[0]["paths"]
    assert isinstance(paths, OutputPaths)
    assert paths.mapcut == case.resolve() / "output" / "decomp_fcf" / "mapcut.rv0"


def test_the_revision_flag_must_agree_with_the_case_name(
    case: Path, settings_file: Path, recorded: list[dict[str, object]]
) -> None:
    code = cli.main([str(case), "--settings", str(settings_file), "--revision", "rv1"])
    assert code == cli.EXIT_PATH
    assert not recorded, "nothing runs when the revision disagrees"


def test_a_missing_settings_file_returns_the_config_code(
    case: Path, tmp_path: Path, recorded: list[dict[str, object]]
) -> None:
    code = cli.main([str(case), "--settings", str(tmp_path / "absent.json")])
    assert code == cli.EXIT_CONFIG
    assert not recorded


def test_a_malformed_settings_file_returns_the_config_code(
    case: Path, tmp_path: Path, recorded: list[dict[str, object]]
) -> None:
    broken = tmp_path / "broken.json"
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    del payload["output"]["directory"]
    broken.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main([str(case), "--settings", str(broken)]) == cli.EXIT_CONFIG
    assert not recorded


def test_a_case_without_the_policy_manifest_returns_the_path_code(
    tmp_path: Path, settings_file: Path, recorded: list[dict[str, object]]
) -> None:
    bare = tmp_path / CASE_NAME
    bare.mkdir()
    assert cli.main([str(bare), "--settings", str(settings_file)]) == cli.EXIT_PATH
    assert not recorded

    audit = (bare / "output" / "decomp_fcf" / "logs" / "conversor-fcf.log").read_text(
        encoding="utf-8"
    )
    assert "manifest.bin is missing" in audit, "the audit trail must name the missing path"


def test_an_existing_artifact_is_refused_and_the_seam_never_runs(
    case: Path, settings_file: Path, recorded: list[dict[str, object]]
) -> None:
    """K3: the refusal must precede any read, so the conversion cannot have started."""
    existing = case / "output" / "decomp_fcf" / "mapcut.rv0"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"stale")

    assert cli.main([str(case), "--settings", str(settings_file)]) == cli.EXIT_PATH
    assert not recorded, "run_conversion must never be reached"
    assert existing.read_bytes() == b"stale", "and the stale artifact is untouched"


def test_force_allows_the_run_to_proceed(
    case: Path, settings_file: Path, recorded: list[dict[str, object]]
) -> None:
    existing = case / "output" / "decomp_fcf" / "cortdeco.rv0"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"stale")

    assert cli.main([str(case), "--settings", str(settings_file), "--force"]) == cli.EXIT_OK
    assert len(recorded) == 1


def test_the_failure_is_reported_on_stderr_even_before_logging_exists(
    case: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ConfigError fires before configure_logging runs, so the log cannot carry it."""
    cli.main([str(case), "--settings", str(tmp_path / "absent.json")])
    captured = capsys.readouterr()
    assert "ConfigError" in captured.err
    assert captured.out == ""


# --- the exit-code map ------------------------------------------------------


def _log_text(case: Path) -> str:
    return (case / "output" / "decomp_fcf" / "logs" / "conversor-fcf.log").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (InputReadError("bad stages.json"), cli.EXIT_INPUT),
        (PolicyFormatError("bad checkpoint"), cli.EXIT_INPUT),
        (MappingError("bus 5 has no submarket"), cli.EXIT_MAPPING),
        (LayoutError("289 records expected"), cli.EXIT_LAYOUT),
        (ReadError("mapcut.rv0 is 14 bytes, not a whole number of records"), cli.EXIT_LAYOUT),
    ],
    ids=["input_read", "policy_format", "mapping", "layout", "read_error"],
)
def test_every_declared_exit_code_is_reachable_and_logged(
    case: Path,
    settings_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_code: int,
) -> None:
    """Codes 4, 5 and 6 are raised through the seam, which is the only route to them.

    Nothing in `main` itself can raise these: `load_settings` raises only
    `ConfigError` and the four path helpers only `PathError`, so every one of
    them arrives from `run_conversion`. Each is therefore injected through the
    seam and asserted here, rather than left to be discovered as mis-mapped once
    a real failure has to be diagnosed through a wrong exit code.

    The ERROR line is asserted against the log FILE, because `configure_logging`
    has already set `propagate = False` by the time the seam runs.
    """

    def raising(*_args: object, **_kwargs: object) -> int:
        raise error

    monkeypatch.setattr(cli, "run_conversion", raising)
    assert cli.main([str(case), "--settings", str(settings_file)]) == expected_code

    audit = _log_text(case)
    assert type(error).__name__ in audit
    assert str(error) in audit
    assert "ERROR" in audit


def test_the_exit_codes_are_distinct_per_error_class() -> None:
    """A1's point is that a wrapper can branch without parsing text.

    Two classes sharing a code would defeat that, except for the two pairs that
    share it deliberately: a Cobre input and a Cobre checkpoint are the same
    failure to an operator, and so are a DECOMP write-time layout violation and
    a DECOMP read-time decode failure (Fix 3) - the file on disk is not what
    its own layout declares, either way.
    """
    codes = [code for _, code in cli._EXIT_CODES]
    assert codes.count(cli.EXIT_INPUT) == 2, "InputReadError and PolicyFormatError share code 4"
    assert codes.count(cli.EXIT_LAYOUT) == 2, "LayoutError and ReadError share code 6"
    assert len(set(codes)) == len(codes) - 2
    assert 2 not in codes, "2 belongs to argparse"
    assert cli.EXIT_OK not in codes


# --- the include-terminal-pool override (the named mutation) ---------------


def _settings_with_terminal(tmp_path: Path, include: bool) -> Path:
    written = tmp_path / f"settings_{include}.json"
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    payload["conversion"]["include_terminal_pool"] = include
    written.write_text(json.dumps(payload), encoding="utf-8")
    return written


@pytest.mark.parametrize("configured", [True, False])
def test_an_unset_flag_leaves_the_settings_value_alone(
    case: Path, tmp_path: Path, recorded: list[dict[str, object]], configured: bool
) -> None:
    """The mutation guard: a `False` default would override a settings file saying true.

    This is the one defect in this ticket that changes what a run converts rather
    than failing, so both settings values are exercised with the flag unset.
    """
    settings_path = _settings_with_terminal(tmp_path, configured)
    assert cli.main([str(case), "--settings", str(settings_path)]) == cli.EXIT_OK
    assert recorded[0]["include_terminal"] is configured


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--include-terminal-pool", True), ("--no-include-terminal-pool", False)],
)
def test_either_flag_overrides_the_settings_value(
    case: Path, tmp_path: Path, recorded: list[dict[str, object]], flag: str, expected: bool
) -> None:
    """K4 in both directions, so neither flag is a no-op against a matching setting."""
    settings_path = _settings_with_terminal(tmp_path, not expected)
    assert cli.main([str(case), "--settings", str(settings_path), flag]) == cli.EXIT_OK
    assert recorded[0]["include_terminal"] is expected


def test_the_two_flags_are_mutually_exclusive(case: Path, settings_file: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                str(case),
                "--settings",
                str(settings_file),
                "--include-terminal-pool",
                "--no-include-terminal-pool",
            ]
        )
    assert exit_info.value.code == 2, "argparse owns the usage exit code"


# --- ordering, versioning and the seam -------------------------------------


def test_the_operation_order_is_the_declared_one(
    case: Path, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 6. Logging is configured before the refusal so the refusal is audited,
    and the refusal precedes case validation because it is the cheaper check.

    `run_conversion` is stubbed rather than called through: since `ticket-012`
    wired the real pipeline behind it, calling through would need a full,
    valid Cobre case, which is not this test's concern - `run_conversion`'s
    place in the order is. The other six names are cheap, pure resolution steps
    and are still exercised for real.
    """
    order: list[str] = []
    for name in (
        "load_settings",
        "resolve_revision",
        "resolve_output_paths",
        "configure_logging",
        "assert_outputs_absent",
        "assert_case_readable",
    ):
        original = getattr(cli, name)

        def spy(*args: object, _name: str = name, _original: object = original, **kwargs: object):  # type: ignore[no-untyped-def]
            order.append(_name)
            return _original(*args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(cli, name, spy)

    def run_conversion_spy(*_args: object, **_kwargs: object) -> int:
        order.append("run_conversion")
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "run_conversion", run_conversion_spy)

    assert cli.main([str(case), "--settings", str(settings_file)]) == cli.EXIT_OK
    assert order == [
        "load_settings",
        "resolve_revision",
        "resolve_output_paths",
        "configure_logging",
        "assert_outputs_absent",
        "assert_case_readable",
        "run_conversion",
    ]


@pytest.mark.filterwarnings(
    "ignore:.*found in sys.modules after import of package.*:RuntimeWarning"
)
def test_the_module_runs_as_a_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`python -m conversor_fcf.cli` is a real invocation path, so it is tested as one.

    Run through `runpy` rather than a subprocess so the `__main__` guard executes
    in-process and coverage sees it; a subprocess would need coverage's own
    subprocess support to count. `--version` is used because it exits 0 without
    touching the filesystem.
    """
    monkeypatch.setattr("sys.argv", ["conversor-fcf", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("conversor_fcf.cli", run_name="__main__")
    assert exit_info.value.code == 0
    assert "conversor-fcf" in capsys.readouterr().out


def test_the_version_flag_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "conversor-fcf" in capsys.readouterr().out


def test_the_seam_logs_the_resolved_plan_before_the_real_pipeline_runs(
    case: Path, settings_file: Path
) -> None:
    """Asserted against the log FILE, not caplog, and that is the stronger check.

    `configure_logging` sets `propagate = False`, so caplog stops seeing anything
    emitted after it runs - which is most of a real invocation. Reading the file
    the run actually wrote proves the audit trail carries the resolved plan,
    which is what an auditor would consult.

    Since `ticket-012` wired the real pipeline behind this seam, and the minimal
    `case` fixture carries no `stages.json`, the run fails at its very first step
    (`InputReadError`) rather than reaching `EXIT_OK` - a full successful run
    belongs to `tests/integration/test_pipeline_end_to_end.py`. What this test
    isolates is that the resolved plan reaches the audit log, and the run
    manifest records the failure, even though the conversion itself failed.
    """
    assert cli.main([str(case), "--settings", str(settings_file)]) == cli.EXIT_INPUT

    output_root = case / "output" / "decomp_fcf"
    log_file = output_root / "logs" / "conversor-fcf.log"
    assert log_file.is_file(), "A2: a relative log path lands under the output root"
    audit = log_file.read_text(encoding="utf-8")
    assert "resolved plan:" in audit
    assert "revision=rv0" in audit
    assert "mapcut=mapcut.rv0" in audit
    assert "InputReadError" in audit

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_step"] == "read settings-derived inputs"
