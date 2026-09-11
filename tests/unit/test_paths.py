"""Unit tests for revision and output-path resolution.

The transcribed anchor is the reference case's own directory name,
`DEC_ONS_052026_RV0_VE_CONVERTIDO`, and the `rv0` it yields. Nothing else here is
numeric.

Resolution must create nothing, so every test that resolves paths also asserts
the filesystem is unchanged: a run refused for an existing artifact has to leave
nothing behind, and only the writers create directories.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from conversor_fcf.config import Settings, load_settings
from conversor_fcf.paths import (
    ACCEPTED_REVISIONS,
    OutputPaths,
    PathError,
    assert_case_readable,
    assert_outputs_absent,
    normalize_revision,
    resolve_output_paths,
    resolve_revision,
    revision_from_case_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_SETTINGS = REPO_ROOT / "settings.json"

# The reference case's real directory name (transcribed).
REFERENCE_CASE_NAME = "DEC_ONS_052026_RV0_VE_CONVERTIDO"


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


@pytest.fixture
def settings() -> Settings:
    return load_settings(TRACKED_SETTINGS)


# --- the revision -----------------------------------------------------------


def test_the_reference_case_name_yields_rv0() -> None:
    """The transcribed anchor: upper case in the directory, lower case on the files."""
    assert revision_from_case_dir(Path("/cases") / REFERENCE_CASE_NAME) == "rv0"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("DEC_ONS_052026_RV0_VE_CONVERTIDO", "rv0"),
        ("DEC_ONS_052026_RV4_VE", "rv4"),
        ("DEC_ONS_052026_RVF_VE", "rvf"),
        ("case_RV9", "rv9"),
        ("RV0_leading", "rv0"),
        ("no_revision_here", None),
        ("DEC_RV10_VE", None),
        ("DEC_ARV0_VE", None),
    ],
)
def test_revision_extraction(name: str, expected: str | None) -> None:
    assert revision_from_case_dir(Path(name)) == expected


def test_a_name_carrying_two_revisions_is_refused() -> None:
    """The lookahead matters here: consuming the delimiter would hide the second token."""
    with pytest.raises(PathError, match="more than one revision"):
        revision_from_case_dir(Path("DEC_RV0_RV1_VE"))


def test_a_name_repeating_one_revision_is_accepted() -> None:
    assert revision_from_case_dir(Path("DEC_RV0_middle_RV0_VE")) == "rv0"


@pytest.mark.parametrize("declared", ["rv0", "RV0", " Rv0 "])
def test_a_declared_revision_is_normalized(declared: str) -> None:
    assert normalize_revision(declared) == "rv0"


@pytest.mark.parametrize("declared", ["rv12", "x", "", "rv", "0", "rvg"])
def test_an_unparseable_revision_names_the_accepted_forms(declared: str) -> None:
    with pytest.raises(PathError, match="rv0 through rv9"):
        normalize_revision(declared)
    assert "rvf" in ACCEPTED_REVISIONS


def test_the_case_name_and_the_flag_must_agree() -> None:
    with pytest.raises(PathError) as error:
        resolve_revision(Path(REFERENCE_CASE_NAME), "rv1")
    message = str(error.value)
    assert "rv0" in message, "the directory's value must be named"
    assert "rv1" in message, "and the flag's"


def test_the_flag_is_required_when_the_name_carries_none() -> None:
    with pytest.raises(PathError, match="--revision is required"):
        resolve_revision(Path("some_case_directory"), None)


def test_the_flag_alone_is_accepted_for_an_unnamed_case() -> None:
    assert resolve_revision(Path("some_case_directory"), "RV3") == "rv3"


def test_the_name_alone_is_accepted(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="conversor_fcf"):
        assert resolve_revision(Path(REFERENCE_CASE_NAME), None) == "rv0"
    assert any("derived from the case directory name" in r.getMessage() for r in caplog.records)


def test_agreement_is_accepted() -> None:
    assert resolve_revision(Path(REFERENCE_CASE_NAME), "rv0") == "rv0"


# --- ticket-019: resolve before reading the name ----------------------------


def test_a_symlink_to_a_differently_revisioned_case_is_not_bypassed(tmp_path: Path) -> None:
    """The cross-check `resolve_revision`'s docstring promises must reach the
    directory the symlink points at, not the symlink's own unrevisioned name.
    """
    target = tmp_path / "DEC_ONS_052026_RV1_VE_CONVERTIDO"
    target.mkdir()
    link = tmp_path / "latest"
    link.symlink_to(target)

    with pytest.raises(PathError) as error:
        resolve_revision(link, "rv0")
    message = str(error.value)
    assert "rv1" in message, "the resolved target's revision must be named"
    assert "rv0" in message, "and the declared one"


def test_the_working_directory_alone_resolves_its_own_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Path(".").name` is `''`; the resolved directory carries the revision."""
    case = tmp_path / REFERENCE_CASE_NAME
    case.mkdir()
    monkeypatch.chdir(case)
    assert resolve_revision(Path("."), None) == "rv0"


def test_the_working_directory_without_a_revision_names_itself_not_the_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pitfall this fix exists to avoid: a raised message quoting `''`
    instead of the resolved directory it actually refused.
    """
    case = tmp_path / "no_revision_here"
    case.mkdir()
    monkeypatch.chdir(case)
    with pytest.raises(PathError) as error:
        resolve_revision(Path("."), None)
    message = str(error.value)
    assert "no_revision_here" in message
    assert "''" not in message


# --- the output paths -------------------------------------------------------


def test_the_output_root_sits_inside_the_case(tmp_path: Path, settings: Settings) -> None:
    """K2: a relative settings.output.directory resolves against the case."""
    case = tmp_path / REFERENCE_CASE_NAME
    case.mkdir()
    paths = resolve_output_paths(case, "rv0", settings)

    assert paths.root == case / "output" / "decomp_fcf"
    assert paths.mapcut == paths.root / "mapcut.rv0"
    assert paths.cortdeco == paths.root / "cortdeco.rv0"
    assert paths.run_manifest == paths.root / "run_manifest.json"
    assert paths.eco_dir == paths.root / "eco"
    assert paths.content_dir == paths.root / "content"
    assert paths.log == paths.root / "logs" / "conversor-fcf.log"


def test_resolution_creates_nothing(tmp_path: Path, settings: Settings) -> None:
    case = tmp_path / REFERENCE_CASE_NAME
    case.mkdir()
    resolve_output_paths(case, "rv0", settings)
    assert list(case.iterdir()) == [], "resolution must not touch the filesystem"


def test_every_path_is_absolute_for_a_relative_case(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = tmp_path / REFERENCE_CASE_NAME
    case.mkdir()
    monkeypatch.chdir(tmp_path)
    paths = resolve_output_paths(Path(REFERENCE_CASE_NAME), "rv0", settings)

    for name, value in vars(paths).items():
        assert value.is_absolute(), f"{name} is relative: {value}"
    assert paths.root.is_relative_to(tmp_path.resolve())


def test_the_output_override_replaces_the_root_but_not_the_filenames(
    tmp_path: Path, settings: Settings
) -> None:
    case = tmp_path / REFERENCE_CASE_NAME
    case.mkdir()
    elsewhere = tmp_path / "elsewhere"
    paths = resolve_output_paths(case, "rv0", settings, output_override=elsewhere)

    assert paths.root == elsewhere.resolve()
    assert paths.mapcut.name == "mapcut.rv0"
    assert not elsewhere.exists()


def test_an_absolute_configured_directory_is_honoured(tmp_path: Path, settings: Settings) -> None:
    absolute = tmp_path / "absolute_output"
    written = tmp_path / "settings.json"
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    payload["output"]["directory"] = str(absolute)
    written.write_text(json.dumps(payload), encoding="utf-8")

    paths = resolve_output_paths(tmp_path / REFERENCE_CASE_NAME, "rv0", load_settings(written))
    assert paths.root == absolute.resolve()


def test_an_absolute_log_path_is_honoured(tmp_path: Path) -> None:
    """A2 resolves a relative log path against the root; an absolute one is left alone."""
    absolute_log = tmp_path / "audit" / "run.log"
    written = tmp_path / "settings.json"
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    payload["logging"]["file_path"] = str(absolute_log)
    written.write_text(json.dumps(payload), encoding="utf-8")

    paths = resolve_output_paths(tmp_path / REFERENCE_CASE_NAME, "rv0", load_settings(written))
    assert paths.log == absolute_log
    assert not absolute_log.parent.exists()


def test_the_revision_appears_only_in_the_binary_filenames(
    tmp_path: Path, settings: Settings
) -> None:
    """The names DECOMP reads carry it; the root is already per-case and per-revision."""
    paths = resolve_output_paths(tmp_path / REFERENCE_CASE_NAME, "rv4", settings)
    assert paths.mapcut.name == "mapcut.rv4"
    assert paths.cortdeco.name == "cortdeco.rv4"
    for other in (paths.run_manifest, paths.eco_dir, paths.content_dir, paths.log):
        assert "rv4" not in other.name


# --- refusing an existing artifact -----------------------------------------


def _paths(root: Path) -> OutputPaths:
    return OutputPaths(
        root=root,
        mapcut=root / "mapcut.rv0",
        cortdeco=root / "cortdeco.rv0",
        run_manifest=root / "run_manifest.json",
        eco_dir=root / "eco",
        content_dir=root / "content",
        log=root / "logs" / "conversor-fcf.log",
    )


def test_absent_outputs_are_accepted(tmp_path: Path) -> None:
    assert_outputs_absent(_paths(tmp_path), force=False)


def test_every_existing_artifact_is_named_not_just_the_first(tmp_path: Path) -> None:
    """One run must tell the operator everything they have to move."""
    paths = _paths(tmp_path)
    paths.mapcut.write_bytes(b"x")
    paths.run_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(PathError) as error:
        assert_outputs_absent(paths, force=False)
    message = str(error.value)
    assert "mapcut.rv0" in message
    assert "run_manifest.json" in message
    assert "2 existing artifact" in message
    assert "--force" in message


def test_force_replaces_and_warns_for_each(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _paths(tmp_path)
    paths.mapcut.write_bytes(b"x")
    paths.cortdeco.write_bytes(b"y")

    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        assert_outputs_absent(paths, force=True)
    warned = [r.getMessage() for r in caplog.records if "--force given" in r.getMessage()]
    assert len(warned) == 2


# --- validating the case ---------------------------------------------------


def test_a_missing_case_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PathError, match="case directory not found"):
        assert_case_readable(tmp_path / "absent")


def test_a_directory_without_the_policy_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PathError, match="does not look like a converted Cobre case"):
        assert_case_readable(tmp_path)


def test_a_case_carrying_the_policy_manifest_is_accepted(tmp_path: Path) -> None:
    manifest = tmp_path / "output" / "policy" / "manifest.bin"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"")
    assert assert_case_readable(tmp_path) == tmp_path.resolve()
