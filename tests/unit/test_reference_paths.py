"""Unit tests for `tests/reference.py` and the session-level guard it feeds.

Three groups: resolution (`reference_root`, `reference_artifacts`), the pure
judgment (`unverified_reason`, every branch), and the ratchet that keeps a
twelfth hardcoded module from going unnoticed. The three `pytester` sessions
at the bottom prove the hook wiring in `tests/conftest.py` under pytest
itself, each against its own temporary root rather than this machine's.

Every one of those three sessions - and this module's own ratchet grep -
uses an empty `legacy_modules` set / avoids writing the default root's own
literal, for the same reason `reference.py` builds `DEFAULT_REFERENCE_ROOT`
from parts: a module that names the default root verbatim would flag itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reference import (
    ALLOW_MISSING_VAR,
    CASE_NAME,
    CORTDECO_NAME,
    DEFAULT_REFERENCE_ROOT,
    LEGACY_HARDCODED_MODULES,
    MAPCUT_NAME,
    REFERENCE_ROOT_VAR,
    ReferenceArtifacts,
    ReferenceRootError,
    reference_artifacts,
    reference_root,
    unresolvable_root_reason,
    unverified_reason,
)


def _populate(root: Path, *, omit: str | None = None) -> None:
    """Create all three artifacts under `root`, skipping `omit` if given."""
    for name in (CASE_NAME, MAPCUT_NAME, CORTDECO_NAME):
        if name == omit:
            continue
        target = root / name
        if name == CASE_NAME:
            target.mkdir()
        else:
            target.write_bytes(b"")


def _install_wiring(pytester: pytest.Pytester) -> None:
    """Install the real `reference.py` and the real `tests/conftest.py`, verbatim.

    A regression in either file's actual source is what these three sessions
    must catch, so nothing here is reimplemented. `LEGACY_HARDCODED_MODULES` is
    overridden to empty for these sessions only, appended after the real
    module body: they exist to prove the artifact-presence/opt-out wiring, not
    the legacy sweep, which every temporary `pytester` root would otherwise
    always trip (Requirement 5: any root other than `DEFAULT_REFERENCE_ROOT`
    is "relocated").
    """
    tests_dir = Path(__file__).resolve().parents[1]
    reference_source = (tests_dir / "reference.py").read_text(encoding="utf-8")
    reference_source += "\nLEGACY_HARDCODED_MODULES = frozenset()\n"
    pytester.makefile(".py", reference=reference_source)
    pytester.makeconftest((tests_dir / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(test_something="def test_something():\n    assert True\n")


# --- resolution --------------------------------------------------------------


def test_the_default_reference_root_is_built_from_the_documented_parts() -> None:
    """Independently re-derives the value `reference.py` builds from parts."""
    assert DEFAULT_REFERENCE_ROOT == Path("/home", "carlosribeiro", "git")


def test_reference_root_defaults_when_the_override_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REFERENCE_ROOT_VAR, raising=False)
    assert reference_root() == DEFAULT_REFERENCE_ROOT


def test_reference_root_reads_an_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REFERENCE_ROOT_VAR, "/srv/decks")
    assert reference_root() == Path("/srv/decks").resolve()


def test_reference_root_expands_a_tilde_in_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REFERENCE_ROOT_VAR, "~/decks")
    assert reference_root() == (Path.home() / "decks").resolve()


def test_an_override_naming_a_user_with_no_home_raises_rather_than_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Path.expanduser` raises `RuntimeError` for a `~user` with no passwd entry.

    Unconverted, that escapes both session hooks and takes the whole test
    report with it - pytest catches only `exit.Exception` there.
    """
    monkeypatch.setenv(REFERENCE_ROOT_VAR, "~nosuchuser42/decks")
    with pytest.raises(ReferenceRootError, match=REFERENCE_ROOT_VAR):
        reference_root()


def test_an_unresolvable_override_reports_as_unverified_and_names_both_vars() -> None:
    reason = unresolvable_root_reason("root is nonsense", allow_missing=False)
    assert "root is nonsense" in reason
    assert "verified nothing reference-dependent" in reason
    assert REFERENCE_ROOT_VAR in reason
    assert ALLOW_MISSING_VAR in reason


def test_reference_artifacts_use_the_documented_names() -> None:
    assert CASE_NAME == "DEC_ONS_052026_RV0_VE_CONVERTIDO"
    assert MAPCUT_NAME == "mapcut.rv0"
    assert CORTDECO_NAME == "cortdeco.rv0"


def test_reference_artifacts_are_children_of_the_default_root() -> None:
    assert reference_artifacts(DEFAULT_REFERENCE_ROOT) == ReferenceArtifacts(
        case=DEFAULT_REFERENCE_ROOT / CASE_NAME,
        mapcut=DEFAULT_REFERENCE_ROOT / MAPCUT_NAME,
        cortdeco=DEFAULT_REFERENCE_ROOT / CORTDECO_NAME,
    )


def test_reference_artifacts_are_children_of_an_overridden_root() -> None:
    root = Path("/srv/decks")
    assert reference_artifacts(root) == ReferenceArtifacts(
        case=root / CASE_NAME, mapcut=root / MAPCUT_NAME, cortdeco=root / CORTDECO_NAME
    )


# --- unverified_reason ---------------------------------------------------------


def test_unverified_reason_is_none_when_all_present_and_legacy_is_empty(tmp_path: Path) -> None:
    assert unverified_reason(tmp_path, [], allow_missing=False, legacy_modules=frozenset()) is None


def test_unverified_reason_names_every_absent_artifact_and_both_env_vars(tmp_path: Path) -> None:
    cortdeco = tmp_path / CORTDECO_NAME
    reason = unverified_reason(
        tmp_path, [cortdeco], allow_missing=False, legacy_modules=frozenset()
    )
    assert reason is not None
    assert str(tmp_path) in reason
    assert str(cortdeco) in reason
    assert REFERENCE_ROOT_VAR in reason
    assert ALLOW_MISSING_VAR in reason


def test_the_opt_out_moves_the_status_without_silencing_the_finding(tmp_path: Path) -> None:
    """Requirement 4: the two calls differ only in `allow_missing`, and both report."""
    cortdeco = tmp_path / CORTDECO_NAME
    refused = unverified_reason(
        tmp_path, [cortdeco], allow_missing=False, legacy_modules=frozenset()
    )
    accepted = unverified_reason(
        tmp_path, [cortdeco], allow_missing=True, legacy_modules=frozenset()
    )
    assert refused is not None
    assert accepted is not None
    assert str(cortdeco) in refused
    assert str(cortdeco) in accepted
    assert ALLOW_MISSING_VAR in accepted
    assert REFERENCE_ROOT_VAR in accepted


def test_a_relocated_root_with_legacy_modules_is_unverified_even_when_all_present() -> None:
    """Requirement 5: those eleven modules would still skip against the default root."""
    assert len(LEGACY_HARDCODED_MODULES) == 11
    reason = unverified_reason(
        Path("/srv/decks"), [], allow_missing=False, legacy_modules=LEGACY_HARDCODED_MODULES
    )
    assert reason is not None
    assert str(len(LEGACY_HARDCODED_MODULES)) in reason
    assert "ticket-022" in reason


def test_a_relocated_root_whose_legacy_modules_ran_does_not_claim_everything_skipped() -> None:
    """The summary must not say a verified session verified nothing.

    With the default root populated, the eleven modules that hardcode it RUN
    against it and the override is simply ignored: nothing skips. Claiming
    "every reference-gated test skipped" is then false, and a false line in
    an audit trail is the defect class this whole epic exists to remove. The
    status stays 6 regardless - the override was still not honored.
    """
    reason = unverified_reason(
        Path("/srv/decks"),
        [],
        allow_missing=False,
        legacy_modules=LEGACY_HARDCODED_MODULES,
        legacy_root_absent=[],
    )
    assert reason is not None
    assert "every reference-gated test skipped" not in reason
    assert "verified nothing" not in reason
    assert "ran against the default root instead" in reason
    assert "ticket-022" in reason


def test_a_relocated_root_whose_legacy_modules_also_skipped_says_so() -> None:
    """The mirror case: nothing ran anywhere, so the claim is true and is made."""
    reason = unverified_reason(
        Path("/srv/decks"),
        [],
        allow_missing=False,
        legacy_modules=LEGACY_HARDCODED_MODULES,
        legacy_root_absent=[DEFAULT_REFERENCE_ROOT / CORTDECO_NAME],
    )
    assert reason is not None
    assert "every reference-gated test skipped" in reason
    assert "skip there regardless" in reason


def test_the_default_root_is_unaffected_by_a_non_empty_legacy_list() -> None:
    """The regression-safety property this machine's own suite depends on."""
    reason = unverified_reason(
        DEFAULT_REFERENCE_ROOT, [], allow_missing=False, legacy_modules=LEGACY_HARDCODED_MODULES
    )
    assert reason is None


def test_unverified_reason_touches_no_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 2, verified rather than trusted: no `Path` probe fires."""

    def _explode(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("unverified_reason must not touch the filesystem")

    monkeypatch.setattr(Path, "exists", _explode)
    monkeypatch.setattr(Path, "is_dir", _explode)
    monkeypatch.setattr(Path, "is_file", _explode)

    assert (
        unverified_reason(Path("/anywhere"), [], allow_missing=False, legacy_modules=frozenset())
        is None
    )


# --- the hardcoded-path ratchet ------------------------------------------------


def test_the_hardcoded_reference_path_ratchet_matches_legacy_modules_exactly() -> None:
    """A twelfth module adding the literal fails here by name; a converted
    module must be removed from `LEGACY_HARDCODED_MODULES`.

    Targets `str(DEFAULT_REFERENCE_ROOT)` rather than a literal: this grep
    runs over every module under `tests/`, itself included, and this file is
    not one of the eleven.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tests_root = repo_root / "tests"
    literal = str(DEFAULT_REFERENCE_ROOT)

    found = {
        path.relative_to(repo_root).as_posix()
        for path in tests_root.rglob("*.py")
        if literal in path.read_text(encoding="utf-8")
    }

    extra = found - LEGACY_HARDCODED_MODULES
    missing = LEGACY_HARDCODED_MODULES - found
    assert not extra, (
        f"new hardcoded reference path(s), add to LEGACY_HARDCODED_MODULES: {sorted(extra)}"
    )
    assert not missing, (
        f"no longer hardcoded, remove from LEGACY_HARDCODED_MODULES: {sorted(missing)}"
    )


# --- the hook wiring, under pytest itself --------------------------------------


def test_a_session_with_every_artifact_present_reports_nothing(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = pytester.mkdir("reference_root")
    _populate(root)
    monkeypatch.setenv(REFERENCE_ROOT_VAR, str(root))
    monkeypatch.delenv(ALLOW_MISSING_VAR, raising=False)
    _install_wiring(pytester)

    result = pytester.runpytest_subprocess()

    result.assert_outcomes(passed=1)
    assert result.ret == 0
    combined = str(result.stdout)
    assert REFERENCE_ROOT_VAR not in combined
    assert ALLOW_MISSING_VAR not in combined


def test_an_absent_artifact_forces_exit_six_and_names_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = pytester.mkdir("reference_root")
    _populate(root, omit=CORTDECO_NAME)
    monkeypatch.setenv(REFERENCE_ROOT_VAR, str(root))
    monkeypatch.delenv(ALLOW_MISSING_VAR, raising=False)
    _install_wiring(pytester)

    result = pytester.runpytest_subprocess()

    result.assert_outcomes(passed=1)
    assert result.ret == 6
    combined = str(result.stdout)
    assert str(root / CORTDECO_NAME) in combined
    assert str(root) in combined
    assert ALLOW_MISSING_VAR in combined


def test_the_opt_out_moves_the_status_without_silencing_the_report(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = pytester.mkdir("reference_root")
    _populate(root, omit=CORTDECO_NAME)
    monkeypatch.setenv(REFERENCE_ROOT_VAR, str(root))
    monkeypatch.setenv(ALLOW_MISSING_VAR, "1")
    _install_wiring(pytester)

    result = pytester.runpytest_subprocess()

    result.assert_outcomes(passed=1)
    assert result.ret == 0
    combined = str(result.stdout)
    assert str(root / CORTDECO_NAME) in combined


def test_an_unresolvable_override_still_reports_the_outcomes_it_ran(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement: a malformed override degrades to unverified, never to a traceback.

    Before the guard caught it, the `RuntimeError` from `expanduser` escaped
    both hooks: the `1 passed` summary never printed and the exit status was
    1, indistinguishable from a real test failure.
    """
    monkeypatch.setenv(REFERENCE_ROOT_VAR, "~nosuchuser42/decks")
    monkeypatch.delenv(ALLOW_MISSING_VAR, raising=False)
    _install_wiring(pytester)

    result = pytester.runpytest_subprocess()

    result.assert_outcomes(passed=1)
    assert result.ret == 6
    combined = str(result.stdout)
    assert "Traceback" not in combined
    assert REFERENCE_ROOT_VAR in combined
    assert ALLOW_MISSING_VAR in combined
