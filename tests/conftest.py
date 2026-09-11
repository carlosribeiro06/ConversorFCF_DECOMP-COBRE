"""Session-level verdict: a suite that verified nothing must not report success.

Two hooks and nothing else. `reference.py` carries every filesystem- and
`pytest`-independent decision; this file only resolves the live filesystem
and environment, calls into it, and reacts. Nothing here propagates an
exception: the single call that can raise, `reference_root`, is caught and
reported like any other unverified session, because an exception escaping a
session hook aborts the whole test report.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from reference import (
    ALLOW_MISSING_VAR,
    DEFAULT_REFERENCE_ROOT,
    LEGACY_HARDCODED_MODULES,
    ReferenceRootError,
    reference_artifacts,
    reference_root,
    unresolvable_root_reason,
    unverified_reason,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.terminal import TerminalReporter
    from reference import ReferenceArtifacts

pytest_plugins = ["pytester"]


def _absent(artifacts: ReferenceArtifacts) -> list[Path]:
    """Which of the three artifacts are missing, in declaration order."""
    return [
        path
        for path, present in (
            (artifacts.case, artifacts.case.is_dir()),
            (artifacts.mapcut, artifacts.mapcut.is_file()),
            (artifacts.cortdeco, artifacts.cortdeco.is_file()),
        )
        if not present
    ]


def _session_reason() -> tuple[str | None, bool]:
    """The terminal-summary text (or `None`) and whether the opt-out is set."""
    allow_missing = os.environ.get(ALLOW_MISSING_VAR) == "1"
    try:
        root = reference_root()
    except ReferenceRootError as error:
        return unresolvable_root_reason(str(error), allow_missing), allow_missing

    absent = _absent(reference_artifacts(root))
    # Where the eleven legacy modules actually look, which decides whether the
    # summary may claim every reference-gated test skipped.
    legacy_root_absent = (
        absent
        if root == DEFAULT_REFERENCE_ROOT
        else _absent(reference_artifacts(DEFAULT_REFERENCE_ROOT))
    )
    reason = unverified_reason(
        root,
        absent,
        allow_missing,
        LEGACY_HARDCODED_MODULES,
        legacy_root_absent=legacy_root_absent,
    )
    return reason, allow_missing


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: pytest.ExitCode, config: pytest.Config
) -> None:
    """Print the unverified-session summary, if any."""
    reason, _ = _session_reason()
    if reason is not None:
        terminalreporter.write_line(reason)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """Force exit 6 for an unverified session, unless the opt-out is set."""
    reason, allow_missing = _session_reason()
    if reason is not None and not allow_missing:
        session.exitstatus = 6
