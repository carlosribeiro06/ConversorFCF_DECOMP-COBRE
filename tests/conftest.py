"""Session-level verdict: a suite that verified nothing must not report success.

Two hooks and nothing else. `reference.py` carries every filesystem- and
`pytest`-independent decision; this file only resolves the live filesystem
and environment, calls `unverified_reason`, and reacts - short enough that
reading it proves it cannot raise.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from reference import (
    ALLOW_MISSING_VAR,
    LEGACY_HARDCODED_MODULES,
    reference_artifacts,
    reference_root,
    unverified_reason,
)

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter

pytest_plugins = ["pytester"]


def _session_reason() -> tuple[str | None, bool]:
    """The terminal-summary text (or `None`) and whether the opt-out is set."""
    root = reference_root()
    artifacts = reference_artifacts(root)
    absent = [
        path
        for path, present in (
            (artifacts.case, artifacts.case.is_dir()),
            (artifacts.mapcut, artifacts.mapcut.is_file()),
            (artifacts.cortdeco, artifacts.cortdeco.is_file()),
        )
        if not present
    ]
    allow_missing = os.environ.get(ALLOW_MISSING_VAR) == "1"
    reason = unverified_reason(root, absent, allow_missing, LEGACY_HARDCODED_MODULES)
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
