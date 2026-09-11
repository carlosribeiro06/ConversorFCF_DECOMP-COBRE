"""Drift tests: the README must quote what the code actually says and prints.

Two guarantees, one per pure guard. `missing_premises` holds the README's
`## Known premises` section to every string `run_manifest.PREMISES` carries.
`undocumented_guard_lines` holds the captured transcript in `### Running
against the reference artifacts` to what `reference.unverified_reason` really
emits - the defect class that recurred five times in this plan was a document
asserting something nobody re-derived from the code, and a hand-written
transcript is exactly that.

Each guard comes with the executed-mutation pair this project requires of any
new check (Requirement 6): one test shows the real README passes, the other
drifts a stub and shows the guard names what went missing rather than passing
silently. Every README read in the suite lives in this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from reference import LEGACY_HARDCODED_MODULES, reference_artifacts, unverified_reason

from conversor_fcf.run_manifest import PREMISES

_WHITESPACE = re.compile(r"\s+")

_README = Path(__file__).resolve().parents[2] / "README.md"


def _normalize(text: str) -> str:
    """Collapse every run of whitespace to a single space."""
    return _WHITESPACE.sub(" ", text).strip()


def missing_premises(readme_text: str, premises: Sequence[str]) -> tuple[int, ...]:
    """0-based indices of `premises` absent from `readme_text`, whitespace-normalized.

    Pure: no filesystem access, no `pytest` import. The caller resolves
    `readme_text`, so this stays testable against both the real README and a
    stub. Normalizing both sides is what lets a premise wrapped across README
    lines still match the single-line string `PREMISES` holds.
    """
    normalized_readme = _normalize(readme_text)
    return tuple(
        index
        for index, premise in enumerate(premises)
        if _normalize(premise) not in normalized_readme
    )


def test_every_premise_is_quoted_in_the_real_readme() -> None:
    readme_text = _README.read_text(encoding="utf-8")
    absent = missing_premises(readme_text, PREMISES)
    assert not absent, "\n".join(
        f"premise index {index} is not quoted in {_README.name}: {PREMISES[index][:80]!r}..."
        for index in absent
    )


def test_a_removed_premise_is_reported_by_its_own_index() -> None:
    """The executed mutation: drop one premise from a stub and expect its index back."""
    removed_index = 4
    stub = "\n\n".join(premise for index, premise in enumerate(PREMISES) if index != removed_index)
    assert missing_premises(stub, PREMISES) == (removed_index,)


# --- the captured guard transcript ---------------------------------------------

_DOCUMENTED_OVERRIDE_ROOT = Path("/srv/decks")


def undocumented_guard_lines(readme_text: str, reason: str) -> tuple[str, ...]:
    """Lines of `reason` the README does not quote, whitespace-normalized.

    Pure, for the same reason `missing_premises` is. Each line is compared
    only up to the module list it may end with: the README deliberately
    elides those eleven paths, so just the sentence introducing them has to
    match. Normalizing both sides lets a line the README wrapped still match
    the single line the guard prints.
    """
    normalized_readme = _normalize(readme_text)
    return tuple(
        line
        for line in reason.splitlines()
        if _normalize(line.split(": tests/")[0]) not in normalized_readme
    )


def documented_guard_reason() -> str:
    """The guard summary the README's captured transcript claims to show.

    The same inputs as the documented command: an override naming a root that
    holds none of the three artifacts, while the eleven legacy modules still
    read a populated default root and therefore run against it.
    """
    artifacts = reference_artifacts(_DOCUMENTED_OVERRIDE_ROOT)
    reason = unverified_reason(
        _DOCUMENTED_OVERRIDE_ROOT,
        [artifacts.case, artifacts.mapcut, artifacts.cortdeco],
        allow_missing=False,
        legacy_modules=LEGACY_HARDCODED_MODULES,
        legacy_root_absent=[],
    )
    assert reason is not None, "the documented scenario must produce a summary"
    return reason


def test_the_readme_transcript_is_what_the_guard_actually_prints() -> None:
    readme_text = _README.read_text(encoding="utf-8")
    undocumented = undocumented_guard_lines(readme_text, documented_guard_reason())
    assert not undocumented, "\n".join(
        f"the guard prints a line {_README.name} does not quote: {line[:100]!r}..."
        for line in undocumented
    )


def test_a_reworded_guard_line_is_reported_back() -> None:
    """The executed mutation: drop one line from a stub and expect it named."""
    reason = documented_guard_reason()
    lines = reason.splitlines()
    dropped = lines[-1]
    stub = "\n".join(lines[:-1])
    assert undocumented_guard_lines(stub, reason) == (dropped,)
