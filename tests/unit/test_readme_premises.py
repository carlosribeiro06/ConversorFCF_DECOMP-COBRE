"""Drift test: `## Known premises` must quote every string `run_manifest.PREMISES` holds.

`missing_premises` is the pure guard behind the guarantee; the two tests below
are the executed-mutation pair this project requires of any new check
(Requirement 6): one shows the real README passes, the other removes a real
premise and shows the guard names it back by index rather than passing
silently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

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
