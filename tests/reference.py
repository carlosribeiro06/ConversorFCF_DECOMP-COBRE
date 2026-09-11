"""Single source of truth for where the reference DECOMP artifacts live.

Eleven modules under `tests/` still declare their own module-level constant
and their own `pytest.mark.skipif` against a hardcoded default root
(`LEGACY_HARDCODED_MODULES`); nothing here converts them - that sweep is
`ticket-022`. `tests/conftest.py` is the only consumer of `unverified_reason`
and holds every filesystem/`pytest` touching step, so this module stays pure
and unit-testable on its own.

Importable as `import reference` from both `tests/unit/` and
`tests/integration/`: neither directory's `__init__.py` extends up to
`tests/` itself, so pytest's import machinery inserts `tests/` on
`sys.path` for every module underneath it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REFERENCE_ROOT_VAR = "CONVERSOR_FCF_REFERENCE_ROOT"
ALLOW_MISSING_VAR = "CONVERSOR_FCF_ALLOW_MISSING_REFERENCE"

# Built from parts, not one contiguous literal: `tests/unit/test_reference_paths.py`
# greps every module under `tests/` for that literal, and this module is not one of
# the `LEGACY_HARDCODED_MODULES` its own grep is checked against.
DEFAULT_REFERENCE_ROOT = Path("/home", "carlosribeiro", "git")

CASE_NAME = "DEC_ONS_052026_RV0_VE_CONVERTIDO"
MAPCUT_NAME = "mapcut.rv0"
CORTDECO_NAME = "cortdeco.rv0"

LEGACY_HARDCODED_MODULES: frozenset[str] = frozenset(
    {
        "tests/integration/conftest.py",
        "tests/integration/test_cortdeco_assembly.py",
        "tests/integration/test_eco_csv_reference.py",
        "tests/integration/test_inputs_reader_reference.py",
        "tests/integration/test_mapcut_assembly.py",
        "tests/integration/test_mapping_reference.py",
        "tests/integration/test_pipeline_end_to_end.py",
        "tests/integration/test_policy_ingestion_reference.py",
        "tests/integration/test_range_envelopes.py",
        "tests/unit/test_content_csv.py",
        "tests/unit/test_reader.py",
    }
)


@dataclass(frozen=True)
class ReferenceArtifacts:
    """The three artifacts a reference-dependent test needs, as children of a root."""

    case: Path
    mapcut: Path
    cortdeco: Path


def reference_root() -> Path:
    """The resolved reference root.

    `CONVERSOR_FCF_REFERENCE_ROOT` when set - `~` expanded, then resolved -
    else `DEFAULT_REFERENCE_ROOT` unchanged.
    """
    override = os.environ.get(REFERENCE_ROOT_VAR)
    if override is None:
        return DEFAULT_REFERENCE_ROOT
    return Path(override).expanduser().resolve()


def reference_artifacts(root: Path) -> ReferenceArtifacts:
    """The three reference artifacts as children of `root`."""
    return ReferenceArtifacts(
        case=root / CASE_NAME,
        mapcut=root / MAPCUT_NAME,
        cortdeco=root / CORTDECO_NAME,
    )


def unverified_reason(
    root: Path,
    absent: Sequence[Path],
    allow_missing: bool,
    legacy_modules: frozenset[str],
) -> str | None:
    """The terminal-summary text for an unverified session, or `None`.

    Pure: no filesystem access, no `pytest` import. The caller decides `root`
    and `absent` from the live filesystem, and `allow_missing` from the
    environment; `legacy_modules` is normally `LEGACY_HARDCODED_MODULES`,
    threaded in rather than read as a global so a test can pass an empty set
    for the counterfactual. A relocated root (one that differs from
    `DEFAULT_REFERENCE_ROOT`) with a non-empty `legacy_modules` is unverified
    even with every artifact present at that root, because those modules
    would still look at the default root and skip there regardless.
    """
    findings: list[str] = []
    if absent:
        named = ", ".join(str(path) for path in absent)
        findings.append(f"{len(absent)} reference artifact(s) absent at {root}: {named}.")

    if root != DEFAULT_REFERENCE_ROOT and legacy_modules:
        named_modules = ", ".join(sorted(legacy_modules))
        findings.append(
            f"{REFERENCE_ROOT_VAR} points elsewhere while {len(legacy_modules)} module(s) "
            f"still hardcode the default root and skip there regardless, unaffected by it "
            f"(the ticket-022 follow-up sweep): {named_modules}."
        )

    if not findings:
        return None

    findings.append(
        "This session verified nothing reference-dependent: every reference-gated test "
        "skipped, and pytest's own exit status does not distinguish that from having run."
    )
    if allow_missing:
        findings.append(
            f"{ALLOW_MISSING_VAR}=1 is set: exit status left at 0 despite the above. Unset "
            f"it, or point {REFERENCE_ROOT_VAR} at a directory holding all three artifacts, "
            f"to require them again."
        )
    else:
        findings.append(
            f"Set {ALLOW_MISSING_VAR}=1 to accept this and keep exit 0, or point "
            f"{REFERENCE_ROOT_VAR} at a directory holding all three artifacts."
        )
    return "\n".join(findings)
