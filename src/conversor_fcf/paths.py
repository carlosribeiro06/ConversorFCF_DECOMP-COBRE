"""Revision and output-path resolution.

Every function here is pure with respect to the filesystem apart from the two
that deliberately inspect it: nothing in this module creates a directory or a
file. Resolution has to stay side-effect free because a run refused for an
existing artifact must leave nothing behind, and only the writers create
directories.

The revision appears in two cases on disk, which is transcribed from the
reference case rather than assumed: the case directory carries it upper case
(`DEC_ONS_052026_RV0_VE_CONVERTIDO`) while the DECOMP files carry it lower case
(`dadger.rv0`, `mapcut.rv0`). The lower-case form is the one that reaches a
filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from conversor_fcf.config import Settings
from conversor_fcf.logging_setup import get_logger

_logger = get_logger("paths")

POLICY_MANIFEST_RELATIVE = Path("output") / "policy" / "manifest.bin"

ACCEPTED_REVISIONS = "rv0 through rv9, or rvf for the final revision"

_REVISION = re.compile(r"^rv[0-9f]$")

# Delimited by underscores or the ends of the name, matched with a lookahead so a
# trailing underscore is not consumed: without it, `RV0_RV1` would yield only the
# first token and an ambiguous name would read as an unambiguous one.
_CASE_REVISION = re.compile(r"(?:^|_)(RV[0-9F])(?=_|$)")


class PathError(Exception):
    """Raised when a revision or an output path cannot be resolved."""


def normalize_revision(text: str) -> str:
    """Case-fold and validate a declared revision (K1)."""
    revision = text.strip().lower()
    if not _REVISION.match(revision):
        raise PathError(f"revision {text!r} is not one of {ACCEPTED_REVISIONS}")
    return revision


def revision_from_case_dir(case_dir: Path) -> str | None:
    """The revision the case directory name carries, or `None` if it carries none."""
    found: set[str] = {match.lower() for match in _CASE_REVISION.findall(case_dir.name.upper())}
    if not found:
        return None
    if len(found) > 1:
        raise PathError(
            f"the case directory name {case_dir.name!r} carries more than one revision "
            f"({', '.join(sorted(found))}); pass --revision to say which one applies"
        )
    return found.pop()


def resolve_revision(case_dir: Path, declared: str | None) -> str:
    """The revision to use, cross-checking the case name against `--revision` (K1).

    The cross-check is the point: deriving it silently would convert an RV1 case
    into a `mapcut.rv0` with nothing in the artifacts to show it happened, and
    requiring the flag alone would leave the same mistake possible in the other
    direction.
    """
    derived = revision_from_case_dir(case_dir)
    if declared is None:
        if derived is None:
            raise PathError(
                f"the case directory name {case_dir.name!r} carries no revision, so --revision is "
                f"required ({ACCEPTED_REVISIONS})"
            )
        _logger.info("revision %s derived from the case directory name", derived)
        return derived

    normalized = normalize_revision(declared)
    if derived is not None and derived != normalized:
        raise PathError(
            f"the case directory name {case_dir.name!r} says {derived!r} but --revision says "
            f"{normalized!r}; refusing to write {normalized} artifacts from a {derived} case"
        )
    return normalized


@dataclass(frozen=True)
class OutputPaths:
    """Every artifact a run writes, all absolute.

    The revision appears **only** in the two binary filenames, because those are
    the names DECOMP reads. The CSV directories and the manifest do not carry it:
    the root is already per-case and, since a case directory names its own
    revision, per-revision by construction.
    """

    root: Path
    mapcut: Path
    cortdeco: Path
    run_manifest: Path
    eco_dir: Path
    content_dir: Path
    log: Path


def resolve_output_paths(
    case_dir: Path,
    revision: str,
    settings: Settings,
    output_override: Path | None = None,
) -> OutputPaths:
    """Resolve every output path, creating nothing (K2, A2).

    A relative `settings.output.directory` resolves against the **case**, not the
    working directory, so the artifacts stay with the case that produced them and
    a study is self-contained. A relative `settings.logging.file_path` resolves
    against the output root, so a run's audit trail sits with the artifacts it
    describes; an absolute path is honoured as given.
    """
    case = case_dir.resolve()
    if output_override is not None:
        root = output_override.resolve()
    else:
        configured = Path(settings.output.directory)
        root = (configured if configured.is_absolute() else case / configured).resolve()

    log_configured = Path(settings.logging.file_path)
    log = log_configured if log_configured.is_absolute() else root / log_configured

    return OutputPaths(
        root=root,
        mapcut=root / f"mapcut.{revision}",
        cortdeco=root / f"cortdeco.{revision}",
        run_manifest=root / "run_manifest.json",
        eco_dir=root / settings.output.eco_subdirectory,
        content_dir=root / settings.output.content_subdirectory,
        log=log,
    )


def assert_outputs_absent(paths: OutputPaths, force: bool) -> None:
    """Refuse to overwrite an existing artifact unless `force` (K3).

    Names **every** existing artifact rather than the first, so one run tells the
    operator everything they have to move. Called before any ingestion, so a
    repeated command costs milliseconds rather than a 0.5 GB read.
    """
    existing = [
        path for path in (paths.mapcut, paths.cortdeco, paths.run_manifest) if path.exists()
    ]
    if not existing:
        return
    if not force:
        raise PathError(
            f"refusing to overwrite {len(existing)} existing artifact(s): "
            f"{', '.join(str(path) for path in existing)}. Move them aside or pass --force"
        )
    for path in existing:
        _logger.warning("--force given: replacing existing artifact %s", path)


def assert_case_readable(case_dir: Path) -> Path:
    """Check the case exists and looks converted, returning its absolute path.

    Checked before any read so a wrong path fails in milliseconds. `manifest.bin`
    is the file the policy ingestion needs first, so its absence is the cheapest
    proof that this directory is not a converted Cobre case.
    """
    case = case_dir.resolve()
    if not case.is_dir():
        raise PathError(f"case directory not found: {case}")
    manifest = case / POLICY_MANIFEST_RELATIVE
    if not manifest.is_file():
        raise PathError(f"{case} does not look like a converted Cobre case: {manifest} is missing")
    return case
