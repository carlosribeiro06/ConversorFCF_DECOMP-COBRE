"""Command-line entry point.

Argument parsing, revision resolution and output-path resolution only: the
conversion itself lives behind `pipeline.run_conversion`, imported here as the
package's single call site into it. That seam is what turns "premise logged
once per run" from a per-call guarantee into a real one (the caveat
`ticket-008` recorded as item C8).

Exit codes are per error class, so an official wrapper can branch on the failure
without parsing text. `2` is argparse's own usage error and is left to it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from conversor_fcf import __version__
from conversor_fcf.cobre.inputs_reader import InputReadError
from conversor_fcf.cobre.policy_reader import PolicyFormatError
from conversor_fcf.config import ConfigError, load_settings
from conversor_fcf.decomp.layout import LayoutError
from conversor_fcf.decomp.reader import ReadError
from conversor_fcf.logging_setup import configure_logging, get_logger
from conversor_fcf.mapping.rules import MappingError
from conversor_fcf.paths import (
    PathError,
    assert_case_readable,
    assert_outputs_absent,
    resolve_output_paths,
    resolve_revision,
)
from conversor_fcf.pipeline import run_conversion

EXIT_OK = 0
EXIT_CONFIG = 3
EXIT_INPUT = 4
EXIT_MAPPING = 5
EXIT_LAYOUT = 6
EXIT_PATH = 7

# Order matters only for readability: the classes are disjoint.
_EXIT_CODES: tuple[tuple[type[Exception], int], ...] = (
    (ConfigError, EXIT_CONFIG),
    (InputReadError, EXIT_INPUT),
    (PolicyFormatError, EXIT_INPUT),
    (MappingError, EXIT_MAPPING),
    (LayoutError, EXIT_LAYOUT),
    # ReadError shares LayoutError's code: the file on disk is not what the
    # layout it declares says it should be, which is the same class of
    # failure to an operator (Fix 3). No new code is added to A1's table.
    (ReadError, EXIT_LAYOUT),
    (PathError, EXIT_PATH),
)
_HANDLED = tuple(kind for kind, _ in _EXIT_CODES)

_logger = get_logger("cli")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conversor-fcf",
        description=(
            "Convert a Cobre case into a DECOMP-format mapcut/cortdeco pair, so DESSEM can "
            "couple to a Cobre-produced future cost function."
        ),
    )
    parser.add_argument("case", type=Path, help="path to the converted Cobre case directory")
    parser.add_argument(
        "--revision",
        help=(
            "PMO revision (rv0..rv9, rvf). Derived from the case directory name when omitted; "
            "when given, it must agree with the name."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output root, overriding settings.output.directory",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.json"),
        help="path to the settings file (default: settings.json)",
    )
    terminal = parser.add_mutually_exclusive_group()
    terminal.add_argument(
        "--include-terminal-pool",
        dest="include_terminal",
        action="store_true",
        help="convert the terminal pool as well (overrides settings)",
    )
    terminal.add_argument(
        "--no-include-terminal-pool",
        dest="include_terminal",
        action="store_false",
        help="exclude the terminal pool (overrides settings)",
    )
    # None, not False: an unset flag must not override a settings file saying true,
    # which is the opposite of what K4 asks for.
    parser.set_defaults(include_terminal=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing artifacts instead of refusing",
    )
    parser.add_argument("--version", action="version", version=f"conversor-fcf {__version__}")
    return parser


def _exit_code_for(error: Exception) -> int:
    """The exit code for a handled error. Only ever called with one of `_HANDLED`."""
    return next(code for kind, code in _EXIT_CODES if isinstance(error, kind))


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve everything, refuse what must be refused, then hand off.

    The order is load-bearing: logging is configured **before** the
    existing-artifact refusal so that refusal lands in the audit trail, and the
    refusal comes before case validation because it is the cheaper of the two
    and both precede any read.
    """
    arguments = _parser().parse_args(argv)
    try:
        settings = load_settings(arguments.settings)
        revision = resolve_revision(arguments.case, arguments.revision)
        paths = resolve_output_paths(arguments.case, revision, settings, arguments.output)
        # The resolved log path, not the settings' own: a relative one belongs with
        # the artifacts it describes (A2). This is the first step that touches the
        # filesystem, and it deliberately creates the log's directory even for a run
        # about to be refused - an audit trail that does not exist cannot record why.
        configure_logging(replace(settings.logging, file_path=str(paths.log)))
        assert_outputs_absent(paths, arguments.force)
        case_dir = assert_case_readable(arguments.case)
        include_terminal = (
            settings.conversion.include_terminal_pool
            if arguments.include_terminal is None
            else arguments.include_terminal
        )
        return run_conversion(case_dir, revision, paths, settings, include_terminal)
    except _HANDLED as error:
        # Both, not either: the audit log is the record, but a ConfigError can fire
        # before logging is configured at all, and an operator must still see why
        # the run refused. stderr is the only channel guaranteed to exist here.
        _logger.error("%s: %s", type(error).__name__, error)
        print(f"conversor-fcf: {type(error).__name__}: {error}", file=sys.stderr)
        return _exit_code_for(error)


if __name__ == "__main__":
    raise SystemExit(main())
