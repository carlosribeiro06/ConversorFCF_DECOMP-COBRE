"""Shared fixtures for `tests/integration/`.

`converted_pair` is the only fixture here. Every existing module in this
directory keeps its own `REFERENCE_CASE` constant and its own
`pytest.mark.skipif`, and this file does not touch any of them - it exists so
a module that needs the emitted pair does not have to run
`pipeline.run_conversion` itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conversor_fcf.config import load_settings
from conversor_fcf.paths import OutputPaths, resolve_output_paths
from conversor_fcf.pipeline import run_conversion

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CASE = Path("/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO")
TRACKED_SETTINGS = REPO_ROOT / "settings.json"


def _settings_with_absolute_hydro_codes(directory: Path) -> Path:
    """A settings copy whose `hydro_codes_path` does not depend on the CWD a
    test happens to run from: `conversion.hydro_codes_path` is resolved
    exactly as given, the same way `--settings settings.json`'s own default
    resolves against the working directory, so a test must supply an
    absolute one. Copied from
    `test_pipeline_end_to_end.py::_settings_with_absolute_hydro_codes`."""
    written = directory / "settings.json"
    payload = json.loads(TRACKED_SETTINGS.read_text(encoding="utf-8"))
    payload["conversion"]["hydro_codes_path"] = str(REPO_ROOT / "decomp_hydro_codes.json")
    written.write_text(json.dumps(payload), encoding="utf-8")
    return written


@pytest.fixture(scope="session")
def converted_pair(tmp_path_factory: pytest.TempPathFactory) -> OutputPaths:
    """Run `pipeline.run_conversion` on the reference Cobre case exactly once.

    Session-scoped: every consumer reads the emitted pair back through its
    own, different code path (an `idecomp` oracle or the native reader), so a
    second run would only spend the ~2.5 s conversion again for no additional
    evidence. Skips by name rather than passing silently when the reference
    case is absent.
    """
    if not REFERENCE_CASE.is_dir():
        pytest.skip(f"Cobre reference case not present at {REFERENCE_CASE}")

    settings_path = _settings_with_absolute_hydro_codes(tmp_path_factory.mktemp("settings"))
    settings = load_settings(settings_path)
    output_root = tmp_path_factory.mktemp("output") / "decomp_fcf"
    paths = resolve_output_paths(REFERENCE_CASE, "rv0", settings, output_override=output_root)
    code = run_conversion(REFERENCE_CASE, "rv0", paths, settings, include_terminal=False)
    assert code == 0, f"run_conversion exited {code} converting the reference case"
    return paths
