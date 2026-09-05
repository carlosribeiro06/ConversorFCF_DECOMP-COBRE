import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from conversor_fcf import run_manifest as run_manifest_module
from conversor_fcf.config import load_settings
from conversor_fcf.run_manifest import (
    PREMISES,
    build_run_manifest,
    write_run_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_SETTINGS = REPO_ROOT / "settings.json"

EXPECTED_KEYS = {
    "tool_version",
    "created_at",
    "case_path",
    "revision",
    "settings_snapshot",
    "premises",
    "library_versions",
    "outputs",
}


def _manifest_payload(tmp_path: Path) -> dict[str, object]:
    settings = load_settings(TRACKED_SETTINGS)
    manifest = build_run_manifest(
        case_path=Path("/cases/DEC_ONS_052026_RV0_VE_CONVERTIDO"),
        revision="rv0",
        settings=settings,
        outputs={"mapcut": Path("output/mapcut.rv0"), "cortdeco": Path("output/cortdeco.rv0")},
    )
    path = tmp_path / "nested" / "run_manifest.json"
    write_run_manifest(manifest, path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_premises_are_numbered_contiguously_from_one() -> None:
    assert len(PREMISES) == 12
    assert [entry.split(":", 1)[0] for entry in PREMISES] == [f"P{n}" for n in range(1, 13)]


def test_premise_eleven_names_every_zero_filled_reg_ten_field() -> None:
    """A count is not a name: an auditor must be able to see which fields are zero."""
    p11 = next(entry for entry in PREMISES if entry.startswith("P11:"))
    for field in (
        "parcela_custo_geracao_termica_minima",
        "parcela_custo_contrato_importacao_minimo",
        "parcela_custo_contrato_exportacao_minimo",
        "geracao_termica_minima_sinalizada_gnl",
        "geracao_termica_minima_gerada_gnl",
    ):
        assert field in p11, field
    assert "taxa_desconto" in p11


def test_premise_twelve_states_why_reg_nine_is_not_populated() -> None:
    """An unsettled axis is the reason, so the premise must say which two readings."""
    p12 = next(entry for entry in PREMISES if entry.startswith("P12:"))
    assert "730.5" in p12, "the reference's own per-submarket total anchors the claim"
    assert "ngnl*npat" in p12
    assert "ngnl*n_estagios" in p12, "both candidate axes must be named, not just the chosen one"


def test_manifest_has_all_eight_top_level_keys(tmp_path: Path) -> None:
    assert set(_manifest_payload(tmp_path)) == EXPECTED_KEYS


def test_manifest_records_every_premise(tmp_path: Path) -> None:
    assert len(_manifest_payload(tmp_path)["premises"]) == len(PREMISES)  # type: ignore[arg-type]


def test_manifest_records_tracked_library_versions(tmp_path: Path) -> None:
    versions = _manifest_payload(tmp_path)["library_versions"]
    assert isinstance(versions, dict)
    assert set(versions) == {"numpy", "pandas", "flatbuffers"}


def test_manifest_records_inputs_and_outputs(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    assert payload["revision"] == "rv0"
    assert payload["case_path"] == "/cases/DEC_ONS_052026_RV0_VE_CONVERTIDO"
    assert payload["outputs"] == {
        "mapcut": "output/mapcut.rv0",
        "cortdeco": "output/cortdeco.rv0",
    }


def test_absent_library_is_recorded_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _absent(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(run_manifest_module, "version", _absent)
    versions = _manifest_payload(tmp_path)["library_versions"]
    assert isinstance(versions, dict)
    assert set(versions.values()) == {"unknown"}


def test_manifest_json_is_key_sorted_and_indented(tmp_path: Path) -> None:
    settings = load_settings(TRACKED_SETTINGS)
    manifest = build_run_manifest(
        case_path=Path("/cases/x"), revision="rv0", settings=settings, outputs={}
    )
    path = tmp_path / "run_manifest.json"
    write_run_manifest(manifest, path)
    text = path.read_text(encoding="utf-8")
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert text.startswith('{\n  "case_path"')
    assert text.endswith("\n")


def test_manifest_is_deterministic_apart_from_created_at(tmp_path: Path) -> None:
    first = _manifest_payload(tmp_path / "a")
    second = _manifest_payload(tmp_path / "b")
    del first["created_at"], second["created_at"]
    assert first == second
