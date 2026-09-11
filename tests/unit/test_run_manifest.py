import json
import math
import os
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from conversor_fcf import run_manifest as run_manifest_module
from conversor_fcf.config import load_settings
from conversor_fcf.decomp import layout
from conversor_fcf.mapping.rules import (
    DAYS_PER_YEAR,
    EVIDENCED_GNL_LAG_MESES,
    EVIDENCED_GNL_LEAD_TIME_HOURS,
)
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
    "status",
    "failed_step",
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
    assert len(PREMISES) == 15
    assert [entry.split(":", 1)[0] for entry in PREMISES] == [f"P{n}" for n in range(1, 16)]


def test_premise_fifteen_names_the_evidenced_lead_time_and_both_candidate_formulas() -> None:
    """The premise must say why guessing is refused, not just that it is."""
    p15 = next(entry for entry in PREMISES if entry.startswith("P15:"))
    assert "1608.0" in p15, "the one evidenced lead time"
    assert "floor" in p15 and "round" in p15, "both candidate formulas must be named"
    assert "2.5" in p15, "where the two formulas would diverge"


def test_premise_fifteen_prints_the_quotient_it_actually_claims() -> None:
    """Recomputed from the named constants, never read back from the prose.

    Every other P15 assertion guards a claim and none guarded the arithmetic,
    which is how a wrong fifth digit reached committed source.
    """
    p15 = next(entry for entry in PREMISES if entry.startswith("P15:"))
    hours_per_month = DAYS_PER_YEAR * 24.0 / 12.0
    assert f"{hours_per_month:g}" in p15, "the month length the quotient divides by"
    quotient = EVIDENCED_GNL_LEAD_TIME_HOURS / hours_per_month
    assert f"{quotient:.4f}" in p15, "the quotient, to the digits the premise prints"


def test_both_candidate_formulas_map_the_evidenced_point_to_the_declared_lag() -> None:
    """P15 rests on floor and round agreeing at 1608.0. Check that, not the wording."""
    quotient = EVIDENCED_GNL_LEAD_TIME_HOURS / (DAYS_PER_YEAR * 24.0 / 12.0)
    assert math.floor(quotient) == EVIDENCED_GNL_LAG_MESES
    assert round(quotient) == EVIDENCED_GNL_LAG_MESES


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
    """Two reasons, and the decisive one is that the values are not derivable at all.

    A reader who sees only the axis ambiguity would conclude that settling the axis
    unblocks the block. It does not: the split belongs to the DECOMP deck.
    """
    p12 = next(entry for entry in PREMISES if entry.startswith("P12:"))
    assert "730.5" in p12, "the reference's own per-submarket total anchors the claim"
    assert "not derivable" in p12, "the decisive reason must be stated"
    assert "monthly" in p12 and "weekly" in p12, "the mismatch that makes it non-derivable"
    assert "ngnl*npat" in p12
    assert "ngnl*n_estagios" in p12, "both candidate axes must be named, not just the chosen one"


def test_premise_five_does_not_present_one_stage_as_the_whole_study() -> None:
    """24/65/79 is stage 0's split alone; naming it bare misled a ticket once already."""
    p5 = next(entry for entry in PREMISES if entry.startswith("P5:"))
    assert "per-stage" in p5
    for split in ("24/65/79", "15/64/89", "12/61/95", "51/226/323"):
        assert split in p5, f"the premise must name {split}"
    assert "600" in p5, "stage 6 spans 600 hours, not 168"


def test_premise_fourteen_separates_the_zeroed_values_from_the_dimensioned_block() -> None:
    """The distinction P14 lives or dies on.

    "GNL emitted as zeros" read as "GNL removed" would drop the block from
    `NCOEF` and shorten the coefficient span, producing a differently shaped file
    that still reads. The premise must therefore say what is unchanged, not only
    what is zero.
    """
    p14 = next(entry for entry in PREMISES if entry.startswith("P14:"))
    assert "Only the values are zero" in p14
    for unchanged in ("n_sbm_gnl", "codigos_submercados_gnl", "NCOEF", "file size"):
        assert unchanged in p14, f"P14 must state that {unchanged} is unchanged"
    assert "ticket-016" in p14, "a deferred premise must name its follow-up"
    assert "invisibly" in p14, "why this differs from P1/P11/P12's declared zeros"


def test_premises_four_and_five_declare_themselves_dormant() -> None:
    """A premise with no effect on the output pair must say so.

    P4 negates GNL coefficients and P5 spreads them across load blocks. With the
    whole block zeroed by P14, neither changes a byte, and a manifest that
    asserts them flatly would overstate what the file honours.
    """
    for number in ("P4", "P5"):
        entry = next(item for item in PREMISES if item.startswith(f"{number}:"))
        assert "DORMANT while P14 holds" in entry, f"{number} must declare its dormancy"
        assert "ticket-016" in entry, f"{number} must say why the code is retained"


def test_no_premise_text_states_how_many_premises_there_are() -> None:
    """A count in prose beside the tuple that defines it has gone stale once already.

    Both spellings are rejected: a bare numeral form like "all 14 premises" slips
    past a word list. "ten premises" carries its noun because bare "ten" matches
    "written" inside P13, which is why this list is deliberately asymmetric.
    """
    words = ("ten premises", "eleven", "twelve", "thirteen", "fourteen", "fifteen")
    numerals = tuple(f"{n} premises" for n in range(2, 31))
    for entry in PREMISES:
        for count in words + numerals:
            assert count not in entry.lower(), f"{entry.split(':', 1)[0]} carries a count"


def test_manifest_has_every_top_level_key(tmp_path: Path) -> None:
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


# --- M7: status and failed_step ---------------------------------------------


def test_the_default_manifest_reports_success_with_no_failed_step(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    assert payload["status"] == "ok"
    assert payload["failed_step"] is None


def test_a_failed_run_records_its_status_and_the_step_that_raised(tmp_path: Path) -> None:
    settings = load_settings(TRACKED_SETTINGS)
    manifest = build_run_manifest(
        case_path=Path("/cases/x"),
        revision="rv0",
        settings=settings,
        outputs={"mapcut": Path("output/mapcut.rv0")},
        status="failed",
        failed_step="write cortdeco",
    )
    path = tmp_path / "run_manifest.json"
    write_run_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failed_step"] == "write cortdeco"
    # M7: outputs are still recorded, even though the run never reached the end.
    assert payload["outputs"] == {"mapcut": "output/mapcut.rv0"}


# --- atomicity ---------------------------------------------------------------


def test_the_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run_manifest.json"
    settings = load_settings(TRACKED_SETTINGS)
    manifest = build_run_manifest(
        case_path=Path("/cases/x"), revision="rv0", settings=settings, outputs={}
    )
    write_run_manifest(manifest, path)
    assert path.is_file()
    assert not list(path.parent.glob("*.partial"))


def test_the_payload_is_synced_before_the_rename_and_the_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durability, in the order that makes it durability rather than decoration.

    The same idiom `cortdeco_writer` and `mapcut_writer` already use: `os.replace`
    is atomic for readers but not against a machine crash, so the payload must
    reach stable storage, and the rename's own directory entry after it, before
    either writer or the manifest can be trusted to have really landed.
    """
    events: list[str] = []
    real_replace = os.replace
    real_write_durably = layout.write_durably
    real_sync_directory = layout.sync_directory

    def spy_write(target: Path, payload: bytes) -> None:
        real_write_durably(target, payload)
        events.append(f"fsync:{target.name}")

    def spy_replace(source: object, destination: object) -> None:
        events.append("replace")
        real_replace(source, destination)  # type: ignore[arg-type]

    def spy_sync_directory(target: Path) -> None:
        real_sync_directory(target)
        events.append("fsync:dir")

    monkeypatch.setattr(run_manifest_module, "write_durably", spy_write)
    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(run_manifest_module, "sync_directory", spy_sync_directory)

    path = tmp_path / "run_manifest.json"
    settings = load_settings(TRACKED_SETTINGS)
    manifest = build_run_manifest(
        case_path=Path("/cases/x"), revision="rv0", settings=settings, outputs={}
    )
    write_run_manifest(manifest, path)

    assert events == ["fsync:run_manifest.json.partial", "replace", "fsync:dir"]
    assert path.is_file()


def test_a_failure_between_write_and_rename_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run_manifest.json"

    def raising_replace(source: object, destination_: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", raising_replace)
    settings = load_settings(TRACKED_SETTINGS)
    manifest = build_run_manifest(
        case_path=Path("/cases/x"), revision="rv0", settings=settings, outputs={}
    )
    with pytest.raises(OSError, match="simulated rename failure"):
        write_run_manifest(manifest, destination)

    assert not destination.exists()
    assert not list(tmp_path.iterdir()), "the partial file must be cleaned up"
