"""Unit tests for the `mapcut` layout invariant.

Per the master plan's Numeric Path Testing Policy, the two named deliberate
mutations for this ticket are **a file missing one record** and **a dirty byte
inside the zero-filled physical span**. Both are built from a real written file
rather than simulated, and both were demonstrated by removing the corresponding
check from `assert_mapcut_layout` and observing the failures.

The anchor 356 is the reference `mapcut.rv0`'s own record count, transcribed
from its 17,095,120-byte size rather than computed by the code under test.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from conversor_fcf.cobre.policy_reader import EntitySlotRecord
from conversor_fcf.decomp import mapcut_writer
from conversor_fcf.decomp.layout import (
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    LayoutError,
    assert_mapcut_layout,
    assert_no_travel_time,
    assert_record_multiple,
    derive_n_utv,
    mapcut_record_count,
)
from conversor_fcf.decomp.mapcut_writer import MapcutHeader, write_mapcut

# The reference deck's own dimensions and the file they describe.
REFERENCE_DIMENSIONS = {"n_utv": 2, "n_estagios": 7, "max_lag": 3, "n_cenarios": 273}
REFERENCE_RECORD_COUNT = 356
REFERENCE_FILE_BYTES = 17_095_120

# This project's own dimensions, with premise P3's n_utv = 0.
PROJECT_RECORD_COUNT = 300
PROJECT_FILE_BYTES = 14_406_000

HeaderFactory = Callable[..., MapcutHeader]


@pytest.fixture(autouse=True)
def propagating_package_logger() -> Iterator[None]:
    """caplog reads through the root logger, so propagation must be on."""
    logger = logging.getLogger("conversor_fcf")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


def _slot(entity_type: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=entity_type, entity_id=0, subindex=0, was_active=True, delivery_date=0
    )


def test_record_count_reproduces_the_reference_file() -> None:
    """356 is the reference file's real record count, not a computed expectation."""
    assert mapcut_record_count(**REFERENCE_DIMENSIONS) == REFERENCE_RECORD_COUNT
    assert REFERENCE_FILE_BYTES == REFERENCE_RECORD_COUNT * RECORD_SIZE


def test_record_count_for_this_project_omits_the_travel_time_span() -> None:
    assert mapcut_record_count(n_utv=0, n_estagios=7, max_lag=0, n_cenarios=273) == (
        PROJECT_RECORD_COUNT
    )
    assert PROJECT_FILE_BYTES == PROJECT_RECORD_COUNT * RECORD_SIZE


def test_max_lag_only_matters_when_there_is_travel_time() -> None:
    assert mapcut_record_count(0, 7, 0, 273) == mapcut_record_count(0, 7, 99, 273)
    assert mapcut_record_count(2, 7, 3, 273) != mapcut_record_count(2, 7, 4, 273)


@pytest.mark.parametrize("field", ["n_utv", "n_estagios", "max_lag", "n_cenarios"])
def test_negative_dimensions_are_rejected(field: str) -> None:
    kwargs = dict(REFERENCE_DIMENSIONS)
    kwargs[field] = -1
    with pytest.raises(LayoutError, match=f"{field} must be non-negative"):
        mapcut_record_count(**kwargs)


def test_a_size_that_is_not_a_whole_record_is_rejected() -> None:
    assert assert_record_multiple(RECORD_SIZE * 3, "x") == 3
    with pytest.raises(LayoutError, match="trailing bytes"):
        assert_record_multiple(RECORD_SIZE * 3 + 1, "x")


# --- premise P3 as a guard, not a constant ---------------------------------


def test_travel_time_axes_are_counted_not_assumed() -> None:
    """All four axis types, so the count cannot be right by accident."""
    every_type = [_slot(0), _slot(1), _slot(2), _slot(3)]
    assert derive_n_utv(every_type) == 1
    assert derive_n_utv([_slot(0), _slot(1), _slot(2)]) == 0
    assert derive_n_utv([_slot(3), _slot(3), _slot(3)]) == 3
    assert derive_n_utv([]) == 0


def test_a_case_with_travel_time_state_is_refused() -> None:
    with pytest.raises(LayoutError) as excinfo:
        assert_no_travel_time([_slot(0), _slot(3)])
    message = str(excinfo.value)
    assert "carries 1 " in message, "the count must be named, not merely present as a digit"
    assert "NCOEF" in message
    assert "P3" in message


def test_a_case_without_travel_time_state_logs_premise_p3_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="conversor_fcf"):
        assert_no_travel_time([_slot(0), _slot(2)])
    premise = [r.getMessage() for r in caplog.records if "premise P3 holds" in r.getMessage()]
    assert len(premise) == 1, "one INFO per call, not one per slot"
    assert "regs 7/8" in premise[0]
    assert "NCOEF" in premise[0]


def test_a_written_file_satisfies_the_layout_its_header_declares(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    count = write_mapcut(header, path)
    assert_mapcut_layout(path, header)
    assert path.stat().st_size == count * RECORD_SIZE


def test_a_file_missing_one_record_is_refused(tmp_path: Path, mapcut_header: HeaderFactory) -> None:
    """First named mutation: a truncated file, built from a real written one."""
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    count = write_mapcut(header, path)

    truncated = tmp_path / "truncated.rv0"
    truncated.write_bytes(path.read_bytes()[: (count - 1) * RECORD_SIZE])
    with pytest.raises(LayoutError) as excinfo:
        assert_mapcut_layout(truncated, header)
    message = str(excinfo.value)
    assert f"holds {count - 1} records" in message
    assert f"declares {count}" in message


def test_a_file_with_trailing_bytes_is_refused(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    write_mapcut(header, path)

    ragged = tmp_path / "ragged.rv0"
    ragged.write_bytes(path.read_bytes() + b"\x01")
    with pytest.raises(LayoutError, match="records plus 1 trailing bytes"):
        assert_mapcut_layout(ragged, header)


def test_a_header_that_disagrees_with_a_correct_file_is_refused(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    """The reason the check reads the header: otherwise both could be wrong together."""
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    count = write_mapcut(header, path)

    overstated = replace(header, numero_cenarios=header.numero_cenarios + 1)
    with pytest.raises(LayoutError) as excinfo:
        assert_mapcut_layout(path, overstated)
    message = str(excinfo.value)
    assert f"holds {count} records" in message
    assert f"declares {count + 1}" in message
    assert f"n_cenarios={header.numero_cenarios + 1}" in message


def test_a_dirty_byte_in_the_physical_span_is_refused(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    """Second named mutation: no input can dirty that span, so it is set by hand."""
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    write_mapcut(header, path)

    raw = bytearray(path.read_bytes())
    raw[9 * RECORD_SIZE + 17] = 1
    dirty = tmp_path / "dirty.rv0"
    dirty.write_bytes(bytes(raw))

    with pytest.raises(LayoutError) as excinfo:
        assert_mapcut_layout(dirty, header)
    message = str(excinfo.value)
    assert "record 9" in message
    assert "byte 17" in message
    assert f"records {PHYSICAL_RECORD_FIRST}-{PHYSICAL_RECORD_LAST}" in message


@pytest.mark.parametrize("index", [PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST])
def test_both_ends_of_the_physical_span_are_scanned(
    tmp_path: Path, mapcut_header: HeaderFactory, index: int
) -> None:
    """An off-by-one in the span bounds would leave one end unchecked."""
    path = tmp_path / "mapcut.rv0"
    header = mapcut_header()
    write_mapcut(header, path)

    raw = bytearray(path.read_bytes())
    raw[(index + 1) * RECORD_SIZE - 1] = 255
    dirty = tmp_path / "dirty.rv0"
    dirty.write_bytes(bytes(raw))

    with pytest.raises(LayoutError, match=f"record {index} is inside"):
        assert_mapcut_layout(dirty, header)


def test_the_layout_check_runs_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mapcut_header: HeaderFactory
) -> None:
    """A file that fails its own invariant must never appear at the destination."""
    destination = tmp_path / "mapcut.rv0"
    observed: dict[str, object] = {}

    def spy(path: Path, header: MapcutHeader) -> None:
        observed["checked"] = path.name
        observed["destination_existed"] = destination.exists()
        raise LayoutError("simulated invariant failure")

    monkeypatch.setattr(mapcut_writer, "assert_mapcut_layout", spy)
    with pytest.raises(LayoutError, match="simulated invariant failure"):
        write_mapcut(mapcut_header(), destination)

    assert observed["checked"] == "mapcut.rv0.partial", "the partial file is what gets checked"
    assert observed["destination_existed"] is False
    assert not destination.exists()
    assert not list(tmp_path.iterdir()), "the partial file must be cleaned up"


def test_max_lag_defaults_to_zero_when_there_is_no_travel_time(
    mapcut_header: HeaderFactory,
) -> None:
    assert mapcut_header().max_lag == 0


def test_a_negative_max_lag_is_refused_before_any_byte(
    tmp_path: Path, mapcut_header: HeaderFactory
) -> None:
    path = tmp_path / "mapcut.rv0"
    with pytest.raises(LayoutError, match="max_lag is -1"):
        write_mapcut(mapcut_header(max_lag=-1), path)
    assert not list(tmp_path.iterdir())
