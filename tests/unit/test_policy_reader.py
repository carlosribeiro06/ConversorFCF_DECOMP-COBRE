import logging
from pathlib import Path

import flatbuffers
import numpy as np
import pytest

from Cobre.IO.Policy import AffinePiece as affine_piece
from Cobre.IO.Policy import CheckpointManifest as checkpoint_manifest
from Cobre.IO.Policy import EntitySlot as entity_slot
from Cobre.IO.Policy import ManifestNode as manifest_node
from Cobre.IO.Policy import StageCuts as stage_cuts
from conversor_fcf.cobre.policy_reader import (
    SUPPORTED_FORMAT_VERSION,
    VALID_ENTITY_TYPES,
    PolicyFormatError,
    read_policy_manifest,
    read_stage_cuts,
)


def _manifest_bytes(format_version: int) -> bytes:
    builder = flatbuffers.Builder(256)
    checkpoint_manifest.CheckpointManifestStart(builder)
    checkpoint_manifest.CheckpointManifestAddFormatVersion(builder, format_version)
    builder.Finish(checkpoint_manifest.CheckpointManifestEnd(builder))
    return bytes(builder.Output())


def _stage_cuts_bytes(manifest_length: int, coefficient_length: int, entity_type: int = 0) -> bytes:
    """Build a StageCuts buffer whose piece width need not match its manifest width."""
    builder = flatbuffers.Builder(4096)

    affine_piece.AffinePieceStartCoefficientsVector(builder, coefficient_length)
    for value in reversed(range(coefficient_length)):
        builder.PrependFloat64(float(value))
    coefficients = builder.EndVector()

    affine_piece.AffinePieceStart(builder)
    affine_piece.AffinePieceAddPieceId(builder, 1)
    affine_piece.AffinePieceAddIntercept(builder, 2.5)
    affine_piece.AffinePieceAddCoefficients(builder, coefficients)
    affine_piece.AffinePieceAddIsActive(builder, True)
    piece = affine_piece.AffinePieceEnd(builder)

    slots = []
    for index in range(manifest_length):
        entity_slot.EntitySlotStart(builder)
        entity_slot.EntitySlotAddEntityType(builder, entity_type)
        entity_slot.EntitySlotAddEntityId(builder, index)
        entity_slot.EntitySlotAddDeliveryDate(builder, -2147483648)
        slots.append(entity_slot.EntitySlotEnd(builder))

    stage_cuts.StageCutsStartEntityManifestVector(builder, manifest_length)
    for offset in reversed(slots):
        builder.PrependUOffsetTRelative(offset)
    entity_manifest = builder.EndVector()

    stage_cuts.StageCutsStartCutsVector(builder, 1)
    builder.PrependUOffsetTRelative(piece)
    cuts = builder.EndVector()

    stage_cuts.StageCutsStart(builder)
    stage_cuts.StageCutsAddStageId(builder, 0)
    stage_cuts.StageCutsAddStateDimension(builder, 999)
    stage_cuts.StageCutsAddEntityManifest(builder, entity_manifest)
    stage_cuts.StageCutsAddCuts(builder, cuts)
    stage_cuts.StageCutsAddPopulatedCount(builder, 1)
    stage_cuts.StageCutsAddCostScaleFactor(builder, 1.0)
    stage_cuts.StageCutsAddNodeId(builder, 0)
    builder.Finish(stage_cuts.StageCutsEnd(builder))
    return bytes(builder.Output())


def _write(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "buffer.bin"
    path.write_bytes(payload)
    return path


def test_missing_manifest_names_the_path(tmp_path: Path) -> None:
    absent = tmp_path / "manifest.bin"
    with pytest.raises(PolicyFormatError) as excinfo:
        read_policy_manifest(absent)
    assert str(absent) in str(excinfo.value)


def test_missing_stage_cuts_names_the_path(tmp_path: Path) -> None:
    absent = tmp_path / "000.bin"
    with pytest.raises(PolicyFormatError) as excinfo:
        read_stage_cuts(absent)
    assert str(absent) in str(excinfo.value)


@pytest.mark.parametrize("size", [0, 1, 7])
def test_truncated_buffer_is_rejected_before_decoding(tmp_path: Path, size: int) -> None:
    path = _write(tmp_path, b"\x00" * size)
    with pytest.raises(PolicyFormatError, match="too short"):
        read_policy_manifest(path)


def test_unsupported_format_version_names_found_and_supported(tmp_path: Path) -> None:
    path = _write(tmp_path, _manifest_bytes(99))
    with pytest.raises(PolicyFormatError) as excinfo:
        read_policy_manifest(path)
    message = str(excinfo.value)
    assert "99" in message
    assert str(SUPPORTED_FORMAT_VERSION) in message


def test_supported_format_version_is_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path, _manifest_bytes(SUPPORTED_FORMAT_VERSION))
    manifest = read_policy_manifest(path)
    assert manifest.format_version == SUPPORTED_FORMAT_VERSION
    assert manifest.nodes == ()
    assert manifest.edges == ()
    assert manifest.cobre_version == ""


def test_coefficient_manifest_mismatch_names_the_index_and_both_lengths(tmp_path: Path) -> None:
    path = _write(tmp_path, _stage_cuts_bytes(manifest_length=3, coefficient_length=2))
    with pytest.raises(PolicyFormatError) as excinfo:
        read_stage_cuts(path)
    message = str(excinfo.value)
    assert "piece index 0" in message
    assert "2 coefficients" in message
    assert "3 entries" in message


def test_matching_widths_are_accepted_and_state_dimension_never_sizes_the_array(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _stage_cuts_bytes(manifest_length=3, coefficient_length=3))
    pool = read_stage_cuts(path)
    # The builder sets state_dimension to 999 deliberately: the manifest governs.
    assert pool.state_dimension == 999
    assert len(pool.slots) == 3
    assert len(pool.pieces) == 1
    assert pool.pieces[0].intercept == 2.5
    # Values, not just the length: asserting the length alone leaves a float32
    # dtype or a reversed axis undetectable.
    assert list(pool.pieces[0].coefficients) == [0.0, 1.0, 2.0]
    assert pool.pieces[0].coefficients.dtype == np.float64


def test_coefficients_are_not_writeable(tmp_path: Path) -> None:
    path = _write(tmp_path, _stage_cuts_bytes(manifest_length=2, coefficient_length=2))
    pool = read_stage_cuts(path)
    assert pool.pieces[0].coefficients.flags.writeable is False


def test_garbage_buffer_is_wrapped_as_a_policy_format_error(tmp_path: Path) -> None:
    path = _write(tmp_path, b"\xff" * 64)
    with pytest.raises(PolicyFormatError):
        read_stage_cuts(path)


def test_garbage_manifest_is_wrapped_as_a_policy_format_error(tmp_path: Path) -> None:
    """Callers handle one exception type, whatever flatbuffers raises underneath."""
    path = _write(tmp_path, b"\xff" * 64)
    with pytest.raises(PolicyFormatError, match="cannot decode policy manifest"):
        read_policy_manifest(path)


def test_empty_entity_manifest_yields_empty_coefficient_arrays(tmp_path: Path) -> None:
    path = _write(tmp_path, _stage_cuts_bytes(manifest_length=0, coefficient_length=0))
    pool = read_stage_cuts(path)
    assert pool.slots == ()
    assert len(pool.pieces) == 1
    assert pool.pieces[0].coefficients.shape == (0,)
    assert pool.pieces[0].coefficients.dtype.name == "float64"


def _manifest_bytes_with_nodes(node_count: int, num_stages: int | None = None) -> bytes:
    builder = flatbuffers.Builder(1024)

    nodes = []
    for index in range(node_count):
        manifest_node.ManifestNodeStart(builder)
        manifest_node.ManifestNodeAddId(builder, index)
        manifest_node.ManifestNodeAddStageId(builder, index)
        manifest_node.ManifestNodeAddPoolId(builder, index)
        nodes.append(manifest_node.ManifestNodeEnd(builder))

    checkpoint_manifest.CheckpointManifestStartNodesVector(builder, node_count)
    for offset in reversed(nodes):
        builder.PrependUOffsetTRelative(offset)
    node_vector = builder.EndVector()

    checkpoint_manifest.CheckpointManifestStart(builder)
    checkpoint_manifest.CheckpointManifestAddFormatVersion(builder, SUPPORTED_FORMAT_VERSION)
    checkpoint_manifest.CheckpointManifestAddNodes(builder, node_vector)
    checkpoint_manifest.CheckpointManifestAddNumStages(
        builder, node_count if num_stages is None else num_stages
    )
    builder.Finish(checkpoint_manifest.CheckpointManifestEnd(builder))
    return bytes(builder.Output())


def test_manifest_with_nodes_round_trips(tmp_path: Path) -> None:
    manifest = read_policy_manifest(_write(tmp_path, _manifest_bytes_with_nodes(3)))
    assert [node.id for node in manifest.nodes] == [0, 1, 2]
    assert manifest.num_stages == 3


def test_decode_failure_after_the_version_guard_is_wrapped(tmp_path: Path) -> None:
    """A buffer whose version reads cleanly but whose node vector dangles."""
    payload = _manifest_bytes_with_nodes(6)
    truncated = payload[: len(payload) // 2]
    path = _write(tmp_path, truncated)
    with pytest.raises(PolicyFormatError, match="cannot decode policy manifest"):
        read_policy_manifest(path)


def test_coefficients_cannot_be_mutated_through_the_base_chain(tmp_path: Path) -> None:
    """A bytearray buffer would leave a writeable intermediate under the frozen view."""
    path = _write(tmp_path, _stage_cuts_bytes(manifest_length=3, coefficient_length=3))
    coefficients = read_stage_cuts(path).pieces[0].coefficients
    assert coefficients.flags.writeable is False
    base = coefficients.base
    assert base is None or not getattr(base, "flags", _Writeable()).writeable


class _Writeable:
    writeable = True


def test_valid_entity_types_mirror_the_entity_type_enum() -> None:
    """The set is duplicated to break an import cycle, so pin it against the enum."""
    from conversor_fcf.cobre.entities import EntityType

    assert VALID_ENTITY_TYPES == {member.value for member in EntityType}


def test_unknown_entity_type_names_the_position_and_the_path(tmp_path: Path) -> None:
    path = _write(
        tmp_path, _stage_cuts_bytes(manifest_length=1, coefficient_length=1, entity_type=7)
    )
    with pytest.raises(PolicyFormatError) as excinfo:
        read_stage_cuts(path)
    message = str(excinfo.value)
    assert "entity_type 7" in message
    assert "position 0" in message
    assert str(path) in message


def test_manifest_globals_disagreeing_with_the_node_set_are_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same discipline read_stage_cuts applies to state_dimension.

    The builder leaves n_pools at its default 0 while writing three nodes with
    pool ids 0-2, and overrides num_stages to 99 against three distinct stage ids,
    so both cross-checks fire on one buffer.
    """
    logging.getLogger("conversor_fcf").propagate = True
    path = _write(tmp_path, _manifest_bytes_with_nodes(3, num_stages=99))
    with caplog.at_level(logging.WARNING, logger="conversor_fcf"):
        manifest = read_policy_manifest(path)

    assert manifest.n_pools == 0
    assert manifest.num_stages == 99
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "n_pools 0 does not match the 3 distinct pool ids" in messages
    assert "num_stages 99 does not match the 3 distinct stage ids" in messages
    assert str(path) in messages
