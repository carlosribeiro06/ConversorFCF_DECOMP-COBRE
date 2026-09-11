"""Readers for the Cobre FlatBuffers policy checkpoint.

Both readers materialize their tables into frozen dataclasses at this boundary;
no live FlatBuffers table travels deeper into the package, because a table is a
view over a buffer whose lifetime the caller cannot see.

Coefficient vectors are sized by the pool's own entity manifest, never by
`StageCuts.state_dimension`, which is the study-global dimension: in the
reference case it reads 2211 in every pool while the trunk pools carry 183.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from Cobre.IO.Policy.CheckpointManifest import CheckpointManifest
from Cobre.IO.Policy.StageCuts import StageCuts
from conversor_fcf.logging_setup import get_logger

SUPPORTED_FORMAT_VERSION = 1

_MIN_BUFFER_BYTES = 8

# FlatBuffers surfaces a malformed buffer as whatever the underlying struct or index
# operation raises. Enumerated rather than caught blanket, so a genuine bug in this
# module is not swallowed as a format error.
_DECODE_ERRORS = (
    IndexError,
    OverflowError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    struct.error,
)

# Mirrors conversor_fcf.cobre.entities.EntityType, which cannot be imported
# here because that module imports PolicyFormatError from this one. A test
# asserts the two stay equal.
VALID_ENTITY_TYPES = frozenset({0, 1, 2, 3})

_logger = get_logger("policy_reader")


class PolicyFormatError(Exception):
    """Raised when a policy file is absent, truncated, or not the supported format."""


@dataclass(frozen=True)
class ManifestNodeRecord:
    """One node of the policy graph."""

    id: int
    stage_id: int
    pool_id: int


@dataclass(frozen=True)
class ManifestEdgeRecord:
    """One transition of the policy graph, with its probability."""

    source_id: int
    target_id: int
    probability: float


@dataclass(frozen=True)
class PolicyManifest:
    """Study-level checkpoint metadata and the policy graph."""

    format_version: int
    cobre_version: str
    created_at: str
    num_stages: int
    n_pools: int
    completed_iterations: int
    cost_scale_factor: float
    nodes: tuple[ManifestNodeRecord, ...]
    edges: tuple[ManifestEdgeRecord, ...]


@dataclass(frozen=True)
class EntitySlotRecord:
    """One state axis of a cut pool, in entity-manifest order."""

    entity_type: int
    entity_id: int
    subindex: int
    was_active: bool
    delivery_date: int


@dataclass(frozen=True, eq=False)
class AffinePieceRecord:
    """One affine piece of the value function.

    `eq=False` because `coefficients` is a numpy array, which would make a
    generated `__eq__` return an array rather than a bool.

    `piece_id` and `iteration` are read through unsigned accessors and carry
    sentinels on warm-start pieces (18446744073709551549 and 4294967295 in the
    reference case), so neither is range-validated.
    """

    piece_id: int
    slot_index: int
    iteration: int
    forward_pass_index: int
    intercept: float
    coefficients: NDArray[np.float64]
    is_active: bool


@dataclass(frozen=True)
class StageCutPool:
    """Every cut of one pool, plus the axes those cuts are expressed over.

    Three field names invite misreading, so state plainly what they hold:

    - `stage_id` is the schema's **pool id**, the `cuts/<pool>.bin` key, not a
      stage.
    - `graph_stage_id` is the actual stage key. In the reference deck pool id
      and stage id coincide for all seven pools, so conflating them passes
      there and breaks on any graph with more pools than stages.
    - `node_id` is `-1` for a **shared** pool owned by several nodes; the
      reference terminal pool really is `-1`, owned by 267 nodes.

    `state_dimension` is recorded for audit only; `slots` is the authority on
    coefficient count.
    """

    stage_id: int
    node_id: int
    graph_stage_id: int
    state_dimension: int
    capacity: int
    warm_start_count: int
    populated_count: int
    cost_scale_factor: float
    slots: tuple[EntitySlotRecord, ...]
    pieces: tuple[AffinePieceRecord, ...]
    active_cut_indices: tuple[int, ...]


def _read_buffer(path: Path) -> bytes:
    if not path.is_file():
        raise PolicyFormatError(f"policy file not found: {path}")
    raw = path.read_bytes()
    if len(raw) < _MIN_BUFFER_BYTES:
        raise PolicyFormatError(
            f"policy file too short to be a FlatBuffers buffer: {path} holds {len(raw)} bytes, "
            f"minimum is {_MIN_BUFFER_BYTES}"
        )
    # Deliberately not copied into a bytearray: a mutable buffer makes
    # np.frombuffer yield a writeable intermediate, so the coefficient arrays
    # would stay reachable for mutation through the base chain despite
    # setflags. Keeping the immutable bytes also halves peak memory on the
    # 177 MB terminal pool.
    return raw


def _decode_text(value: bytes | None) -> str:
    return value.decode("utf-8") if value is not None else ""


def _read_coefficients(piece: Any, length: int) -> NDArray[np.float64]:
    if length == 0:
        return np.empty(0, dtype=np.float64)
    array: NDArray[np.float64] = np.asarray(piece.CoefficientsAsNumpy(), dtype=np.float64)
    # A read-only view over the immutable source bytes; the flag is
    # belt-and-braces on top of that immutability, not the only thing
    # preventing mutation.
    array.setflags(write=False)
    return array


def read_policy_manifest(path: Path) -> PolicyManifest:
    """Read `manifest.bin`, rejecting any format version other than the supported one."""
    buffer = _read_buffer(path)
    try:
        table = CheckpointManifest.GetRootAs(buffer, 0)
        format_version = table.FormatVersion()
    except _DECODE_ERRORS as exc:
        raise PolicyFormatError(f"cannot decode policy manifest {path}: {exc}") from exc

    # Checked before any other field: on an incompatible buffer the other reads are
    # precisely the ones that yield nonsense.
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise PolicyFormatError(
            f"unsupported policy format_version {format_version} in {path}; "
            f"this tool supports version {SUPPORTED_FORMAT_VERSION}"
        )

    try:
        nodes = tuple(
            ManifestNodeRecord(id=node.Id(), stage_id=node.StageId(), pool_id=node.PoolId())
            for node in (table.Nodes(i) for i in range(table.NodesLength()))
        )
        edges = tuple(
            ManifestEdgeRecord(
                source_id=edge.SourceId(),
                target_id=edge.TargetId(),
                probability=edge.Probability(),
            )
            for edge in (table.Edges(i) for i in range(table.EdgesLength()))
        )
        manifest = PolicyManifest(
            format_version=format_version,
            cobre_version=_decode_text(table.CobreVersion()),
            created_at=_decode_text(table.CreatedAt()),
            num_stages=table.NumStages(),
            n_pools=table.NPools(),
            completed_iterations=table.CompletedIterations(),
            cost_scale_factor=table.CostScaleFactor(),
            nodes=nodes,
            edges=edges,
        )
    except _DECODE_ERRORS as exc:
        raise PolicyFormatError(f"cannot decode policy manifest {path}: {exc}") from exc

    pool_count = len(nodes_by_pool(manifest))
    if manifest.n_pools != pool_count:
        _logger.warning(
            "n_pools %d does not match the %d distinct pool ids in the node set of %s; "
            "the node set governs",
            manifest.n_pools,
            pool_count,
            path,
        )
    stage_count = len({node.stage_id for node in manifest.nodes})
    if manifest.num_stages != stage_count:
        _logger.warning(
            "num_stages %d does not match the %d distinct stage ids in the node set of "
            "%s; the node set governs",
            manifest.num_stages,
            stage_count,
            path,
        )

    _logger.info(
        "read policy manifest %s format_version=%d num_stages=%d n_pools=%d nodes=%d edges=%d "
        "completed_iterations=%d",
        path,
        manifest.format_version,
        manifest.num_stages,
        manifest.n_pools,
        len(manifest.nodes),
        len(manifest.edges),
        manifest.completed_iterations,
    )
    return manifest


def nodes_by_pool(manifest: PolicyManifest) -> dict[int, tuple[int, ...]]:
    """Group node ids by the pool that serves them, in ascending pool order."""
    grouped: dict[int, list[int]] = {}
    for node in manifest.nodes:
        grouped.setdefault(node.pool_id, []).append(node.id)
    return {pool_id: tuple(ids) for pool_id, ids in sorted(grouped.items())}


def read_stage_cuts(path: Path) -> StageCutPool:
    """Read one `cuts/<pool>.bin`, sizing every coefficient vector by the entity manifest."""
    buffer = _read_buffer(path)
    try:
        table = StageCuts.GetRootAs(buffer, 0)
        manifest_length = table.EntityManifestLength()

        slots = tuple(
            EntitySlotRecord(
                entity_type=slot.EntityType(),
                entity_id=slot.EntityId(),
                subindex=slot.Subindex(),
                was_active=bool(slot.WasActive()),
                delivery_date=slot.DeliveryDate(),
            )
            for slot in (table.EntityManifest(i) for i in range(manifest_length))
        )
        for position, entity_slot in enumerate(slots):
            if entity_slot.entity_type not in VALID_ENTITY_TYPES:
                raise PolicyFormatError(
                    f"unknown entity_type {entity_slot.entity_type} at entity manifest "
                    f"position {position} of {path}"
                )

        pieces: list[AffinePieceRecord] = []
        for index in range(table.CutsLength()):
            piece = table.Cuts(index)
            coefficient_length = piece.CoefficientsLength()
            if coefficient_length != manifest_length:
                raise PolicyFormatError(
                    f"coefficient/manifest length mismatch in {path}: piece index {index} carries "
                    f"{coefficient_length} coefficients but the entity manifest has "
                    f"{manifest_length} entries"
                )
            pieces.append(
                AffinePieceRecord(
                    piece_id=piece.PieceId(),
                    slot_index=piece.SlotIndex(),
                    iteration=piece.Iteration(),
                    forward_pass_index=piece.ForwardPassIndex(),
                    intercept=piece.Intercept(),
                    coefficients=_read_coefficients(piece, coefficient_length),
                    is_active=bool(piece.IsActive()),
                )
            )

        pool = StageCutPool(
            stage_id=table.StageId(),
            node_id=table.NodeId(),
            graph_stage_id=table.GraphStageId(),
            state_dimension=table.StateDimension(),
            capacity=table.Capacity(),
            warm_start_count=table.WarmStartCount(),
            populated_count=table.PopulatedCount(),
            cost_scale_factor=table.CostScaleFactor(),
            slots=slots,
            pieces=tuple(pieces),
            active_cut_indices=tuple(
                int(table.ActiveCutIndices(i)) for i in range(table.ActiveCutIndicesLength())
            ),
        )
    except _DECODE_ERRORS as exc:
        raise PolicyFormatError(f"cannot decode stage cuts {path}: {exc}") from exc

    _logger.info(
        "read stage cuts %s stage_id=%d node_id=%d manifest_length=%d cuts=%d populated=%d "
        "warm_start=%d",
        path,
        pool.stage_id,
        pool.node_id,
        len(pool.slots),
        len(pool.pieces),
        pool.populated_count,
        pool.warm_start_count,
    )
    if pool.node_id < 0:
        _logger.warning(
            "%s is a shared pool: node_id=%d means several nodes own these cuts, so "
            "node_id is not a usable node key downstream",
            path,
            pool.node_id,
        )
    if pool.state_dimension != len(pool.slots):
        _logger.warning(
            "state_dimension %d does not match the entity manifest length %d in %s; the entity "
            "manifest length governs every coefficient array",
            pool.state_dimension,
            len(pool.slots),
            path,
        )
    return pool
