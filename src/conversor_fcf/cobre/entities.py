"""Typed interpretation of the Cobre cut state axes.

`subindex` is **0-based** for `AnticipatedThermalState`. In the reference case
entity-manifest position 171 is thermal 112 at ring position 1; reading it as
1-based would shift the entire GNL stage axis downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from conversor_fcf.cobre.policy_reader import EntitySlotRecord, PolicyFormatError

# Storage slots carry this in place of a delivery date.
DELIVERY_DATE_SENTINEL = -2147483648


class EntityType(IntEnum):
    """The four state axes a Cobre cut can be expressed over."""

    HYDRO_STORAGE = 0
    HYDRO_INFLOW_LAG = 1
    ANTICIPATED_THERMAL_STATE = 2
    HYDRO_TRANSIT_BUCKET = 3


_LABEL_FORMATS = {
    EntityType.HYDRO_STORAGE: "storage_h{entity_id}",
    EntityType.HYDRO_INFLOW_LAG: "inflow_lag_h{entity_id}_l{subindex}",
    EntityType.ANTICIPATED_THERMAL_STATE: "anticipated_thermal_t{entity_id}_s{subindex}",
    EntityType.HYDRO_TRANSIT_BUCKET: "transit_bucket_h{entity_id}_d{subindex}",
}


@dataclass(frozen=True)
class SlotIndex:
    """Entity-manifest positions grouped by axis, ascending within each group."""

    storage: tuple[int, ...]
    inflow_lag: tuple[int, ...]
    anticipated_thermal: tuple[int, ...]
    transit_bucket: tuple[int, ...]


def has_delivery_date(slot: EntitySlotRecord) -> bool:
    """Whether the slot carries a real delivery date.

    Stated positively on purpose. `EntitySlot.DeliveryDate()` yields `0` when
    the field is absent from the buffer, which a negative test against
    `DELIVERY_DATE_SENTINEL` alone would accept as a real date, and a `0` date
    fed to the GNL delivery-month logic is silent corruption, not a crash.
    """
    return slot.delivery_date > 0


def _as_entity_type(value: int, position: int | None = None) -> EntityType:
    try:
        return EntityType(value)
    except ValueError as exc:
        where = "" if position is None else f" at entity manifest position {position}"
        raise PolicyFormatError(f"unknown entity_type {value}{where}") from exc


def index_slots(slots: Sequence[EntitySlotRecord]) -> SlotIndex:
    """Group slot positions by axis, rejecting any unrecognised `entity_type`.

    An unknown axis is an error rather than a skip: silently dropping it would
    discard the matching coefficients without a trace.
    """
    buckets: dict[EntityType, list[int]] = {member: [] for member in EntityType}
    for position, slot in enumerate(slots):
        buckets[_as_entity_type(slot.entity_type, position)].append(position)
    return SlotIndex(
        storage=tuple(buckets[EntityType.HYDRO_STORAGE]),
        inflow_lag=tuple(buckets[EntityType.HYDRO_INFLOW_LAG]),
        anticipated_thermal=tuple(buckets[EntityType.ANTICIPATED_THERMAL_STATE]),
        transit_bucket=tuple(buckets[EntityType.HYDRO_TRANSIT_BUCKET]),
    )


def slot_label(slot: EntitySlotRecord) -> str:
    """Return the canonical column name for a slot.

    Single source of coefficient column naming: the ECO and content CSVs both
    derive their headers from here, so the two cannot disagree.
    """
    return _LABEL_FORMATS[_as_entity_type(slot.entity_type)].format(
        entity_id=slot.entity_id, subindex=slot.subindex
    )
