import pytest

from conversor_fcf.cobre.entities import (
    DELIVERY_DATE_SENTINEL,
    EntityType,
    SlotIndex,
    has_delivery_date,
    index_slots,
    slot_label,
)
from conversor_fcf.cobre.policy_reader import EntitySlotRecord, PolicyFormatError


def _slot(
    entity_type: int, entity_id: int, subindex: int = 0, delivery_date: int = 0
) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        subindex=subindex,
        was_active=True,
        delivery_date=delivery_date,
    )


def test_entity_type_values_match_the_schema() -> None:
    assert EntityType.HYDRO_STORAGE.value == 0
    assert EntityType.HYDRO_INFLOW_LAG.value == 1
    assert EntityType.ANTICIPATED_THERMAL_STATE.value == 2
    assert EntityType.HYDRO_TRANSIT_BUCKET.value == 3


@pytest.mark.parametrize(
    ("entity_type", "entity_id", "subindex", "expected"),
    [
        (0, 7, 0, "storage_h7"),
        (1, 7, 3, "inflow_lag_h7_l3"),
        (2, 112, 1, "anticipated_thermal_t112_s1"),
        (3, 7, 2, "transit_bucket_h7_d2"),
    ],
)
def test_slot_label_covers_all_four_axes(
    entity_type: int, entity_id: int, subindex: int, expected: str
) -> None:
    assert slot_label(_slot(entity_type, entity_id, subindex)) == expected


def test_slot_label_rejects_an_unknown_axis() -> None:
    with pytest.raises(PolicyFormatError, match="unknown entity_type 9"):
        slot_label(_slot(9, 1))


def test_has_delivery_date_distinguishes_the_sentinel_from_a_real_date() -> None:
    assert has_delivery_date(_slot(2, 112, delivery_date=20260425)) is True
    assert has_delivery_date(_slot(0, 0, delivery_date=DELIVERY_DATE_SENTINEL)) is False


def test_has_delivery_date_rejects_the_absent_field_default() -> None:
    """EntitySlot.DeliveryDate() yields 0 when the field is absent, not the sentinel."""
    assert has_delivery_date(_slot(2, 112, delivery_date=0)) is False


def test_index_slots_groups_every_axis_by_position() -> None:
    slots = [
        _slot(0, 0),
        _slot(2, 112, 0),
        _slot(1, 5, 2),
        _slot(0, 1),
        _slot(3, 9, 1),
        _slot(2, 113, 0),
    ]
    assert index_slots(slots) == SlotIndex(
        storage=(0, 3),
        inflow_lag=(2,),
        anticipated_thermal=(1, 5),
        transit_bucket=(4,),
    )


def test_index_slots_returns_empty_groups_for_absent_axes() -> None:
    result = index_slots([_slot(0, 0), _slot(0, 1)])
    assert result.storage == (0, 1)
    assert result.inflow_lag == ()
    assert result.anticipated_thermal == ()
    assert result.transit_bucket == ()


def test_index_slots_names_the_position_and_value_of_an_unknown_axis() -> None:
    slots = [_slot(0, 0), _slot(0, 1), _slot(7, 3)]
    with pytest.raises(PolicyFormatError) as excinfo:
        index_slots(slots)
    message = str(excinfo.value)
    assert "7" in message
    assert "position 2" in message


def test_index_slots_accepts_an_empty_manifest() -> None:
    assert index_slots([]) == SlotIndex((), (), (), ())
