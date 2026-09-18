"""Analytic contract fixtures; these are not RGB reconstruction/navigation trials."""

import json

import pytest

from bb8_rl.mapping import (
    Bounds3D,
    EvidenceSource,
    ObjectBounds,
    Provenance,
    RoomMemory,
    SpaceState,
)


def source(name="scan-1", error=0.0):
    return EvidenceSource(
        name, (f"{name}-view",), 1.0, Provenance.ANALYTIC_SYNTHETIC, error
    )


def room():
    return RoomMemory(
        Bounds3D((-1, -1, 0), (1, 1, 0.5)),
        0.05,
        scene_version="static-scene-v1",
        calibration_version="rig-v1",
    )


def corridor(memory):
    memory.observe_volume(
        Bounds3D((-0.95, -0.3, 0), (0.95, 0.3, 0.3)), SpaceState.FREE, source()
    )


def passage(memory, y=0):
    return memory.segment_free((-0.7, y), (0.7, y), radius_m=0.05, height_m=0.15)


def test_visible_corridor_certifies_full_body_height_and_occupied_wins():
    memory = room()
    corridor(memory)
    assert passage(memory)
    memory.remember_object(
        ObjectBounds(
            "low-obstacle",
            Bounds3D((-0.1, -0.1, 0.01), (0.1, 0.1, 0.08)),
            source("object"),
        )
    )
    assert not passage(memory)
    corridor(memory)  # Later free evidence cannot erase an observed obstacle.
    assert not passage(memory)
    assert memory.objects[0].bounds.size == pytest.approx((0.2, 0.2, 0.07))


def test_remembered_hidden_corridor_retains_source_evidence_after_round_trip(tmp_path):
    memory = room()
    corridor(memory)
    path = tmp_path / "map.json"
    memory.save(path)
    remembered = RoomMemory.load(path)
    # No current-view observation is supplied. Prior certified volume persists.
    assert passage(remembered)
    assert remembered.evidence[0].source.view_ids == ("scan-1-view",)
    assert remembered.evidence[0].source.observed_at == 1.0
    assert remembered.evidence[0].source.provenance is Provenance.ANALYTIC_SYNTHETIC
    assert remembered.to_dict() == memory.to_dict()


def test_never_observed_corridor_and_thin_unknown_gap_are_not_free():
    memory = room()
    query = Bounds3D((-0.5, -0.1, 0), (0.5, 0.1, 0.1))
    assert memory.query(query) is SpaceState.UNKNOWN
    for lo, hi in ((-0.95, -0.001), (0.001, 0.95)):
        memory.observe_volume(
            Bounds3D((lo, -0.3, 0), (hi, 0.3, 0.3)), SpaceState.FREE, source(str(lo))
        )
    assert not passage(memory), "A 2 mm unknown slit cannot disappear through sampling"


def test_ambiguous_fork_has_only_one_evidenced_branch():
    memory = room()
    memory.observe_volume(
        Bounds3D((-0.95, -0.8, 0), (0.95, -0.2, 0.3)), SpaceState.FREE, source("south")
    )
    assert passage(memory, y=-0.5)
    assert not passage(memory, y=0.5)
    # An isolated far endpoint does not establish the connecting branch.
    memory.observe_volume(
        Bounds3D((0.5, 0.2, 0), (0.95, 0.8, 0.3)), SpaceState.FREE, source("endpoint")
    )
    assert not passage(memory, y=0.5)


def test_changed_obstacle_invalidates_entire_map_and_survives_serialization():
    memory = room()
    corridor(memory)
    before = memory.map_version
    assert not memory.invalidate_if_changed(
        scene_version="obstacle-moved-v2", calibration_version="rig-v1", at_time=2.0
    )
    assert memory.map_version == before + 1
    assert not passage(memory)
    restored = RoomMemory.from_dict(json.loads(json.dumps(memory.to_dict())))
    assert not restored.valid
    assert (
        restored.query(Bounds3D((-0.1, -0.1, 0), (0.1, 0.1, 0.1))) is SpaceState.UNKNOWN
    )
    with pytest.raises(ValueError, match="new registered reconstruction"):
        corridor(restored)


def test_changed_camera_registration_invalidates_old_geometry():
    memory = room()
    corridor(memory)
    assert memory.invalidate_if_changed(
        scene_version="static-scene-v1", calibration_version="rig-v1", at_time=2.0
    )
    assert passage(memory)
    assert not memory.invalidate_if_changed(
        scene_version="static-scene-v1", calibration_version="rig-v2-moved", at_time=3.0
    )
    assert not passage(memory)
    # Returning the old version string cannot revive evidence invalidated earlier.
    assert not memory.invalidate_if_changed(
        scene_version="static-scene-v1", calibration_version="rig-v1", at_time=4.0
    )


def test_error_bounds_shrink_free_space_expand_obstacles_and_validate_inputs():
    memory = room()
    memory.observe_volume(
        Bounds3D((-0.95, -0.2, 0), (0.95, 0.2, 0.3)),
        SpaceState.FREE,
        source(error=0.06),
    )
    # Eroding geometry vertically also refuses to certify unobserved floor contact.
    assert not passage(memory)
    with pytest.raises(ValueError):
        EvidenceSource("bad", ("view",), 0, Provenance.RGB_RECONSTRUCTION, -1)
    with pytest.raises(ValueError):
        Bounds3D((0, 0, 0), (float("nan"), 1, 1))
    raw = room().to_dict()
    raw["units"] = "centimeters"
    with pytest.raises(ValueError, match="non-metric"):
        RoomMemory.from_dict(raw)
    assert room().query(Bounds3D((-2, -2, 0), (2, 2, 0.2))) is SpaceState.UNKNOWN


def test_object_and_free_volume_order_does_not_change_occupancy():
    memory = room()
    obj = ObjectBounds(
        "box", Bounds3D((0.1, -0.1, 0), (0.3, 0.1, 0.2)), source("box", error=0.05)
    )
    memory.remember_object(obj)
    corridor(memory)
    query = Bounds3D((0.06, -0.02, 0.01), (0.09, 0.02, 0.1))
    assert memory.query(query) is SpaceState.OCCUPIED
    restored = RoomMemory.from_dict(memory.to_dict())
    assert restored.query(query) is SpaceState.OCCUPIED
    assert restored.objects == memory.objects
    with pytest.raises(ValueError, match="already exists"):
        restored.remember_object(obj)
