"""Analytic contract fixtures; these are not RGB reconstruction/navigation trials."""

import json

import numpy as np
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


def decimal_room(ceiling=0.12):
    return RoomMemory(
        Bounds3D((0, 0, 0), (0.04, 0.04, ceiling)),
        0.02,
        scene_version="decimal-grid",
        calibration_version="analytic",
    )


def test_decimal_grid_fills_true_top_voxel_without_rounding_past_ceiling():
    memory = decimal_room()
    memory.observe_volume(memory.bounds, SpaceState.FREE, source())
    assert memory._cells.shape == (2, 2, 6)
    assert np.all(memory._cells == memory._FREE)
    assert (
        memory.query(Bounds3D((0.001, 0.001, 0.101), (0.019, 0.019, 0.12)))
        is SpaceState.FREE
    )
    restored = RoomMemory.from_dict(memory.to_dict())
    np.testing.assert_array_equal(restored._cells, memory._cells)


@pytest.mark.parametrize("short_bound", ["room", "evidence"])
def test_one_ulp_short_ceiling_or_evidence_never_certifies_partial_voxel(short_bound):
    short = np.nextafter(0.12, -np.inf)
    memory = decimal_room(short if short_bound == "room" else 0.12)
    observation = Bounds3D(
        (0, 0, 0), (0.04, 0.04, short if short_bound == "evidence" else 0.12)
    )
    memory.observe_volume(observation, SpaceState.FREE, source())
    assert np.all(memory._cells[:, :, :5] == memory._FREE)
    assert np.all(memory._cells[:, :, 5] == memory._UNKNOWN)
    assert (
        memory.query(Bounds3D((0.001, 0.001, 0.101), (0.019, 0.019, short)))
        is SpaceState.UNKNOWN
    )


def test_adjacent_voxels_share_exact_face_without_overlap_or_gap():
    memory = decimal_room(0.14)
    edge = 6 * memory.resolution_m
    touching = Bounds3D((0.001, 0.001, edge), (0.019, 0.019, 0.13))
    assert memory._axes(touching, fully_contained=False)[2].tolist() == [5, 6]
    above = Bounds3D((0.001, 0.001, np.nextafter(edge, np.inf)), (0.019, 0.019, 0.13))
    assert memory._axes(above, fully_contained=False)[2].tolist() == [6]
    below = Bounds3D((0.001, 0.001, 0.11), (0.019, 0.019, np.nextafter(edge, -np.inf)))
    assert memory._axes(below, fully_contained=False)[2].tolist() == [5]


def test_occupied_face_contact_blocks_both_canonical_neighbor_voxels():
    memory = decimal_room(0.14)
    memory.observe_volume(memory.bounds, SpaceState.FREE, source())
    memory.observe_volume(
        Bounds3D((0.001, 0.001, 0.12), (0.019, 0.019, 0.125)),
        SpaceState.OCCUPIED,
        source("touching-object"),
    )
    memory.observe_volume(memory.bounds, SpaceState.FREE, source("later-free"))
    assert memory._cells[0, 0, 5] == memory._OCCUPIED
    assert memory._cells[0, 0, 6] == memory._OCCUPIED
    assert (
        memory.query(Bounds3D((0.001, 0.001, 0.111), (0.019, 0.019, 0.119)))
        is SpaceState.OCCUPIED
    )
