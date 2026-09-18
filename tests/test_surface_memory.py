import numpy as np
import pytest

from bb8_rl.mapping import Bounds3D, RoomMemory, SpaceState
from bb8_rl.mapping.surface_memory import import_surfaces


def build(points, masks):
    return import_surfaces(
        points,
        masks,
        ["A", "B"],
        bounds=Bounds3D((0, 0, 0), (1, 1, 1)),
        resolution_m=0.05,
        uncertainty_m=0.02,
        scene_version="scan1",
        calibration_version="v1",
    )


def test_estimated_surface_import_deduplicates_voxels_without_free_space():
    points = np.array(
        [
            [[[0.33, 0.33, 0.33], [0.34, 0.34, 0.34]]],
            [[[0.33, 0.33, 0.33], [0.8, 0.8, 0.0]]],
        ]
    )
    memory, report = build(points, np.ones(points.shape[:-1], bool))
    assert (
        report["retained_surface_samples"] == 3
        and report["occupied_source_voxels"] == 1
    )
    assert memory.evidence[0].source.view_ids == ("A", "B")
    assert (
        memory.query(Bounds3D((0.32, 0.32, 0.32), (0.34, 0.34, 0.34)))
        is SpaceState.OCCUPIED
    )
    assert (
        memory.query(Bounds3D((0.7, 0.7, 0.2), (0.8, 0.8, 0.3))) is SpaceState.UNKNOWN
    )
    assert not memory.segment_free((0.7, 0.7), (0.8, 0.8), radius_m=0.05, height_m=0.1)
    assert all(e.state is SpaceState.OCCUPIED for e in memory.evidence)
    restored = RoomMemory.from_dict(memory.to_dict())
    assert restored.to_dict() == memory.to_dict()


def test_invalid_and_outside_points_are_not_persisted():
    points = np.array(
        [[[[np.nan, 0, 0.1], [0.4, 0.4, 0.4]]], [[[2, 2, 0.4], [0.5, 0.5, 0.5]]]]
    )
    masks = np.ones(points.shape[:-1], bool)
    masks[0, 0, 1] = False
    memory, report = build(points, masks)
    assert report["retained_surface_samples"] == len(memory.evidence) == 1
    with pytest.raises(ValueError, match="Invalid"):
        build(points, masks.astype(int))
