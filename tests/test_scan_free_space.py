import cv2
import numpy as np
import pytest

from bb8_rl.mapping import Bounds3D, EvidenceSource, Provenance, RoomMemory, SpaceState
from bb8_rl.mapping.scan_free_space import (
    ScanFreeMemory,
    ScanRGBView,
    build_scan_free_memory,
    capsule_free,
    floor_homography_masks,
    prism_floor_support,
    refine_occupied_conflicts,
    synthetic_floor_mask,
)


def view(name, camera_x):
    hsv = np.zeros((80, 80, 3), np.uint8)
    hsv[:] = [105, 150, 70]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = [-camera_x, 0.0, 3.0]
    return ScanRGBView(
        name,
        rgb,
        np.array([[60.0, 0.0, 40.0], [0.0, 60.0, 40.0], [0.0, 0.0, 1.0]]),
        pose,
    )


def build(views, **kwargs):
    return build_scan_free_memory(
        views,
        extent=0.2,
        resolution=0.1,
        body_height_m=0.1,
        projection_margin_m=0.0,
        minimum_baseline_m=0.5,
        **kwargs,
    )


def test_floor_palette_requires_known_floor_colors_and_never_robot_whites():
    hsv = np.array(
        [[[105, 150, 70], [112, 150, 70], [105, 20, 150], [105, 150, 5]]], np.uint8
    )
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    assert synthetic_floor_mask(rgb).tolist() == [[True, False, False, False]]


def test_convex_projection_excludes_unrelated_rectangle_corners_but_keeps_touched_pixels():
    camera = view("a", 0.0)
    angle = np.pi / 4
    rotate = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    camera.world_to_camera[:3, :3] = rotate @ camera.world_to_camera[:3, :3]
    low, high = np.array([[-0.5, -0.5]]), np.array([[0.5, 0.5]])
    camera.rgb[26, 26] = 255
    assert not prism_floor_support(camera, low, high, floor_z=0, top_z=0.3).item()
    assert prism_floor_support(
        camera, low, high, floor_z=0, top_z=0.3, projection_shape="convex_hull"
    ).item()
    camera.rgb[24, 40] = 255
    assert not prism_floor_support(
        camera, low, high, floor_z=0, top_z=0.3, projection_shape="convex_hull"
    ).item()


def test_occupied_refinement_requires_distinct_view_geometry_without_creating_free():
    points = np.array(
        [
            [[[0, 0, 0.1], [1, 0, 0.1]]],
            [[[0.01, 0, 0.1], [0, 0, 0]]],
            [[[0, 0.01, 0.1], [0, 0, 0]]],
        ],
        dtype=float,
    )
    valid = np.array([[[True, True]], [[True, False]], [[True, False]]])
    memory, report, accepted, counts = refine_occupied_conflicts(
        points, valid, ["a", "b", "c"]
    )
    assert accepted[:, 0, 0].all()
    assert not accepted[0, 0, 1]
    assert counts[0, 0, 1] == 1
    assert report["uncertainty_m"] == 0.05
    assert all(e.state is SpaceState.OCCUPIED for e in memory.evidence)
    assert (
        memory.query(Bounds3D((0.97, -0.01, 0.09), (1.03, 0.01, 0.11)))
        is SpaceState.UNKNOWN
    )


def test_complete_prism_rejects_interior_nonfloor_pixel_not_only_corners():
    camera = view("a", 0.0)
    low = np.array([[-0.1, -0.1]])
    high = np.array([[0.1, 0.1]])
    assert prism_floor_support(camera, low, high, floor_z=0.0, top_z=0.3).item()
    camera.rgb[40, 40] = [255, 255, 255]
    assert not prism_floor_support(camera, low, high, floor_z=0.0, top_z=0.3).item()


def test_homography_rejects_wrong_plane_even_when_all_colors_match_floor_palette():
    cameras = [view("a", -0.8), view("b", 0.8)]
    for camera in cameras:
        y, x = np.indices(camera.rgb.shape[:2])
        homogeneous = (
            camera.intrinsics
            @ np.c_[camera.world_to_camera[:3, :2], camera.world_to_camera[:3, 3]]
        )
        rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(homogeneous).T
        xy = rays[..., :2] / rays[..., 2:]
        checker = (np.floor(xy[..., 0] / 0.25) + np.floor(xy[..., 1] / 0.25)) % 2
        hsv = np.empty(camera.rgb.shape, np.uint8)
        hsv[:] = [105, 150, 70]
        hsv[checker.astype(bool)] = [105, 120, 95]
        camera.rgb[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    correct, _ = floor_homography_masks(
        cameras, minimum_matches=1, minimum_angle_degrees=10, patch_size=5
    )
    bad_pose = cameras[1].world_to_camera.copy()
    bad_pose[0, 3] += 0.125
    wrong = ScanRGBView("b", cameras[1].rgb, cameras[1].intrinsics, bad_pose)
    incorrect, _ = floor_homography_masks(
        [cameras[0], wrong], minimum_matches=1, minimum_angle_degrees=10, patch_size=5
    )
    # This shared field of view excludes the first 32 columns, which the
    # translated second camera correctly leaves unobserved.
    assert correct[0][20:60, 40:60].mean() > 0.8
    assert incorrect[0][20:60, 40:60].mean() < 0.6


def test_supplied_mask_requires_whole_prism_and_insufficient_homography_stays_unknown():
    cameras = [view("a", -0.8), view("b", 0.8)]
    masks, _ = floor_homography_masks(
        cameras, minimum_matches=1, minimum_angle_degrees=10
    )
    assert not any(m.any() for m in masks)
    assert not build(cameras, floor_masks=masks).free_mask.any()
    # The preserved pointwise ablation accepts uniform ambiguous colors.
    weak_masks, _ = floor_homography_masks(
        cameras, minimum_matches=1, minimum_angle_degrees=10, patch_size=None
    )
    assert any(m.any() for m in weak_masks)


def test_free_requires_two_separated_views_and_absent_views_stay_unknown():
    assert not build([]).free_mask.any()
    assert not build([view("a", -0.8)]).free_mask.any()
    assert not build([view("a", -0.8), view("b", -0.79)]).free_mask.any()
    result = build([view("a", -0.8), view("b", 0.8)])
    assert result.candidate_free_mask.all()
    assert result.free_mask[:3, :3].all()
    # Floating-point trailing voxels may extend beyond the declared map bounds;
    # RoomMemory deliberately leaves them unknown instead of claiming outside space.
    assert np.all(result.free_mask <= result.candidate_free_mask)
    assert result.supporting_views((1, 1)) == ("a", "b")
    assert not build(
        [view("a", -0.8), view("b", 0.8), view("c", 0.81)], minimum_views=3
    ).free_mask.any()
    assert build(
        [view("a", -1.6), view("b", 0), view("c", 1.6)], minimum_views=3
    ).free_mask[1, 1]


def test_occupied_conflicts_survive_positive_floor_evidence():
    memory = RoomMemory(
        Bounds3D((-0.2, -0.2, 0.0), (0.2, 0.2, 0.3)),
        0.1,
        scene_version="s",
        calibration_version="c",
    )
    source = EvidenceSource("occupied", ("a",), 0.0, Provenance.RGB_RECONSTRUCTION)
    memory.observe_volume(
        Bounds3D((-0.19, -0.19, 0.01), (-0.11, -0.11, 0.09)),
        SpaceState.OCCUPIED,
        source,
    )
    result = build([view("a", -0.8), view("b", 0.8)], occupied_memory=memory)
    assert result.candidate_free_mask[0, 0]
    assert result.occupied_mask[0, 0]
    assert not result.free_mask[0, 0]
    assert result.free_mask.sum() > 0
    assert all(e.state is SpaceState.OCCUPIED for e in memory.evidence)


def test_capsule_checks_whole_segment_radius_and_unknown_corners():
    free = np.ones((8, 8), bool)
    free[4, 4] = False
    assert not capsule_free(free, 1.0, 0.25, [-0.5, 0.1], [0.5, 0.1], 0.01)
    assert not capsule_free(free, 1.0, 0.25, [-0.5, -0.1], [0.5, -0.1], 0.11)
    assert capsule_free(free, 1.0, 0.25, [-0.5, -0.2], [0.5, -0.2], 0.1)
    assert not capsule_free(free, 1.0, 0.25, [-0.9, -0.5], [0.5, -0.5], 0.11)
    assert not capsule_free(free, 1.0, 0.25, [-0.1, -0.1], [-0.1, -0.1], 0.15)


def test_serialization_invalidation_and_planning_radius_fail_closed(tmp_path):
    result = build([view("a", -0.8), view("b", 0.8)])
    assert result.segment_free([-0.05, 0.0], [0.05, 0.0], 0.02)
    assert result.planning_grid(0.1).blocked.sum() > result.grid.blocked.sum()
    output = tmp_path / "memory"
    result.save(output)
    loaded = ScanFreeMemory.load(output)
    np.testing.assert_array_equal(loaded.free_mask, result.free_mask)
    loaded.memory.invalidate_if_changed(
        scene_version="moved",
        calibration_version=loaded.memory.calibration_version,
        at_time=1.0,
    )
    assert not loaded.segment_free([-0.05, 0.0], [0.05, 0.0], 0.02)
    assert loaded.grid.blocked.all()


def test_frozen_queries_and_non_occupied_conflict_layer_are_rejected():
    with pytest.raises(ValueError, match="Frozen query"):
        build([view("scan-014", -0.8), view("a", 0.8)])
    result = build([view("a", -0.8), view("b", 0.8)])
    with pytest.raises(ValueError, match="occupied-only"):
        build([view("a", -0.8), view("b", 0.8)], occupied_memory=result.memory)
