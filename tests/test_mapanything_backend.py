"""Geometry/contract tests without downloading or importing learned models."""

import numpy as np
import pytest

from bb8_rl.mapping.backends.mapanything import (
    CalibratedRGBView,
    add_pose_alignment,
    calibrated_world_points,
    fit_camera_center_similarity,
    transform_points,
    write_ply,
)


def test_calibrated_depth_uses_optical_z_and_known_metric_pose():
    k = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = [10, 20, 30]
    points = calibrated_world_points(np.full((3, 3), 4.0), k, c2w)
    np.testing.assert_allclose(points[1, 1], [10, 20, 34])
    np.testing.assert_allclose(points[0, 0], [8, 18, 34])


def test_pose_convention_round_trip_and_anchor():
    c2w = np.array(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    local = np.array([[1.0, 0.0, 2.0], [-2.0, 3.0, 1.0]])
    np.testing.assert_allclose(
        transform_points(transform_points(local, c2w), np.linalg.inv(c2w)), local
    )
    estimated_first = np.eye(4)
    estimated_first[:3, 3] = [1, 2, 3]
    anchor = c2w @ np.linalg.inv(estimated_first)
    np.testing.assert_allclose(anchor @ estimated_first, c2w)


def test_contract_rejects_reflection_and_non_rgb():
    pose = np.eye(4)
    pose[0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        CalibratedRGBView(
            "a", np.zeros((2, 2, 3), np.uint8), np.eye(3), pose, "synthetic_exact"
        )
    with pytest.raises(ValueError, match="RGB"):
        CalibratedRGBView(
            "a", np.zeros((2, 2, 4), np.uint8), np.eye(3), np.eye(4), "synthetic_exact"
        )


def test_ply_excludes_invalid_and_nonfinite_points(tmp_path):
    points = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [np.nan, 1.0, 2.0]]])
    rgb = np.zeros((1, 3, 3), np.uint8)
    path = tmp_path / "points.ply"
    assert write_ply(path, points, rgb, np.array([[True, False, True]])) == 1
    header, body = path.read_bytes().split(b"end_header\n")
    assert b"element vertex 1\n" in header
    assert len(body) == 15


def test_similarity_recovers_camera_only_scale_rotation_translation():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    expected = np.array(
        [
            [0.0, -2.0, 0.0, 3.0],
            [2.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    similarity, details = fit_camera_center_similarity(
        source, transform_points(source, expected)
    )
    np.testing.assert_allclose(similarity, expected, atol=1e-12)
    assert details["scale"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Collinear"):
        fit_camera_center_similarity(
            source[:2].repeat(2, axis=0), source[:2].repeat(2, axis=0)
        )


def test_pose_alignment_preserves_raw_depth_and_enforces_only_supplied_scale():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    known = poses.copy()
    known[:, :3, 3] *= 2
    depth = np.ones((3, 1, 1), np.float32)
    arrays = {
        "camera_to_world_input": known,
        "camera_to_model_predicted": poses,
        "points_model_predicted": np.zeros((3, 1, 1, 3)),
        "depth_z_m": depth.copy(),
        "intrinsics_input_processed": np.repeat(np.eye(3)[None], 3, axis=0),
        "view_ids": np.array(["a", "b", "c"]),
    }
    details = add_pose_alignment(arrays)
    np.testing.assert_array_equal(arrays["depth_z_m"], depth)
    np.testing.assert_allclose(arrays["depth_z_pose_scale_m"], 2)
    np.testing.assert_allclose(
        arrays["points_world_calibrated_pose_scale"][1, 0, 0], [2, 0, 2]
    )
    assert not details["oracle_geometry_fit"]
