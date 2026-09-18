import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "scan_registration_script",
    Path(__file__).resolve().parents[1] / "scripts/register-scan-camera.py",
)
registration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registration)


def dense_plane():
    yy, xx = np.indices((12, 12))
    points = np.stack([xx * 0.01, yy * 0.01, np.ones_like(xx)], axis=-1)
    return points, np.ones((12, 12)), np.ones((12, 12), bool)


def test_feature_sampling_interpolates_without_rounding_camera_feature():
    points, depth, valid = dense_plane()
    indices, xyz, rejected = registration.sample_feature_points(
        [[4.25, 5.5]], points, depth, valid
    )
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(xyz, [[0.0425, 0.055, 1]])
    assert not any(rejected.values())


def test_unknown_patch_and_depth_edge_never_supply_registration_landmark():
    points, depth, valid = dense_plane()
    valid[3, 3] = False
    indices, xyz, rejected = registration.sample_feature_points(
        [[5, 5]], points, depth, valid
    )
    assert len(indices) == len(xyz) == 0 and rejected["invalid_neighborhood"] == 1
    valid[:] = True
    depth[:, 6:] = 1.5
    indices, xyz, rejected = registration.sample_feature_points(
        [[5, 5]], points, depth, valid
    )
    assert len(indices) == len(xyz) == 0 and rejected["depth_discontinuity"] == 1


def test_nonfinite_and_boundary_feature_do_not_get_clamped_to_valid_pixel():
    points, depth, valid = dense_plane()
    indices, xyz, rejected = registration.sample_feature_points(
        [[0, 0], [np.nan, 5], [11, 11]], points, depth, valid
    )
    assert not len(indices) and not len(xyz) and rejected["boundary_or_nonfinite"] == 3


def test_elevation_summary_does_not_filter_or_reclassify_predicted_points():
    points = np.array([[0, 0, -0.2], [0, 0, 0.01], [0, 0, 0.4]])
    original = points.copy()
    result = registration.elevation_summary(points)
    assert result["near_floor_abs_z_le_005m"] == 1
    assert result["elevated_z_gt_005m"] == 1
    assert result["below_support_plane_z_lt_minus005m"] == 1
    np.testing.assert_array_equal(points, original)


def candidate(view_id, x, inliers=20):
    transform = np.eye(4)
    transform[0, 3] = -x
    return {
        "view_id": view_id,
        "registration": {
            "status": "candidate",
            "world_to_camera": transform.tolist(),
            "inliers": inliers,
            "reprojection_p95_px": 0.5,
        },
    }


def test_global_fit_is_not_independent_pose_consensus():
    result = registration.select_consensus(
        [candidate("scan-000", 0), candidate("global", 0, 1000)]
    )
    assert (
        result["status"] == "registration_pending"
        and result["selected_view_id"] is None
    )


def test_consensus_rejects_disagreeing_views_even_if_they_have_more_matches():
    result = registration.select_consensus(
        [
            candidate("scan-000", 0),
            candidate("scan-001", 0.05, 25),
            candidate("scan-002", 2, 200),
        ]
    )
    assert result["status"] == "candidate"
    assert result["selected_view_id"] == "scan-001"
    assert set(result["support_view_ids"]) == {"scan-000", "scan-001"}
