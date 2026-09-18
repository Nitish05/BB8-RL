import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refined_audit", ROOT / "scripts/audit-refined-scan.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def box():
    return {"name": "room_obstacle_0", "position": [0, 0, 0.5], "size": [1, 1, 1]}


def protocol():
    return {
        "room_half_extent_m": 2.0,
        "elevated_cutoff_m": 0.05,
        "obstacle_sample_spacing_m": 0.05,
        "spatial_support_voxel_m": 0.05,
        "registration_center_threshold_m": 0.05,
        "registration_rotation_threshold_degrees": 1.0,
        "registration_min_inliers": 12,
    }


def test_mapping_provenance_rejects_heldout_direct_and_ancestor_inputs():
    with pytest.raises(ValueError, match="Held-out"):
        audit.validate_mapping_inputs({"mapping_input_view_ids": [0, 14]})
    with pytest.raises(ValueError, match="ancestry"):
        audit.validate_mapping_inputs(
            {
                "mapping_input_view_ids": [0, 3],
                "source_batches": [{"view_ids": [0, 17]}],
            }
        )
    with pytest.raises(ValueError, match="Nested source input"):
        audit.validate_mapping_inputs(
            {"mapping_input_view_ids": [0, 3], "inputs": [{"view_id": "scan-020"}]}
        )
    assert audit.validate_mapping_inputs(
        {"mapping_input_view_ids": [0, 3], "query_indices": [14, 17, 20, 23]},
        ["scan-003"],
    ) == ["scan-000", "scan-003"]


def test_missing_or_duplicate_mapping_provenance_cannot_pass():
    with pytest.raises(ValueError, match="declare"):
        audit.validate_mapping_inputs({})
    with pytest.raises(ValueError, match="Duplicate"):
        audit.validate_mapping_inputs({"mapping_input_view_ids": [0, "scan-000"]})
    with pytest.raises(ValueError, match="disagree"):
        audit.validate_mapping_inputs({"mapping_input_view_ids": [0]}, ["scan-003"])


def test_upstream_half_pixel_resize_and_actual_integer_size_are_distinguished():
    k = np.array([[865.9429225302833, 0, 640], [0, 865.9429225302833, 480], [0, 0, 1]])
    nominal, actual, size, crop = audit.preprocessing_geometry(
        k, (1280, 960), (518, 392)
    )
    assert size == (522, 392) and crop == (2, 0)
    np.testing.assert_allclose(
        nominal[:2, 2], [258.7041666716667, 195.7041666716667], atol=1e-6
    )
    assert nominal[0, 0] != actual[0, 0]
    # PIL maps input pixel centers with (u+.5)*scale-.5, then crops.
    np.testing.assert_allclose(actual[0, 2], (640 + 0.5) * 522 / 1280 - 0.5 - 2)
    np.testing.assert_allclose(actual[1, 2], (480 + 0.5) * 392 / 960 - 0.5)


def test_retention_counts_every_rejected_and_nonfinite_point(tmp_path):
    p = tmp_path / "points.npz"
    np.savez(
        p,
        points=np.array([[0, 0, 0], [1, 1, 1], [np.nan, 0, 0]]),
        valid_mask=np.array([True, False, True]),
    )
    with np.load(p) as arrays:
        points, counts = audit.load_points(arrays, "points")
    assert len(points) == 1 and counts["input_samples"] == 3
    assert counts["producer_rejected_samples"] == counts["nonfinite_valid_samples"] == 1
    assert counts["retained_fraction"] == pytest.approx(1 / 3)


def test_planar_layers_must_partition_valid_samples(tmp_path):
    p = tmp_path / "layers.npz"
    np.savez(
        p,
        points=np.zeros((3, 3)),
        valid_mask=np.array([True, True, False]),
        anchor_mask=np.array([True, False, False]),
        completed_mask=np.array([False, True, False]),
    )
    with np.load(p) as arrays:
        _, counts = audit.load_points(arrays, "points")
    assert counts["anchor_samples"] == counts["completed_samples"] == 1
    np.savez(
        p,
        points=np.zeros((3, 3)),
        valid_mask=np.array([True, True, False]),
        anchor_mask=np.array([True, False, False]),
        completed_mask=np.array([True, True, False]),
    )
    with np.load(p) as arrays, pytest.raises(ValueError, match="partition"):
        audit.load_points(arrays, "points")


def test_perfect_floor_only_map_has_perfect_precision_but_poor_obstacle_completeness():
    xy = np.linspace(-1.9, 1.9, 40)
    x, y = np.meshgrid(xy, xy)
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    points = points[np.max(abs(points[:, :2]), axis=1) > 0.5]
    score = audit.geometry_scores(points, [box()], protocol())
    assert score["precision_in_room_xy"]["within_2cm_fraction"] == 1
    assert score["macro_obstacle_completeness_5cm"] < 0.1
    assert score["obstacles"][0]["elevated_within_5cm_fraction"] == 0


def test_hallucinated_floor_beneath_box_is_not_true_surface():
    score = audit.geometry_scores(np.array([[0.0, 0, 0]]), [box()], protocol())
    assert score["precision_in_room_xy"]["median_m"] == 0.5
    assert score["floor_like_estimated_samples_inside_box_footprints"] == 1


def test_full_free_volume_is_checked_not_just_center():
    # Center x=.65 is outside cube, but the volume intersects it on x=.4..5.
    claims = {
        "free_lower": np.array([[0.4, -0.1, 0.1], [2, 2, 0]]),
        "free_upper": np.array([[0.9, 0.1, 0.2], [3, 3, 0.2]]),
    }
    result = audit.claim_scores(claims, [box()])
    assert result["free_claims"]["intersecting_solid_or_below_floor"] == 1
    assert not result["free_space_certified"]


def test_empty_claim_artifact_never_means_certified_free_space():
    result = audit.claim_scores(None, [box()])
    assert result["free_claims"] is None and not result["free_space_certified"]


def test_support_under_obstacle_is_false_even_on_z_zero():
    result = audit.claim_scores(
        {"support_points": np.array([[0, 0, 0], [1, 1, 0]])}, [box()]
    )
    assert result["support_claims"]["farther_than_2cm_from_exposed_ground"] == 1


def test_camera_error_is_camera_center_distance_not_translation_vector_difference():
    truth = np.eye(4)
    estimate = np.eye(4)
    estimate[:3, 3] = [0.03, 0.04, 0]
    assert audit.pose_errors(estimate, truth) == pytest.approx((0.05, 0))
    estimate[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        audit.pose_errors(estimate, truth)


def test_registration_all_four_denominator_and_failed_pose_validation(tmp_path):
    truth_file = tmp_path / "truth.json"
    truth_file.write_text(json.dumps({"world_to_camera": np.eye(4).tolist()}))
    p = protocol() | {
        "frames": {
            f"scan-{i:03}": {"calibration": str(truth_file)}
            for i in audit.QUERY_INDICES
        }
    }
    result = audit.registration_scores(
        {
            "mapping_input_view_ids": [0],
            "queries": [
                {
                    "query_index": 14,
                    "status": "candidate",
                    "world_to_camera": np.eye(4).tolist(),
                    "inliers": 20,
                },
                {
                    "query_index": 17,
                    "status": "candidate",
                    "world_to_camera": np.zeros((4, 4)).tolist(),
                    "inliers": 20,
                },
                {"query_index": 20, "status": "rejected"},
            ],
        },
        p,
    )
    assert result["requested_queries"] == 4 and result["passed_queries"] == 1
    assert result["queries"][1]["status"] == "invalid_candidate_pose"
    assert result["queries"][3]["status"] == "missing_result"
    assert not result["all_pass"]


def test_paired_ablation_scores_identical_retained_rays_without_truth_selection(
    tmp_path,
):
    k = np.diag([100.0, 100.0, 1.0])
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 2
    y, x = np.indices((4, 4))
    points = np.stack([2 * x / 100, -2 * y / 100, np.zeros_like(x)], axis=-1)
    prior_path, refined_path = tmp_path / "prior.npz", tmp_path / "refined.npz"
    np.savez(
        prior_path,
        view_ids=["scan-000"],
        depth_z_pose_scale_m=np.ones((1, 4, 4), np.float32),
        intrinsics_input_processed=k[None],
    )
    np.savez(
        refined_path,
        view_ids=["scan-000"],
        valid_mask=np.ones((1, 4, 4), bool),
        intrinsics_input_processed=k[None],
        camera_to_world_input=pose[None],
        points_world_refined=points[None],
    )
    result = audit.paired_prior_scores(
        refined_path, prior_path, "points_world_refined", []
    )
    assert result["retained_rays"] == result["common_in_room_rays"] == 16
    assert result["prior_on_retained_rays"]["median_m"] == 1
    assert result["refined_on_retained_rays"]["median_m"] == 0
    assert result["fraction_with_lower_surface_error"] == 1


def test_correspondence_scoring_retains_wrong_associations_in_denominator(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    landmarks = tmp_path / "landmarks.npz"
    feature = features / "scan-014.npz"
    truth = tmp_path / "truth.json"
    np.savez(landmarks, points=np.array([[0.0, 0, 1], [1.0, 0, 1]]))
    np.savez(feature, keypoints=np.array([[0.0, 0], [100.0, 0]]))
    truth.write_text(
        json.dumps(
            {
                "intrinsics": np.diag([10.0, 10.0, 1.0]).tolist(),
                "world_to_camera": np.eye(4).tolist(),
            }
        )
    )
    report = {
        "landmarks_sha256": audit.digest(landmarks),
        "query_inputs": {"scan-014": {"feature_cache_sha256": audit.digest(feature)}},
        "queries": [
            {
                "query_index": 14,
                "correspondences": [
                    {"landmark": 0, "query_index": 0},
                    {"landmark": 1, "query_index": 1},
                ],
                "inlier_indices": [0],
            }
        ],
    }
    result = audit.correspondence_scores(
        report, tmp_path, {"frames": {"scan-014": {"calibration": str(truth)}}}
    )
    row = result["queries"][0]
    assert row["correspondences"] == 2 and row["within_2px_and_positive_depth"] == 1
    assert row["within_2px_fraction"] == 0.5
    feature.write_bytes(b"changed")
    with pytest.raises(ValueError, match="features changed"):
        audit.correspondence_scores(report, tmp_path, {})
