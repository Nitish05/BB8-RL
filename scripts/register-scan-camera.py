"""Register a held-out RGB camera against predicted dense scan-map landmarks."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.diagnostics import write_report
from bb8_rl.mapping.scan_geometry import extract, register_camera
from bb8_rl.training import file_hash


def sample_feature_points(
    pixels,
    points,
    depth,
    valid,
    *,
    radius=2,
    absolute_spread=0.05,
    relative_spread=0.02,
):
    """Bilinear points only in fully supported, locally continuous depth patches."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    points, depth, valid = np.asarray(points), np.asarray(depth), np.asarray(valid)
    if (
        points.shape != (*depth.shape, 3)
        or valid.shape != depth.shape
        or depth.ndim != 2
    ):
        raise ValueError("Dense point/depth/mask shapes disagree")
    if radius < 1 or absolute_spread <= 0 or relative_spread < 0:
        raise ValueError("Invalid depth-continuity gate")
    height, width = depth.shape
    accepted, xyz = [], []
    rejected = {
        "boundary_or_nonfinite": 0,
        "invalid_neighborhood": 0,
        "depth_discontinuity": 0,
    }
    for index, (u, v) in enumerate(pixels):
        if not np.isfinite([u, v]).all():
            rejected["boundary_or_nonfinite"] += 1
            continue
        x, y = int(np.floor(u)), int(np.floor(v))
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius >= width
            or y + radius >= height
        ):
            rejected["boundary_or_nonfinite"] += 1
            continue
        region = (slice(y - radius, y + radius + 1), slice(x - radius, x + radius + 1))
        local_depth = depth[region]
        if (
            not valid[region].all()
            or not np.isfinite(local_depth).all()
            or np.any(local_depth <= 0)
            or not np.isfinite(points[region]).all()
        ):
            rejected["invalid_neighborhood"] += 1
            continue
        if np.ptp(local_depth) > max(
            absolute_spread, relative_spread * np.median(local_depth)
        ):
            rejected["depth_discontinuity"] += 1
            continue
        du, dv = u - x, v - y
        point = (
            (1 - du) * (1 - dv) * points[y, x]
            + du * (1 - dv) * points[y, x + 1]
            + (1 - du) * dv * points[y + 1, x]
            + du * dv * points[y + 1, x + 1]
        )
        accepted.append(index)
        xyz.append(point)
    return (
        np.asarray(accepted, dtype=int),
        np.asarray(xyz, dtype=float).reshape(-1, 3),
        rejected,
    )


def pose_difference(first, second):
    first, second = np.asarray(first), np.asarray(second)
    centers = [np.linalg.inv(transform)[:3, 3] for transform in (first, second)]
    angle = np.degrees(
        np.arccos(np.clip((np.trace(first[:3, :3] @ second[:3, :3].T) - 1) / 2, -1, 1))
    )
    return float(np.linalg.norm(centers[0] - centers[1])), float(angle)


def elevation_summary(points):
    """Diagnostics relative to the known z=0 support plane; never a point filter."""
    z = np.asarray(points, dtype=float).reshape(-1, 3)[:, 2]
    return {
        "z_min_p05_p50_p95_max_m": np.quantile(z, [0, 0.05, 0.5, 0.95, 1]).tolist()
        if len(z)
        else None,
        "elevation_spread_m": float(np.ptp(z)) if len(z) else None,
        "near_floor_abs_z_le_005m": int((abs(z) <= 0.05).sum()),
        "elevated_z_gt_005m": int((z > 0.05).sum()),
        "below_support_plane_z_lt_minus005m": int((z < -0.05).sum()),
    }


def select_consensus(candidates, *, center_tolerance=0.15, angle_tolerance=3.0):
    """Select only mutually agreeing independent map-view PnP candidates.

    The global pooled fit is diagnostic and cannot count as another independent
    camera. A single-view fit remains unconfirmed, even with many SIFT inliers.
    """
    eligible = [
        row
        for row in candidates
        if row["view_id"] != "global" and row["registration"]["status"] == "candidate"
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -row["registration"]["inliers"],
            row["registration"]["reprojection_p95_px"],
            row["view_id"],
        ),
    )
    clusters = []
    for anchor in ranked:
        cluster = [anchor]
        for proposal in ranked:
            if proposal is anchor:
                continue
            differences = [
                pose_difference(
                    proposal["registration"]["world_to_camera"],
                    row["registration"]["world_to_camera"],
                )
                for row in cluster
            ]
            if all(
                center <= center_tolerance and angle <= angle_tolerance
                for center, angle in differences
            ):
                cluster.append(proposal)
        clusters.append(cluster)
    clusters.sort(
        key=lambda rows: (
            -len(rows),
            -sum(row["registration"]["inliers"] for row in rows),
        )
    )
    if not clusters or len(clusters[0]) < 2:
        return {
            "status": "registration_pending",
            "reason": "No two independent map-view poses agree",
            "selected_view_id": None,
            "support_view_ids": [],
        }
    selected = min(
        clusters[0],
        key=lambda row: (
            -row["registration"]["inliers"],
            row["registration"]["reprojection_p95_px"],
        ),
    )
    return {
        "status": "candidate",
        "selected_view_id": selected["view_id"],
        "support_view_ids": [row["view_id"] for row in clusters[0]],
        "world_to_camera": selected["registration"]["world_to_camera"],
    }


def estimate_registration(query_rgb, query_intrinsics, clouds):
    """Estimator boundary: no query pose, reference geometry or scorer labels."""
    candidates = []
    for view_id, cloud in clouds.items():
        candidates.append(
            {
                "view_id": view_id,
                "registration": register_camera(query_rgb, query_intrinsics, cloud),
            }
        )
    pooled = {
        "points": np.concatenate([cloud["points"] for cloud in clouds.values()]),
        "descriptors": np.concatenate(
            [cloud["descriptors"] for cloud in clouds.values()]
        ),
    }
    pooled["descriptor_landmarks"] = np.arange(len(pooled["points"]))
    candidates.append(
        {
            "view_id": "global",
            "registration": register_camera(query_rgb, query_intrinsics, pooled),
        }
    )
    return {"selection": select_consensus(candidates), "candidates": candidates}


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh registration output")
    cv2.setNumThreads(2)
    manifest_path = args.reconstruction.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_view_ids = [row["view_id"] for row in manifest["inputs"]]
    map_indices = [int(view_id.removeprefix("scan-")) for view_id in expected_view_ids]
    if manifest["status"] != "complete" or args.query_index in map_indices:
        raise ValueError("Need a complete reconstruction excluding the query camera")
    field_pairs = {
        "points_world_calibrated": "depth_z_m",
        "points_world_calibrated_pose_scale": "depth_z_pose_scale_m",
    }
    depth_field = field_pairs[args.point_field]
    if args.point_field.endswith("pose_scale") and (
        manifest.get("pose_alignment", {}).get("held_out_camera_used") is not False
        or manifest.get("pose_alignment", {}).get("oracle_geometry_fit") is not False
    ):
        raise ValueError("Scale enforcement must exclude the query and oracle geometry")
    query_stem = args.scan_dir / f"scan-{args.query_index:03}"
    query_rgb_path, query_meta_path = (
        query_stem.with_suffix(".png"),
        query_stem.with_suffix(".json"),
    )
    query_rgb = cv2.cvtColor(cv2.imread(str(query_rgb_path)), cv2.COLOR_BGR2RGB)
    query_intrinsics = np.array(json.loads(query_meta_path.read_text())["intrinsics"])
    clouds, summaries = {}, []
    with np.load(args.reconstruction) as arrays:
        view_ids = arrays["view_ids"].tolist()
        if view_ids != expected_view_ids or len(set(view_ids)) != len(view_ids):
            raise ValueError("Reconstruction view membership differs from manifest")
        rgbs, depths, valid = (
            arrays["rgb_uint8"],
            arrays[depth_field],
            arrays["valid_mask"],
        )
        points = arrays[args.point_field]
        for index, view_id in enumerate(view_ids):
            features = extract(
                rgbs[index], arrays["intrinsics_input_processed"][index], np.eye(4)
            )
            accepted, xyz, rejected = sample_feature_points(
                features["pixels"], points[index], depths[index], valid[index]
            )
            descriptors = features["descriptors"]
            clouds[view_id] = {
                "points": xyz,
                "descriptors": descriptors[accepted]
                if descriptors is not None
                else np.empty((0, 128), np.float32),
                "descriptor_landmarks": np.arange(len(xyz)),
            }
            summaries.append(
                {
                    "view_id": view_id,
                    "features": len(features["pixels"]),
                    "accepted_landmarks": len(xyz),
                    "rejections": rejected,
                    "accepted_elevation": elevation_summary(xyz),
                }
            )
    # Freeze estimator output before consulting the held-out pose for scoring.
    result = estimate_registration(query_rgb, query_intrinsics, clouds)
    truth = np.array(json.loads(query_meta_path.read_text())["world_to_camera"])
    for candidate in result["candidates"]:
        registration = candidate["registration"]
        if registration["status"] == "candidate":
            center, angle = pose_difference(registration["world_to_camera"], truth)
            candidate["scoring_only"] = {
                "camera_center_error_m": center,
                "rotation_error_degrees": angle,
            }
    selected = result["selection"]
    if selected["status"] == "candidate":
        center, angle = pose_difference(selected["world_to_camera"], truth)
        selected["scoring_only"] = {
            "camera_center_error_m": center,
            "rotation_error_degrees": angle,
        }
    report = {
        "status": "complete",
        "registration_validation": "pending",
        "scope": "actual held-out RGB+K registration against learned dense geometry; candidate poses are not accepted physical calibration",
        "query_index": args.query_index,
        "query_excluded_from_map": True,
        "map_indices": map_indices,
        "map_point_field": args.point_field,
        "map_depth_field": depth_field,
        "map_point_semantics": "Learned optical-Z depth, optionally scaled using scan-camera priors only, reprojected with known processed K and known scan poses",
        "query_pose_estimator_input": False,
        "truth_use": "only independent scoring after the complete estimator returns",
        "reconstruction_sha256": file_hash(args.reconstruction),
        "reconstruction_manifest_sha256": file_hash(manifest_path),
        "query_rgb_sha256": file_hash(query_rgb_path),
        "query_metadata_sha256": file_hash(query_meta_path),
        "parameters": {
            "nfeatures": 5000,
            "sift_contrast_threshold": 0.02,
            "depth_neighborhood_radius_px": 2,
            "depth_spread_max_absolute_m": 0.05,
            "depth_spread_max_relative": 0.02,
            "depth_spread_rule": "max(absolute, relative times local median)",
            "ratio": 0.75,
            "min_inliers": 12,
            "pnp_reprojection_px": 2.0,
            "consensus_center_tolerance_m": 0.15,
            "consensus_angle_tolerance_degrees": 3.0,
            "minimum_independent_view_candidates": 2,
        },
        "map_views": summaries,
        **result,
        "source_sha256": file_hash(Path(__file__)),
        "registration_source_sha256": file_hash(
            Path(__file__).resolve().parents[1] / "src/bb8_rl/mapping/scan_geometry.py"
        ),
        "limitations": [
            "Learned depth is not ground truth; coherent bias can produce a low-reprojection but inaccurate pose.",
            "Repeated checker texture can yield ambiguous descriptor correspondences.",
            "No map space or robot position becomes observed/free through camera registration.",
        ],
    }
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    np.savez_compressed(
        args.output / "landmarks.npz",
        **{
            f"{view_id}_{key}": value
            for view_id, cloud in clouds.items()
            for key, value in cloud.items()
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction",
        type=Path,
        default=Path("work/m76/mapanything/run-8/reconstruction.npz"),
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(
            "work/scan/view-probe"
        ),
    )
    parser.add_argument("--query-index", type=int, default=23)
    parser.add_argument(
        "--point-field",
        choices=("points_world_calibrated", "points_world_calibrated_pose_scale"),
        default="points_world_calibrated",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/m76/fixed-camera-registration")
    )
    main(parser.parse_args())
