"""Frozen, scoring-only M7.7 geometry and leave-four-out registration audit.

This script never changes a reconstruction, fits calibration or returns a map to
an estimator. No absence of reconstructed points is interpreted as free space.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

QUERY_INDICES = (14, 17, 20, 23)
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scan_surface_scoring", ROOT / "scripts/evaluate-scan-geometry.py"
)
SURFACES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SURFACES)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def view_id(value):
    if isinstance(value, int) and not isinstance(value, bool):
        index = value
    elif isinstance(value, str) and value.startswith("scan-"):
        index = int(value[5:])
    else:
        raise ValueError("Expected scan-NNN view ID or integer index")
    if not 0 <= index < 24:
        raise ValueError("View index outside frozen 24-frame scan")
    return f"scan-{index:03}"


def validate_mapping_inputs(manifest, array_view_ids=None):
    """Require complete declared ancestry; independently check explicit sublists.

    This validates machine-readable declarations, not arbitrary hidden producer
    behavior. Source review and hashes remain necessary provenance evidence.
    """
    declared = manifest.get("mapping_input_view_ids")
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            "Manifest must declare complete mapping_input_view_ids ancestry"
        )
    names = [view_id(v) for v in declared]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate mapping input view IDs")
    forbidden = {view_id(v) for v in QUERY_INDICES}
    if forbidden.intersection(names):
        raise ValueError("Held-out query RGB/pose leaked into mapping ancestry")
    allowed = set(names)
    if array_view_ids is not None and not {view_id(v) for v in array_view_ids}.issubset(
        allowed
    ):
        raise ValueError(
            "Reconstruction view IDs disagree with declared mapping inputs"
        )

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in (
                    "mapping_input_view_ids",
                    "view_ids",
                    "mapping_views",
                    "indices",
                    "prior_view_ids",
                    "source_view_ids",
                    "refinement_view_ids",
                ) and isinstance(child, list):
                    if not {view_id(v) for v in child}.issubset(allowed):
                        raise ValueError(
                            "Nested source ancestry includes undeclared/held-out views"
                        )
                elif key == "view_id" and view_id(child) not in allowed:
                    raise ValueError(
                        "Nested source input includes undeclared/held-out view"
                    )
                elif key not in ("queries", "query_indices", "held_out_indices"):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    return sorted(names)


def freeze_protocol(scan_dir, truth_scene, output):
    if output.exists():
        raise ValueError("Frozen protocol path must be new")
    frames = {}
    for index in range(24):
        stem = scan_dir / view_id(index)
        frames[stem.name] = {
            "rgb": str(stem.with_suffix(".png").resolve()),
            "rgb_sha256": digest(stem.with_suffix(".png")),
            "calibration": str(stem.with_suffix(".json").resolve()),
            "calibration_sha256": digest(stem.with_suffix(".json")),
        }
    protocol = {
        "schema": "bb8.refined-scan-audit.v1",
        "query_indices": list(QUERY_INDICES),
        "query_roles": {
            "14": "new query scoring",
            "17": "new query scoring",
            "20": "new query scoring",
            "23": "previously inspected development failure",
        },
        "map_allowed_view_ids": [
            view_id(i) for i in range(24) if i not in QUERY_INDICES
        ],
        "truth_scene": str(truth_scene.resolve()),
        "truth_scene_sha256": digest(truth_scene),
        "frames": frames,
        "room_half_extent_m": 2.0,
        "surface_thresholds_m": [0.02, 0.05],
        "obstacle_sample_spacing_m": 0.05,
        "elevated_cutoff_m": 0.05,
        "spatial_support_voxel_m": 0.05,
        "registration_center_threshold_m": 0.05,
        "registration_rotation_threshold_degrees": 1.0,
        "registration_min_inliers": 12,
        "rules": [
            "No scene, obstacle, depth, label, or held-out-pose fitting; truth is scorer-only.",
            "All four query RGB and poses excluded from every map/refinement/track/scale ancestor, including unsuccessful inference inputs if their outputs are reused.",
            "Original classical23 includes 14/17/20: rebuild before using its priors.",
            "Report precision jointly with all-obstacle and elevated-surface completeness, valid sample retention and fixed spatial support; deletion alone is not improvement.",
            "All four queries count. Rejected, missing and malformed poses are failures; truth never selects candidates.",
            "Comparisons use identical source RGB and frozen thresholds. Report individual configurations; no truth-based winner promotion.",
            "This same-room synthetic development comparison is not a new-room generalization or navigation success test.",
            "Support and free claims require explicit artifacts. Missing points and rejected pixels remain unknown; no certified free volume follows from this audit.",
        ],
        "protocol_source_sha256": digest(Path(__file__)),
        "surface_scorer_sha256": digest(ROOT / "scripts/evaluate-scan-geometry.py"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, protocol)
    return protocol


def validate_protocol(protocol):
    if protocol.get("query_indices") != list(QUERY_INDICES):
        raise ValueError("Query set differs from frozen M7.7 protocol")
    if digest(protocol["truth_scene"]) != protocol["truth_scene_sha256"]:
        raise ValueError("Scoring scene changed since protocol freeze")
    for row in protocol["frames"].values():
        for name in ("rgb", "calibration"):
            if digest(row[name]) != row[f"{name}_sha256"]:
                raise ValueError("Scan input changed since protocol freeze")


def preprocessing_geometry(k, original_wh, target_wh):
    """Pinned upstream PIL-resize/crop K, plus the actual integer resize affine.

    Upstream uses one nominal scale before integer rounding the resized width.
    The second K describes PIL's actual independent x/y pixel-center scaling.
    Their small difference is diagnostic only; neither is fitted to truth.
    """
    original, target = np.asarray(original_wh), np.asarray(target_wh)
    scale = max(target / original) + 1e-8
    if scale >= 1:
        raise ValueError("This audit reproduces downsampling only")
    resized = np.floor(original * scale).astype(int)
    offset = np.rint((resized - target) / 2).astype(int)
    nominal = np.array(k, dtype=float, copy=True)
    nominal[:2, 2] += 0.5
    nominal[:2] *= scale
    nominal[:2, 2] -= (original * scale - resized) / 2 + 0.5 + offset
    actual = np.array(k, dtype=float, copy=True)
    actual[:2, 2] += 0.5
    actual[:2] *= (resized / original)[:, None]
    actual[:2, 2] -= 0.5 + offset
    return nominal, actual, tuple(resized), tuple(offset)


def calibration_audit(arrays, protocol):
    keys = {
        "view_ids",
        "intrinsics_input_original",
        "intrinsics_input_processed",
        "camera_to_world_input",
        "rgb_uint8",
    }
    if not keys.issubset(arrays.files):
        return {
            "status": "not_available",
            "reason": "Artifact lacks saved per-view preprocessing arrays",
        }
    arrays = {key: arrays[key] for key in keys}
    rows = []
    for i, name in enumerate(arrays["view_ids"].tolist()):
        original_rgb = Image.open(protocol["frames"][name]["rgb"]).convert("RGB")
        metadata = json.loads(Path(protocol["frames"][name]["calibration"]).read_text())
        k, t = np.array(metadata["intrinsics"]), np.array(metadata["world_to_camera"])
        saved_rgb = arrays["rgb_uint8"][i]
        h, w = saved_rgb.shape[:2]
        nominal, actual, resized, offset = preprocessing_geometry(
            k, original_rgb.size, (w, h)
        )
        expected = np.array(
            original_rgb.resize(resized, Image.Resampling.LANCZOS).crop(
                (*offset, offset[0] + w, offset[1] + h)
            )
        )
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]])
        rays = corners @ np.linalg.inv(nominal).T
        actual_pixels = rays @ actual.T
        delta = np.linalg.norm(
            actual_pixels[:, :2] / actual_pixels[:, 2:] - corners[:, :2], axis=1
        )
        rows.append(
            {
                "view_id": name,
                "original_K_max_abs_difference": float(
                    np.max(abs(arrays["intrinsics_input_original"][i] - k))
                ),
                "input_pose_max_abs_difference": float(
                    np.max(abs(arrays["camera_to_world_input"][i] - np.linalg.inv(t)))
                ),
                "processed_K_vs_pinned_upstream_max_abs_difference": float(
                    np.max(abs(arrays["intrinsics_input_processed"][i] - nominal))
                ),
                "processed_rgb_max_byte_difference": int(
                    np.max(abs(saved_rgb.astype(int) - expected.astype(int)))
                ),
                "processed_rgb_mean_abs_byte_difference": float(
                    np.mean(abs(saved_rgb.astype(int) - expected.astype(int)))
                ),
                "nominal_vs_integer_resize_max_corner_error_processed_px": float(
                    delta.max()
                ),
                "resized_wh": [int(v) for v in resized],
                "crop_left_top": [int(v) for v in offset],
                "native_camera_center_vs_declared_position_m": float(
                    np.linalg.norm(np.linalg.inv(t)[:3, 3] - metadata["position"])
                ),
                "rotation_orthogonality_max_error": float(
                    np.max(abs(t[:3, :3] @ t[:3, :3].T - np.eye(3)))
                ),
            }
        )
    return {
        "status": "complete",
        "rows": rows,
        "scope": "Independent saved-RGB/preprocessing/transform consistency; no oracle calibration fit. Subpixel integer-resize discrepancy is distinguished from a frame/axis mismatch.",
    }


def load_points(arrays, field):
    values = np.asarray(arrays[field])
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("Point field must end in dimension 3")
    mask = (
        np.asarray(arrays["valid_mask"])
        if "valid_mask" in arrays.files
        else np.ones(values.shape[:-1], bool)
    )
    if mask.shape != values.shape[:-1] or mask.dtype != np.bool_:
        raise ValueError("Valid mask must be boolean and match point field")
    finite = np.isfinite(values).all(-1)
    counts = {
        "input_samples": int(mask.size),
        "producer_valid_samples": int(mask.sum()),
        "finite_valid_samples": int((mask & finite).sum()),
        "producer_rejected_samples": int((~mask).sum()),
        "nonfinite_valid_samples": int((mask & ~finite).sum()),
        "retained_fraction": float((mask & finite).sum() / mask.size)
        if mask.size
        else 0.0,
    }
    if "prior_valid_mask" in arrays.files:
        prior = np.asarray(arrays["prior_valid_mask"])
        if prior.dtype != np.bool_ or prior.shape != mask.shape:
            raise ValueError("Prior valid mask must be boolean and match point field")
        counts["prior_valid_samples"] = int(prior.sum())
        counts["retained_from_prior_samples"] = int((prior & mask & finite).sum())
        counts["retained_from_prior_fraction"] = (
            float((prior & mask & finite).sum() / prior.sum()) if prior.any() else 0.0
        )
        counts["valid_samples_outside_prior_mask"] = int((~prior & mask & finite).sum())
    if "anchor_mask" in arrays.files and "completed_mask" in arrays.files:
        anchors, completed = arrays["anchor_mask"], arrays["completed_mask"]
        if (
            any(
                a.shape != mask.shape or a.dtype != np.bool_
                for a in (anchors, completed)
            )
            or np.any(anchors & completed)
            or not np.array_equal(anchors | completed, mask)
        ):
            raise ValueError("Anchor/completed masks must partition the valid mask")
        counts["anchor_samples"] = int((anchors & finite).sum())
        counts["completed_samples"] = int((completed & finite).sum())
    return values[mask & finite], counts


def geometry_scores(points, boxes, protocol):
    SURFACES.validate_boxes(boxes)
    extent = protocol["room_half_extent_m"]
    inside = np.max(abs(points[:, :2]), axis=1) < extent
    selected = points[inside]
    distances, _ = SURFACES.nearest_surface(selected, boxes)
    elevated = selected[:, 2] > protocol["elevated_cutoff_m"]
    tree = cKDTree(points) if len(points) else None
    completeness = []
    for box in boxes:
        if not box["name"].startswith("room_obstacle"):
            continue
        samples = SURFACES.sample_exterior(box, protocol["obstacle_sample_spacing_m"])
        errors = tree.query(samples)[0] if tree else np.full(len(samples), np.inf)
        raised = samples[:, 2] > protocol["elevated_cutoff_m"]
        completeness.append(
            {
                "name": box["name"],
                "sample_count": len(samples),
                "within_2cm_fraction": float(np.mean(errors <= 0.02)),
                "within_5cm_fraction": float(np.mean(errors <= 0.05)),
                "elevated_sample_count": int(raised.sum()),
                "elevated_within_2cm_fraction": float(np.mean(errors[raised] <= 0.02)),
                "elevated_within_5cm_fraction": float(np.mean(errors[raised] <= 0.05)),
            }
        )
    floor_like = abs(selected[:, 2]) <= 0.02
    footprint = np.zeros(len(selected), bool)
    for box in boxes:
        lo = np.array(box["position"]) - np.array(box["size"]) / 2
        hi = lo + box["size"]
        footprint |= np.all(
            (selected[:, :2] > lo[:2]) & (selected[:, :2] < hi[:2]), axis=1
        )
    voxel = protocol["spatial_support_voxel_m"]
    support = (
        np.unique(np.floor(selected / voxel).astype(np.int64), axis=0)
        if len(selected)
        else np.empty((0, 3))
    )
    elevated_support = support[support[:, 2] * voxel > protocol["elevated_cutoff_m"]]
    return {
        "precision_in_room_xy": SURFACES.summarize(distances),
        "precision_elevated_in_room_xy": SURFACES.summarize(distances[elevated]),
        "outside_primary_room_xy_points": int((~inside).sum()),
        "below_support_plane_by_more_than_2cm_points": int(
            (selected[:, 2] < -0.02).sum()
        ),
        "floor_like_estimated_samples_inside_box_footprints": int(
            (floor_like & footprint).sum()
        ),
        "floor_like_inside_note": "Scoring diagnostic on estimates, not producer support/free claims; no masks created for the estimator.",
        "occupied_sample_voxels_5cm": len(support),
        "elevated_sample_voxels_5cm": len(elevated_support),
        "obstacles": completeness,
        "macro_obstacle_completeness_5cm": float(
            np.mean([r["within_5cm_fraction"] for r in completeness])
        )
        if completeness
        else None,
        "minimum_obstacle_completeness_5cm": min(
            (r["within_5cm_fraction"] for r in completeness), default=None
        ),
        "completeness_note": "Equal-obstacle summary of <=5cm spaced four sides/top, including unobserved faces; not surface-area weighting or visibility-adjusted. Elevated samples exclude the floor-adjacent band.",
    }


def pose_errors(estimate, truth):
    estimate, truth = np.asarray(estimate, float), np.asarray(truth, float)
    if (
        estimate.shape != (4, 4)
        or not np.isfinite(estimate).all()
        or not np.allclose(estimate[3], [0, 0, 0, 1])
        or not np.allclose(estimate[:3, :3].T @ estimate[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(estimate[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("Registration must be a finite rigid world-to-camera pose")
    center = np.linalg.norm(
        np.linalg.inv(estimate)[:3, 3] - np.linalg.inv(truth)[:3, 3]
    )
    rotation = np.degrees(
        np.arccos(
            np.clip((np.trace(estimate[:3, :3] @ truth[:3, :3].T) - 1) / 2, -1, 1)
        )
    )
    return float(center), float(rotation)


def registration_scores(report, protocol):
    validate_mapping_inputs(report)
    rows = report.get("queries", [])
    ids = [r["query_index"] for r in rows]
    if len(ids) != len(set(ids)) or any(i not in QUERY_INDICES for i in ids):
        raise ValueError("Duplicate or unexpected registration query")
    by_id = {r["query_index"]: r for r in rows}
    results = []
    for index in QUERY_INDICES:
        row = by_id.get(index, {"status": "missing_result"})
        result = {
            "query_index": index,
            "status": row["status"],
            "passes_pose_gate": False,
        }
        if row["status"] == "candidate":
            truth = json.loads(
                Path(protocol["frames"][view_id(index)]["calibration"]).read_text()
            )["world_to_camera"]
            try:
                center, angle = pose_errors(row.get("world_to_camera"), truth)
            except (TypeError, ValueError, np.linalg.LinAlgError):
                result["status"] = "invalid_candidate_pose"
            else:
                result.update(
                    camera_center_error_m=center, rotation_error_degrees=angle
                )
                result["passes_pose_gate"] = (
                    center <= protocol["registration_center_threshold_m"]
                    and angle <= protocol["registration_rotation_threshold_degrees"]
                    and row.get("inliers", 0) >= protocol["registration_min_inliers"]
                )
        results.append(result)
    return {
        "requested_queries": len(QUERY_INDICES),
        "passed_queries": sum(r["passes_pose_gate"] for r in results),
        "queries": results,
        "all_pass": all(r["passes_pose_gate"] for r in results),
    }


def correspondence_scores(report, directory, protocol):
    """Posthoc query-pose projection; jointly tests association and map geometry."""
    map_path = directory / "landmarks.npz"
    if not map_path.is_file() or "landmarks_sha256" not in report:
        return {"status": "not_available"}
    if digest(map_path) != report["landmarks_sha256"]:
        raise ValueError("Frozen registration landmarks changed")
    with np.load(map_path, allow_pickle=False) as arrays:
        points = arrays["points"]
    rows = []
    for query in report["queries"]:
        index = query["query_index"]
        name = view_id(index)
        path = directory / "features" / f"{name}.npz"
        if digest(path) != report["query_inputs"][name]["feature_cache_sha256"]:
            raise ValueError("Frozen query features changed")
        with np.load(path, allow_pickle=False) as arrays:
            pixels = arrays["keypoints"]
        corr = query.get("correspondences", [])
        if not corr:
            rows.append({"query_index": index, "correspondences": 0})
            continue
        landmarks = points[[c["landmark"] for c in corr]]
        measured = pixels[[c["query_index"] for c in corr]]
        truth = json.loads(Path(protocol["frames"][name]["calibration"]).read_text())
        t, k = np.array(truth["world_to_camera"]), np.array(truth["intrinsics"])
        camera = landmarks @ t[:3, :3].T + t[:3, 3]
        projected = camera @ k.T
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.linalg.norm(
                projected[:, :2] / projected[:, 2:] - measured, axis=1
            )
        correct = np.isfinite(errors) & (camera[:, 2] > 0) & (errors <= 2)
        inliers = np.array(query.get("inlier_indices", []), dtype=int)
        finite = errors[np.isfinite(errors)]
        rows.append(
            {
                "query_index": index,
                "correspondences": len(corr),
                "within_2px_and_positive_depth": int(correct.sum()),
                "within_2px_fraction": float(correct.mean()),
                "finite_error_p50_p95_max_px": np.quantile(
                    finite, [0.5, 0.95, 1]
                ).tolist()
                if len(finite)
                else None,
                "estimator_selected_inliers": len(inliers),
                "selected_inliers_within_2px": int(correct[inliers].sum()),
                "selected_inlier_truth_pose_error_p95_px": float(
                    np.quantile(errors[inliers], 0.95)
                )
                if len(inliers) and np.isfinite(errors[inliers]).all()
                else None,
            }
        )
    return {
        "status": "complete",
        "scope": "Known query pose used only after estimator freeze. Reprojection consistency combines map-point error and association error; it does not alone prove semantic correspondence identity.",
        "queries": rows,
    }


def paired_prior_scores(
    reconstruction,
    prior_path,
    field,
    boxes,
    *,
    prior_depth_field="depth_z_pose_scale_m",
):
    """Compare old/new depth on identical producer-retained rays, without fitting."""
    import cv2

    with np.load(prior_path, allow_pickle=False) as arrays:
        prior = {
            key: arrays[key]
            for key in (
                "view_ids",
                prior_depth_field,
                "intrinsics_input_processed",
            )
        }
    before, after = [], []
    with np.load(reconstruction, allow_pickle=False) as arrays:
        new = {
            key: arrays[key]
            for key in (
                "view_ids",
                "valid_mask",
                "intrinsics_input_processed",
                "camera_to_world_input",
                field,
            )
        }
    for i, name in enumerate(new["view_ids"].tolist()):
        slot = prior["view_ids"].tolist().index(name)
        mask = new["valid_mask"][i] & np.isfinite(new[field][i]).all(-1)
        y, x = np.nonzero(mask)
        if not len(x):
            continue
        rays = (
            np.c_[x, y, np.ones(len(x))]
            @ np.linalg.inv(new["intrinsics_input_processed"][i]).T
        )
        uv = rays @ prior["intrinsics_input_processed"][slot].T
        uv = (uv[:, :2] / uv[:, 2:]).astype(np.float32)
        depth = cv2.remap(
            prior[prior_depth_field][slot],
            uv[:, 0:1],
            uv[:, 1:2],
            cv2.INTER_LINEAR,
        ).ravel()
        pose = new["camera_to_world_input"][i]
        before.append((rays * depth[:, None]) @ pose[:3, :3].T + pose[:3, 3])
        after.append(new[field][i][mask])
    before = np.concatenate(before) if before else np.empty((0, 3))
    after = np.concatenate(after) if after else np.empty((0, 3))
    shared = (
        np.isfinite(before).all(1)
        & (np.max(abs(before[:, :2]), axis=1) < 2)
        & (np.max(abs(after[:, :2]), axis=1) < 2)
    )
    old_error, _ = SURFACES.nearest_surface(before[shared], boxes)
    new_error, _ = SURFACES.nearest_surface(after[shared], boxes)
    return {
        "scope": "Identical producer-retained rays; deterministic original-prior resampling with saved K/poses, no truth-based selection. Primary-room summaries require both old/new estimates within xy(-2,2).",
        "prior_depth_field": prior_depth_field,
        "retained_rays": len(before),
        "common_in_room_rays": int(shared.sum()),
        "prior_on_retained_rays": SURFACES.summarize(old_error),
        "refined_on_retained_rays": SURFACES.summarize(new_error),
        "fraction_with_lower_surface_error": float(np.mean(new_error < old_error))
        if len(old_error)
        else None,
        "median_surface_error_reduction_m": float(np.median(old_error - new_error))
        if len(old_error)
        else None,
    }


def claim_scores(claims, boxes):
    result = {
        "free_space_certified": False,
        "claim_artifact_supplied": claims is not None,
    }
    if claims is None:
        return result | {
            "free_claims": None,
            "support_claims": None,
            "note": "No claim artifact; absence is not zero false claims and is not free evidence.",
        }
    if "free_lower" in claims or "free_upper" in claims:
        lower, upper = (
            np.asarray(claims["free_lower"]),
            np.asarray(claims["free_upper"]),
        )
        if (
            lower.ndim != 2
            or lower.shape[1] != 3
            or upper.shape != lower.shape
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(upper <= lower)
        ):
            raise ValueError("Free claims require finite nonempty axis-aligned volumes")
        intersects = lower[:, 2] < 0
        for box in boxes:
            lo = np.array(box["position"]) - np.array(box["size"]) / 2
            hi = lo + box["size"]
            intersects |= np.all(np.minimum(upper, hi) > np.maximum(lower, lo), axis=1)
        result["free_claims"] = {
            "volumes": len(lower),
            "intersecting_solid_or_below_floor": int(intersects.sum()),
            "false_claim_fraction": float(intersects.mean()) if len(lower) else None,
            "scope": "Exact positive-volume intersection, not cell-center checks; no certification of unmodelled contents.",
        }
    if "support_points" in claims:
        points = np.asarray(claims["support_points"])
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Support claims require finite Nx3 points")
        invalid = SURFACES.exposed_floor_distance(points, boxes) > 0.02
        result["support_claims"] = {
            "points": len(points),
            "farther_than_2cm_from_exposed_ground": int(invalid.sum()),
        }
    return result


def evaluate(args):
    if args.output.exists():
        raise ValueError("Use a fresh audit output")
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    manifest = json.loads(args.manifest.read_text())
    hashes = {
        "protocol": digest(args.protocol),
        "reconstruction": digest(args.reconstruction),
        "manifest": digest(args.manifest),
        "audit_source": digest(Path(__file__)),
        "surface_scorer": digest(ROOT / "scripts/evaluate-scan-geometry.py"),
    }
    with np.load(args.reconstruction, allow_pickle=False) as arrays:
        names = arrays["view_ids"].tolist() if "view_ids" in arrays.files else None
        provenance = validate_mapping_inputs(manifest, names)
        points, retention = load_points(arrays, args.point_field)
        preprocessing = calibration_audit(arrays, protocol)
        layers = {}
        for name in ("anchor", "completed"):
            if f"{name}_mask" in arrays.files:
                values = arrays[args.point_field]
                layers[name] = values[
                    arrays[f"{name}_mask"] & np.isfinite(values).all(-1)
                ]
    # No truth geometry is read until the finished producer artifact is loaded.
    scene = json.loads(Path(protocol["truth_scene"]).read_text())
    boxes = [b for b in scene["objects"] if b["format"] == "box" and b.get("fixed")]
    report = {
        "status": "complete",
        "scope": "Frozen synthetic development scoring only; no estimator fitting or motion authorization",
        "sha256": hashes,
        "mapping_input_view_ids": provenance,
        "point_field": args.point_field,
        "retention": retention,
        "preprocessing": preprocessing,
        "geometry": geometry_scores(points, boxes, protocol),
    }
    if layers:
        report["geometry_by_layer"] = {
            name: geometry_scores(values, boxes, protocol)
            for name, values in layers.items()
        }
    if args.prior_reconstruction:
        hashes["paired_prior_reconstruction"] = digest(args.prior_reconstruction)
        if manifest.get("prior_sha256") != hashes["paired_prior_reconstruction"]:
            raise ValueError("Paired prior differs from producer's declared prior hash")
        report["paired_prior_ablation"] = paired_prior_scores(
            args.reconstruction,
            args.prior_reconstruction,
            args.point_field,
            boxes,
            prior_depth_field=manifest.get("prior_depth_field", "depth_z_pose_scale_m"),
        )
    if args.registrations:
        hashes["registrations"] = digest(args.registrations)
        registration_report = json.loads(args.registrations.read_text())
        report["registration"] = registration_scores(registration_report, protocol)
        report["registration_correspondences"] = correspondence_scores(
            registration_report, args.registrations.parent, protocol
        )
    else:
        report["registration"] = {
            "status": "not_supplied",
            "requested_queries": 4,
            "passed_queries": 0,
        }
    if args.claims:
        hashes["claims"] = digest(args.claims)
        with np.load(args.claims, allow_pickle=False) as claims:
            report["claims"] = claim_scores(claims, boxes)
    else:
        report["claims"] = claim_scores(None, boxes)
    args.output.mkdir(parents=True)
    (args.output / "audit-refined-scan.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "surface-scorer.py").write_bytes(
        (ROOT / "scripts/evaluate-scan-geometry.py").read_bytes()
    )
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "retention", "registration", "claims")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--scan-dir", type=Path, required=True)
    freeze.add_argument(
        "--truth-scene",
        type=Path,
        default=ROOT / "projects/bb8/synthetic-room/room.genesis.json",
    )
    freeze.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    for name in ("protocol", "reconstruction", "manifest", "output"):
        score.add_argument(f"--{name}", type=Path, required=True)
    score.add_argument("--point-field", default="points_world_refined")
    score.add_argument("--registrations", type=Path)
    score.add_argument("--prior-reconstruction", type=Path)
    score.add_argument("--claims", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_protocol(args.scan_dir, args.truth_scene, args.output)
    else:
        evaluate(args)
