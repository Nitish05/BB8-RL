"""Correct learned scan depth using independent RGB triangulation and calibration."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bb8_rl.mapping.backends.mapanything import calibrated_world_points, write_ply
from bb8_rl.mapping.depth_calibration import (
    FROZEN_QUERY_IDS,
    apply_depth_transform,
    choose_depth_transform,
    deduplicate_rgb_tracks,
    error_summary,
    sample_bilinear,
    triangulated_rgb_anchors,
)
from bb8_rl.mapping.scan_geometry import extract, project, triangulate_track


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_preprocessing(scan, arrays):
    """Replay official preprocessing only; no model weights or query inputs."""
    from mapanything.utils.image import preprocess_inputs, rgb

    result = []
    for i, stem in enumerate(arrays["view_ids"].tolist()):
        metadata = json.loads((scan / f"{stem}.json").read_text())
        original = np.array(Image.open(scan / f"{stem}.png").convert("RGB"))
        processed = preprocess_inputs(
            [
                {
                    "img": original,
                    "intrinsics": np.array(metadata["intrinsics"], np.float32),
                    "camera_poses": np.linalg.inv(
                        np.array(metadata["world_to_camera"], np.float64)
                    ).astype(np.float32),
                    "is_metric_scale": True,
                }
            ]
        )[0]
        repeated = np.rint(rgb(processed["img"], "dinov2") * 255).astype(np.uint8)[0]
        k = processed["intrinsics"].numpy()[0]
        result.append(
            {
                "view_id": stem,
                "rgb_max_abs_uint8_difference": int(
                    np.max(
                        abs(repeated.astype(int) - arrays["rgb_uint8"][i].astype(int))
                    )
                ),
                "intrinsics_max_abs_difference": float(
                    np.max(abs(k - arrays["intrinsics_input_processed"][i]))
                ),
            }
        )
    return result


def load_rgb_anchor_map(anchor_dir, scan_dir):
    """Consume only frozen map ancestry/features, never query estimates or truth."""
    manifest_path = anchor_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    input_ids = [int(x.split("-")[-1]) for x in manifest["mapping_input_view_ids"]]
    if (
        set(input_ids) & FROZEN_QUERY_IDS
        or not manifest["map_frozen_before_query_matching"]
    ):
        raise ValueError(
            "Anchor map contains frozen queries or is not frozen before queries"
        )
    landmark_path = anchor_dir / "landmarks.npz"
    if digest(landmark_path) != manifest["landmarks_sha256"]:
        raise ValueError("Anchor map hash no longer matches frozen manifest")
    source_tracks = json.loads((anchor_dir / "tracks.json").read_text())["tracks"]
    with np.load(landmark_path, allow_pickle=False) as archive:
        source_points = archive["points"]
    if len(source_points) != len(source_tracks):
        raise ValueError("Anchor point/track count mismatch")
    features = {}
    calibration = {}
    input_hashes = {}
    for view_id in input_ids:
        stem = f"scan-{view_id:03}"
        feature_path = anchor_dir / "features" / f"{stem}.npz"
        with np.load(feature_path, allow_pickle=False) as archive:
            features[view_id] = archive["keypoints"]
        meta_path = scan_dir / f"{stem}.json"
        calibration[view_id] = json.loads(meta_path.read_text())
        input_hashes[stem] = {
            "rgb_features": digest(feature_path),
            "calibration": digest(meta_path),
        }
    accepted = []
    for landmark_id, track in enumerate(source_tracks):
        views = [int(str(o[0]).split("-")[-1]) for o in track["observations"]]
        if set(views) & FROZEN_QUERY_IDS or not set(views) <= set(input_ids):
            raise ValueError("Forbidden or unregistered view in anchor track")
        if len(views) < 3:
            continue
        pixels = [
            features[v][int(o[1])]
            for v, o in zip(views, track["observations"], strict=True)
        ]
        solved = triangulate_track(
            pixels,
            [calibration[v]["intrinsics"] for v in views],
            [calibration[v]["world_to_camera"] for v in views],
            max_error_px=1.0,
            min_angle_degrees=4.0,
        )
        if solved is None:
            continue
        xyz, errors, angle = solved
        accepted.append(
            {
                "point": xyz.tolist(),
                "view_ids": views,
                "pixels": np.asarray(pixels).tolist(),
                "reprojection_max_px": float(max(errors)),
                "parallax_degrees": angle,
                "source_landmark_index": landmark_id,
            }
        )
    kept = deduplicate_rgb_tracks(accepted)
    return (
        kept,
        {
            "raw_tracks": len(source_tracks),
            "quality_accepted": len(accepted),
            "deduplicated_tracks": len(kept),
            "source_manifest": str(manifest_path.resolve()),
            "source_landmarks_sha256": digest(landmark_path),
            "source_tracks_sha256": digest(anchor_dir / "tracks.json"),
            "source_mapping_ids": input_ids,
        },
        input_hashes,
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cv2.setNumThreads(2)
    with np.load(args.reconstruction, allow_pickle=False) as source:
        keys = [
            "view_ids",
            "rgb_uint8",
            "valid_mask",
            "depth_z_pose_scale_m",
            "camera_to_world_input",
            "intrinsics_input_processed",
        ]
        arrays = {key: source[key] for key in keys}
        arrays["source_batch_index"] = (
            source["source_batch_index"]
            if "source_batch_index" in source
            else np.zeros(len(arrays["view_ids"]), dtype=np.int32)
        )
    ids = [int(s.split("-")[-1]) for s in arrays["view_ids"].tolist()]
    if set(ids) & FROZEN_QUERY_IDS:
        raise ValueError("Frozen query contamination in learned reconstruction")
    report = {
        "scope": "RGB triangulation depth calibration; no scene/floor/object/held-out-query truth",
        "source_npz": str(args.reconstruction.resolve()),
        "source_npz_sha256": digest(args.reconstruction),
        "query_ids_excluded": sorted(FROZEN_QUERY_IDS),
        "preprocessing_audit": audit_preprocessing(args.scan_dir, arrays),
        "gates": {
            "track_min_views": 3,
            "epipolar_max_px": 1,
            "reprojection_max_px": 1,
            "parallax_min_degrees": 4,
            "spatial_dedup_m": 0.02,
            "validation_hash_modulus": 5,
        },
    }
    features, scan_ids, hashes = [], [], {}
    for index in [] if args.anchor_dir else range(23):
        if index in FROZEN_QUERY_IDS:
            continue
        stem = f"scan-{index:03}"
        meta_path = args.scan_dir / f"{stem}.json"
        rgb_path = args.scan_dir / f"{stem}.png"
        metadata = json.loads(meta_path.read_text())
        image = np.array(Image.open(rgb_path).convert("RGB"))
        features.append(
            extract(
                image,
                metadata["intrinsics"],
                metadata["world_to_camera"],
                nfeatures=6000,
            )
        )
        scan_ids.append(index)
        hashes[stem] = {"rgb": digest(rgb_path), "calibration": digest(meta_path)}
        print(f"RGB features {stem}: {len(features[-1]['pixels'])}", flush=True)
    if args.anchor_dir:
        tracks, stats, hashes = load_rgb_anchor_map(args.anchor_dir, args.scan_dir)
    else:
        tracks, stats = triangulated_rgb_anchors(features, scan_ids)
    report["input_hashes"] = hashes
    report["track_statistics"] = stats
    rows = []
    for landmark_id, track in enumerate(tracks):
        for view_id in track["view_ids"]:
            if view_id not in ids:
                continue
            i = ids.index(view_id)
            pixel, target = project(
                np.array(track["point"]),
                arrays["intrinsics_input_processed"][i],
                np.linalg.inv(arrays["camera_to_world_input"][i]),
            )
            predicted = sample_bilinear(
                arrays["depth_z_pose_scale_m"][i], arrays["valid_mask"][i], pixel
            )[0]
            if np.isfinite(predicted) and predicted > 0 and target[0] > 0:
                rows.append(
                    {
                        "landmark_id": landmark_id,
                        "view_index": i,
                        "view_id": view_id,
                        "predicted_m": float(predicted),
                        "triangulated_m": float(target[0]),
                        "validation": track["validation"],
                        "pixel": pixel[0].tolist(),
                    }
                )
    if len(rows) < 12:
        raise RuntimeError("Insufficient RGB anchor observations")
    x = np.array([r["predicted_m"] for r in rows])
    y = np.array([r["triangulated_m"] for r in rows])
    validation = np.array([r["validation"] for r in rows])
    row_views = np.array([r["view_index"] for r in rows])
    global_transform, selection = choose_depth_transform(x, y, validation)
    report["global_selection"] = selection
    report["anchors"] = rows
    report["tracks"] = tracks
    report["anchor_count"] = len(rows)
    report["unique_anchor_tracks"] = len({r["landmark_id"] for r in rows})
    report["calibration_choice_uses_only_rgb_validation"] = True
    report["validation_role"] = (
        "Landmarks held out of fitting but reused for model selection; this is validation, not an unbiased final test. Repeated observations of a landmark are correlated."
    )
    report["validation_landmark_groups"] = len(
        {r["landmark_id"] for r in rows if r["validation"]}
    )
    report["training_landmark_groups"] = len(
        {r["landmark_id"] for r in rows if not r["validation"]}
    )
    batch_transforms = {}
    report["batch_selection"] = {}
    for batch in np.unique(arrays["source_batch_index"]):
        selected = arrays["source_batch_index"][row_views] == batch
        candidate, batch_report = choose_depth_transform(
            x[selected], y[selected], validation[selected]
        )
        chosen_batch = global_transform
        if batch_report["status"] == "selected_on_rgb_anchor_validation_only":
            held_out = selected & validation
            base_error = error_summary(
                apply_depth_transform(x[held_out], global_transform), y[held_out]
            )
            candidate_error = error_summary(
                apply_depth_transform(x[held_out], candidate), y[held_out]
            )
            if (
                candidate_error["median_m"] < base_error["median_m"]
                and candidate_error["p95_m"] <= base_error["p95_m"]
            ):
                chosen_batch = candidate
        batch_transforms[int(batch)] = chosen_batch
        report["batch_selection"][str(batch)] = {
            "chosen": chosen_batch,
            "selection": batch_report,
        }
    corrected = arrays["depth_z_pose_scale_m"].copy()
    per_view = []
    for i, view_id in enumerate(ids):
        selected = row_views == i
        local, local_report = choose_depth_transform(
            x[selected], y[selected], validation[selected]
        )
        chosen = batch_transforms[int(arrays["source_batch_index"][i])]
        reason = "source-batch/global RGB-anchor validation selection"
        local_val = selected & validation
        if local_report["status"] == "selected_on_rgb_anchor_validation_only":
            global_errors = error_summary(
                apply_depth_transform(x[local_val], chosen), y[local_val]
            )
            local_errors = error_summary(
                apply_depth_transform(x[local_val], local), y[local_val]
            )
            if (
                local_errors["median_m"] < global_errors["median_m"]
                and local_errors["p95_m"] <= global_errors["p95_m"]
            ):
                chosen = local
                reason = "per-view model improves both median and p95 on RGB validation anchors"
        proposed = apply_depth_transform(corrected[i], chosen)
        mask = arrays["valid_mask"][i]
        if not np.isfinite(proposed[mask]).all() or np.any(proposed[mask] <= 0):
            chosen = {"kind": "identity", "scale": 1.0, "offset": 0.0}
            reason = "reject nonpositive/nonfinite dense extrapolation"
            proposed = corrected[i]
        corrected[i] = proposed
        per_view.append(
            {
                "view_id": view_id,
                "transform": chosen,
                "reason": reason,
                "local_selection": local_report,
            }
        )
    arrays["depth_z_rgb_calibrated_m"] = corrected
    arrays["points_world_rgb_calibrated"] = np.stack(
        [
            calibrated_world_points(d, k, p)
            for d, k, p in zip(
                corrected,
                arrays["intrinsics_input_processed"],
                arrays["camera_to_world_input"],
                strict=True,
            )
        ]
    )
    report["per_view"] = per_view
    report["before_held_out_rgb_anchor_error"] = error_summary(
        x[validation], y[validation]
    )
    calibrated_anchor = np.array(
        [
            float(
                apply_depth_transform(
                    np.array(r["predicted_m"]), per_view[r["view_index"]]["transform"]
                )
            )
            for r in rows
        ]
    )
    report["after_held_out_rgb_anchor_error"] = error_summary(
        calibrated_anchor[validation], y[validation]
    )
    report["mask_unchanged"] = True
    report["runtime_seconds"] = time.perf_counter() - start
    report["geometry_scored_with_truth"] = False
    report["free_space_certified"] = False
    np.savez_compressed(args.output / "reconstruction.npz", **arrays)
    report["point_count"] = write_ply(
        args.output / "points-world.ply",
        arrays["points_world_rgb_calibrated"],
        arrays["rgb_uint8"],
        arrays["valid_mask"],
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    manifest = {
        "status": "complete",
        "mapping_input_view_ids": sorted(hashes),
        "source_view_ids": arrays["view_ids"].tolist(),
        "excluded_query_view_ids": [f"scan-{i:03}" for i in sorted(FROZEN_QUERY_IDS)],
        "source_reconstruction_npz": str(args.reconstruction.resolve()),
        "source_reconstruction_npz_sha256": report["source_npz_sha256"],
        "source_anchor_provenance": stats,
        "artifact_npz_sha256": digest(args.output / "reconstruction.npz"),
        "report_sha256": digest(args.output / "report.json"),
        "geometry_truth_used_for_calibration": False,
        "query_data_used_for_calibration": False,
        "implementation_sha256": digest(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "anchor_count",
                    "unique_anchor_tracks",
                    "before_held_out_rgb_anchor_error",
                    "after_held_out_rgb_anchor_error",
                    "runtime_seconds",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path)
    main(parser.parse_args())
