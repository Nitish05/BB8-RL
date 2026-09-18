"""Frozen-split multiview RGB depth refinement; no scorer truth is imported."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.mapping.photometric_refinement import (
    backproject,
    ray_grid,
    refine,
    resample_prior,
    resized_intrinsics,
)

QUERY_IDS = {14, 17, 20, 23}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    arrays = np.load(args.prior)
    ids = arrays["view_ids"].tolist()
    ancestry = set(ids)
    if args.prior_manifest is not None:
        prior_manifest = json.loads(args.prior_manifest.read_text())
        declared = prior_manifest.get("mapping_input_view_ids")
        if (
            not isinstance(declared, list)
            or not declared
            or not ancestry.issubset(declared)
        ):
            raise ValueError("Prior manifest must contain complete mapping ancestry")
        ancestry.update(declared)
    elif args.depth_field != "depth_z_pose_scale_m":
        raise ValueError("A calibrated/refined prior requires --prior-manifest")
    if any(int(v.removeprefix("scan-")) in QUERY_IDS for v in ancestry):
        raise ValueError("Prior includes a held-out query")
    all_views = {}
    hashes = {}
    for index in range(24):
        if index in QUERY_IDS:
            continue
        stem = args.scan_dir / f"scan-{index:03}"
        metadata = json.loads(stem.with_suffix(".json").read_text())
        rgb = cv2.cvtColor(cv2.imread(str(stem.with_suffix(".png"))), cv2.COLOR_BGR2RGB)
        scale = args.width / rgb.shape[1]
        resized = cv2.resize(
            rgb, (args.width, round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )
        k = resized_intrinsics(
            metadata["intrinsics"], rgb.shape[1::-1], resized.shape[1::-1]
        )
        all_views[index] = {
            "rgb": resized,
            "intrinsics": k,
            "world_to_camera": np.asarray(metadata["world_to_camera"]),
        }
        for ext in (".png", ".json"):
            p = stem.with_suffix(ext)
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    output = {
        k: []
        for k in [
            "points_world_refined",
            "depth_z_refined_m",
            "valid_mask",
            "prior_valid_mask",
            "rgb_uint8",
            "view_ids",
            "intrinsics_input_processed",
            "camera_to_world_input",
            "best_cost",
            "second_cost",
        ]
    }
    records = []
    indices = (
        args.indices
        if args.indices is not None
        else [int(v.removeprefix("scan-")) for v in ids]
    )
    for index in indices:
        if index in QUERY_IDS:
            raise ValueError("Query cannot become reference")
        slot = ids.index(f"scan-{index:03}")
        view = all_views[index]
        rays = ray_grid(view["rgb"].shape[:2], view["intrinsics"])
        uv = rays @ arrays["intrinsics_input_processed"][slot].T
        u, v = (uv[..., :2] / uv[..., 2:]).transpose(2, 0, 1).astype(np.float32)
        prior, prior_valid = resample_prior(
            arrays[args.depth_field][slot], arrays["valid_mask"][slot], u, v
        )
        prior_points = backproject(
            prior, view["intrinsics"], np.linalg.inv(view["world_to_camera"])
        )
        prior_valid &= (
            (abs(prior_points[..., :2]).max(-1) < 2.3)
            & (prior_points[..., 2] > -0.3)
            & (prior_points[..., 2] < 1.5)
        )
        neighbors = sorted(
            (i for i in all_views if i != index),
            key=lambda i: min(abs(i - index), 24 - abs(i - index)),
        )[:4]
        tick = time.perf_counter()
        result = refine(
            view["rgb"],
            view["intrinsics"],
            view["world_to_camera"],
            prior,
            prior_valid,
            [all_views[i] for i in neighbors],
        )
        fields = {
            "points_world_refined": backproject(
                result["depth"],
                view["intrinsics"],
                np.linalg.inv(view["world_to_camera"]),
            ),
            "depth_z_refined_m": result["depth"],
            "valid_mask": result["valid"],
            "prior_valid_mask": prior_valid,
            "rgb_uint8": view["rgb"],
            "view_ids": f"scan-{index:03}",
            "intrinsics_input_processed": view["intrinsics"],
            "camera_to_world_input": np.linalg.inv(view["world_to_camera"]),
            "best_cost": result["best_cost"],
            "second_cost": result["second_cost"],
        }
        for key, value in fields.items():
            output[key].append(value)
        row = {
            "view_id": f"scan-{index:03}",
            "neighbor_indices": neighbors,
            "prior_valid": int(prior_valid.sum()),
            "retained": int(result["valid"].sum()),
            "seconds": time.perf_counter() - tick,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
    np.savez_compressed(
        args.output / "reconstruction.npz",
        **{k: np.stack(v) for k, v in output.items()},
    )
    manifest = {
        "status": "complete",
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]
                / "src/bb8_rl/mapping/photometric_refinement.py",
            ]
        },
        "scope": "RGB photometric refinement candidates, not free-space or control acceptance",
        "mapping_input_view_ids": [f"scan-{i:03}" for i in sorted(all_views)],
        "held_out_view_ids": [f"scan-{i:03}" for i in sorted(QUERY_IDS)],
        "input_sha256": hashes,
        "prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        "prior_depth_field": args.depth_field,
        "prior_manifest_sha256": hashlib.sha256(
            args.prior_manifest.read_bytes()
        ).hexdigest()
        if args.prior_manifest
        else None,
        "prior_view_ids": ids,
        "source_view_ids": sorted(ancestry),
        "records": records,
        "parameters": {
            "half_range_m": 0.45,
            "step_m": 0.015,
            "maximum_cost": 0.08,
            "minimum_gap": 0.02,
            "patch_radius": 2,
            "best_neighbors": 2,
            "resize_geometry": "OpenCV pixel centers: u_new = (u_old + 0.5) * sx - 0.5, likewise v",
        },
        "geometry_inputs": [
            "RGB",
            "known synthetic scan K and poses",
            "MapAnything depth with camera-prior scale",
        ],
        "oracle_geometry_fit": False,
        "free_space_certified": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("work/m76/mapanything/run-8-pose-aligned/reconstruction.npz"),
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(
            "work/scan/view-probe"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--depth-field", default="depth_z_pose_scale_m")
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--indices", nargs="+", type=int)
    main(parser.parse_args())
