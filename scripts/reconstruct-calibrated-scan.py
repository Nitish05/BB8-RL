"""Actual RGB scan triangulation; last scan view is a held-out fixed-camera query."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.mapping import Bounds3D, EvidenceSource, Provenance, RoomMemory, SpaceState
from bb8_rl.mapping.scan_geometry import extract, reconstruct, register_camera, save_ply


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh reconstruction output")
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    features, inputs = [], {}
    # Query view 23 never contributes map points, matching or track validation.
    for index in range(23):
        stem = args.scan_dir / f"scan-{index:03}"
        metadata = json.loads(stem.with_suffix(".json").read_text())
        rgb = cv2.cvtColor(cv2.imread(str(stem.with_suffix(".png"))), cv2.COLOR_BGR2RGB)
        features.append(
            extract(rgb, metadata["intrinsics"], metadata["world_to_camera"])
        )
        for suffix in (".png", ".json"):
            path = stem.with_suffix(suffix)
            inputs[path.name] = digest(path)
        print(
            json.dumps({"view": index, "features": len(features[-1]["pixels"])}),
            flush=True,
        )
    cloud = reconstruct(features)
    np.savez_compressed(
        args.output / "landmarks.npz",
        **{
            k: cloud[k]
            for k in ("points", "colors", "descriptors", "descriptor_landmarks")
        },
    )
    save_ply(args.output / "reconstruction.ply", cloud["points"], cloud["colors"])
    query_rgb_path = args.scan_dir / "scan-023.png"
    query_meta_path = args.scan_dir / "scan-023.json"
    query_rgb = cv2.cvtColor(cv2.imread(str(query_rgb_path)), cv2.COLOR_BGR2RGB)
    query_meta = json.loads(query_meta_path.read_text())
    for path in (query_rgb_path, query_meta_path):
        inputs[path.name] = digest(path)
    registration = register_camera(query_rgb, np.array(query_meta["intrinsics"]), cloud)
    # This truth is read only after the RGB+K+landmark registration returned.
    if registration["status"] == "candidate":
        estimate = np.array(registration["world_to_camera"])
        truth = np.array(query_meta["world_to_camera"])
        c_est, c_true = np.linalg.inv(estimate)[:3, 3], np.linalg.inv(truth)[:3, 3]
        angle = np.degrees(
            np.arccos(
                np.clip((np.trace(estimate[:3, :3] @ truth[:3, :3].T) - 1) / 2, -1, 1)
            )
        )
        registration["scoring_only"] = {
            "camera_center_error_m": float(np.linalg.norm(c_est - c_true)),
            "rotation_error_degrees": float(angle),
        }
    memory = RoomMemory(
        Bounds3D((-2, -2, 0), (2, 2, 1.5)),
        0.05,
        scene_version="scan-m76-classical",
        calibration_version="known-metric-synthetic-scan",
    )
    occupied = 0
    for index, (point, track) in enumerate(zip(cloud["points"], cloud["tracks"])):
        # Floor support points remain in the point cloud. Sparse elevated surface
        # samples can block space, but can never certify the volume between rays.
        if np.max(abs(point[:2])) >= 2 or not 0.04 <= point[2] < 1.45:
            continue
        source = EvidenceSource(
            f"landmark-{index}",
            tuple(f"scan-{v:03}" for v in track["views"]),
            0.0,
            Provenance.RGB_RECONSTRUCTION,
            0.025,
        )
        memory.observe_volume(
            Bounds3D(tuple(point - 0.001), tuple(point + 0.001)),
            SpaceState.OCCUPIED,
            source,
        )
        occupied += 1
    memory.save(args.output / "room-memory.json")
    report = {
        "status": "complete",
        "scope": "calibrated RGB sparse reconstruction and held-out final-camera registration, not unknown-pose SLAM or navigability acceptance",
        "mapping_views": list(range(23)),
        "query_view": 23,
        "query_excluded_from_map": True,
        "geometry_inputs": [
            "RGB",
            "known synthetic intrinsics",
            "known metric synthetic scan camera poses",
        ],
        "depth_or_segmentation_inputs": False,
        "input_sha256": inputs,
        "opencv_version": cv2.__version__,
        "algorithm": "SIFT mutual ratio/epipolar matches; at least three views per landmark; DLT with reprojection/parallax rejection; PnP final camera",
        "parameters": {
            "ratio": 0.75,
            "epipolar_px": 1.5,
            "reprojection_max_px": 2.0,
            "min_parallax_degrees": 2.0,
            "min_views": 3,
            "pair_neighbor_window": 3,
        },
        "candidate_tracks": cloud["candidate_tracks"],
        "landmarks": len(cloud["points"]),
        "elevated_memory_surface_samples": occupied,
        "free_space_certified": False,
        "surface_semantics": "Unclassified scan-time surfaces; may include parked robot. Not certified persistent room obstacles.",
        "memory_geometry_uncertainty_m": 0.025,
        "uncertainty_status": "assumed import margin, not calibrated coverage",
        "registration": registration,
        "pairs": cloud["pairs"],
        "tracks": cloud["tracks"],
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": digest(Path(__file__)),
        "geometry_source_sha256": digest(
            Path(__file__).resolve().parents[1] / "src/bb8_rl/mapping/scan_geometry.py"
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("tracks", "pairs", "input_sha256")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
