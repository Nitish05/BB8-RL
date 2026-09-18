"""Frozen SuperPoint+LightGlue metric-map registration; query poses never consumed."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.mapping.registration import (
    build_landmarks,
    collect_query_correspondences,
    solve_registration,
)

QUERIES = (14, 17, 20, 23)
LIGHTGLUE_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
WEIGHTS = {
    "superpoint_v1.pth": "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
    "superpoint_lightglue_v0-1_arxiv.pth": "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh registration output")
    vendor = args.runtime_root / "vendor/LightGlue"
    commit = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != LIGHTGLUE_COMMIT:
        raise ValueError("LightGlue source differs from the pinned revision")
    for name, expected in WEIGHTS.items():
        if digest(args.runtime_root / "weights/hub/checkpoints" / name) != expected:
            raise ValueError("Official pretrained weight content changed")
    os.environ["TORCH_HOME"] = str((args.runtime_root / "weights").resolve())
    sys.path.insert(0, str(vendor.resolve()))
    import torch
    from lightglue import LightGlue, SuperPoint

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    extractor = SuperPoint(max_num_keypoints=2048).eval()
    matcher = LightGlue(
        features="superpoint", depth_confidence=-1, width_confidence=-1, flash=False
    ).eval()
    args.output.mkdir(parents=True)
    cache = args.output / "features"
    cache.mkdir()
    map_indices = [i for i in range(24) if i not in QUERIES]
    map_ids = [f"scan-{i:03}" for i in map_indices]
    manifest = {
        "status": "started",
        "mapping_input_view_ids": map_ids,
        "query_indices": list(QUERIES),
        "query_pose_consumed": False,
        "lightglue_commit": commit,
        "weight_sha256": WEIGHTS,
        "official_sources": [
            "https://github.com/cvg/LightGlue",
            "https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html",
        ],
        "licenses": {
            "LightGlue": "Apache-2.0",
            "SuperPoint": "Magic Leap restrictive pretrained research license; isolated experiment only",
        },
        "parameters": {
            "max_keypoints": 2048,
            "resize": None,
            "depth_confidence": -1,
            "width_confidence": -1,
            "filter_threshold": 0.1,
            "map_pair_circular_distance_max": 3,
            "epipolar_px": 1.5,
            "triangulation_max_reprojection_px": 2.0,
            "triangulation_min_views": 3,
            "triangulation_min_parallax_degrees": 2,
            "query_landmark_min_support_views": 2,
            "pnp_min_inliers": 12,
            "pnp_reprojection_px": 2.0,
        },
        "mapping_inputs": {},
        "query_inputs": {},
        "native_resolution_features": True,
        "python": sys.version,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
    }
    write(args.output / "manifest.json", manifest)
    views, tensor_features, query_intrinsics = {}, {}, {}
    for index in range(24):
        name = f"scan-{index:03}"
        rgb_path, metadata_path = (
            args.scan_dir / f"{name}.png",
            args.scan_dir / f"{name}.json",
        )
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        metadata = json.loads(metadata_path.read_text())
        # Query camera poses are deliberately never accessed. Their JSON hash
        # authenticates K; only map indices contribute known metric extrinsics.
        intrinsics = np.asarray(metadata["intrinsics"], dtype=float)
        with torch.inference_mode():
            features = extractor.extract(
                torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255,
                resize=None,
            )
        tensor_features[name] = features
        arrays = {key: value[0].cpu().numpy() for key, value in features.items()}
        np.savez_compressed(cache / f"{name}.npz", **arrays)
        if np.any(arrays["keypoints"] < 0) or np.any(
            arrays["keypoints"] >= [rgb.shape[1], rgb.shape[0]]
        ):
            raise ValueError("Feature coordinate outside native image")
        record = {
            "rgb_sha256": digest(rgb_path),
            "metadata_sha256": digest(metadata_path),
            "features": len(arrays["keypoints"]),
            "image_size": [rgb.shape[1], rgb.shape[0]],
            "feature_cache_sha256": digest(cache / f"{name}.npz"),
        }
        if index in QUERIES:
            query_intrinsics[name] = intrinsics
            manifest["query_inputs"][name] = record
        else:
            views[name] = {
                **arrays,
                "intrinsics": intrinsics,
                "world_to_camera": np.asarray(metadata["world_to_camera"], dtype=float),
            }
            manifest["mapping_inputs"][name] = record
        print(
            json.dumps({"extracted": name, "features": record["features"]}), flush=True
        )
    write(args.output / "manifest.json", manifest)

    def match(first, second):
        with torch.inference_mode():
            result = matcher(
                {"image0": tensor_features[first], "image1": tensor_features[second]}
            )
        return result["matches"][0].cpu().numpy(), result["scores"][0].cpu().numpy()

    pairs = []
    pair_cache = args.output / "pairs"
    pair_cache.mkdir()
    for first, second in combinations(map_indices, 2):
        if min(second - first, 24 - second + first) > 3:
            continue
        a, b = f"scan-{first:03}", f"scan-{second:03}"
        matches, scores = match(a, b)
        pairs.append({"first": a, "second": b, "matches": matches, "scores": scores})
        np.savez_compressed(
            pair_cache / f"{a}--{b}.npz", matches=matches, scores=scores
        )
        print(json.dumps({"map_pair": [a, b], "matches": len(matches)}), flush=True)
    cloud = build_landmarks(views, pairs)
    np.savez_compressed(args.output / "landmarks.npz", points=cloud["points"])
    write(
        args.output / "tracks.json",
        {"tracks": cloud["tracks"], "pairs": cloud["pair_statistics"]},
    )
    # Freeze the metric map before query matching; query images cannot add points.
    manifest["landmarks_sha256"] = digest(args.output / "landmarks.npz")
    manifest["map_frozen_before_query_matching"] = True
    manifest["landmarks"] = len(cloud["points"])
    write(args.output / "manifest.json", manifest)
    queries = []
    for index in QUERIES:
        name = f"scan-{index:03}"
        query_pairs = []
        for map_view in map_ids:
            matches, scores = match(name, map_view)
            query_pairs.append(
                {"map_view": map_view, "matches": matches, "scores": scores}
            )
            np.savez_compressed(
                pair_cache / f"{name}--{map_view}.npz", matches=matches, scores=scores
            )
        correspondences = collect_query_correspondences(query_pairs, cloud["lookup"])
        keypoints = tensor_features[name]["keypoints"][0].cpu().numpy()
        registration = solve_registration(
            keypoints, query_intrinsics[name], cloud["points"], correspondences
        )
        queries.append(
            {
                "query_index": index,
                **registration,
                "correspondences": correspondences,
                "pair_match_counts": {
                    pair["map_view"]: len(pair["matches"]) for pair in query_pairs
                },
            }
        )
        print(
            json.dumps(
                {
                    "query": index,
                    "status": registration["status"],
                    "matches": registration.get("matches"),
                    "inliers": registration.get("inliers"),
                }
            ),
            flush=True,
        )
    manifest.update(
        status="complete",
        source_sha256=digest(Path(__file__)),
        registration_source_sha256=digest(
            Path(__file__).resolve().parents[1] / "src/bb8_rl/mapping/registration.py"
        ),
    )
    write(args.output / "manifest.json", manifest)
    report = {
        **manifest,
        "scope": "RGB-only learned matching, known-pose metric triangulation, held-out RGB+K PnP candidates; independent scoring required",
        "candidate_tracks": cloud["candidate_tracks"],
        "pair_statistics": cloud["pair_statistics"],
        "queries": queries,
    }
    write(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root", type=Path, default=Path("work/m77/registration")
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(
            "work/scan/view-probe"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/m77/registration/superpoint-lightglue"),
    )
    main(parser.parse_args())
