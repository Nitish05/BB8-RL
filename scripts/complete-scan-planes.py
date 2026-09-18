#!/usr/bin/env python3
"""Freeze an RGB/anchor-only planar completion candidate for separate scoring."""

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bb8_rl.mapping.photometric_refinement import backproject
from bb8_rl.mapping.planar_completion import complete_depth


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "Output must be fresh so frozen experiments cannot be overwritten"
        )
    parent = json.loads(args.manifest.read_text())
    mapping_ids = parent["mapping_input_view_ids"]
    heldout = {"scan-014", "scan-017", "scan-020", "scan-023"}
    if not mapping_ids or heldout.intersection(mapping_ids):
        raise ValueError("Missing map ancestry or held-out query leakage")
    source = np.load(args.input, allow_pickle=False)
    required = {
        "rgb_uint8",
        "depth_z_refined_m",
        "valid_mask",
        "points_world_refined",
        "intrinsics_input_processed",
        "camera_to_world_input",
        "view_ids",
    }
    if not required.issubset(source.files):
        raise ValueError(
            f"Missing input fields: {sorted(required - set(source.files))}"
        )
    if not set(source["view_ids"]).issubset(mapping_ids):
        raise ValueError("Output view IDs are absent from declared map ancestry")
    args.output.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        repository / "src/bb8_rl/mapping/planar_completion.py",
        repository / "src/bb8_rl/mapping/photometric_refinement.py",
        repository / "tests/test_planar_completion.py",
    ]
    source_output = args.output / "source"
    source_output.mkdir()
    for path in source_paths:
        shutil.copy2(path, source_output / path.name)
    manifest = {
        "status": "running",
        "mapping_input_view_ids": mapping_ids,
        "held_out_view_ids": sorted(heldout),
        "input": str(args.input.resolve()),
        "input_sha256": digest(args.input),
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": digest(args.manifest),
        "parent_manifest": parent,
        "source_sha256": {path.name: digest(path) for path in source_paths},
        "geometry_inputs": "RGB connected regions and trusted photometric depth anchors; known map-view K and poses only for backprojection",
        "plane_fit": "inverse optical-Z plane, robust fit on deterministic 8px spatial tiles; other tiles reserved for validation, no final refit",
        "completion_support": "same 4-connected quantized RGB region and exact fit-anchor convex hull pixel centers",
        "scope": "planar-assumption surface candidates, not observed geometry or navigation acceptance",
        "oracle_geometry_fit": False,
        "query_inputs_used": False,
        "planar_assumption": True,
        "free_space_certified": False,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    all_depth, all_points, all_completed, reports = [], [], [], []
    for index, view_id in enumerate(source["view_ids"]):
        anchor_mask = source["valid_mask"][index]
        depth, completed, report = complete_depth(
            source["rgb_uint8"][index], source["depth_z_refined_m"][index], anchor_mask
        )
        points = backproject(
            depth,
            source["intrinsics_input_processed"][index],
            source["camera_to_world_input"][index],
        )
        points[anchor_mask] = source["points_world_refined"][index][anchor_mask]
        report["view_id"] = str(view_id)
        report["region_status_counts"] = dict(
            Counter(r["status"] for r in report["regions"])
        )
        reports.append(report)
        all_depth.append(depth)
        all_points.append(points)
        all_completed.append(completed)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "view_id",
                        "anchor_pixels",
                        "completed_pixels",
                        "region_status_counts",
                    )
                }
            ),
            flush=True,
        )
    completed = np.stack(all_completed)
    arrays = {key: source[key] for key in source.files}
    arrays.update(
        anchor_mask=source["valid_mask"],
        completed_mask=completed,
        valid_mask=source["valid_mask"] | completed,
        depth_z_completed_m=np.stack(all_depth),
        points_world_completed=np.stack(all_points),
    )
    destination = args.output / "reconstruction.npz"
    np.savez_compressed(destination, **arrays)
    report = {
        "views": reports,
        "anchor_points": int(arrays["anchor_mask"].sum()),
        "completed_points": int(completed.sum()),
        "combined_points": int(arrays["valid_mask"].sum()),
        "elapsed_seconds": time.monotonic() - started,
        "planar_assumption": True,
        "free_space_certified": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest.update(
        status="complete",
        output_sha256=digest(destination),
        summary=report
        | {
            "views": [
                {key: value for key, value in item.items() if key != "regions"}
                for item in reports
            ]
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
