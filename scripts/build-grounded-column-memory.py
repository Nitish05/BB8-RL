"""Frozen opt-in floor-column experiment; no truth or query-camera inputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bb8_rl.mapping.contracts import SpaceState
from bb8_rl.mapping.room_memory import RoomMemory
from bb8_rl.mapping.scan_free_space import (
    EXCLUDED_QUERY_IDS,
    ScanFreeMemory,
    ScanRGBView,
    build_scan_free_memory,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    cv2.setNumThreads(1)
    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    source = ScanFreeMemory.load(args.input_memory)
    metadata = source.metadata
    expected = [f"scan-{i:03}" for i in range(24) if i not in EXCLUDED_QUERY_IDS]
    if (
        list(source.view_ids) != expected
        or metadata["mapping_input_view_ids"] != expected
        or not source.memory.valid
    ):
        raise ValueError("This frozen experiment requires the 20 original map views")
    mask_path = args.input_memory / "floor-masks.npz"
    if digest(mask_path) != metadata["artifact_sha256"]["floor-masks.npz"]:
        raise ValueError("Retained floor-mask hash mismatch")
    with np.load(mask_path, allow_pickle=False) as data:
        masks = data["masks"]
        if data["view_ids"].tolist() != expected or masks.dtype != bool:
            raise ValueError("Retained masks have incompatible view IDs or dtype")
    views = []
    for index, name in enumerate(expected):
        rgb_path, calibration_path = (
            args.scan_dir / f"{name}.png",
            args.scan_dir / f"{name}.json",
        )
        declared = metadata["input_sha256"][name]
        if (
            digest(rgb_path) != declared["rgb"]
            or digest(calibration_path) != declared["calibration"]
        ):
            raise ValueError(f"Original RGB/calibration hash mismatch: {name}")
        calibration = json.loads(calibration_path.read_text())
        if calibration["calibration"] != "synthetic_exact":
            raise ValueError("Frozen prototype requires declared synthetic calibration")
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        if masks[index].shape != rgb.shape[:2]:
            raise ValueError("Retained masks must use original native RGB resolution")
        views.append(
            ScanRGBView(
                name,
                rgb,
                np.asarray(calibration["intrinsics"]),
                np.asarray(calibration["world_to_camera"]),
            )
        )
    # Re-integrate all original occupied evidence, including object records;
    # existing FREE evidence is not used to rescue a failed footprint test.
    original = source.memory
    occupied = RoomMemory(
        original.bounds,
        original.resolution_m,
        scene_version=original.scene_version,
        calibration_version=original.calibration_version,
        world_frame=original.world_frame,
    )
    for evidence in original.evidence:
        if evidence.state is SpaceState.OCCUPIED:
            occupied.observe_volume(evidence.bounds, evidence.state, evidence.source)
    for obj in original.objects:
        occupied.remember_object(obj)
    result = build_scan_free_memory(
        views,
        extent=metadata["extent_m"],
        resolution=metadata["resolution_m"],
        body_height_m=metadata["requested_body_height_m"],
        floor_z=metadata["floor_z_m"],
        projection_margin_m=metadata["projection_margin_m"],
        pixel_guard=metadata["pixel_guard"],
        minimum_views=metadata["minimum_views"],
        minimum_baseline_m=metadata["minimum_baseline_m"],
        minimum_angle_degrees=metadata["minimum_angle_degrees"],
        occupied_memory=occupied,
        floor_masks=masks,
        projection_shape=metadata["projection_shape"],
        evidence_mode="grounded_column",
        assume_floor_connected_columns=True,
    )
    if not np.array_equal(source.occupied_mask, result.occupied_mask):
        raise ValueError("Original occupied conflicts changed")
    result.metadata.update(
        {
            key: metadata[key]
            for key in (
                "input_sha256",
                "floor_evidence",
                "homography_evidence",
                "learned_mask_intersection",
                "occupied_refinement",
                "occupied_source",
                "occupied_source_sha256",
                "occupied_source_semantics",
            )
            if key in metadata
        }
    )
    result.metadata.update(
        experiment="m712-grounded-column-v1",
        promotion_status="candidate_requires_independent_whole_map_audit",
        source_memory_manifest_sha256=digest(args.input_memory / "manifest.json"),
        source_free_grid_sha256=digest(args.input_memory / "free-grid.npz"),
        source_floor_masks_sha256=digest(mask_path),
        scan_dir=str(args.scan_dir.resolve()),
        query_inputs_used=False,
        truth_inputs_used=False,
        thresholds_selected_using_truth=False,
        floor_masks_recomputed=False,
        free_added_cells=int((result.free_mask & ~source.free_mask).sum()),
        free_removed_cells=int((source.free_mask & ~result.free_mask).sum()),
        producer_source_sha256=digest(__file__),
        producer_module_sha256=digest(build_scan_free_memory.__code__.co_filename),
    )
    result.save(args.output)
    shutil.copyfile(mask_path, args.output / "floor-masks.npz")
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["floor-masks.npz"] = digest(
        args.output / "floor-masks.npz"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "free_evidence_mode",
                    "free_cells",
                    "free_added_cells",
                    "free_removed_cells",
                    "occupied_cells",
                    "unknown_cells",
                    "promotion_status",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-memory", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
