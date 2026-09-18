"""Frozen48-view revision with unchanged four-view full-prism addition gates."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bb8_rl.mapping.room_memory import RoomMemory
from bb8_rl.mapping.scan_free_space import (
    ScanFreeMemory,
    ScanRGBView,
    build_scan_free_memory,
    floor_homography_masks,
)
from bb8_rl.mapping.scan_revision import retain_accepted_free

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/build-extended-scan-memory.py"
SPEC = importlib.util.spec_from_file_location("extended_builder", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
digest = HELPER.digest
PREVIOUS_IDS = (*HELPER.ORIGINAL_IDS, *HELPER.SUPPLEMENTAL_IDS)
NEW_IDS = tuple(f"scan-{i:03}" for i in range(36, 52))
ALL_IDS = (*PREVIOUS_IDS, *NEW_IDS)
SETTINGS = {
    "mode": "full_prism",
    "body_height_m": 0.12,
    "metric_projection_margin_m": 0.05,
    "minimum_pairwise_separated_views": 4,
    "minimum_baseline_m": 0.5,
    "minimum_angle_degrees": 15.0,
    "pixel_guard": 2,
    "projection_shape": "convex_hull",
}


def validate_plan(args, baseline, previous):
    plan = json.loads(args.revision_plan.read_text())
    capture_path = args.scan_dir / "capture-manifest.json"
    capture = json.loads(capture_path.read_text())
    pinned = {
        "capture_plan_sha256": digest(args.capture_plan),
        "capture_manifest_sha256": digest(capture_path),
        "retained_manifest_sha256": digest(args.retained_memory / "manifest.json"),
        "retained_floor_masks_sha256": digest(args.retained_memory / "floor-masks.npz"),
        "baseline_manifest_sha256": digest(args.input_memory / "manifest.json"),
        "checkpoint_sha256": digest(args.checkpoint),
    }
    if (
        plan.get("revision") != "full-prism-extended-v3"
        or plan.get("new_evidence") != SETTINGS
        or plan.get("truth_geometry_or_failed_cell_locations_used") is not False
        or tuple(plan.get("new_view_ids", [])) != NEW_IDS
        or tuple(plan.get("retained_view_ids", [])) != PREVIOUS_IDS
        or any(plan.get(key) != value for key, value in pinned.items())
        or tuple(baseline.view_ids) != HELPER.ORIGINAL_IDS
        or tuple(previous.view_ids) != PREVIOUS_IDS
        or capture.get("status") != "complete"
        or capture.get("plan_sha256") != pinned["capture_plan_sha256"]
        or capture.get("depth_used") is not False
        or capture.get("segmentation_used") is not False
        or tuple(capture.get("additional_view_ids", [])) != NEW_IDS
        or tuple(capture.get("mapping_input_view_ids", [])) != ALL_IDS
        or set(capture.get("input_sha256", {})) != set(ALL_IDS)
        or pinned["checkpoint_sha256"]
        != baseline.metadata["learned_mask_intersection"]["checkpoint_sha256"]
    ):
        raise ValueError(
            "Dense capture or revision differs from the frozen RGB-only plan"
        )
    return plan, capture, pinned


def main(args):
    cv2.setNumThreads(1)
    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    baseline = ScanFreeMemory.load(args.input_memory)
    previous = ScanFreeMemory.load(args.retained_memory)
    old = baseline.metadata
    _, capture, pinned = validate_plan(args, baseline, previous)
    mask_path = args.retained_memory / "floor-masks.npz"
    if digest(mask_path) != previous.metadata["artifact_sha256"]["floor-masks.npz"]:
        raise ValueError("Prior32 floor evidence changed")
    with np.load(mask_path, allow_pickle=False) as data:
        retained_masks = data["masks"]
        if (
            retained_masks.dtype != bool
            or len(retained_masks) != 32
            or tuple(data["view_ids"].tolist()) != PREVIOUS_IDS
        ):
            raise ValueError("Prior mask IDs or dtype mismatch")
    views, hashes, input_paths = [], capture["input_sha256"], {}
    for name in ALL_IDS:
        paths = {
            "rgb": args.scan_dir / f"{name}.png",
            "calibration": args.scan_dir / f"{name}.json",
        }
        actual = {key: digest(path) for key, path in paths.items()}
        if actual != hashes[name]:
            raise ValueError(f"Frozen capture input changed: {name}")
        if name in PREVIOUS_IDS and actual != previous.metadata["input_sha256"][name]:
            raise ValueError("An earlier scan was replaced")
        if name in HELPER.ORIGINAL_IDS and actual != old["input_sha256"][name]:
            raise ValueError("An original baseline scan was replaced")
        record = json.loads(paths["calibration"].read_text())
        if (
            record.get("calibration") != "synthetic_exact"
            or record.get("depth_used") is not False
            or record.get("segmentation_used") is not False
        ):
            raise ValueError("Only declared RGB-only calibrated scans are supported")
        rgb = np.array(Image.open(paths["rgb"]).convert("RGB"))
        if rgb.shape[:2] != retained_masks.shape[1:]:
            raise ValueError("New frames must use retained native resolution")
        views.append(
            ScanRGBView(
                name,
                rgb,
                np.asarray(record["intrinsics"]),
                np.asarray(record["world_to_camera"]),
            )
        )
        input_paths[name] = paths
    occupied_path = args.occupied_source or Path(previous.metadata["occupied_source"])
    if digest(occupied_path) != old["occupied_source_sha256"]:
        raise ValueError("Original occupied source changed")
    occupied = RoomMemory.load(occupied_path)
    print(json.dumps({"phase": "inputs_verified", "views": len(views)}), flush=True)
    # The selected16 sources are checked against every other allowed scan; this
    # is identical to selecting their outputs from the full48-source operation.
    plane, agreement = floor_homography_masks(
        views,
        floor_z=old["floor_z_m"],
        maximum_channel_error=6.0,
        minimum_matches=2,
        minimum_baseline_m=0.5,
        minimum_angle_degrees=30.0,
        patch_size=None,
        source_indices=range(len(PREVIOUS_IDS), len(ALL_IDS)),
    )
    plane_counts = [int(mask.sum()) for mask in plane]
    agreement_max = [int(a.max()) for a in agreement]
    del agreement
    print(
        json.dumps({"phase": "homography_complete", "new_plane_pixels": plane_counts}),
        flush=True,
    )
    learned, learned_record = HELPER.learned_floor_masks(
        views[len(PREVIOUS_IDS) :],
        args.checkpoint,
        pinned["checkpoint_sha256"],
        baseline.extent,
    )
    masks = [mask.copy() for mask in retained_masks]
    masks.extend(a & b for a, b in zip(plane, learned, strict=True))
    del plane
    strict = build_scan_free_memory(
        views,
        extent=old["extent_m"],
        resolution=old["resolution_m"],
        body_height_m=0.12,
        floor_z=old["floor_z_m"],
        projection_margin_m=0.05,
        pixel_guard=2,
        minimum_views=4,
        minimum_baseline_m=0.5,
        minimum_angle_degrees=15.0,
        occupied_memory=occupied,
        floor_masks=masks,
        projection_shape="convex_hull",
        evidence_mode="full_prism",
    )
    result = retain_accepted_free(strict, baseline)
    if np.any(previous.free_mask & ~result.free_mask):
        raise ValueError("A previously geometry-accepted FREE cell was lost")
    result.metadata.update(
        experiment="m712-full-prism-extended-v3",
        promotion_status="candidate_requires_independent_whole_map_audit",
        original_mapping_input_view_ids=list(HELPER.ORIGINAL_IDS),
        supplemental_mapping_input_view_ids=[*HELPER.SUPPLEMENTAL_IDS, *NEW_IDS],
        supplemental_capture_manifest_path=str(
            (args.output / "capture-manifest.json").resolve()
        ),
        supplemental_capture_manifest_sha256=pinned["capture_manifest_sha256"],
        capture_plan_path=str((args.output / "capture-plan.json").resolve()),
        capture_plan_sha256=pinned["capture_plan_sha256"],
        scan_dir=str((args.output / "scan-inputs").resolve()),
        input_sha256=hashes,
        occupied_source=str((args.output / "occupied-source.json").resolve()),
        occupied_source_sha256=digest(occupied_path),
        occupied_source_semantics=old["occupied_source_semantics"],
        occupied_refinement=old["occupied_refinement"],
        source_memory_manifest_sha256=pinned["baseline_manifest_sha256"],
        source_floor_masks_sha256=old["artifact_sha256"]["floor-masks.npz"],
        reused_floor_evidence_manifest_sha256=pinned["retained_manifest_sha256"],
        reused_floor_evidence_masks_sha256=pinned["retained_floor_masks_sha256"],
        retained_previous_memory_path=str(args.retained_memory.resolve()),
        retained_baseline_memory_path=str(args.input_memory.resolve()),
        original_floor_masks_retained_exactly=True,
        prior32_floor_masks_retained_exactly=True,
        revision_plan_path=str((args.output / "revision-plan.json").resolve()),
        revision_plan_sha256=digest(args.revision_plan),
        floor_evidence="homography",
        homography_evidence=dict(
            old["homography_evidence"],
            new_view_ids=list(NEW_IDS),
            new_views_plane_mask_pixels=plane_counts,
            new_views_maximum_agreement=agreement_max,
            corroborating_views="all48 allowed views; prior32 final masks not replaced",
        ),
        learned_mask_intersection=learned_record,
        query_inputs_used=False,
        truth_inputs_used=False,
        thresholds_selected_using_truth=False,
        free_added_cells=int((result.free_mask & ~baseline.free_mask).sum()),
        free_removed_cells=0,
        free_added_over_v2_cells=int((result.free_mask & ~previous.free_mask).sum()),
        producer_source_sha256=digest(__file__),
        producer_module_sha256=digest(build_scan_free_memory.__code__.co_filename),
        helper_script_sha256=digest(HELPER_PATH),
        revision_module_sha256=digest(ROOT / "src/bb8_rl/mapping/scan_revision.py"),
        vision_source_sha256=digest(ROOT / "src/bb8_rl/vision.py"),
    )
    result.save(args.output)
    (args.output / "scan-inputs").mkdir()
    for paths in input_paths.values():
        for path in paths.values():
            shutil.copyfile(path, args.output / "scan-inputs" / path.name)
    for source, name in (
        (occupied_path, "occupied-source.json"),
        (args.scan_dir / "capture-manifest.json", "capture-manifest.json"),
        (args.capture_plan, "capture-plan.json"),
        (args.revision_plan, "revision-plan.json"),
    ):
        shutil.copyfile(source, args.output / name)
    np.savez_compressed(
        args.output / "floor-masks.npz",
        masks=np.stack(masks),
        view_ids=np.array(ALL_IDS),
    )
    np.savez_compressed(
        args.output / "supplemental-learned-masks.npz",
        masks=np.stack(learned),
        view_ids=np.array(NEW_IDS),
    )
    np.savez_compressed(
        args.output / "supplemental-additions.npz",
        added_mask=result.free_mask & ~baseline.free_mask,
        strict_candidate_mask=strict.candidate_free_mask,
        strict_support_bits=strict.support_bits,
        retained_baseline_free_mask=baseline.free_mask,
    )
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in (
        "floor-masks.npz",
        "supplemental-learned-masks.npz",
        "supplemental-additions.npz",
    ):
        manifest["artifact_sha256"][name] = digest(args.output / name)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "free_cells",
                    "free_added_cells",
                    "free_removed_cells",
                    "occupied_cells",
                    "unknown_cells",
                    "free_added_over_v2_cells",
                    "promotion_status",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-memory", type=Path, required=True)
    parser.add_argument("--retained-memory", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--revision-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--occupied-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
