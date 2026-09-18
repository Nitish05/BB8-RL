"""Extend positive full-prism scan evidence with twelve frozen overhead RGB views.

Original final floor masks are retained exactly. Only new-view masks are produced
from the unchanged floor-plane agreement and frozen calibrated learned observer.
No scene geometry, depth/segmentation labels, query cameras, or truth fit is used.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bb8_rl.mapping.room_memory import RoomMemory
from bb8_rl.mapping.scan_free_space import (
    EXCLUDED_QUERY_IDS,
    ScanFreeMemory,
    ScanRGBView,
    build_scan_free_memory,
    floor_homography_masks,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_IDS = tuple(f"scan-{i:03}" for i in range(24) if i not in EXCLUDED_QUERY_IDS)
SUPPLEMENTAL_IDS = tuple(f"scan-{i:03}" for i in range(24, 36))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    return (
        ROOT / path.removeprefix("<BB8-RL>/")
        if path.startswith("<BB8-RL>/")
        else Path(path)
    )


def read_view(directory, name, expected_hashes=None):
    if name not in (*ORIGINAL_IDS, *SUPPLEMENTAL_IDS):
        raise ValueError("Only frozen original and supplemental scan IDs are allowed")
    paths = {
        "rgb": directory / f"{name}.png",
        "calibration": directory / f"{name}.json",
    }
    hashes = {key: digest(path) for key, path in paths.items()}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError(f"Frozen RGB/calibration hash mismatch: {name}")
    record = json.loads(paths["calibration"].read_text())
    if (
        record.get("calibration") != "synthetic_exact"
        or record.get("depth_used") is not False
        or record.get("segmentation_used") is not False
    ):
        raise ValueError("Require declared RGB-only synthetic scan calibration")
    rgb = np.array(Image.open(paths["rgb"]).convert("RGB"))
    return (
        ScanRGBView(
            name,
            rgb,
            np.asarray(record["intrinsics"]),
            np.asarray(record["world_to_camera"]),
        ),
        hashes,
        paths,
    )


def capture_input_hashes(manifest_path, plan_path):
    record = json.loads(manifest_path.read_text())
    expected_ids = (*ORIGINAL_IDS, *SUPPLEMENTAL_IDS)
    if (
        record.get("status") != "complete"
        or record.get("plan_sha256") != digest(plan_path)
        or record.get("depth_used") is not False
        or record.get("segmentation_used") is not False
        or tuple(record.get("additional_view_ids", [])) != SUPPLEMENTAL_IDS
        or tuple(record.get("mapping_input_view_ids", [])) != expected_ids
        or set(record.get("input_sha256", {})) != set(expected_ids)
    ):
        raise ValueError("Supplemental capture differs from the frozen RGB-only plan")
    return record["input_sha256"]


def learned_floor_masks(views, checkpoint, expected_checkpoint_sha256, extent):
    # Heavy optional dependency stays out of import-time/test-only paths.
    import torch

    from bb8_rl.camera import Calibration
    from bb8_rl.vision import LearnedObserver, guided_floor_probability

    if digest(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("Frozen calibrated vision checkpoint hash mismatch")
    torch.set_num_threads(2)
    observer, masks, counts = None, [], []
    for index, view in enumerate(views):
        calibration = Calibration(
            view.intrinsics,
            view.world_to_camera,
            (view.rgb.shape[1], view.rgb.shape[0]),
            extent,
            provenance="synthetic_known_supplemental_scan_camera",
        )
        if observer is None:
            observer = LearnedObserver(
                calibration, checkpoint, floor_refinement="guided"
            )
            if observer.floor_target != "traversable":
                raise ValueError("Unexpected frozen checkpoint floor target")
        observer.calibration = calibration
        observer._validate_frame(view.rgb, float(index))
        small = cv2.resize(view.rgb, (320, 240), interpolation=cv2.INTER_AREA)
        probability = observer.probabilities([small])[0]
        probability = cv2.resize(
            probability, calibration.resolution, interpolation=cv2.INTER_LINEAR
        )
        probability = guided_floor_probability(view.rgb, probability)
        mask = probability >= observer.floor_threshold
        if index == 0 and not np.array_equal(
            mask, observer.observe(view.rgb, float(index)).visible_floor
        ):
            raise ValueError(
                "Floor-only inference differs from public calibrated observer"
            )
        masks.append(mask)
        counts.append(int(mask.sum()))
    return masks, {
        "checkpoint_sha256": digest(checkpoint),
        "floor_threshold": float(observer.floor_threshold),
        "floor_refinement": "guided-radius4-epsilon1e-4",
        "target_semantics": "visible ground plus visible self (as trained); not floor-only",
        "operation": "intersection with native palette and unchanged planar agreement; no self clearing",
        "public_observe_mask_parity_first_new_view": True,
        "training_or_threshold_selection": False,
        "mask_pixels": counts,
    }


def frozen_revision_masks(args, old, hashes, retained_masks):
    """Read one predeclared strict revision without refitting any RGB evidence."""
    if args.reuse_floor_evidence is None or args.revision_plan is None:
        raise ValueError(
            "Strict revision needs frozen floor evidence and a revision plan"
        )
    plan = json.loads(args.revision_plan.read_text())
    prior_path = args.reuse_floor_evidence / "manifest.json"
    prior = json.loads(prior_path.read_text())
    expected_settings = {
        "mode": "full_prism",
        "body_height_m": 0.12,
        "metric_projection_margin_m": 0.05,
        "minimum_pairwise_separated_views": 4,
        "minimum_baseline_m": 0.5,
        "minimum_angle_degrees": 15.0,
        "pixel_guard": 2,
        "projection_shape": "convex_hull",
    }
    pinned = plan.get("inputs_sha256", {})
    if (
        plan.get("revision") != "full-prism-extended-v2"
        or plan.get("new_evidence") != expected_settings
        or plan.get("truth_geometry_or_failed_cell_locations_used") is not False
        or pinned.get("work/m712-map-improvement/full-prism-extended-v1/manifest.json")
        != digest(prior_path)
        or pinned.get("work/interactive-assets/memory/manifest.json")
        != digest(args.input_memory / "manifest.json")
        or prior.get("input_sha256") != hashes
        or prior.get("free_evidence_mode") != "full_prism"
        or prior.get("truth_inputs_used") is not False
        or prior.get("thresholds_selected_using_truth") is not False
        or prior.get("capture_plan_sha256") != digest(args.capture_plan)
        or digest(args.checkpoint)
        != old["learned_mask_intersection"]["checkpoint_sha256"]
    ):
        raise ValueError("Strict revision inputs differ from the frozen plan")
    mask_path = args.reuse_floor_evidence / "floor-masks.npz"
    if (
        digest(mask_path) != prior["artifact_sha256"]["floor-masks.npz"]
        or digest(mask_path)
        != pinned["work/m712-map-improvement/full-prism-extended-v1/floor-masks.npz"]
    ):
        raise ValueError("Frozen32 floor evidence changed")
    with np.load(mask_path, allow_pickle=False) as data:
        masks = data["masks"]
        if (
            tuple(data["view_ids"].tolist()) != (*ORIGINAL_IDS, *SUPPLEMENTAL_IDS)
            or masks.dtype != bool
            or len(masks) != 32
            or not np.array_equal(masks[:20], retained_masks)
        ):
            raise ValueError(
                "Frozen mask IDs, native shape or retained evidence changed"
            )
    learned_path = args.reuse_floor_evidence / "supplemental-learned-masks.npz"
    if (
        digest(learned_path)
        != prior["artifact_sha256"]["supplemental-learned-masks.npz"]
    ):
        raise ValueError("Frozen supplemental learned masks changed")
    with np.load(learned_path, allow_pickle=False) as data:
        learned = data["masks"]
        if (
            tuple(data["view_ids"].tolist()) != SUPPLEMENTAL_IDS
            or learned.dtype != bool
        ):
            raise ValueError("Frozen learned mask IDs/dtype mismatch")
    return list(masks), list(learned), prior


def main(args):
    cv2.setNumThreads(1)
    if args.output.exists():
        raise ValueError("Choose a fresh candidate output directory")
    source = ScanFreeMemory.load(args.input_memory)
    old = source.metadata
    if (
        tuple(source.view_ids) != ORIGINAL_IDS
        or old.get("free_evidence_mode", "full_prism") != "full_prism"
    ):
        raise ValueError("Retained memory must be the original20 full-prism baseline")
    if not source.memory.valid:
        raise ValueError("Cannot extend invalidated memory")
    mask_path = args.input_memory / "floor-masks.npz"
    if digest(mask_path) != old["artifact_sha256"]["floor-masks.npz"]:
        raise ValueError("Original retained floor masks changed")
    with np.load(mask_path, allow_pickle=False) as data:
        retained_masks = data["masks"]
        if (
            tuple(data["view_ids"].tolist()) != ORIGINAL_IDS
            or retained_masks.dtype != bool
        ):
            raise ValueError("Original mask IDs/dtype mismatch")
    capture_manifest = (
        args.capture_manifest or args.extended_scan_dir / "capture-manifest.json"
    )
    capture_plan = args.capture_plan
    # Capture metadata is provenance only. Its scene/robot fixture fields never
    # participate in floor inference or geometry production.
    capture_manifest_hash, capture_plan_hash = (
        digest(capture_manifest),
        digest(capture_plan),
    )
    capture_hashes = capture_input_hashes(capture_manifest, capture_plan)
    views, hashes, inputs = [], {}, {}
    for name in (*ORIGINAL_IDS, *SUPPLEMENTAL_IDS):
        original = name in ORIGINAL_IDS
        if original and capture_hashes[name] != old["input_sha256"][name]:
            raise ValueError("Capture changed an original frozen scan")
        view, record, paths = read_view(
            args.original_scan_dir if original else args.extended_scan_dir,
            name,
            capture_hashes[name],
        )
        if view.rgb.shape[:2] != retained_masks.shape[1:]:
            raise ValueError(
                "All extended views require original native RGB resolution"
            )
        views.append(view)
        hashes[name], inputs[name] = record, paths
    occupied_path = args.occupied_source or resolve(old["occupied_source"])
    if digest(occupied_path) != old["occupied_source_sha256"]:
        raise ValueError("Frozen occupied source changed")
    occupied = RoomMemory.load(occupied_path)
    print(json.dumps({"phase": "inputs_verified", "views": len(views)}), flush=True)
    # This recomputes all32 to use exactly the shared implementation. New masks
    # use all32 as geometric corroboration; old20 masks are discarded here and
    # replaced byte-for-byte with the frozen final masks, never expanded.
    strict_revision = (
        args.revision_plan is not None or args.reuse_floor_evidence is not None
    )
    if strict_revision:
        masks, learned, prior = frozen_revision_masks(args, old, hashes, retained_masks)
        plane_counts = prior["homography_evidence"]["new_views_plane_mask_pixels"]
        agreement_max = prior["homography_evidence"]["new_views_maximum_agreement"]
        learned_record = prior["learned_mask_intersection"]
    else:
        plane, agreement = floor_homography_masks(
            views,
            floor_z=old["floor_z_m"],
            maximum_channel_error=6.0,
            minimum_matches=2,
            minimum_baseline_m=0.5,
            minimum_angle_degrees=30.0,
            patch_size=None,
        )
        plane_counts = [int(mask.sum()) for mask in plane[len(ORIGINAL_IDS) :]]
        agreement_max = [int(value.max()) for value in agreement[len(ORIGINAL_IDS) :]]
        del agreement
        print(
            json.dumps(
                {"phase": "homography_complete", "new_plane_pixels": plane_counts}
            ),
            flush=True,
        )
        learned, learned_record = learned_floor_masks(
            views[len(ORIGINAL_IDS) :],
            args.checkpoint,
            old["learned_mask_intersection"]["checkpoint_sha256"],
            source.extent,
        )
        masks = [mask.copy() for mask in retained_masks]
        masks.extend(
            a & b for a, b in zip(plane[len(ORIGINAL_IDS) :], learned, strict=True)
        )
        del plane
    result = build_scan_free_memory(
        views,
        extent=old["extent_m"],
        resolution=old["resolution_m"],
        body_height_m=old["requested_body_height_m"],
        floor_z=old["floor_z_m"],
        projection_margin_m=old["projection_margin_m"],
        pixel_guard=2 if strict_revision else old["pixel_guard"],
        minimum_views=4 if strict_revision else old["minimum_views"],
        minimum_baseline_m=old["minimum_baseline_m"],
        minimum_angle_degrees=old["minimum_angle_degrees"],
        occupied_memory=occupied,
        floor_masks=masks,
        projection_shape=old["projection_shape"],
        evidence_mode="full_prism",
    )
    strict_support = strict_candidate = None
    if strict_revision:
        from bb8_rl.mapping.scan_revision import retain_accepted_free

        strict_support, strict_candidate = (
            result.support_bits.copy(),
            result.candidate_free_mask.copy(),
        )
        result = retain_accepted_free(result, source)
    if not np.array_equal(source.occupied_mask, result.occupied_mask):
        raise ValueError("Original occupied cells changed")
    if np.any(source.free_mask & ~result.free_mask):
        raise ValueError("Retained full-prism evidence lost previously FREE cells")
    result.metadata.update(
        experiment="m712-full-prism-extended-v2"
        if strict_revision
        else "m712-full-prism-extended-v1",
        promotion_status="candidate_requires_independent_whole_map_audit",
        original_mapping_input_view_ids=list(ORIGINAL_IDS),
        supplemental_mapping_input_view_ids=list(SUPPLEMENTAL_IDS),
        supplemental_capture_manifest_path=str(
            (args.output / "capture-manifest.json").resolve()
        ),
        supplemental_capture_manifest_sha256=capture_manifest_hash,
        capture_plan_path=str((args.output / "capture-plan.json").resolve()),
        capture_plan_sha256=capture_plan_hash,
        scan_dir=str((args.output / "scan-inputs").resolve()),
        input_sha256=hashes,
        occupied_source=str((args.output / "occupied-source.json").resolve()),
        occupied_source_sha256=digest(occupied_path),
        occupied_source_semantics=old["occupied_source_semantics"],
        occupied_refinement=old["occupied_refinement"],
        source_memory_manifest_sha256=digest(args.input_memory / "manifest.json"),
        source_floor_masks_sha256=digest(mask_path),
        original_floor_masks_retained_exactly=True,
        floor_evidence="homography",
        homography_evidence=dict(
            old["homography_evidence"],
            new_views_plane_mask_pixels=plane_counts,
            new_views_maximum_agreement=agreement_max,
            corroborating_views="all32allowedviews; original20 final masks not replaced",
        ),
        learned_mask_intersection=dict(
            learned_record,
            original_manifest_sha256=old["learned_mask_intersection"][
                "manifest_sha256"
            ],
        ),
        query_inputs_used=False,
        truth_inputs_used=False,
        thresholds_selected_using_truth=False,
        free_added_cells=int((result.free_mask & ~source.free_mask).sum()),
        free_removed_cells=0,
        producer_source_sha256=digest(__file__),
        producer_module_sha256=digest(build_scan_free_memory.__code__.co_filename),
        vision_source_sha256=digest(ROOT / "src/bb8_rl/vision.py"),
    )
    result.save(args.output)
    if strict_revision:
        shutil.copyfile(args.revision_plan, args.output / "revision-plan.json")
        np.savez_compressed(
            args.output / "supplemental-additions.npz",
            added_mask=result.free_mask & ~source.free_mask,
            strict_candidate_mask=strict_candidate,
            strict_support_bits=strict_support,
            retained_baseline_free_mask=source.free_mask,
        )
    scan_output = args.output / "scan-inputs"
    scan_output.mkdir()
    for name, paths in inputs.items():
        for path in paths.values():
            shutil.copyfile(path, scan_output / path.name)
    shutil.copyfile(occupied_path, args.output / "occupied-source.json")
    shutil.copyfile(capture_manifest, args.output / "capture-manifest.json")
    shutil.copyfile(capture_plan, args.output / "capture-plan.json")
    np.savez_compressed(
        args.output / "floor-masks.npz",
        masks=np.stack(masks),
        view_ids=np.array(result.view_ids),
    )
    np.savez_compressed(
        args.output / "supplemental-learned-masks.npz",
        masks=np.stack(learned),
        view_ids=np.array(SUPPLEMENTAL_IDS),
    )
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if strict_revision:
        manifest.update(
            revision_plan_path=str((args.output / "revision-plan.json").resolve()),
            revision_plan_sha256=digest(args.revision_plan),
            reused_floor_evidence_manifest_sha256=digest(
                args.reuse_floor_evidence / "manifest.json"
            ),
            retained_baseline_memory_path=str(args.input_memory.resolve()),
            revision_module_sha256=digest(ROOT / "src/bb8_rl/mapping/scan_revision.py"),
        )
        manifest["artifact_sha256"]["supplemental-additions.npz"] = digest(
            args.output / "supplemental-additions.npz"
        )
    for name in ("floor-masks.npz", "supplemental-learned-masks.npz"):
        manifest["artifact_sha256"][name] = digest(args.output / name)
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
    parser.add_argument("--original-scan-dir", type=Path, required=True)
    parser.add_argument("--extended-scan-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--occupied-source", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reuse-floor-evidence", type=Path)
    parser.add_argument("--revision-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
