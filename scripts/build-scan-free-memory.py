"""Build conditional scan-once free volume from original RGB, preserving occupancy."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bb8_rl.mapping.room_memory import RoomMemory
from bb8_rl.mapping.scan_free_space import (
    EXCLUDED_QUERY_IDS,
    ScanRGBView,
    build_scan_free_memory,
    floor_homography_masks,
    refine_occupied_conflicts,
    synthetic_floor_mask,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preview(result, views, output, floor_masks=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for axis, index in zip(axes[0], (0, len(views) // 2), strict=True):
        view = views[index]
        mask = (
            synthetic_floor_mask(view.rgb)
            if floor_masks is None
            else floor_masks[index]
        )
        overlay = view.rgb.astype(float) / 255
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([0.15, 0.9, 0.4])
        axis.imshow(overlay)
        axis.set_title(f"{view.view_id}: floor classification in green")
        axis.axis("off")
    states = np.zeros((*result.free_mask.shape, 3)) + 0.65
    states[result.free_mask] = [0.25, 0.72, 0.5]
    states[result.occupied_mask] = [0.22, 0.25, 0.3]
    e = result.extent
    axes[1, 0].imshow(
        states, origin="lower", extent=[-e, e, -e, e], interpolation="nearest"
    )
    axes[1, 0].set(
        xlabel="world x (m)",
        ylabel="world y (m)",
        title="Raw free grid: green FREE / dark OCCUPIED / gray UNKNOWN",
    )
    supported = np.array(
        [int(v).bit_count() for v in result.support_bits.ravel()]
    ).reshape(result.free_mask.shape)
    im = axes[1, 1].imshow(
        supported,
        origin="lower",
        extent=[-e, e, -e, e],
        vmin=0,
        vmax=len(views),
        cmap="viridis",
    )
    fig.colorbar(
        im, ax=axes[1, 1], label="views observing the complete projected prism as floor"
    )
    axes[1, 1].set(
        xlabel="world x (m)",
        ylabel="world y (m)",
        title="Positive RGB support before occupied conflicts",
    )
    fig.suptitle(
        "Conditional synthetic scan memory — whole body-height volume, RGB evidence only\nGrounded objects / no overhangs / known palette and calibration / static scene",
        fontsize=12,
    )
    fig.savefig(output / "preview.png", dpi=150)
    plt.close(fig)


def main(args):
    start = time.perf_counter()
    views = []
    hashes = {}
    palette = []
    cv2.setNumThreads(2)
    for index in range(24):
        if index in EXCLUDED_QUERY_IDS:
            continue
        stem = f"scan-{index:03}"
        rgb_path = args.scan_dir / f"{stem}.png"
        meta_path = args.scan_dir / f"{stem}.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("calibration") != "synthetic_exact":
            raise ValueError(
                "This baseline requires explicitly synthetic scan calibration"
            )
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        views.append(
            ScanRGBView(
                stem,
                rgb,
                np.asarray(meta["intrinsics"]),
                np.asarray(meta["world_to_camera"]),
            )
        )
        mask = synthetic_floor_mask(rgb)
        palette.append(
            {
                "view_id": stem,
                "floor_pixels": int(mask.sum()),
                "total_pixels": int(mask.size),
                "unclassified_pixels": int((~mask).sum()),
            }
        )
        hashes[stem] = {"rgb": digest(rgb_path), "calibration": digest(meta_path)}
    occupied_refinement = None
    if args.occupied_reconstruction is not None:
        if args.occupied_output is None:
            raise ValueError(
                "Provide a fresh output directory for refined occupied evidence"
            )
        source_manifest_path = args.occupied_reconstruction.with_name("manifest.json")
        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest["mapping_input_view_ids"] != [v.view_id for v in views]:
            raise ValueError(
                "Occupied reconstruction ancestry differs from frozen map scans"
            )
        with np.load(args.occupied_reconstruction, allow_pickle=False) as data:
            occupied, occupied_refinement, accepted, support = (
                refine_occupied_conflicts(
                    data["points_world_refined"],
                    data["valid_mask"],
                    data["view_ids"].tolist(),
                    extent=args.extent,
                    resolution_m=args.resolution,
                )
            )
        args.occupied_output.mkdir(parents=True, exist_ok=False)
        args.occupied_memory = args.occupied_output / "room-memory.json"
        occupied.save(args.occupied_memory)
        np.savez_compressed(
            args.occupied_output / "surface-support.npz",
            accepted=accepted,
            supporting_view_count=support,
        )
        occupied_refinement.update(
            {
                "source_npz_path": str(args.occupied_reconstruction.resolve()),
                "source_npz_sha256": digest(args.occupied_reconstruction),
                "source_manifest_path": str(source_manifest_path.resolve()),
                "source_manifest_sha256": digest(source_manifest_path),
                "mapping_input_view_ids": source_manifest["mapping_input_view_ids"],
                "source_occupancy_sha256": digest(args.occupied_memory),
                "surface_support_sha256": digest(
                    args.occupied_output / "surface-support.npz"
                ),
            }
        )
        (args.occupied_output / "manifest.json").write_text(
            json.dumps(occupied_refinement, indent=2, allow_nan=False) + "\n"
        )
    else:
        occupied = (
            RoomMemory.load(args.occupied_memory) if args.occupied_memory else None
        )
    masks = None
    homography_report = None
    if args.floor_evidence in {"homography", "texture"}:
        patch_size = 31 if args.floor_evidence == "texture" else None
        masks, agreement = floor_homography_masks(views, patch_size=patch_size)
        homography_report = {
            "maximum_channel_error_0_255": 6.0,
            "minimum_independent_matching_views": 2,
            "minimum_baseline_m": 0.5,
            "minimum_angle_degrees": 30.0,
            "patch_size_pixels": patch_size,
            "minimum_patch_std_0_255": 3.0 if patch_size is not None else None,
            "minimum_patch_correlation": 0.98 if patch_size is not None else None,
            "patch_all_pixels_within_error_gate": patch_size is not None,
            "prefilter": "3x3 Gaussian sigma .6 before warp; complete bilinear native palette-mask support required",
            "agreement_pixels": [int(m.sum()) for m in masks],
            "maximum_agreement_views": [int(a.max()) for a in agreement],
            "limitation": "uniform identical elevated surfaces can remain ambiguous; independent zero-false-FREE audit required before promotion",
        }
    learned_report = None
    if args.learned_floor_dir is not None:
        learned_path = args.learned_floor_dir / "manifest.json"
        learned = json.loads(learned_path.read_text())
        if (
            learned.get("status") != "complete"
            or learned.get("mapping_input_view_ids") != [v.view_id for v in views]
            or learned.get("query_inputs_used") is not False
            or learned.get("truth_labels_used") is not False
            or learned.get("training_or_threshold_selection") is not False
        ):
            raise ValueError(
                "Learned masks require complete frozen map-only provenance"
            )
        if masks is None:
            masks = [synthetic_floor_mask(v.rgb) for v in views]
        by_id = {v["view_id"]: v for v in learned["views"]}
        for i, view in enumerate(views):
            record = by_id[view.view_id]
            mask_path = args.learned_floor_dir / record["mask_file"]
            if (
                digest(mask_path) != record["mask_file_sha256"]
                or hashes[view.view_id]["rgb"] != record["rgb_sha256"]
                or hashes[view.view_id]["calibration"]
                != record["calibration_json_sha256"]
            ):
                raise ValueError("Learned mask or scan input hash mismatch")
            with np.load(mask_path, allow_pickle=False) as data:
                mask = data["mask"]
                if mask.dtype != bool or mask.shape != masks[i].shape:
                    raise ValueError(
                        "Learned masks must preserve native RGB resolution"
                    )
                masks[i] &= mask
        learned_report = {
            "manifest_path": str(learned_path.resolve()),
            "manifest_sha256": digest(learned_path),
            "checkpoint_sha256": learned["checkpoint_sha256"],
            "target_semantics": learned["target_semantics"],
            "operation": "intersection with palette and selected plane evidence only; no mask union, threshold refit, robot or self clearing",
            "retained_pixels": [int(mask.sum()) for mask in masks],
        }
    result = build_scan_free_memory(
        views,
        extent=args.extent,
        resolution=args.resolution,
        body_height_m=args.body_height,
        floor_z=0.0,
        projection_margin_m=args.projection_margin,
        pixel_guard=args.pixel_guard,
        minimum_baseline_m=args.minimum_baseline,
        minimum_angle_degrees=args.minimum_angle,
        minimum_views=args.minimum_views,
        occupied_memory=occupied,
        floor_masks=masks,
        projection_shape=args.projection_shape,
    )
    result.metadata.update(
        {
            "input_sha256": hashes,
            "scan_dir": str(args.scan_dir.resolve()),
            "floor_evidence": args.floor_evidence,
            "homography_evidence": homography_report,
            "learned_mask_intersection": learned_report,
            "occupied_refinement": occupied_refinement,
            "palette_observations": palette,
            "occupied_source": str(args.occupied_memory.resolve())
            if args.occupied_memory
            else None,
            "occupied_source_sha256": digest(args.occupied_memory)
            if args.occupied_memory
            else None,
            "occupied_source_semantics": "Refined sourcephoto conflict evidence: three-view metric support, finer collision-height voxelization, unchanged assumed .05m margin; rejected samples do not create FREE"
            if occupied_refinement
            else "M77 RGB-supported photometric surfaces, unmodified conflict evidence; broader dense geometry is not imported",
            "producer_source_sha256": digest(Path(__file__)),
            "producer_module_sha256": digest(
                Path(build_scan_free_memory.__code__.co_filename)
            ),
            "elapsed_seconds": time.perf_counter() - start,
            "floor_palette_note": "candidate floor from existing synthetic HSV rule; unknown colors remain nonfloor but overlapping object colors may be accepted; independent zero-false-FREE audit is required before driving promotion",
        }
    )
    result.save(args.output)
    if masks is not None:
        np.savez_compressed(
            args.output / "floor-masks.npz",
            masks=np.stack(masks),
            view_ids=np.asarray(result.view_ids),
        )
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"]["floor-masks.npz"] = digest(
            args.output / "floor-masks.npz"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    preview(result, views, args.output, masks)
    print(
        json.dumps(
            {
                k: result.metadata[k]
                for k in [
                    "candidate_free_cells",
                    "free_cells",
                    "occupied_cells",
                    "unknown_cells",
                    "free_candidates_blocked_by_occupied",
                    "elapsed_seconds",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(
            "work/scan/view-probe"
        ),
    )
    parser.add_argument(
        "--occupied-memory",
        type=Path,
        default=Path("work/m77/memory/photometric/room-memory.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extent", type=float, default=2.0)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--body-height", type=float, default=0.12)
    parser.add_argument("--projection-margin", type=float, default=0.015)
    parser.add_argument("--pixel-guard", type=int, default=1)
    parser.add_argument("--minimum-baseline", type=float, default=0.5)
    parser.add_argument("--minimum-angle", type=float, default=15.0)
    parser.add_argument("--minimum-views", type=int, choices=[2, 3], default=2)
    parser.add_argument(
        "--projection-shape", choices=["rectangle", "convex_hull"], default="rectangle"
    )
    parser.add_argument("--occupied-reconstruction", type=Path)
    parser.add_argument("--occupied-output", type=Path)
    parser.add_argument(
        "--floor-evidence",
        choices=["palette", "homography", "texture"],
        default="texture",
    )
    parser.add_argument("--learned-floor-dir", type=Path)
    main(parser.parse_args())
