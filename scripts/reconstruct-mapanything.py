"""Run isolated, calibrated RGB-only MapAnything inference on an existing scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from bb8_rl.mapping.backends.mapanything import (
    CalibratedRGBView,
    add_pose_alignment,
    reconstruct,
    write_ply,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def preview(arrays, output, pose_scale=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    point_key = (
        "points_world_calibrated_pose_scale"
        if pose_scale
        else "points_world_calibrated"
    )
    depth_key = "depth_z_pose_scale_m" if pose_scale else "depth_z_m"
    points = arrays[point_key][arrays["valid_mask"]]
    colors = arrays["rgb_uint8"][arrays["valid_mask"]]
    stride = max(1, len(points) // 35000)
    xyz = points[::stride]
    ax.scatter(*xyz.T, c=colors[::stride] / 255, s=0.4, rasterized=True)
    cameras = arrays["camera_to_world_input"][:, :3, 3]
    ax.scatter(*cameras.T, c="red", marker="^", s=35, label="supplied scan cameras")
    ax.set(
        xlabel="world x (m)",
        ylabel="world y (m)",
        zlabel="world z (m)",
        title="Estimated surfaces in supplied metric frame",
    )
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(222)
    ax2.imshow(arrays["rgb_uint8"][0])
    ax2.set(title="First input RGB, processed resolution")
    ax2.axis("off")
    ax3 = fig.add_subplot(224)
    depth = np.where(arrays["valid_mask"][0], arrays[depth_key][0], np.nan)
    im = ax3.imshow(depth, cmap="viridis")
    fig.colorbar(im, ax=ax3, label="predicted optical-Z depth (m)")
    ax3.set(title="Learned depth; invalid pixels blank")
    ax3.axis("off")
    subtitle = (
        "Metric scale enforced from input scan-camera centers; no surface fit"
        if pose_scale
        else "Estimated geometry; reconstruction accuracy and free space not established"
    )
    fig.suptitle(
        "Actual pretrained MapAnything inference — RGB + synthetic metric calibration\n"
        + subtitle,
        fontsize=12,
    )
    fig.savefig(output / "preview.png", dpi=150)
    plt.close(fig)


def export_pose_aligned_existing(source, output):
    """Derived artifact only; inference source is immutable and fully referenced."""
    started = time.perf_counter()
    with np.load(source / "reconstruction.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    details = add_pose_alignment(arrays)
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "schema": "bb8.mapanything-reconstruction.v1",
        "status": "complete",
        "derivation": "camera-prior-only Sim(3) scale enforcement; no new inference",
        "source_manifest": str((source / "manifest.json").resolve()),
        "source_npz_sha256": sha256(source / "reconstruction.npz"),
        "pose_alignment": details,
        "known_calibration_conditioning": True,
        "free_space_claim": False,
        "reconstruction_accuracy_evaluated": False,
        "inputs": source_manifest["inputs"],
        "held_out_index": source_manifest["held_out_index"],
        "checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "source_commits": source_manifest["source_commits"],
    }
    np.savez_compressed(output / "reconstruction.npz", **arrays)
    manifest["point_count"] = write_ply(
        output / "points-calibrated-pose-scale-world.ply",
        arrays["points_world_calibrated_pose_scale"],
        arrays["rgb_uint8"],
        arrays["valid_mask"],
    )
    write_ply(
        output / "points-model-pose-aligned-world.ply",
        arrays["points_world_model_pose_aligned"],
        arrays["rgb_uint8"],
        arrays["valid_mask"],
    )
    manifest["array_schema"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }
    preview(arrays, output, pose_scale=True)
    manifest["postprocess_seconds"] = time.perf_counter() - started
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "complete", "pose_alignment": details}), flush=True)


def merge_batches(sources, output, held_out_index):
    """Concatenate calibrated estimates; retain first occurrence of overlap views.

    This does not optimize a joint map or fuse disagreement away. Every retained
    view has a source batch, and all individual raw artifacts remain inspectable.
    """
    selected = {}
    batch_records = []
    shared_keys = None
    for batch_index, source in enumerate(sources):
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["status"] != "complete" or "pose_alignment" not in manifest:
            raise ValueError(f"Batch lacks completed camera-prior alignment: {source}")
        with np.load(source / "reconstruction.npz", allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        keys = set(arrays) - {"world_from_model_anchor", "world_from_model_similarity"}
        if shared_keys is None:
            shared_keys = keys
        elif keys != shared_keys:
            raise ValueError("Incompatible batch output schemas")
        for i, view_id in enumerate(arrays["view_ids"].tolist()):
            if view_id == f"scan-{held_out_index:03}":
                raise ValueError("Held-out view present in batch")
            if view_id in selected:
                continue
            item = {key: arrays[key][i] for key in keys}
            item["source_batch_index"] = np.int32(batch_index)
            item["source_batch_view_index"] = np.int32(i)
            item["world_from_model_similarity_per_view"] = arrays[
                "world_from_model_similarity"
            ]
            item["world_from_model_anchor_per_view"] = arrays["world_from_model_anchor"]
            item["camera_prior_scale_per_view"] = np.float32(
                manifest["pose_alignment"]["scale"]
            )
            selected[view_id] = item
        batch_records.append(
            {
                "manifest": str((source / "manifest.json").resolve()),
                "npz_sha256": sha256(source / "reconstruction.npz"),
                "view_ids": arrays["view_ids"].tolist(),
                "pose_alignment": manifest["pose_alignment"],
            }
        )
    ordered = [selected[key] for key in sorted(selected)]
    arrays = {key: np.stack([item[key] for item in ordered]) for key in ordered[0]}
    np.savez_compressed(output / "reconstruction.npz", **arrays)
    count = write_ply(
        output / "points-calibrated-pose-scale-world.ply",
        arrays["points_world_calibrated_pose_scale"],
        arrays["rgb_uint8"],
        arrays["valid_mask"],
    )
    write_ply(
        output / "points-model-pose-aligned-world.ply",
        arrays["points_world_model_pose_aligned"],
        arrays["rgb_uint8"],
        arrays["valid_mask"],
    )
    preview(arrays, output, pose_scale=True)
    manifest = {
        "schema": "bb8.mapanything-reconstruction.v1",
        "status": "complete",
        "derivation": "overlapping independently inferred batches; first occurrence retained for duplicate views; no joint 23-view attention or surface fusion",
        "source_batches": batch_records,
        "view_ids": sorted(selected),
        "held_out_index": held_out_index,
        "missing_scan_indices_before_held_out": [
            i for i in range(held_out_index) if f"scan-{i:03}" not in selected
        ],
        "point_count": count,
        "known_calibration_conditioning": True,
        "free_space_claim": False,
        "reconstruction_accuracy_evaluated": False,
        "model_frame_warning": "raw model coordinates/poses are batch-local; use per-view source batch or exported world-frame arrays",
        "array_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"status": "complete", "views": len(selected), "points": count}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--pose-align-existing", type=Path)
    parser.add_argument("--merge-batches", type=Path, nargs="+")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--held-out-index", type=int, default=23)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.merge_batches:
        args.output.mkdir(parents=True, exist_ok=False)
        merge_batches(args.merge_batches, args.output, args.held_out_index)
        return
    if args.pose_align_existing:
        args.output.mkdir(parents=True, exist_ok=False)
        export_pose_aligned_existing(args.pose_align_existing, args.output)
        return
    if args.scan is None or args.runtime_root is None:
        parser.error("Inference requires --scan and --runtime-root")
    if args.held_out_index in args.indices:
        parser.error("Held-out registration image cannot be a reconstruction input")
    if len(set(args.indices)) != len(args.indices):
        parser.error("Duplicate view indices")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "schema": "bb8.mapanything-reconstruction.v1",
        "status": "started",
        "held_out_index": args.held_out_index,
        "indices": args.indices,
        "license": "Apache-2.0 code and facebook/map-anything-apache checkpoint",
        "checkpoint_repository": "facebook/map-anything-apache",
        "checkpoint_revision": "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
        "runtime_context": "offline elapsed time; other project CPU work may run concurrently; not an isolated speed benchmark",
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {
            key: os.environ.get(key)
            for key in [
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            ]
        },
    }
    manifest_path = args.output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    save()
    try:
        root = args.runtime_root.resolve()
        checkpoint = root / "checkpoint"
        manifest["source_commits"] = {
            name: git_sha(root / "vendor" / name) for name in ["map-anything", "dinov2"]
        }
        expected_commits = {
            "map-anything": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
            "dinov2": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        }
        if manifest["source_commits"] != expected_commits:
            raise ValueError(
                "Source checkout does not match the recorded experiment pins"
            )
        manifest["official_sources"] = [
            "https://github.com/facebookresearch/map-anything",
            "https://huggingface.co/facebook/map-anything-apache",
            "https://github.com/facebookresearch/dinov2",
        ]
        (args.output / "dependencies.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        )
        manifest["checkpoint_sha256"] = {
            name: sha256(checkpoint / name)
            for name in ["config.json", "model.safetensors"]
        }
        expected_hashes = {
            "config.json": "65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5",
            "model.safetensors": "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201",
        }
        if manifest["checkpoint_sha256"] != expected_hashes:
            raise ValueError(
                "Checkpoint content does not match the pinned Apache experiment"
            )
        views = []
        manifest["inputs"] = []
        for index in args.indices:
            stem = f"scan-{index:03}"
            image_path, calibration_path = (
                args.scan / f"{stem}.png",
                args.scan / f"{stem}.json",
            )
            calibration = json.loads(calibration_path.read_text())
            provenance = calibration.get("calibration", "unspecified")
            if provenance != "synthetic_exact":
                raise ValueError(
                    "This experimental runner requires explicit synthetic_exact calibration"
                )
            views.append(
                CalibratedRGBView(
                    stem,
                    np.asarray(Image.open(image_path).convert("RGB")),
                    np.asarray(calibration["intrinsics"], dtype=np.float32),
                    np.linalg.inv(
                        np.asarray(calibration["world_to_camera"], dtype=np.float64)
                    ),
                    provenance,
                )
            )
            manifest["inputs"].append(
                {
                    "view_id": stem,
                    "rgb": str(image_path),
                    "calibration": str(calibration_path),
                    "rgb_sha256": sha256(image_path),
                    "calibration_sha256": sha256(calibration_path),
                    "calibration_provenance": provenance,
                }
            )
        save()
        print(
            f"Starting actual inference for {len(views)} RGB views on {args.device}",
            flush=True,
        )
        arrays, details = reconstruct(
            views,
            checkpoint=checkpoint,
            dinov2_source=root / "vendor/dinov2",
            device=args.device,
            amp=args.amp,
        )
        manifest.update(details)
        # Preserve actual inference even if calibration enforcement rejects it.
        np.savez_compressed(args.output / "reconstruction-raw.npz", **arrays)
        manifest["pose_alignment"] = add_pose_alignment(arrays)
        np.savez_compressed(args.output / "reconstruction.npz", **arrays)
        manifest["point_count"] = write_ply(
            args.output / "points-calibrated-world.ply",
            arrays["points_world_calibrated"],
            arrays["rgb_uint8"],
            arrays["valid_mask"],
        )
        write_ply(
            args.output / "points-model-anchored-world.ply",
            arrays["points_world_model_anchored"],
            arrays["rgb_uint8"],
            arrays["valid_mask"],
        )
        write_ply(
            args.output / "points-calibrated-pose-scale-world.ply",
            arrays["points_world_calibrated_pose_scale"],
            arrays["rgb_uint8"],
            arrays["valid_mask"],
        )
        write_ply(
            args.output / "points-model-pose-aligned-world.ply",
            arrays["points_world_model_pose_aligned"],
            arrays["rgb_uint8"],
            arrays["valid_mask"],
        )
        manifest["array_schema"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }
        manifest["per_view"] = [
            {
                "view_id": v.view_id,
                "valid_pixels": int(mask.sum()),
                "total_pixels": int(mask.size),
                "depth_z_m_percentiles_5_50_95": np.percentile(
                    depth[mask], [5, 50, 95]
                ).tolist()
                if mask.any()
                else None,
            }
            for v, mask, depth in zip(
                views, arrays["valid_mask"], arrays["depth_z_m"], strict=True
            )
        ]
        preview(arrays, args.output)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest["process_peak_rss_bytes"] = (
            rss if sys.platform == "darwin" else rss * 1024
        )
        manifest["total_wall_seconds"] = time.perf_counter() - started
        save()
        print(
            json.dumps(
                {
                    key: manifest.get(key)
                    for key in [
                        "status",
                        "point_count",
                        "total_wall_seconds",
                        "process_peak_rss_bytes",
                        "error",
                    ]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
