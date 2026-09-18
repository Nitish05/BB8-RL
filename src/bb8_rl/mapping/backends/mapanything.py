"""Offline MapAnything adapter with explicit synthetic metric calibration.

Torch and the upstream model remain optional. This module cannot actuate a robot
or consume simulator depth, segmentation, or obstacle geometry. Model outputs are
estimated surfaces, never evidence of free space without further validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


@dataclass(frozen=True)
class CalibratedRGBView:
    view_id: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    calibration_provenance: str

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("RGB must be HxWx3 uint8")
        k = np.asarray(self.intrinsics)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
        ):
            raise ValueError("Invalid pinhole intrinsics")
        validate_rigid_pose(self.camera_to_world)
        if not self.view_id or not self.calibration_provenance:
            raise ValueError("View ID and calibration provenance are required")


def validate_rigid_pose(pose: np.ndarray) -> None:
    p = np.asarray(pose)
    if (
        p.shape != (4, 4)
        or not np.isfinite(p).all()
        or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(p[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("Pose must be a finite rigid OpenCV camera-to-world transform")


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a rigid affine transform to arbitrary (..., 3) points."""
    return np.asarray(points) @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]


def calibrated_world_points(
    depth_z: np.ndarray, k: np.ndarray, c2w: np.ndarray
) -> np.ndarray:
    """Reproject estimated optical Z using the supplied, resized calibration."""
    y, x = np.indices(depth_z.shape, dtype=np.float32)
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    camera = (pixels @ np.linalg.inv(k).T) * depth_z[..., None]
    return transform_points(camera, c2w).astype(np.float32)


def fit_camera_center_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Least-squares Sim(3), using only supplied scan-camera positions as targets.

    Reject collinear centers: they cannot determine the 3D orientation. Reflections
    are prohibited. No surface, floor, object or held-out-camera data is accepted.
    """
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Need at least three finite corresponding camera centers")
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    if np.linalg.matrix_rank(x, tol=1e-7) < 2 or np.linalg.matrix_rank(y, tol=1e-7) < 2:
        raise ValueError("Collinear scan-camera centers cannot determine Sim(3)")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = float(np.sum(singular * signs) / np.mean(np.sum(x * x, axis=1)))
    if scale <= 0:
        raise ValueError("Nonpositive camera-derived metric scale")
    translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    similarity = np.eye(4)
    similarity[:3, :3] = scale * rotation
    similarity[:3, 3] = translation
    residuals = np.linalg.norm(transform_points(source, similarity) - target, axis=1)
    return similarity, {
        "method": "Umeyama least-squares proper Sim(3), all supplied scan-camera centers, equal weights",
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "camera_center_residuals_m": residuals.tolist(),
        "camera_center_residual_p50_p95_max_m": [
            float(np.median(residuals)),
            float(np.percentile(residuals, 95)),
            float(np.max(residuals)),
        ],
        "oracle_geometry_fit": False,
        "held_out_camera_used": False,
        "interpretation": "enforces supplied metric calibration after learned inference; not evidence of metric accuracy from RGB alone",
    }


def add_pose_alignment(arrays: dict[str, np.ndarray]) -> dict:
    """Add camera-prior enforcement without modifying any raw prediction arrays."""
    known = arrays["camera_to_world_input"]
    predicted = arrays["camera_to_model_predicted"]
    similarity, details = fit_camera_center_similarity(
        predicted[:, :3, 3], known[:, :3, 3]
    )
    scale = details["scale"]
    arrays["world_from_model_similarity"] = similarity.astype(np.float32)
    arrays["points_world_model_pose_aligned"] = transform_points(
        arrays["points_model_predicted"], similarity
    ).astype(np.float32)
    arrays["depth_z_pose_scale_m"] = arrays["depth_z_m"] * np.float32(scale)
    arrays["points_world_calibrated_pose_scale"] = np.stack(
        [
            calibrated_world_points(d, k, pose)
            for d, k, pose in zip(
                arrays["depth_z_pose_scale_m"],
                arrays["intrinsics_input_processed"],
                known,
                strict=True,
            )
        ]
    )
    aligned_cameras = predicted.copy()
    aligned_cameras[:, :3, :3] = (similarity[:3, :3] / scale)[None] @ predicted[
        :, :3, :3
    ]
    aligned_cameras[:, :3, 3] = transform_points(predicted[:, :3, 3], similarity)
    arrays["camera_to_world_predicted_pose_aligned"] = aligned_cameras
    details["view_ids"] = arrays["view_ids"].tolist()
    return details


def reconstruct(
    views: list[CalibratedRGBView],
    *,
    checkpoint: Path,
    dinov2_source: Path,
    device: str = "mps",
    amp: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run official pretrained inference; download/install is deliberately external.

    Both output frames are preserved: calibrated reprojection fixes supplied K
    and poses, while model geometry is rigidly anchored by the first camera only.
    The latter retains estimated relative-pose and ray errors. No geometry fit,
    scale fit, depth hint, simulator surface, or held-out image is used.
    """
    if not views or len({v.view_id for v in views}) != len(views):
        raise ValueError("Need a nonempty set of uniquely named RGB views")
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; no silent CPU fallback")
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    if not (dinov2_source / "hubconf.py").is_file():
        raise FileNotFoundError(dinov2_source / "hubconf.py")

    original_hub_load = torch.hub.load

    def pinned_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov2":
            raise RuntimeError(f"Unexpected upstream hub dependency: {repo_or_dir}")
        kwargs["source"] = "local"
        kwargs["pretrained"] = False  # full Apache checkpoint includes encoder
        return original_hub_load(str(dinov2_source), model, *args, **kwargs)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

    started = time.perf_counter()
    with patch.object(torch.hub, "load", pinned_hub_load):
        model = MapAnything.from_pretrained(str(checkpoint), local_files_only=True)
    model = model.to(device).eval()
    sync()
    loaded = time.perf_counter()
    # Whitelist the complete model input; no scan metadata dictionary is passed.
    model_inputs = [
        {
            "img": v.rgb,
            "intrinsics": v.intrinsics.astype(np.float32),
            "camera_poses": v.camera_to_world.astype(np.float32),
            "is_metric_scale": True,
        }
        for v in views
    ]
    processed = preprocess_inputs(model_inputs, resolution_set=518, verbose=True)
    processed_k = [
        p["intrinsics"].detach().cpu().numpy().reshape(3, 3).copy() for p in processed
    ]
    preprocessed = time.perf_counter()
    with torch.inference_mode():
        predictions = model.infer(
            processed,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=amp,
            amp_dtype="fp16" if amp else "fp32",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            use_multiview_confidence=False,
            ignore_depth_inputs=True,
            ignore_calibration_inputs=False,
            ignore_pose_inputs=False,
            ignore_pose_scale_inputs=False,
        )
    sync()
    inferred = time.perf_counter()

    def cpu(key):
        return np.stack([p[key].detach().float().cpu().numpy()[0] for p in predictions])

    depth = cpu("depth_z")[..., 0]
    predicted_pose = cpu("camera_poses")
    anchor = views[0].camera_to_world @ np.linalg.inv(predicted_pose[0])
    predicted_camera = cpu("pts3d_cam")
    predicted_global = cpu("pts3d")
    valid = cpu("mask")[..., 0].astype(bool)
    valid &= np.isfinite(depth) & (depth > 0)
    confidence = cpu("conf")
    valid &= np.isfinite(confidence) & np.isfinite(predicted_camera).all(axis=-1)
    points_calibrated = np.stack(
        [
            calibrated_world_points(d, k, v.camera_to_world)
            for d, k, v in zip(depth, processed_k, views, strict=True)
        ]
    )
    arrays = {
        "view_ids": np.asarray([v.view_id for v in views]),
        "depth_z_m": depth.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "valid_mask": valid,
        "rgb_uint8": np.rint(cpu("img_no_norm").clip(0, 1) * 255).astype(np.uint8),
        "points_world_calibrated": points_calibrated,
        "points_camera_predicted": predicted_camera,
        "points_model_predicted": predicted_global,
        "points_world_model_anchored": transform_points(
            predicted_global, anchor
        ).astype(np.float32),
        "intrinsics_input_original": np.stack([v.intrinsics for v in views]).astype(
            np.float32
        ),
        "intrinsics_input_processed": np.stack(processed_k).astype(np.float32),
        "intrinsics_predicted": cpu("intrinsics"),
        "camera_to_world_input": np.stack([v.camera_to_world for v in views]).astype(
            np.float32
        ),
        "camera_to_model_predicted": predicted_pose,
        "camera_to_world_predicted_anchored": (anchor[None] @ predicted_pose).astype(
            np.float32
        ),
        "world_from_model_anchor": anchor.astype(np.float32),
        "metric_scaling_factor_predicted": cpu("metric_scaling_factor"),
    }
    metadata = {
        "device": device,
        "amp": amp,
        "amp_dtype": "fp16" if amp else "fp32",
        "torch_version": torch.__version__,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": {
            "load_model": loaded - started,
            "preprocess": preprocessed - loaded,
            "inference": inferred - preprocessed,
            "arrays_to_cpu": time.perf_counter() - inferred,
        },
        "processed_resolution_hw": list(depth.shape[1:]),
        "model_inputs": [
            "RGB",
            "synthetic metric camera-to-world poses",
            "synthetic intrinsics",
        ],
        "depth_input": False,
        "segmentation_input": False,
        "obstacle_geometry_input": False,
        "confidence_semantics": "uncalibrated learned confidence, not covariance or collision probability",
        "points_world_calibrated_semantics": "estimated optical-Z depth reprojected with supplied processed K and supplied synthetic metric poses",
        "points_world_model_anchored_semantics": "model points rigidly anchored by input C2W[0] @ inverse(predicted C2W[0]); no scale or surface fit",
        "known_calibration_conditioning": True,
        "free_space_claim": False,
        "reconstruction_accuracy_evaluated": False,
        "mask_settings": {
            "non_ambiguous": True,
            "edges": True,
            "confidence_percentile": False,
        },
    }
    if device == "mps":
        metadata["mps_end_allocated_bytes"] = torch.mps.current_allocated_memory()
        metadata["mps_end_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return arrays, metadata


def write_ply(
    path: Path, points: np.ndarray, rgb: np.ndarray, valid: np.ndarray
) -> int:
    """Write all finite, mask-valid points in binary little-endian PLY format."""
    keep = valid & np.isfinite(points).all(axis=-1)
    xyz = points[keep]
    color = rgb[keep]
    records = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for index, key in enumerate(["x", "y", "z"]):
        records[key] = xyz[:, index]
    for index, key in enumerate(["red", "green", "blue"]):
        records[key] = color[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(records)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        records.tofile(stream)
    return len(records)
