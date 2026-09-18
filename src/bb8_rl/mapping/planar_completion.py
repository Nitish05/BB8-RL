"""Conservative RGB-region planar depth completion, not free-space evidence.

An accepted inverse-Z plane is an explicit geometric assumption, checked on
spatially held-out anchor tiles. Completion stays within the fit-anchor convex
hull and the same 4-connected quantized RGB region. Even a validated plane can
miss an unobserved object of the same color; callers must not interpret these
points as observed surfaces or certified free volume.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CompletionParameters:
    color_bin_width: int = 32
    tile_size: int = 8
    minimum_fit: int = 20
    minimum_validation: int = 8
    maximum_median_m: float = 0.015
    maximum_p95_m: float = 0.05
    robust_scale_m: float = 0.02

    def __post_init__(self):
        if not 1 <= self.color_bin_width <= 256 or self.tile_size < 1:
            raise ValueError("Color bin width and tile size must be positive")
        if self.minimum_fit < 20 or self.minimum_validation < 8:
            raise ValueError("Need at least 20 fit and 8 validation anchors")
        if not (
            0 < self.maximum_median_m <= 0.015
            and 0 < self.maximum_p95_m <= 0.05
            and self.robust_scale_m > 0
        ):
            raise ValueError("Validation gates may only be made stricter")


def validation_tiles(xy, tile_size=8):
    """Deterministic tile split; a tile never appears in both fit and validation."""
    xy = np.asarray(xy)
    tile = np.floor_divide(xy, tile_size).astype(np.int64)
    return (tile[:, 0] + 2 * tile[:, 1]) % 4 == 0


def _design(xy, center, scale):
    return np.column_stack(((xy - center) / scale, np.ones(len(xy))))


def _fit_plane(xy, depth, robust_scale):
    center = xy.mean(axis=0)
    centered = xy - center
    singular = np.linalg.svd(centered, compute_uv=False)
    # Reject near-lines at image pixel precision as well as exact collinearity.
    if len(singular) < 2 or singular[1] / np.sqrt(len(xy)) < 0.5:
        return None
    scale = max(float(np.max(np.abs(centered))), 1.0)
    design = _design(xy, center, scale)
    inverse = 1 / depth
    coefficients = np.linalg.lstsq(design, inverse, rcond=None)[0]
    for _ in range(10):
        prediction = design @ coefficients
        if np.any(prediction <= 0) or not np.all(np.isfinite(prediction)):
            return None
        residual = np.abs(1 / prediction - depth)
        weight = np.sqrt(np.minimum(1, robust_scale / np.maximum(residual, 1e-12)))
        updated = np.linalg.lstsq(
            design * weight[:, None], inverse * weight, rcond=None
        )[0]
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return coefficients, center, scale


def complete_depth(rgb, depth, anchor_mask, *, parameters=None):
    """Return depth, completed mask, and validation report for one RGB view.

    Trusted anchors remain bit-for-bit unchanged. Other returned depths are NaN
    unless completed. Only anchor depths are read for fitting/validation; values
    outside anchor_mask never supply geometry. Validation samples are not used
    to refit the plane after acceptance. No calibration or world truth is used.
    """
    p = parameters or CompletionParameters()
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    anchors = np.asarray(anchor_mask, bool)
    if (
        rgb.dtype != np.uint8
        or rgb.shape != (*depth.shape, 3)
        or depth.ndim != 2
        or anchors.shape != depth.shape
        or not np.issubdtype(depth.dtype, np.floating)
    ):
        raise ValueError("Need uint8 RGB and aligned floating depth / anchor mask")
    if np.any(anchors & (~np.isfinite(depth) | (depth <= 0))):
        raise ValueError("Every trusted anchor must have finite positive depth")
    result = np.full_like(depth, np.nan)
    result[anchors] = depth[anchors]
    completed = np.zeros(depth.shape, bool)
    bins = rgb.astype(np.int32) // p.color_bin_width
    base = (255 // p.color_bin_width) + 1
    color_codes = (bins[..., 0] * base + bins[..., 1]) * base + bins[..., 2]
    values, anchor_counts = np.unique(color_codes[anchors], return_counts=True)
    records = []
    skipped_color_bins = 0
    insufficient_components = 0
    for code, count in zip(values, anchor_counts, strict=True):
        if count < p.minimum_fit + p.minimum_validation:
            skipped_color_bins += 1
            continue
        region_mask = color_codes == code
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            region_mask.astype(np.uint8), connectivity=4
        )
        component_counts = np.bincount(labels[anchors & region_mask])
        for label in range(1, len(stats)):
            count = component_counts[label] if label < len(component_counts) else 0
            if count < p.minimum_fit + p.minimum_validation:
                insufficient_components += 1
                continue
            x, y, width, height, area = map(int, stats[label])
            component = labels[y : y + height, x : x + width] == label
            component_anchors = component & anchors[y : y + height, x : x + width]
            ay, ax = np.nonzero(component_anchors)
            xy = np.column_stack((ax + x, ay + y)).astype(float)
            values_depth = depth[ay + y, ax + x].astype(float)
            validation = validation_tiles(xy, p.tile_size)
            fit = ~validation
            record = {
                "color_code": int(code),
                "component": int(label),
                "bbox_xywh": [x, y, width, height],
                "region_pixels": area,
                "fit_anchors": int(fit.sum()),
                "validation_anchors": int(validation.sum()),
                "completed_pixels": 0,
            }
            records.append(record)
            if fit.sum() < p.minimum_fit or validation.sum() < p.minimum_validation:
                record["status"] = "insufficient_split_support"
                continue
            fitted = _fit_plane(xy[fit], values_depth[fit], p.robust_scale_m)
            if fitted is None:
                record["status"] = "degenerate_fit"
                continue
            coefficients, center, scale = fitted
            predicted_inverse = _design(xy[validation], center, scale) @ coefficients
            if np.any(predicted_inverse <= 0) or not np.all(
                np.isfinite(predicted_inverse)
            ):
                record["status"] = "invalid_validation_prediction"
                continue
            residual = np.abs(1 / predicted_inverse - values_depth[validation])
            median, p95 = np.percentile(residual, [50, 95])
            record.update(
                validation_median_m=float(median), validation_p95_m=float(p95)
            )
            if median > p.maximum_median_m or p95 > p.maximum_p95_m:
                record["status"] = "validation_failed"
                continue
            hull = cv2.convexHull((xy[fit] - (x, y)).astype(np.int32))
            support = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(support, hull, 1)
            candidates = component & support.astype(bool) & ~component_anchors
            cy, cx = np.nonzero(candidates)
            candidate_xy = np.column_stack((cx + x, cy + y)).astype(float)
            # OpenCV polygon rasterization may round an edge outwards. Check
            # pixel centers against every exact convex half-plane as well.
            vertices = hull[:, 0].astype(float) + (x, y)
            inside = np.ones(len(candidate_xy), bool)
            for first, second in zip(
                vertices, np.roll(vertices, -1, axis=0), strict=True
            ):
                edge = second - first
                delta = candidate_xy - first
                inside &= edge[0] * delta[:, 1] - edge[1] * delta[:, 0] >= -1e-9
            inverse = _design(candidate_xy, center, scale) @ coefficients
            usable = inside & np.isfinite(inverse) & (inverse > 0)
            result[cy[usable] + y, cx[usable] + x] = 1 / inverse[usable]
            completed[cy[usable] + y, cx[usable] + x] = True
            record.update(
                status="accepted",
                completed_pixels=int(usable.sum()),
                inverse_depth_coefficients=coefficients.tolist(),
                fit_center_xy=center.tolist(),
                fit_coordinate_scale=scale,
                fit_hull_xy=(hull[:, 0] + (x, y)).tolist(),
            )
    return (
        result,
        completed,
        {
            "parameters": asdict(p),
            "anchor_pixels": int(anchors.sum()),
            "completed_pixels": int(completed.sum()),
            "skipped_color_bins_insufficient_anchors": skipped_color_bins,
            "skipped_components_insufficient_anchors": insufficient_components,
            "regions": records,
            "planar_assumption": True,
            "free_space_certified": False,
        },
    )
