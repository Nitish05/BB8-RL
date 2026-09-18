"""RGB multiview depth search around an estimated calibrated depth prior.

Local normalized correlation supplies candidate geometry, not free volume.
Textureless, ambiguous and unsupported estimates remain rejected.
"""

import cv2
import numpy as np


def resized_intrinsics(intrinsics, original_wh, resized_wh):
    """Preserve OpenCV's pixel-center geometry under image resize."""
    original = np.asarray(original_wh, float)
    resized = np.asarray(resized_wh, float)
    if (
        original.shape != (2,)
        or resized.shape != (2,)
        or np.any(original <= 0)
        or np.any(resized <= 0)
    ):
        raise ValueError("Image dimensions must be positive width/height pairs")
    scale = resized / original
    transform = np.eye(3)
    transform[0, 0], transform[1, 1] = scale
    transform[:2, 2] = (scale - 1) / 2
    return transform @ np.asarray(intrinsics, float)


def ray_grid(shape, intrinsics):
    y, x = np.indices(shape, dtype=np.float32)
    return np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(intrinsics).T


def backproject(depth, intrinsics, camera_to_world):
    camera = ray_grid(depth.shape, intrinsics) * depth[..., None]
    return (camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]).astype(
        np.float32
    )


def resample_prior(depth, valid, u, v):
    """Bilinear depth is usable only when all contributing support is valid."""
    depth = np.asarray(depth, np.float32)
    support = np.asarray(valid, bool) & np.isfinite(depth) & (depth > 0)
    coordinates = [np.asarray(value, np.float32) for value in (u, v)]
    finite = np.isfinite(coordinates[0]) & np.isfinite(coordinates[1])
    coordinates = [
        np.where(finite, value, -1).astype(np.float32) for value in coordinates
    ]
    kwargs = {
        "interpolation": cv2.INTER_LINEAR,
        "borderMode": cv2.BORDER_CONSTANT,
        "borderValue": 0,
    }
    sampled = cv2.remap(np.where(support, depth, 0), *coordinates, **kwargs)
    weight = cv2.remap(support.astype(np.float32), *coordinates, **kwargs)
    usable = finite & (weight >= 1 - 1e-6) & np.isfinite(sampled) & (sampled > 0)
    return sampled, usable


def warped_correlation(reference, warped, valid, *, radius=2):
    """ZNCC costs plus full-patch validity and two-sided texture support."""
    kernel = (2 * radius + 1,) * 2

    def mean(value):
        return cv2.boxFilter(value, cv2.CV_32F, kernel, normalize=True)

    a, b = mean(reference), mean(warped)
    variance_a = np.maximum(mean(reference**2) - a * a, 0)
    variance_b = np.maximum(mean(warped**2) - b * b, 0)
    correlation = (mean(reference * warped) - a * b) / np.sqrt(
        variance_a * variance_b + 1e-10
    )
    supported = cv2.erode(
        valid.astype(np.uint8),
        np.ones(kernel, np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    supported &= (variance_a > (2 / 255) ** 2) & (variance_b > (2 / 255) ** 2)
    return np.where(supported, (1 - np.clip(correlation, -1, 1)) * 0.5, np.inf).astype(
        np.float32
    )


def select_depth(
    costs, prior, offsets, *, maximum_cost=0.08, minimum_gap=0.02, exclusion_bins=3
):
    """Require a distinct, interior minimum before sub-bin quadratic refinement."""
    costs, prior, offsets = np.asarray(costs), np.asarray(prior), np.asarray(offsets)
    if (
        costs.ndim != 3
        or costs.shape[1:] != prior.shape
        or costs.shape[0] != len(offsets)
        or len(offsets) < 2 * exclusion_bins + 3
        or not np.allclose(np.diff(offsets), offsets[1] - offsets[0])
        or offsets[1] <= offsets[0]
    ):
        raise ValueError("Need uniform depth offsets and aligned cost volume")
    best_index = costs.argmin(0)
    yy, xx = np.indices(prior.shape)
    best_cost = costs[best_index, yy, xx]
    alternatives = costs.copy()
    alternatives[
        abs(np.arange(len(offsets))[:, None, None] - best_index) <= exclusion_bins
    ] = np.inf
    second = alternatives.min(0)
    selected = np.isfinite(best_cost) & np.isfinite(second)
    gap = np.zeros_like(best_cost)
    np.subtract(second, best_cost, out=gap, where=selected)
    selected &= (best_cost <= maximum_cost) & (gap >= minimum_gap)
    selected &= (best_index > 0) & (best_index < len(offsets) - 1)
    previous = costs[np.maximum(best_index - 1, 0), yy, xx]
    following = costs[np.minimum(best_index + 1, len(offsets) - 1), yy, xx]
    with np.errstate(invalid="ignore", divide="ignore"):
        subpixel = 0.5 * (previous - following) / (previous - 2 * best_cost + following)
    subpixel = np.where(np.isfinite(subpixel), np.clip(subpixel, -0.5, 0.5), 0)
    refined = prior + offsets[best_index] + subpixel * (offsets[1] - offsets[0])
    selected &= np.isfinite(refined) & (refined > 0)
    return refined.astype(np.float32), selected, best_cost, second


def refine(
    reference_rgb,
    intrinsics,
    world_to_camera,
    prior_depth,
    prior_valid,
    neighbors,
    *,
    half_range_m=0.45,
    step_m=0.015,
    maximum_cost=0.08,
    minimum_gap=0.02,
):
    """Neighbors contain only RGB and calibrated camera matrices.

    Each hypothesis shifts optical-Z depth. The best two of at least three
    candidate cameras support it; correlation and uniqueness reject ambiguity.
    There is no geometry, renderer depth, object class or query-camera input.
    """
    if len(neighbors) < 3 or half_range_m <= 0 or step_m <= 0:
        raise ValueError(
            "Need at least three neighbor cameras and positive depth search"
        )
    gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    prior_depth, prior_valid = (
        np.asarray(prior_depth, np.float32),
        np.asarray(prior_valid, bool),
    )
    if gray.shape != prior_depth.shape or prior_valid.shape != gray.shape:
        raise ValueError("Reference image and depth arrays disagree")
    rays = ray_grid(gray.shape, intrinsics).astype(np.float32)
    c2w = np.linalg.inv(world_to_camera)
    prepared = []
    for neighbor in neighbors:
        other = (
            cv2.cvtColor(neighbor["rgb"], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        )
        relative = np.asarray(neighbor["world_to_camera"]) @ c2w
        prepared.append(
            (other, neighbor["intrinsics"], rays @ relative[:3, :3].T, relative[:3, 3])
        )
    count = int(np.ceil(half_range_m / step_m))
    offsets = np.arange(-count, count + 1, dtype=np.float32) * step_m
    costs = []
    for offset in offsets:
        depth = prior_depth + offset
        view_costs = []
        for other, k, rotated_rays, translation in prepared:
            camera = rotated_rays * depth[..., None] + translation
            with np.errstate(invalid="ignore", divide="ignore"):
                uv = camera @ k.T
                u, v = uv[..., 0] / uv[..., 2], uv[..., 1] / uv[..., 2]
            valid = prior_valid & (depth > 0) & (camera[..., 2] > 0)
            valid &= (
                np.isfinite(u)
                & np.isfinite(v)
                & (u >= 2)
                & (v >= 2)
                & (u < other.shape[1] - 3)
                & (v < other.shape[0] - 3)
            )
            warped = cv2.remap(
                other,
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            view_costs.append(warped_correlation(gray, warped, valid))
        ranked = np.sort(view_costs, axis=0)
        costs.append((ranked[0] + ranked[1]) * 0.5)
    costs = np.asarray(costs, np.float32)
    depth, valid, best, second = select_depth(
        costs, prior_depth, offsets, maximum_cost=maximum_cost, minimum_gap=minimum_gap
    )
    valid &= prior_valid
    return {
        "depth": depth,
        "valid": valid,
        "best_cost": best,
        "second_cost": second,
        "offsets": offsets,
    }
