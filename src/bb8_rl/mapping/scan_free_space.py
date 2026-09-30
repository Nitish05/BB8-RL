"""Conditional free volume from a static synthetic RGB scan.

Assumptions are explicit: opaque grounded scene objects with no overhangs; a
known planar support plane; calibrated scan cameras; and no scene change since
scanning. The coarse synthetic floor-color candidate range overlaps obstacle
colors and cannot distinguish floor by itself. This is not generic or physical
free-space certification. Missing samples and unseen geometry are never free
evidence.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations, pairwise, product
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.planner import OccupancyGrid

from .contracts import Bounds3D, EvidenceSource, Provenance, SpaceState
from .room_memory import RoomMemory

EXCLUDED_QUERY_IDS = frozenset({14, 17, 20, 23})


@dataclass(frozen=True)
class ScanRGBView:
    view_id: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray

    def __post_init__(self):
        k, t = np.asarray(self.intrinsics), np.asarray(self.world_to_camera)
        if (
            not self.view_id
            or self.rgb.dtype != np.uint8
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
            or k.shape != (3, 3)
            or t.shape != (4, 4)
            or not np.isfinite(k).all()
            or not np.isfinite(t).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError("Need uint8 RGB and proper finite OpenCV scan calibration")

    @property
    def center(self):
        return np.linalg.inv(self.world_to_camera)[:3, 3]


def synthetic_floor_mask(rgb):
    """Exactly the existing SyntheticPaletteObserver floor rule, no robot clearing."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Floor classification requires uint8 RGB")
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    return (h >= 102) & (h <= 109) & (s >= 112) & (s <= 190) & (v >= 15) & (v <= 120)


def floor_homography_masks(
    views,
    *,
    floor_z=0.0,
    maximum_channel_error=6.0,
    minimum_matches=2,
    minimum_baseline_m=0.5,
    minimum_angle_degrees=30.0,
    patch_size=31,
    minimum_patch_std=3.0,
    minimum_patch_correlation=0.98,
    source_indices=None,
):
    """Reject floor-colored pixels lacking independent planar RGB agreement.

    Known scan poses induce homographies only for z=floor_z. Elevated surfaces
    generally disagree after this warp, even when their hue resembles floor.
    A 3x3 Gaussian reduces rendering/raster aliasing. By default, every pixel
    in a 31x31 informative patch must meet the color residual and its normalized
    correlation must pass. Uniform matching colors are insufficient evidence.
    patch_size=None retains the failed pointwise-only ablation for reproducibility.
    """
    if (
        maximum_channel_error <= 0
        or minimum_matches < 1
        or minimum_baseline_m <= 0
        or not 0 < minimum_angle_degrees < 180
        or (
            patch_size is not None
            and (type(patch_size) is not int or patch_size < 3 or patch_size % 2 != 1)
        )
        or minimum_patch_std <= 0
        or not 0 < minimum_patch_correlation <= 1
    ):
        raise ValueError("Invalid homography evidence gates")
    selected_sources = (
        tuple(range(len(views))) if source_indices is None else tuple(source_indices)
    )
    if len(set(selected_sources)) != len(selected_sources) or any(
        type(i) is not int or not 0 <= i < len(views) for i in selected_sources
    ):
        raise ValueError("Source indices must be distinct existing scan views")
    masks = [synthetic_floor_mask(v.rgb) for v in views]
    smooth = [cv2.GaussianBlur(v.rgb.astype(np.float32), (3, 3), 0.6) for v in views]
    homographies = [
        v.intrinsics
        @ np.c_[
            v.world_to_camera[:3, :2],
            v.world_to_camera[:3, 3] + floor_z * v.world_to_camera[:3, 2],
        ]
        for v in views
    ]
    result, counts = [], []
    for i in selected_sources:
        source = views[i]
        height, width = source.rgb.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        rays = (
            np.stack((xx, yy, np.ones_like(xx)), axis=-1)
            @ np.linalg.inv(homographies[i]).T
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = rays[..., :2] / rays[..., 2:]
        world = np.concatenate((xy, np.full((*xy.shape[:2], 1), floor_z)), axis=-1)
        a = source.center - world
        a_norm = np.linalg.norm(a, axis=-1)
        agreement = np.zeros((height, width), np.uint8)
        if patch_size is not None:
            kernel = np.ones((patch_size, patch_size), np.uint8)
            source_gray = smooth[i].mean(axis=-1)

            def average(values):
                return cv2.boxFilter(
                    values, -1, (patch_size, patch_size), borderType=cv2.BORDER_CONSTANT
                )

            mean_source = average(source_gray)
            variance_source = average(source_gray**2) - mean_source**2
        for j, target in enumerate(views):
            if (
                i == j
                or np.linalg.norm(source.center - target.center) < minimum_baseline_m
            ):
                continue
            transform = homographies[i] @ np.linalg.inv(homographies[j])
            warped = cv2.warpPerspective(
                smooth[j],
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid = (
                cv2.warpPerspective(
                    masks[j].astype(np.float32),
                    transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                >= 1 - 1e-6
            )
            b = target.center - world
            with np.errstate(divide="ignore", invalid="ignore"):
                cosine = np.sum(a * b, axis=-1) / (a_norm * np.linalg.norm(b, axis=-1))
            target_z = (
                world @ target.world_to_camera[2, :3] + target.world_to_camera[2, 3]
            )
            valid &= target_z > 0
            valid &= cosine <= np.cos(np.radians(minimum_angle_degrees))
            valid &= (
                np.max(np.abs(smooth[i] - warped), axis=-1) <= maximum_channel_error
            )
            if patch_size is not None:
                valid &= masks[i]
                # One view pair must explain the complete informative patch;
                # matching isolated constant-color pixels is insufficient.
                valid = cv2.erode(
                    valid.astype(np.uint8),
                    kernel,
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)
                target_gray = warped.mean(axis=-1)
                mean_target = average(target_gray)
                variance_target = average(target_gray**2) - mean_target**2
                covariance = (
                    average(source_gray * target_gray) - mean_source * mean_target
                )
                correlation = covariance / np.sqrt(
                    np.maximum(variance_source * variance_target, 1e-12)
                )
                valid &= (variance_source >= minimum_patch_std**2) & (
                    variance_target >= minimum_patch_std**2
                )
                valid &= correlation >= minimum_patch_correlation
            agreement += valid
        source_z = world @ source.world_to_camera[2, :3] + source.world_to_camera[2, 3]
        result.append(masks[i] & (agreement >= minimum_matches) & (source_z > 0))
        counts.append(agreement)
    return result, counts


def prism_floor_support(
    view,
    low_xy,
    high_xy,
    *,
    floor_z,
    top_z,
    pixel_guard=1,
    floor_mask=None,
    projection_shape="rectangle",
    evidence_mode="full_prism",
):
    """Every pixel in a conservative projected-prism rectangle must be floor.

    Eight 3D corners bound a convex prism wholly in front of the camera. Their
    image bounds contain its entire perspective projection. Pixel bounds round
    outward and include an explicit extra pixel guard. No corner-only sampling.

    The opt-in grounded_column mode tests only the four floor corners. It is
    footprint evidence, not observed free volume; the caller must separately
    declare that every occupied column extends continuously to the floor.
    """
    if top_z <= floor_z or type(pixel_guard) is not int or pixel_guard < 0:
        raise ValueError(
            "Need positive prism height and nonnegative integer pixel guard"
        )
    if evidence_mode not in {"full_prism", "grounded_column"}:
        raise ValueError("Unknown scan free-evidence mode")
    low_xy, high_xy = np.asarray(low_xy, float), np.asarray(high_xy, float)
    if low_xy.shape != high_xy.shape or low_xy.ndim != 2 or low_xy.shape[1] != 2:
        raise ValueError("Cell bounds must be Nx2 arrays")
    corners = []
    heights = (0, 1) if evidence_mode == "full_prism" else (0,)
    for x, y, z in product((0, 1), (0, 1), heights):
        corners.append(
            np.c_[
                np.where(x, high_xy[:, 0], low_xy[:, 0]),
                np.where(y, high_xy[:, 1], low_xy[:, 1]),
                np.full(len(low_xy), top_z if z else floor_z),
            ]
        )
    corners = np.stack(corners, axis=1)
    t, k = np.asarray(view.world_to_camera), np.asarray(view.intrinsics)
    camera = corners @ t[:3, :3].T + t[:3, 3]
    forward = np.isfinite(camera).all(axis=(1, 2)) & (camera[..., 2] > 1e-7).all(axis=1)
    homogeneous = camera @ k.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[..., :2] / homogeneous[..., 2:]
    finite = np.isfinite(uv).all(axis=(1, 2))
    uv = np.where(np.isfinite(uv), uv, -1e6)
    if projection_shape not in {"rectangle", "convex_hull"}:
        raise ValueError("Unknown projected volume coverage method")
    extra = 1 if projection_shape == "convex_hull" else 0
    low = np.floor(uv.min(axis=1)).astype(np.int64) - pixel_guard - extra
    high = np.ceil(uv.max(axis=1)).astype(np.int64) + pixel_guard + extra
    h, w = view.rgb.shape[:2]
    inside = forward & finite & (low >= 0).all(axis=1) & (high < [w, h]).all(axis=1)
    low = np.maximum(low, 0)
    high = np.minimum(high, [w - 1, h - 1])
    result = np.zeros(len(low_xy), bool)
    selected = np.flatnonzero(inside)
    x0, y0 = low[selected].T
    x1, y1 = (high[selected] + 1).T
    floor_mask = (
        synthetic_floor_mask(view.rgb) if floor_mask is None else np.asarray(floor_mask)
    )
    if floor_mask.dtype != bool or floor_mask.shape != view.rgb.shape[:2]:
        raise ValueError("Floor evidence mask must be boolean at native RGB resolution")
    integral = cv2.integral((~floor_mask).astype(np.uint8), sdepth=cv2.CV_32S)
    nonfloor = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    result[selected] = nonfloor == 0
    if projection_shape == "convex_hull":
        # Separating-axis theorem tests each nonfloor pixel square against the
        # exact convex projection. Expand pixel squares by the explicit guard;
        # no polygon raster rounding can omit a touched pixel.
        half_pixel = 0.5 + pixel_guard
        for index in selected[nonfloor != 0]:
            x0, y0 = low[index]
            x1, y1 = high[index] + 1
            y, x = np.nonzero(~floor_mask[y0:y1, x0:x1])
            pixels = np.c_[x + x0, y + y0]
            hull = (
                cv2.convexHull(uv[index].astype(np.float32))
                .reshape(-1, 2)
                .astype(float)
            )
            edges = np.roll(hull, -1, axis=0) - hull
            axes = np.vstack((np.eye(2), np.c_[-edges[:, 1], edges[:, 0]]))
            projected_hull = hull @ axes.T
            projected_pixels = pixels @ axes.T
            radius = half_pixel * np.abs(axes).sum(axis=1)
            epsilon = 1e-4 * np.linalg.norm(axes, axis=1)
            overlap = (
                projected_pixels + radius >= projected_hull.min(axis=0) - epsilon
            ) & (projected_pixels - radius <= projected_hull.max(axis=0) + epsilon)
            result[index] = not np.any(overlap.all(axis=1))
    return result


def _memory_planar_states(memory, extent, resolution, top_z):
    """Query voxel interiors; capsule queries separately include cell boundaries."""
    n = round(2 * extent / resolution)
    free = np.zeros((n, n), bool)
    occupied = free.copy()
    epsilon = min(1e-8, resolution * 1e-5)
    for y, x in product(range(n), repeat=2):
        low = np.array(
            [
                -extent + x * resolution,
                -extent + y * resolution,
                memory.bounds.minimum[2],
            ]
        )
        high = low + [resolution, resolution, top_z - low[2]]
        state = memory.query(Bounds3D(tuple(low + epsilon), tuple(high - epsilon)))
        free[y, x] = state is SpaceState.FREE
        occupied[y, x] = state is SpaceState.OCCUPIED
    return free, occupied


def capsule_free(free_mask, extent, resolution, start, end, radius_m):
    """Exact 2D capsule intersection with closed unknown/occupied cell rectangles."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    if (
        a.shape != (2,)
        or b.shape != (2,)
        or not np.isfinite(np.r_[a, b, radius_m]).all()
        or radius_m < 0
    ):
        return False
    if np.any(np.minimum(a, b) - radius_m <= -extent) or np.any(
        np.maximum(a, b) + radius_m >= extent
    ):
        return False
    cells = np.argwhere(~np.asarray(free_mask, bool))[:, ::-1]
    lower = -extent + cells * resolution
    upper = lower + resolution
    intersects_roi = (upper >= np.minimum(a, b) - radius_m).all(axis=1) & (
        lower <= np.maximum(a, b) + radius_m
    ).all(axis=1)
    lower, upper = lower[intersects_roi], upper[intersects_roi]
    if not len(lower):
        return True
    direction = b - a
    lo, hi, crossed = (
        np.zeros(len(lower)),
        np.ones(len(lower)),
        np.ones(len(lower), bool),
    )
    for axis in (0, 1):
        if abs(direction[axis]) < 1e-14:
            crossed &= (a[axis] >= lower[:, axis]) & (a[axis] <= upper[:, axis])
        else:
            t1, t2 = (
                (lower[:, axis] - a[axis]) / direction[axis],
                (upper[:, axis] - a[axis]) / direction[axis],
            )
            lo = np.maximum(lo, np.minimum(t1, t2))
            hi = np.minimum(hi, np.maximum(t1, t2))
    if np.any(crossed & (lo <= hi + 1e-12)):
        return False
    distance_sq = np.full(len(lower), np.inf)
    for endpoint in (a, b):
        delta = np.maximum(np.maximum(lower - endpoint, endpoint - upper), 0)
        distance_sq = np.minimum(distance_sq, np.sum(delta * delta, axis=1))
    length_sq = direction @ direction
    for x, y in product((0, 1), repeat=2):
        corner = np.c_[
            upper[:, 0] if x else lower[:, 0], upper[:, 1] if y else lower[:, 1]
        ]
        fraction = (
            np.clip((corner - a) @ direction / length_sq, 0, 1)
            if length_sq > 1e-20
            else np.zeros(len(corner))
        )
        delta = corner - (a + fraction[:, None] * direction)
        distance_sq = np.minimum(distance_sq, np.sum(delta * delta, axis=1))
    return not np.any(distance_sq <= radius_m**2 + 1e-12)


class ScanFreeMemory:
    def __init__(
        self,
        memory,
        free_mask,
        candidate_free_mask,
        occupied_mask,
        support_bits,
        view_ids,
        metadata,
    ):
        self.memory = memory
        self.free_mask = np.array(free_mask, bool, copy=True)
        self.candidate_free_mask = np.array(candidate_free_mask, bool, copy=True)
        self.occupied_mask = np.array(occupied_mask, bool, copy=True)
        self.view_ids = tuple(view_ids)
        raw_bits = np.asarray(support_bits)
        if (
            len(self.view_ids) > 64
            or raw_bits.dtype.kind not in "ui"
            or np.any(raw_bits < 0)
            or (
                len(self.view_ids) < 64
                and np.any(raw_bits > (1 << len(self.view_ids)) - 1)
            )
        ):
            raise ValueError(
                "Support bits must refer only to at most64 declared scan views"
            )
        bit_type = np.uint32 if len(self.view_ids) <= 32 else np.uint64
        self.support_bits = np.array(raw_bits, bit_type, copy=True)
        self.metadata = dict(metadata)
        self.extent = float(metadata["extent_m"])
        self.resolution = float(metadata["resolution_m"])
        self.body_height_m = float(metadata["certified_height_m"])
        self.floor_z = float(metadata["floor_z_m"])
        expected = round(2 * self.extent / self.resolution)
        if (
            self.free_mask.shape != (expected, expected)
            or self.support_bits.shape != self.free_mask.shape
            or self.candidate_free_mask.shape != self.free_mask.shape
            or self.occupied_mask.shape != self.free_mask.shape
            or len(self.view_ids) > 64
            or len(set(self.view_ids)) != len(self.view_ids)
            or np.any(self.free_mask & (~self.candidate_free_mask | self.occupied_mask))
        ):
            raise ValueError("Inconsistent free-memory raster or provenance")
        for array in (
            self.free_mask,
            self.candidate_free_mask,
            self.occupied_mask,
            self.support_bits,
        ):
            array.flags.writeable = False

    @property
    def grid(self):
        return self.planning_grid(0.0)

    def planning_grid(self, radius_m):
        if not math.isfinite(radius_m) or radius_m < 0:
            raise ValueError("Planning radius must be finite and nonnegative")
        grid = OccupancyGrid(self.extent, self.resolution, radius_m, [])
        blocked = (
            ~self.free_mask
            if self.memory.valid
            else np.ones(self.free_mask.shape, bool)
        )
        if radius_m:
            reach = int(np.ceil(radius_m / self.resolution)) + 1
            offset = np.arange(-reach, reach + 1)
            x, y = np.meshgrid(offset, offset)
            distance = self.resolution * np.hypot(
                np.maximum(abs(x) - 1, 0), np.maximum(abs(y) - 1, 0)
            )
            kernel = (distance <= radius_m + 1e-12).astype(np.uint8)
            blocked = cv2.dilate(
                blocked.astype(np.uint8),
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=1,
            ).astype(bool)
        grid.blocked = blocked.copy()
        return grid

    def segment_free(self, start, end, radius_m):
        return bool(
            self.memory.valid
            and capsule_free(
                self.free_mask, self.extent, self.resolution, start, end, radius_m
            )
        )

    def certified_route(self, start, goal, radius_m, *, grid=None):
        """Keep conservative grid search, but certify endpoint connectors metrically.

        A literal endpoint can touch an inflated blocked-cell rectangle while
        its full physical footprint remains inside certified free space. Only
        this memory's exact capsule predicate may admit that connector; the
        grid's free-node, diagonal and shortcut checks remain unchanged.
        """
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if (
            not self.memory.valid
            or start.shape != (2,)
            or goal.shape != (2,)
            or not np.isfinite(start).all()
            or not np.isfinite(goal).all()
            or not math.isfinite(radius_m)
            or radius_m < 0
        ):
            raise ValueError("Route needs valid memory, finite endpoints and clearance")
        if grid is None:
            grid = self.planning_grid(radius_m)
        else:
            if (
                not isinstance(grid, OccupancyGrid)
                or grid.extent != self.extent
                or grid.resolution != self.resolution
                or grid.n != self.free_mask.shape[0]
                or not math.isfinite(grid.inflation)
                or grid.inflation < radius_m
                or not isinstance(grid.blocked, np.ndarray)
                or grid.blocked.dtype != np.dtype(bool)
                or grid.blocked.shape != self.free_mask.shape
            ):
                raise ValueError(
                    "Cached grid geometry or clearance does not match memory"
                )
            if not np.array_equal(
                grid.blocked, self.planning_grid(grid.inflation).blocked
            ):
                raise ValueError(
                    "Cached grid does not match this memory's conservative mask"
                )
        source, target = grid.cell(start), grid.cell(goal)
        if not grid.free(source) or not grid.free(target):
            raise ValueError("Start or goal is blocked after footprint inflation")
        if not self.segment_free(start, start, radius_m) or not self.segment_free(
            goal, goal, radius_m
        ):
            raise ValueError("Route endpoint lacks the full requested clearance")
        try:
            # Preserve existing successful routes, including their waypoints.
            route = grid.route(start, goal)
        except ValueError:
            route = grid.route(grid.point(source), grid.point(target))
        # Preserve literal endpoints exactly, even if grid smoothing discarded a
        # sub-nanometer displacement. Only exact duplicate waypoints vanish.
        points = [start, *route, goal]
        route = np.asarray(
            [points[0]] + [b for a, b in pairwise(points) if not np.array_equal(a, b)]
        )
        if not all(self.segment_free(a, b, radius_m) for a, b in pairwise(route)):
            raise ValueError("Planned route lacks the full requested clearance")
        return route

    def supporting_views(self, cell_yx):
        bits = int(self.support_bits[tuple(cell_yx)])
        return tuple(view for i, view in enumerate(self.view_ids) if bits & (1 << i))

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.memory.save(output / "room-memory.json")
        np.savez_compressed(
            output / "free-grid.npz",
            free_mask=self.free_mask,
            candidate_free_mask=self.candidate_free_mask,
            occupied_mask=self.occupied_mask,
            support_bits=self.support_bits,
            view_ids=np.asarray(self.view_ids),
            extent=self.extent,
            resolution=self.resolution,
            body_height_m=self.body_height_m,
            floor_z=self.floor_z,
        )
        manifest = dict(self.metadata)
        manifest["artifact_sha256"] = {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("room-memory.json", "free-grid.npz")
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def load(cls, output):
        output = Path(output)
        metadata = json.loads((output / "manifest.json").read_text())
        for name, expected in metadata["artifact_sha256"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Free-memory artifact hash mismatch")
        memory = RoomMemory.load(output / "room-memory.json")
        with np.load(output / "free-grid.npz", allow_pickle=False) as arrays:
            result = cls(
                memory,
                arrays["free_mask"],
                arrays["candidate_free_mask"],
                arrays["occupied_mask"],
                arrays["support_bits"],
                arrays["view_ids"].tolist(),
                metadata,
            )
        free, occupied = _memory_planar_states(
            memory,
            result.extent,
            result.resolution,
            result.floor_z + result.body_height_m,
        )
        if memory.valid and (
            not np.array_equal(free, result.free_mask)
            or not np.array_equal(occupied, result.occupied_mask)
        ):
            raise ValueError(
                "Serialized raster disagrees with retained volume evidence"
            )
        return result


def refine_occupied_conflicts(
    points_by_view,
    valid_masks,
    view_ids,
    *,
    extent=2.0,
    resolution_m=0.02,
    height_m=0.30,
    uncertainty_m=0.05,
    support_distance_m=0.05,
    minimum_views=3,
):
    """Retain collision-height surfaces supported by independent RGB clouds.

    This never produces FREE evidence. Rejected/above-scope samples remain in
    the immutable source cloud. The assumed .05m margin is retained after finer
    voxelization; radius support is an estimate-consistency check, not truth.
    """
    from scipy.spatial import cKDTree

    from .surface_memory import import_surfaces

    points, masks = np.asarray(points_by_view), np.asarray(valid_masks)
    if (
        points.ndim != 4
        or masks.shape != points.shape[:-1]
        or masks.dtype != bool
        or len(view_ids) != len(points)
    ):
        raise ValueError("Need aligned view clouds and validity masks")
    if minimum_views < 3 or support_distance_m <= 0 or uncertainty_m < 0.05:
        raise ValueError("Require three-view support and preserve assumed .05m margin")
    if any(
        int(view[5:]) in EXCLUDED_QUERY_IDS
        for view in view_ids
        if view.startswith("scan-")
    ):
        raise ValueError("Frozen query cannot enter occupied refinement")
    candidate = masks & np.isfinite(points).all(axis=-1)
    candidate &= (np.abs(points[..., :2]) < extent).all(axis=-1)
    candidate &= (points[..., 2] >= 0.04) & (points[..., 2] < height_m)
    clouds = [cloud[mask] for cloud, mask in zip(points, candidate)]
    all_points = np.concatenate(clouds)
    support = np.zeros(len(all_points), np.uint8)
    for cloud in clouds:
        if len(cloud):
            distances = cKDTree(cloud).query(
                all_points, distance_upper_bound=support_distance_m, workers=1
            )[0]
            support += distances <= support_distance_m
    accepted = candidate.copy()
    support_by_pixel = np.zeros(masks.shape, np.uint8)
    start = 0
    for index, cloud in enumerate(clouds):
        stop = start + len(cloud)
        support_by_pixel[index][candidate[index]] = support[start:stop]
        accepted[index][candidate[index]] = support[start:stop] >= minimum_views
        start = stop
    memory, report = import_surfaces(
        points,
        accepted,
        view_ids,
        bounds=Bounds3D((-extent, -extent, 0.0), (extent, extent, height_m)),
        resolution_m=resolution_m,
        uncertainty_m=uncertainty_m,
        scene_version="m78-multiview-collision-height-surfaces",
        calibration_version="synthetic-scan-poses-metric-registration",
    )
    report.update(
        {
            "candidate_body_height_samples": int(candidate.sum()),
            "minimum_distinct_views": minimum_views,
            "support_distance_m": support_distance_m,
            "resolution_m": resolution_m,
            "height_scope_m": height_m,
            "support_rule": "at least one reconstructed point in each of three distinct source views within the fixed metric radius, including own view",
            "rejected_samples_create_free": False,
            "query_inputs_used": False,
        }
    )
    return memory, report, accepted, support_by_pixel


def build_scan_free_memory(
    views,
    *,
    extent=2.0,
    resolution=0.05,
    body_height_m=0.12,
    floor_z=0.0,
    projection_margin_m=0.015,
    pixel_guard=1,
    minimum_views=2,
    minimum_baseline_m=0.5,
    minimum_angle_degrees=15.0,
    occupied_memory=None,
    floor_masks=None,
    projection_shape="rectangle",
    evidence_mode="full_prism",
    assume_floor_connected_columns=False,
    scene_version="m78-original-static-room",
    calibration_version="synthetic-scan20-calibration",
):
    """Produce raw, uninflated planar freedom from positive multiview RGB evidence."""
    values = [
        extent,
        resolution,
        body_height_m,
        floor_z,
        projection_margin_m,
        minimum_baseline_m,
        minimum_angle_degrees,
    ]
    if (
        not np.isfinite(values).all()
        or min(extent, resolution, body_height_m) <= 0
        or projection_margin_m < 0
        or minimum_baseline_m <= 0
        or not 0 < minimum_angle_degrees < 180
        or type(minimum_views) is not int
        or minimum_views not in (2, 3, 4)
        or len(views) > 64
        or len({v.view_id for v in views}) != len(views)
    ):
        raise ValueError("Invalid scan support or task geometry parameters")
    if evidence_mode not in {"full_prism", "grounded_column"}:
        raise ValueError("Unknown scan free-evidence mode")
    if evidence_mode == "grounded_column" and (
        assume_floor_connected_columns is not True
        or floor_masks is None
        or minimum_views != 3
        or projection_margin_m < 0.05
        or pixel_guard < 1
    ):
        raise ValueError(
            "Grounded-column evidence needs an explicit floor-connected-column "
            "assumption, supplied floor masks, three views, .05m XY and pixel guards"
        )
    if any(
        v.view_id.startswith("scan-") and int(v.view_id[5:]) in EXCLUDED_QUERY_IDS
        for v in views
    ):
        raise ValueError("Frozen query cannot contribute scan free-volume evidence")
    grid = OccupancyGrid(extent, resolution, 0.0, [])
    certified_height = math.ceil(body_height_m / resolution - 1e-10) * resolution
    top_z = floor_z + certified_height
    if occupied_memory is None:
        memory = RoomMemory(
            Bounds3D((-extent, -extent, floor_z), (extent, extent, top_z)),
            resolution,
            scene_version=scene_version,
            calibration_version=calibration_version,
        )
    else:
        if (
            not occupied_memory.valid
            or any(e.state is not SpaceState.OCCUPIED for e in occupied_memory.evidence)
            or not np.allclose(
                occupied_memory.bounds.minimum, [-extent, -extent, floor_z]
            )
            or not np.allclose(occupied_memory.bounds.maximum[:2], [extent, extent])
            or occupied_memory.bounds.maximum[2] < top_z
            or not np.isclose(occupied_memory.resolution_m, resolution)
        ):
            raise ValueError(
                "Conflict layer must be compatible, valid and occupied-only"
            )
        memory = RoomMemory.from_dict(occupied_memory.to_dict())
    yy, xx = np.indices((grid.n, grid.n))
    low = np.c_[-extent + xx.ravel() * resolution, -extent + yy.ravel() * resolution]
    high = low + resolution
    centers = np.c_[(low + high) / 2, np.full(len(low), floor_z + certified_height / 2)]
    support = np.zeros((len(views), len(low)), bool)
    if floor_masks is not None and len(floor_masks) != len(views):
        raise ValueError("Need one floor mask per scan view")
    for i, view in enumerate(views):
        support[i] = prism_floor_support(
            view,
            low - projection_margin_m,
            high + projection_margin_m,
            floor_z=floor_z,
            top_z=top_z + projection_margin_m,
            pixel_guard=pixel_guard,
            floor_mask=None if floor_masks is None else floor_masks[i],
            projection_shape=projection_shape,
            evidence_mode=evidence_mode,
        )
    separated = np.zeros(len(low), bool)
    compatible_pairs = {}
    for i in range(len(views)):
        for j in range(i):
            a, b = views[i].center, views[j].center
            if np.linalg.norm(a - b) < minimum_baseline_m:
                continue
            ra, rb = a - centers, b - centers
            cosine = np.sum(ra * rb, axis=1) / (
                np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
            )
            compatible_pairs[(j, i)] = (
                support[i]
                & support[j]
                & (cosine <= np.cos(np.radians(minimum_angle_degrees)))
            )
    for subset in combinations(range(len(views)), minimum_views):
        pairs = list(combinations(subset, 2))
        if not all(pair in compatible_pairs for pair in pairs):
            continue
        separated |= np.logical_and.reduce([compatible_pairs[pair] for pair in pairs])
    candidate = (support.sum(axis=0) >= minimum_views) & separated
    bit_type = np.uint32 if len(views) <= 32 else np.uint64
    bits = np.zeros(len(low), bit_type)
    for i in range(len(views)):
        bits |= support[i].astype(bit_type) * bit_type(1 << i)
    epsilon = 1e-9
    for cell in np.flatnonzero(candidate):
        ids = tuple(view.view_id for i, view in enumerate(views) if support[i, cell])
        source = EvidenceSource(
            f"scan-floor-prism-{cell}", ids, 0.0, Provenance.RGB_RECONSTRUCTION, 0.0
        )
        memory.observe_volume(
            Bounds3D(
                (low[cell, 0] - epsilon, low[cell, 1] - epsilon, floor_z),
                (high[cell, 0] + epsilon, high[cell, 1] + epsilon, top_z + epsilon),
            ),
            SpaceState.FREE,
            source,
        )
    free, occupied = _memory_planar_states(memory, extent, resolution, top_z)
    metadata = {
        "schema": "bb8.scan-free-memory.v1",
        "status": "complete",
        "extent_m": extent,
        "resolution_m": resolution,
        "floor_z_m": floor_z,
        "requested_body_height_m": body_height_m,
        "certified_height_m": certified_height,
        "projection_margin_m": projection_margin_m,
        "pixel_guard": pixel_guard,
        "projection_shape": projection_shape,
        "free_evidence_mode": evidence_mode,
        "minimum_views": minimum_views,
        "support_separation_rule": "all pairs in a supporting camera clique satisfy baseline and per-cell angle gates",
        "minimum_baseline_m": minimum_baseline_m,
        "minimum_angle_degrees": minimum_angle_degrees,
        "mapping_input_view_ids": [v.view_id for v in views],
        "excluded_query_ids": sorted(EXCLUDED_QUERY_IDS),
        "candidate_free_cells": int(candidate.sum()),
        "free_cells": int(free.sum()),
        "occupied_cells": int(occupied.sum()),
        "unknown_cells": int((~free & ~occupied).sum()),
        "free_candidates_blocked_by_occupied": int(
            (candidate.reshape(free.shape) & occupied).sum()
        ),
        "robot_radius_inflation_m": 0.0,
        "uncertainty_status": "projection/pixel guards are explicit assumptions, not calibrated physical coverage",
        "floor_classifier": "supplied native-resolution positive floor masks"
        if floor_masks is not None
        else "existing SyntheticPaletteObserver HSV floor rule only; robot pixels never cleared",
        "assumptions": [
            "static original room",
            "known planar floor and scan K/poses",
            "known synthetic color candidates overlap obstacle colors; palette alone does not establish floor",
            "opaque grounded scene objects with no overhangs",
            "no scene change since scan",
        ],
        "free_evidence_semantics": (
            "all pixels in expanded ground-footprint projection observed as floor "
            "in three pairwise-separated cameras; body-height freedom inferred "
            "only under the explicit floor-connected-column assumption"
            if evidence_mode == "grounded_column"
            else "all pixels in whole expanded body-prism projection superset "
            "observed as floor in the declared minimum separated cameras"
        ),
        "missing_geometry_creates_free": False,
        "physical_or_general_scene_guarantee": False,
        "occupied_conflict_layer_preserved": occupied_memory is not None,
    }
    if evidence_mode == "grounded_column":
        metadata["assumptions"].append(
            "For every occupied point above the support plane, the entire vertical "
            "column down to that plane is occupied. This stronger assumption "
            "holds for vertical solid fixture boxes, not arbitrary grounded "
            "objects, rounded bodies, overhangs, or suspended objects."
        )
        metadata["floor_connected_columns_assumed"] = True
        metadata["body_volume_observed_directly"] = False
    return ScanFreeMemory(
        memory,
        free,
        candidate.reshape(free.shape),
        occupied,
        bits.reshape(free.shape),
        [v.view_id for v in views],
        metadata,
    )
