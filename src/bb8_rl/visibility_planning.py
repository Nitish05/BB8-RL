"""Bounded route candidates from scanned occluders and registered cameras only.

Predicted visibility is a heuristic: the scan may omit high geometry, calibration
has uncalibrated error, and a clear ray does not guarantee learned RGB detection.
Footprint clearance remains a separate mandatory certificate. No native world,
render labels, evaluation geometry, model inference or actuator calls enter here.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from itertools import pairwise, product

import cv2
import numpy as np

from .mapping.contracts import SpaceState

_LIMITATIONS = [
    "Stored scan geometry may omit obstacles above the reconstructed height range.",
    "Registered camera uncertainty and learned detector failures are not calibrated here.",
    "Blind seconds equal path distance divided by reference speed; slower motion or dwell can take longer.",
    "Predicted visibility is not certified line of sight or permission to relax localization bounds.",
]
_NEIGHBORS = ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))


class VisibilityPlanner:
    """Frozen head-plane shadows plus A* with a finite blind-distance budget.

    ``grid`` is supplied by the caller at its unchanged required clearance.
    A rejected visibility candidate must not fall back to an unrestricted route.
    Default zero blind distance requests an entirely predicted-visible route.
    """

    def __init__(
        self,
        memory,
        calibrations,
        *,
        max_blind_distance_m=0.0,
        reference_speed_m_s=0.10,
        head_radius_m=0.019,
        pixel_guard=2,
        max_expansions=200_000,
        max_geometry_boxes=20_000,
    ):
        values = (max_blind_distance_m, reference_speed_m_s, head_radius_m)
        if (
            not np.isfinite(values).all()
            or not 0 <= max_blind_distance_m <= 1
            or not 0 < reference_speed_m_s <= 0.35
            or not 0 <= head_radius_m <= 0.10
            or type(pixel_guard) is not int
            or not 0 <= pixel_guard <= 32
            or type(max_expansions) is not int
            or not 1 <= max_expansions <= 500_000
            or type(max_geometry_boxes) is not int
            or not 1 <= max_geometry_boxes <= 50_000
            or not isinstance(calibrations, dict)
            or not 1 <= len(calibrations) <= 3
            or any(not isinstance(name, str) or not name for name in calibrations)
        ):
            raise ValueError(
                "Visibility planning requires bounded geometry, cameras and budgets"
            )
        self.memory, self.calibrations = memory, calibrations
        self.extent, self.resolution = float(memory.extent), float(memory.resolution)
        if (
            not np.isfinite([self.extent, self.resolution]).all()
            or min(self.extent, self.resolution) <= 0
        ):
            raise ValueError("Visibility grid requires positive finite geometry")
        self.n = round(2 * self.extent / self.resolution)
        if not 2 <= self.n <= 256 or not math.isclose(
            self.n * self.resolution, 2 * self.extent
        ):
            raise ValueError("Visibility grid must contain 2--256 cells per axis")
        if not memory.memory.valid:
            raise ValueError("Visibility planning requires valid frozen scan memory")
        self.max_blind_distance_m = float(max_blind_distance_m)
        self.reference_speed_m_s = float(reference_speed_m_s)
        self.head_radius_m, self.pixel_guard = float(head_radius_m), pixel_guard
        self.max_expansions = max_expansions
        self._context = self._context_signature()
        bounds = []
        evidence_records, object_records = memory.memory.evidence, memory.memory.objects
        if len(evidence_records) + len(object_records) > 200_000:
            raise ValueError("Visibility evidence exceeds the bounded input budget")
        # Occupied sources already declare uncertainty. FREE support is used by
        # the independent clearance map, never to infer that a high ray is clear.
        for evidence in evidence_records:
            if evidence.state is SpaceState.OCCUPIED:
                bounds.append(evidence.bounds.expanded(evidence.source.uncertainty_m))
        for obj in object_records:
            bounds.append(obj.bounds.expanded(obj.source.uncertainty_m))
        if len(bounds) > max_geometry_boxes:
            raise ValueError("Visibility geometry exceeds the bounded box budget")
        self._bounds = bounds
        self.visibility_by_camera = {
            name: self._camera_visibility(calibrations[name])
            for name in sorted(calibrations)
        }
        self.visible = np.logical_or.reduce(list(self.visibility_by_camera.values()))
        self.visible.flags.writeable = False
        self.diagnostics = self._base_diagnostics()

    def _context_signature(self):
        digest = hashlib.sha256()
        for name, calibration in sorted(self.calibrations.items()):
            digest.update(name.encode())
            for value in (
                calibration.intrinsics,
                calibration.world_to_camera,
                calibration.resolution,
                calibration.head_height,
            ):
                digest.update(np.asarray(value, dtype=float).tobytes())
        room = self.memory.memory
        return (
            room.map_version,
            room.scene_version,
            room.calibration_version,
            digest.hexdigest(),
        )

    def _base_diagnostics(self):
        return {
            "enabled": True,
            "kind": "scan_head_plane_visibility_heuristic",
            "status": "ready",
            "camera_ids": sorted(self.calibrations),
            "max_blind_distance_m": self.max_blind_distance_m,
            "reference_speed_m_s": self.reference_speed_m_s,
            "predicted_blind_budget_s": self.max_blind_distance_m
            / self.reference_speed_m_s,
            "geometry_boxes": len(self._bounds),
            "grid_cells": self.n**2,
            "predicted_visible_cells": int(self.visible.sum()),
            "predicted_visible_cells_by_camera": {
                name: int(mask.sum())
                for name, mask in self.visibility_by_camera.items()
            },
            "scan_vertical_bounds_m": [
                self.memory.memory.bounds.minimum[2],
                self.memory.memory.bounds.maximum[2],
            ],
            "map_version": self._context[0],
            "calibration_digest": self._context[3],
            "visibility_certified": False,
            "clearance_certified": False,
            "limitations": list(_LIMITATIONS),
        }

    def _camera_visibility(self, calibration):
        k, t = (
            np.asarray(calibration.intrinsics, float),
            np.asarray(calibration.world_to_camera, float),
        )
        if (
            k.shape != (3, 3)
            or t.shape != (4, 4)
            or not np.isfinite(k).all()
            or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-5)
        ):
            raise ValueError(
                "Visibility planning requires a finite registered pinhole camera"
            )
        width, height = calibration.resolution
        head_z = float(calibration.head_height)
        if min(width, height) <= 2 * self.pixel_guard or not math.isfinite(head_z):
            raise ValueError(
                "Visibility camera dimensions or target height are invalid"
            )
        center = -t[:3, :3].T @ t[:3, 3]
        # This bounded shadow construction supports elevated cameras. Reject an
        # unsupported camera/box relation conservatively instead of ignoring it.
        if center[2] <= head_z + self.head_radius_m:
            return np.zeros((self.n, self.n), bool)
        axis = -self.extent + (np.arange(self.n) + 0.5) * self.resolution
        x, y = np.meshgrid(axis, axis)
        visible = np.ones((self.n, self.n), bool)
        envelope = self.head_radius_m + self.resolution / math.sqrt(2)
        for dx, dy, dz in product((-1, 1), repeat=3):
            world = np.stack(
                (
                    x + dx * envelope,
                    y + dy * envelope,
                    np.full_like(x, head_z + dz * self.head_radius_m),
                ),
                axis=-1,
            )
            camera = world @ t[:3, :3].T + t[:3, 3]
            pixels = camera @ k.T
            front = pixels[..., 2] > 1e-8
            pixel_xy = pixels[..., :2] / np.maximum(pixels[..., 2:], 1e-8)
            visible &= (
                front
                & (pixel_xy[..., 0] >= self.pixel_guard)
                & (pixel_xy[..., 0] <= width - 1 - self.pixel_guard)
                & (pixel_xy[..., 1] >= self.pixel_guard)
                & (pixel_xy[..., 1] <= height - 1 - self.pixel_guard)
            )
        shadow = np.zeros((self.n, self.n), np.uint8)
        for box in self._bounds:
            expanded = box.expanded(self.head_radius_m)
            low, high = np.array(expanded.minimum), np.array(expanded.maximum)
            if high[2] < head_z:
                continue
            if high[2] >= center[2] - 1e-6:
                return np.zeros((self.n, self.n), bool)
            low[2] = max(low[2], head_z)
            corners = np.array(list(product(*zip(low, high))))
            scale = (head_z - center[2]) / (corners[:, 2] - center[2])
            projected = center[:2] + scale[:, None] * (corners[:, :2] - center[:2])
            pixel = (projected + self.extent) / self.resolution - 0.5
            if not np.isfinite(pixel).all() or np.max(abs(pixel)) > 1_000_000:
                return np.zeros((self.n, self.n), bool)
            polygon = cv2.convexHull(np.rint(pixel * 256).astype(np.int32))
            cv2.fillConvexPoly(shadow, polygon, 1, shift=8)
        # One-cell dilation overapproximates raster rounding and the whole cell,
        # rather than treating a clear cell center as a visible cell certificate.
        shadow = cv2.dilate(shadow, np.ones((3, 3), np.uint8))
        result = visible & ~shadow.astype(bool)
        result.flags.writeable = False
        return result

    def _reject(self, reason):
        self.diagnostics.update(status="rejected", reason=reason)
        raise ValueError(f"Visibility route rejected: {reason}")

    def _cell(self, xy):
        point = np.asarray(xy, float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(
                "Visibility route point must contain two finite coordinates"
            )
        ij = np.floor((point + self.extent) / self.resolution).astype(int)
        return int(ij[1]), int(ij[0])

    def _point_visible(self, xy):
        row, col = self._cell(xy)
        return 0 <= row < self.n and 0 <= col < self.n and bool(self.visible[row, col])

    def _intervals(self, a, b):
        """Exact raster crossing lengths, including paths lying on grid edges."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        delta = b - a
        length = float(np.linalg.norm(delta))
        cuts = [0.0, 1.0]
        boundaries = -self.extent + np.arange(self.n + 1) * self.resolution
        for axis in (0, 1):
            if abs(delta[axis]) > 1e-12:
                cuts.extend(
                    float(value)
                    for value in (boundaries - a[axis]) / delta[axis]
                    if 0 < value < 1
                )
        cuts = sorted(set(cuts))
        for lo, hi in pairwise(cuts):
            midpoint = a + ((lo + hi) / 2) * delta
            indices = (midpoint + self.extent) / self.resolution
            axes = []
            for value in indices:
                cell = math.floor(value)
                if math.isclose(value, round(value), abs_tol=1e-10):
                    boundary = round(value)
                    axes.append((boundary - 1, boundary))
                else:
                    axes.append((cell,))
            visible = all(
                0 <= row < self.n and 0 <= col < self.n and self.visible[row, col]
                for col, row in product(*axes)
            )
            yield (hi - lo) * length, bool(visible)

    def assess(self, route):
        """Nominal path exposure metrics; no claim about future camera detection."""
        route = np.asarray(route, float)
        if (
            route.ndim != 2
            or route.shape[1] != 2
            or not len(route)
            or not np.isfinite(route).all()
        ):
            raise ValueError("Visibility assessment requires a finite Nx2 route")
        total = blind = longest = current = 0.0
        for a, b in pairwise(route):
            for distance, visible in self._intervals(a, b):
                total += distance
                if visible:
                    current = 0.0
                else:
                    blind += distance
                    current += distance
                    longest = max(longest, current)
        return {
            "route_distance_m": total,
            "predicted_blind_distance_m": blind,
            "predicted_max_blind_distance_m": longest,
            "predicted_blind_seconds": blind / self.reference_speed_m_s,
            "predicted_max_blind_seconds": longest / self.reference_speed_m_s,
        }

    def route(self, start, goal, *, grid, radius):
        self.diagnostics = self._base_diagnostics()
        if not self.memory.memory.valid or self._context_signature() != self._context:
            self._reject("map_or_camera_context_changed")
        if not math.isfinite(radius) or radius < 0:
            self._reject("invalid_clearance")
        if (
            grid.n != self.n
            or grid.extent != self.extent
            or grid.resolution != self.resolution
        ):
            self._reject("grid_geometry_mismatch")
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        source, target = self._cell(start), self._cell(goal)
        if not grid.free(source) or not grid.free(target):
            self._reject("blocked_clearance_endpoint")
        if not self.memory.segment_free(
            start, start, radius
        ) or not self.memory.segment_free(goal, goal, radius):
            self._reject("uncertified_clearance_endpoint")
        if not self._point_visible(goal):
            self._reject("predicted_blind_goal_requires_visible_arrival")
        # Conservative quantization rounds every blind edge up to whole cells.
        budget = math.floor(self.max_blind_distance_m / self.resolution)
        prefix = self.assess([start, grid.point(source)])
        initial = math.ceil(prefix["predicted_max_blind_distance_m"] / self.resolution)
        if initial > budget:
            self._reject("predicted_blind_start_exceeds_budget")
        origin = (*source, initial)
        queue = [(0.0, 0.0, origin)]
        costs, previous = {origin: 0.0}, {}
        expansions, reached = 0, None
        while queue:
            _, distance, state = heapq.heappop(queue)
            if distance != costs.get(state):
                continue
            expansions += 1
            if expansions > self.max_expansions or len(costs) > self.max_expansions:
                self.diagnostics["expanded_states"] = expansions
                self._reject("bounded_search_budget_exhausted")
            row, col, blind_bins = state
            if (row, col) == target:
                reached = state
                break
            for dy, dx in _NEIGHBORS:
                nxt = row + dy, col + dx
                if not grid.free(nxt):
                    continue
                if (
                    dx
                    and dy
                    and not (grid.free((row + dy, col)) and grid.free((row, col + dx)))
                ):
                    continue
                visible_edge = self.visible[row, col] and self.visible[nxt]
                if dx and dy:
                    visible_edge = (
                        visible_edge
                        and self.visible[row + dy, col]
                        and self.visible[row, col + dx]
                    )
                step = math.hypot(dx, dy) * self.resolution
                next_blind = 0 if visible_edge else blind_bins + (2 if dx and dy else 1)
                if next_blind > budget:
                    continue
                if self.visible[nxt]:
                    next_blind = 0
                next_state = (*nxt, next_blind)
                new_distance = distance + step
                if new_distance < costs.get(next_state, math.inf):
                    costs[next_state], previous[next_state] = new_distance, state
                    heuristic = (
                        math.hypot(nxt[0] - target[0], nxt[1] - target[1])
                        * self.resolution
                    )
                    heapq.heappush(
                        queue, (new_distance + heuristic, new_distance, next_state)
                    )
        self.diagnostics["expanded_states"] = expansions
        if reached is None:
            self._reject("no_route_within_predicted_visibility_budget")
        cells = [reached[:2]]
        while reached != origin:
            reached = previous[reached]
            cells.append(reached[:2])
        points = [start, *[grid.point(cell) for cell in reversed(cells)], goal]
        points = [
            point
            for i, point in enumerate(points)
            if i == 0 or np.linalg.norm(point - points[i - 1]) > 1e-10
        ]
        # Smooth only wholly predicted-visible shortcuts. Blind portions keep
        # their bounded grid edges; smoothing cannot join hidden intervals.
        route, index, checks = [points[0]], 0, 0
        while index < len(points) - 1:
            selected = index + 1
            candidates = (
                range(len(points) - 1, index + 1, -1)
                if self._point_visible(points[index]) and checks < 256
                else ()
            )
            for candidate in candidates:
                checks += 1
                if checks > 256:
                    break
                a, b = points[index], points[candidate]
                if (
                    self.assess([a, b])["predicted_blind_distance_m"] <= 1e-12
                    and grid.segment_free(a, b)
                    and self.memory.segment_free(a, b, radius)
                ):
                    selected = candidate
                    break
            route.append(points[selected])
            index = selected
        route = np.asarray(route)
        metrics = self.assess(route)
        self.diagnostics.update(
            metrics,
            smoothing_checks=min(checks, 256),
            clearance_radius_m=radius,
            grid_waypoints=len(points),
            route_waypoints=len(route),
        )
        if (
            metrics["predicted_max_blind_distance_m"]
            > self.max_blind_distance_m + 1e-10
        ):
            self._reject("final_route_exceeds_predicted_visibility_budget")
        if not all(self.memory.segment_free(a, b, radius) for a, b in pairwise(route)):
            self._reject("uncertified_clearance_segment")
        self.diagnostics.update(status="accepted", clearance_certified=True)
        return route
