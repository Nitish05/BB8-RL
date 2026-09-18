"""Continuous clearance against the predicted native RGB free-space mask.

The certificate is relative to the mask and supplied calibration, not a claim
that learned labels or calibration match the physical scene. Unknown pixels are
never cleared. A connector can cross a conservative *grid* boundary only when
its entire continuous footprint is certified in the original visible mask.
"""

from itertools import pairwise

import cv2
import numpy as np


class CameraClearance:
    def __init__(
        self,
        calibration,
        visible_mask,
        inflation,
        *,
        sample_step=0.005,
        connector_radius=0.12,
    ):
        width, height = calibration.resolution
        mask = np.asarray(visible_mask)
        if mask.shape != (height, width) or mask.dtype != np.bool_:
            raise ValueError("Visible mask must be boolean at calibrated resolution")
        if (
            not np.isfinite([inflation, sample_step, connector_radius]).all()
            or inflation < 0
            or not 0 < sample_step <= 0.005
            or not 0 < connector_radius <= 0.12
        ):
            raise ValueError("Invalid continuous clearance parameters")
        self.calibration = calibration
        self.inflation = float(inflation)
        self.sample_step = float(sample_step)
        self.minimum_step = 0.0001
        self.max_refinement_depth = 6
        self.connector_radius = float(connector_radius)
        extent = calibration.extent
        corners = calibration.to_pixel(
            [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]]
        )
        room = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(room, np.rint(corners).astype(np.int32), 1)
        self.visible_mask = mask & room.astype(bool)

        # OpenCV follows boundary-pixel centers, rather than pixel-square edges.
        # A digital contour segment stays in the one-pixel boundary strip. Bound
        # its geometric uncertainty by a full projected pixel diameter, including
        # the rounded room polygon. Projective maps send each pixel square to a
        # quadrilateral; its greatest distance from its projected center occurs
        # at a vertex. Twice the maximum such radius bounds every diameter. Use
        # all room pixels, not a sampled Jacobian or a favorable endpoint pixel.
        yy, xx = np.nonzero(room)
        self.pixel_guard_m = 0.0
        for offset in range(0, len(xx), 65536):
            pixels = np.column_stack(
                (xx[offset : offset + 65536], yy[offset : offset + 65536])
            )
            centers = calibration.to_plane(pixels)
            for corner in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)):
                radius = np.linalg.norm(
                    calibration.to_plane(pixels + corner) - centers, axis=1
                )
                self.pixel_guard_m = max(self.pixel_guard_m, 2 * float(radius.max()))
        # Cover float32 contour arithmetic, independently of the pixel guard.
        self.pixel_guard_m += 1e-6
        contours, _hierarchy = cv2.findContours(
            self.visible_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = [
            calibration.to_plane(contour.reshape(-1, 2)).astype(np.float32)
            for contour in contours
        ]

    @staticmethod
    def _point(value):
        point = np.asarray(value, dtype=float)
        return point if point.shape == (2,) and np.isfinite(point).all() else None

    def clearance(self, point):
        """Conservative visible-boundary distance; unknown/outside returns -inf."""
        point = self._point(point)
        if (
            point is None
            or np.max(abs(point)) >= self.calibration.extent
            or not self.contours
        ):
            return float("-inf")
        distances = [
            cv2.pointPolygonTest(contour, tuple(point), True)
            for contour in self.contours
        ]
        # Nested foreground/background/foreground contours alternate occupancy.
        # Points exactly on a contour have zero clearance and cannot be driven.
        if sum(distance > 0 for distance in distances) % 2 != 1:
            return float("-inf")
        return min(
            min(abs(distance) for distance in distances) - self.pixel_guard_m,
            self.calibration.extent - float(np.max(abs(point))),
        )

    def point_free(self, point):
        return self.clearance(point) > self.inflation

    def segment_free(self, start, end):
        """Certify the whole segment using distance's 1-Lipschitz bound.

        Start with gaps no greater than 5 mm. Ambiguous intervals are bisected,
        never accepted merely because their sampled points are clear. Refinement
        stops at a gap <= 0.1 mm or depth 6; any still ambiguous interval fails
        closed. No successful interval has a lower bound below ``inflation``.
        """
        start, end = self._point(start), self._point(end)
        if (
            start is None
            or end is None
            or not self.point_free(start)
            or not self.point_free(end)
        ):
            return False
        distance = float(np.linalg.norm(end - start))
        intervals = max(1, int(np.ceil(distance / self.sample_step)))
        # Every segment point lies at most half the actual sample gap from a
        # sample. Extra clearance pays for that entire unsampled interval.
        gap = distance / intervals
        points = np.linspace(start, end, intervals + 1)
        clearances = np.array([self.clearance(point) for point in points])
        if np.any(clearances <= self.inflation):
            return False
        ambiguous = np.flatnonzero(
            np.minimum(clearances[:-1], clearances[1:]) <= self.inflation + gap / 2
        )
        pending = [
            (
                points[index],
                points[index + 1],
                clearances[index],
                clearances[index + 1],
                gap,
                0,
            )
            for index in ambiguous
        ]
        while pending:
            a, b, clearance_a, clearance_b, gap, depth = pending.pop()
            if min(clearance_a, clearance_b) > self.inflation + gap / 2:
                continue
            if gap <= self.minimum_step or depth >= self.max_refinement_depth:
                return False
            midpoint = (a + b) / 2
            clearance_midpoint = self.clearance(midpoint)
            if clearance_midpoint <= self.inflation:
                return False
            pending.extend(
                [
                    (a, midpoint, clearance_a, clearance_midpoint, gap / 2, depth + 1),
                    (midpoint, b, clearance_midpoint, clearance_b, gap / 2, depth + 1),
                ]
            )
        return True

    def _connectors(self, grid, endpoint):
        cells = np.argwhere(~grid.blocked)
        centers = -grid.extent + (cells[:, ::-1] + 0.5) * grid.resolution
        distances = np.linalg.norm(centers - endpoint, axis=1)
        indices = np.flatnonzero(distances <= self.connector_radius)
        indices = indices[np.argsort(distances[indices], kind="stable")]
        return [
            (cells[index], centers[index], float(distances[index]))
            for index in indices
            if self.segment_free(endpoint, centers[index])
        ]

    def route(self, grid, start, goal):
        """Plan certified endpoint bridges without changing a single grid cell."""
        start, goal = self._point(start), self._point(goal)
        if (
            start is None
            or goal is None
            or not self.point_free(start)
            or not self.point_free(goal)
        ):
            raise ValueError("Start or goal lacks continuous visible clearance")
        try:
            route = grid.route(start, goal)
        except ValueError:
            route = None
        if route is not None and all(
            self.segment_free(a, b) for a, b in pairwise(route)
        ):
            return route

        sources, targets = self._connectors(grid, start), self._connectors(grid, goal)
        if not sources or not targets:
            raise ValueError("No certified visible connector from endpoint to grid")
        # Four-connected components equal reachability for this grid's A*:
        # allowed diagonal steps require both intervening cardinal cells free.
        _, components = cv2.connectedComponents(
            (~grid.blocked).astype(np.uint8), connectivity=4
        )
        pairs = [
            (a[2] + b[2], a[1], b[1])
            for a in sources
            for b in targets
            if components[tuple(a[0])] == components[tuple(b[0])]
        ]
        pairs.sort(key=lambda pair: pair[0])
        # Bounded fallback: failure to find a certified route stops motion; it
        # never weakens the mask, inflation, or endpoint requirements.
        for _, source, target in pairs[:64]:
            try:
                middle = grid.route(source, target)
            except ValueError:
                continue
            route = np.vstack((start, middle, goal))
            route = route[
                np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-9]
            ]
            if all(self.segment_free(a, b) for a, b in pairwise(route)):
                return route
        raise ValueError("No continuously certified route through visible grid")
