"""Conservative metric occupancy and deterministic A*, including corner checks."""

import heapq
import math

import numpy as np


class OccupancyGrid:
    def __init__(self, extent, resolution, inflation, boxes):
        self.extent, self.resolution, self.inflation = extent, resolution, inflation
        self.n = round(2 * extent / resolution)
        if (
            self.n < 2
            or self.n > 256
            or not math.isclose(self.n * resolution, 2 * extent)
        ):
            raise ValueError(
                "Grid must divide the arena exactly into 2–256 cells per axis"
            )
        axis = -extent + (np.arange(self.n) + 0.5) * resolution
        x, y = np.meshgrid(axis, axis)
        # Include a cell half-diagonal so a free cell is entirely outside inflated geometry.
        margin = inflation + resolution / math.sqrt(2)
        self.blocked = np.maximum(abs(x), abs(y)) >= extent - margin
        for cx, cy, width, height in boxes:
            self.blocked |= (abs(x - cx) <= width / 2 + margin) & (
                abs(y - cy) <= height / 2 + margin
            )

    def cell(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Map point must contain two finite coordinates")
        indices = np.floor((point + self.extent) / self.resolution).astype(int)
        return int(indices[1]), int(indices[0])

    def point(self, cell):
        return np.array(
            [
                -self.extent + (cell[1] + 0.5) * self.resolution,
                -self.extent + (cell[0] + 0.5) * self.resolution,
            ]
        )

    def free(self, cell):
        y, x = cell
        return 0 <= y < self.n and 0 <= x < self.n and not self.blocked[y, x]

    def segment_free(self, start, end):
        """Intersect the segment with closed blocked-cell rectangles (no corner grazing)."""
        a, b = np.asarray(start), np.asarray(end)
        if not self.free(self.cell(a)) or not self.free(self.cell(b)):
            return False
        cells = np.argwhere(self.blocked)[:, ::-1]
        lower = -self.extent + cells * self.resolution
        upper = lower + self.resolution
        lo = np.zeros(len(cells))
        hi = np.ones(len(cells))
        valid = np.ones(len(cells), dtype=bool)
        for axis in (0, 1):
            d = b[axis] - a[axis]
            if abs(d) < 1e-12:
                valid &= (a[axis] >= lower[:, axis]) & (a[axis] <= upper[:, axis])
            else:
                t1, t2 = (lower[:, axis] - a[axis]) / d, (upper[:, axis] - a[axis]) / d
                lo = np.maximum(lo, np.minimum(t1, t2))
                hi = np.minimum(hi, np.maximum(t1, t2))
        return not bool(np.any(valid & (lo <= hi + 1e-12)))

    def route(self, start, goal):
        source, target = self.cell(start), self.cell(goal)
        if not self.free(source) or not self.free(target):
            raise ValueError("Start or goal is blocked after footprint inflation")
        queue = [(0.0, source)]
        cost = {source: 0.0}
        previous = {}
        while queue:
            _, cell = heapq.heappop(queue)
            if cell == target:
                break
            for dy, dx in (
                (-1, 0),
                (0, -1),
                (0, 1),
                (1, 0),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                nxt = cell[0] + dy, cell[1] + dx
                if not self.free(nxt):
                    continue
                if (
                    dx
                    and dy
                    and not (
                        self.free((cell[0] + dy, cell[1]))
                        and self.free((cell[0], cell[1] + dx))
                    )
                ):
                    continue
                distance = cost[cell] + math.hypot(dx, dy)
                if distance < cost.get(nxt, float("inf")):
                    cost[nxt] = distance
                    previous[nxt] = cell
                    heuristic = math.hypot(nxt[0] - target[0], nxt[1] - target[1])
                    heapq.heappush(queue, (distance + heuristic, nxt))
        if target not in cost:
            raise ValueError(
                "Start and goal are disconnected after footprint inflation"
            )
        cells = [target]
        while cells[-1] != source:
            cells.append(previous[cells[-1]])
        points = [
            np.asarray(start),
            *[self.point(c) for c in reversed(cells)],
            np.asarray(goal),
        ]
        result = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_free(points[i], points[j]):
                j -= 1
            if not self.segment_free(points[i], points[j]):
                raise ValueError("Route endpoints cannot connect safely to the grid")
            if np.linalg.norm(points[j] - result[-1]) > 1e-9:
                result.append(points[j])
            i = j
        return np.asarray(result)

    def local_map(self, position, size):
        offset = (np.arange(size) - size // 2) * self.resolution
        x, y = np.meshgrid(offset + position[0], offset + position[1])
        ix = np.floor((x + self.extent) / self.resolution).astype(int)
        iy = np.floor((y + self.extent) / self.resolution).astype(int)
        known = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        blocked = self.blocked[np.clip(iy, 0, self.n - 1), np.clip(ix, 0, self.n - 1)]
        return np.stack((known & blocked, known & ~blocked, ~known)).astype(np.float32)
