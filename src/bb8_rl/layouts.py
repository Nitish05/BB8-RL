"""Seeded synthetic layouts in eight saved, fixed-geometry obstacle slots."""

from dataclasses import dataclass

import numpy as np

from .planner import OccupancyGrid
from .task import require_split_seed


@dataclass(frozen=True)
class Layout:
    seed: int
    kind: str
    positions: tuple[tuple[float, float], ...]


def make_layout(seed, split):
    require_split_seed(split, seed)
    kind = ("empty", "barrier", "corridor", "scatter")[seed % 4]
    rng = np.random.default_rng(seed)
    if kind == "empty":
        positions = []
    elif kind == "barrier":
        x, offset = rng.uniform(-0.15, 0.15), rng.uniform(-0.12, 0.12)
        positions = [(float(x), float(y + offset)) for y in (-0.6, -0.3, 0, 0.3, 0.6)]
    elif kind == "corridor":
        dx, dy, gap = (
            rng.uniform(-0.15, 0.15),
            rng.uniform(-0.08, 0.08),
            rng.uniform(0.46, 0.62),
        )
        positions = [
            (float(x + dx), float(y + dy))
            for y in (-gap, gap)
            for x in (-0.45, -0.15, 0.15, 0.45)
        ]
    else:
        positions = []
        for _ in range(100):
            p = tuple(float(v) for v in rng.uniform(-0.9, 0.9, 2))
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 0.42 for q in positions):
                positions.append(p)
            if len(positions) == 6:
                break
        if len(positions) != 6:
            raise ValueError("Could not generate separated obstacle slots")
    return Layout(seed, kind, tuple(positions))


def layout_grid(layout, *, extent, resolution, inflation, sizes):
    if len(layout.positions) > len(sizes):
        raise ValueError("The world has too few obstacle slots")
    boxes = [(*p, *size[:2]) for p, size in zip(layout.positions, sizes)]
    return OccupancyGrid(extent, resolution, inflation, boxes)


def sample_endpoints(grid, rng, layout, minimum, maximum):
    free = np.argwhere(~grid.blocked)
    if len(free) < 2:
        raise ValueError("Layout has too little free space after inflation")
    for _ in range(2000):
        start, goal = [
            grid.point(c) for c in free[rng.choice(len(free), 2, replace=False)]
        ]
        if not minimum <= np.linalg.norm(goal - start) <= maximum:
            continue
        if layout.kind in ("barrier", "corridor") and not (
            start[0] < -0.7 and goal[0] > 0.7
        ):
            continue
        try:
            grid.route(start, goal)
        except ValueError:
            continue
        return start, goal
    raise ValueError(
        "Could not find connected start/goal positions; relax task distances or layout"
    )
