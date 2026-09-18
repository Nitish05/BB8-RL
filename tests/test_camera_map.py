"""Camera-map regressions using analytic image geometry, without simulator truth."""

from itertools import pairwise

import numpy as np
import pytest

from bb8_rl.camera import Calibration, VisualMeasurement, camera_grid


@pytest.fixture
def overhead():
    # One native pixel spans 2 mm on the floor; a 2 cm map cell spans 10 pixels.
    intrinsics = np.array([[1000.0, 0, 600], [0, 1000.0, 600], [0, 0, 1]])
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 2
    return Calibration(intrinsics, transform, (1200, 1200), 1)


def observation(floor):
    return VisualMeasurement(0, None, None, floor, np.zeros_like(floor), "missing")


def floor_x(calibration):
    """Analytic world x coordinate of each pixel in this overhead camera."""
    return (np.arange(calibration.resolution[0]) - calibration.intrinsics[0, 2]) / 500


def vertical_region(calibration, predicate):
    width, height = calibration.resolution
    return np.broadcast_to(predicate(floor_x(calibration)), (height, width)).copy()


def test_metric_map_catches_unknown_pixel_between_legacy_probes(overhead):
    floor = np.ones((1200, 1200), bool)
    # Cell [.02,.04]² has probes at its corners and (.03,.03). This pixel is
    # strictly inside, at (.026,.026), and coincides with none of those probes.
    floor[587, 613] = False
    measured = observation(floor)
    options = {"resolution": 0.02, "inflation": 0}
    default = camera_grid(overhead, measured, **options)
    legacy = camera_grid(overhead, measured, method="legacy", **options)
    metric = camera_grid(overhead, measured, method="metric", **options)
    cell = metric.cell([0.03, 0.03])

    np.testing.assert_array_equal(default.blocked, legacy.blocked)
    assert legacy.free(cell), "The fixture must expose the old five-probe hole"
    assert not metric.free(cell)
    with pytest.raises(ValueError, match="blocked"):
        metric.route([0.03, 0.03], [0.3, 0.3])


def test_metric_corridor_preserves_clearance_and_rejects_narrow_gap(overhead):
    margin = 0.10
    wide_half_width = 0.20
    wide = camera_grid(
        overhead,
        observation(vertical_region(overhead, lambda x: abs(x) < wide_half_width)),
        method="metric",
        resolution=0.02,
        inflation=margin,
    )
    route = wide.route([0, -0.5], [0, 0.5])
    assert all(wide.segment_free(a, b) for a, b in pairwise(route))
    free_centers = np.array([wide.point(cell) for cell in np.argwhere(~wide.blocked)])
    # Test physical clearance of the complete free cells, not just their centers.
    clearance = wide_half_width - abs(free_centers[:, 0]) - wide.resolution / 2
    assert clearance.min() >= margin - 1e-10

    narrow = camera_grid(
        overhead,
        observation(vertical_region(overhead, lambda x: abs(x) < 0.09)),
        method="metric",
        resolution=0.02,
        inflation=margin,
    )
    assert not narrow.free(narrow.cell([0, 0]))
    with pytest.raises(ValueError, match="blocked|disconnected"):
        narrow.route([0, -0.5], [0, 0.5])


def test_metric_inflation_crossing_ten_cm_has_no_five_cm_jump(overhead):
    measured = observation(vertical_region(overhead, lambda x: x < 0))
    grids = [
        camera_grid(
            overhead,
            measured,
            method="metric",
            resolution=0.02,
            inflation=margin,
        )
        for margin in (0.0999, 0.1001)
    ]
    low, high = grids
    assert np.all(~low.blocked | high.blocked), "Inflation must be monotonic"

    nearest_edges = []
    for grid, margin in zip(grids, (0.0999, 0.1001)):
        row = grid.cell([-0.5, 0])[0]
        free_columns = np.flatnonzero(~grid.blocked[row])
        assert len(free_columns)
        centers = np.array([grid.point((row, col)) for col in free_columns])
        right_edges = centers[:, 0] + grid.resolution / 2
        assert (-right_edges).min() >= margin - 1e-10
        nearest_edges.append(right_edges.max())
        assert not grid.free(grid.cell([0.3, 0]))
        with pytest.raises(ValueError, match="blocked"):
            grid.route([-0.4, 0], [0.3, 0])

    assert 0 <= nearest_edges[0] - nearest_edges[1] <= 0.02 + 1e-10
