"""Continuous camera connectors must certify actual endpoints and every bridge."""

from itertools import pairwise

import numpy as np
import pytest

from bb8_rl.camera import Calibration, VisualMeasurement, camera_grid
from bb8_rl.camera_clearance import CameraClearance


@pytest.fixture
def camera():
    intrinsics = np.array([[1000.0, 0, 600], [0, 1000.0, 600], [0, 0, 1]])
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 2
    return Calibration(intrinsics, transform, (1200, 1200), 1)


def vertical_mask(predicate):
    return np.broadcast_to(
        predicate((np.arange(1200) - 600) / 500), (1200, 1200)
    ).copy()


def test_unknown_endpoint_cannot_snap_to_nearby_visible_cell(camera):
    floor = vertical_mask(lambda x: x < 0)
    certificate = CameraClearance(camera, floor, 0.02)
    measured = VisualMeasurement(0, None, None, floor, np.zeros_like(floor), "missing")
    grid = camera_grid(
        camera, measured, resolution=0.02, inflation=0.02, method="metric"
    )
    assert certificate.point_free([-0.1, 0])
    assert not certificate.point_free([0.01, 0])
    with pytest.raises(ValueError, match="continuous visible clearance"):
        certificate.route(grid, [0.01, 0], [-0.5, 0])
    assert not certificate.segment_free([float("nan"), 0], [-0.5, 0])
    assert not certificate.segment_free([-0.5, 0], [-1.1, 0])
    assert not certificate.segment_free([-0.5, 0], [1e308, 0])


def test_segment_cannot_cross_single_unknown_native_pixel_strip(camera):
    # Zoom in so the one-pixel strip is much narrower than the 5 mm sample gap.
    intrinsics = camera.intrinsics.copy()
    intrinsics[0, 0] = intrinsics[1, 1] = 10000
    camera = Calibration(intrinsics, camera.world_to_camera, camera.resolution, 1)
    floor = np.ones((1200, 1200), bool)
    floor[:, 600] = False
    certificate = CameraClearance(camera, floor, 0)
    assert certificate.point_free([-0.0025, 0])
    assert certificate.point_free([0.0025, 0])
    # Even an interval no longer than the sample step cannot skip the strip.
    assert not certificate.segment_free([-0.0025, 0], [0.0025, 0])
    assert not certificate.segment_free([-0.3, 0], [0.3, 0])


def test_blocked_coarse_start_gets_certified_bridge_without_map_mutation(camera):
    floor = vertical_mask(lambda x: x < 0)
    inflation = 0.10
    certificate = CameraClearance(camera, floor, inflation)
    measured = VisualMeasurement(0, None, None, floor, np.zeros_like(floor), "missing")
    grid = camera_grid(
        camera, measured, resolution=0.02, inflation=inflation, method="metric"
    )
    start, goal = np.array([-0.13, 0.05]), np.array([-0.5, 0.45])
    assert not grid.free(grid.cell(start)), "Fixture requires coarse-cell rejection"
    assert certificate.point_free(start)
    before = grid.blocked.copy()
    route = certificate.route(grid, start, goal)
    np.testing.assert_allclose(route[0], start)
    np.testing.assert_allclose(route[-1], goal)
    np.testing.assert_array_equal(grid.blocked, before)
    assert np.linalg.norm(route[1] - start) <= 0.12
    assert all(certificate.segment_free(a, b) for a, b in pairwise(route))
    assert all(-point[0] > inflation for point in route)


def test_adaptive_intervals_certify_small_positive_margin_without_reducing_it(camera):
    intrinsics = camera.intrinsics.copy()
    intrinsics[0, 0] = intrinsics[1, 1] = 10000
    camera = Calibration(intrinsics, camera.world_to_camera, camera.resolution, 1)
    floor = np.broadcast_to(np.arange(1200) < 600, (1200, 1200)).copy()
    certificate = CameraClearance(camera, floor, 0.02)
    start, end = np.array([-0.0211, -0.01]), np.array([-0.0211, 0.01])
    margin = certificate.clearance(start) - certificate.inflation
    assert 0 < margin < certificate.sample_step / 2
    # The old uniform 5 mm proof rejected this entire parallel segment despite
    # its positive clearance. Subdivision must prove its unchanged 2 cm margin.
    assert certificate.segment_free(start, end)
    assert certificate.minimum_step == 0.0001
    assert certificate.max_refinement_depth == 6
    assert not certificate.segment_free([-0.0203, -0.01], [-0.0203, 0.01])
    barely_clear_x = -(0.02 + 0.0002 + certificate.pixel_guard_m + 1e-6)
    assert certificate.point_free([barely_clear_x, 0])
    # Positive sampled clearance alone is insufficient once subdivision reaches
    # its bound. The remaining unresolved interval must still fail closed.
    assert not certificate.segment_free([barely_clear_x, -0.01], [barely_clear_x, 0.01])


def test_hole_and_nested_visible_island_have_correct_parity(camera):
    floor = np.ones((1200, 1200), bool)
    floor[400:800, 400:800] = False
    floor[550:650, 550:650] = True
    certificate = CameraClearance(camera, floor, 0.01)
    assert certificate.point_free([0, 0])
    assert not certificate.point_free([0.2, 0])
    assert certificate.point_free([0.6, 0])
    assert not certificate.segment_free([0, 0], [0.6, 0])
    assert certificate.pixel_guard_m >= np.sqrt(2) * 0.002
