"""Analytic masks test metric connectors; they are not reconstructed map evidence."""

from itertools import pairwise

import numpy as np
import pytest

from bb8_rl.interactive_runtime import ControlSession
from bb8_rl.mapping import Bounds3D, RoomMemory
from bb8_rl.mapping.scan_free_space import ScanFreeMemory


def analytic_memory(*, wall=False):
    free = np.ones((200, 200), bool)
    free[163, 131] = False
    if wall:
        free[:, 100] = False
    memory = RoomMemory(
        Bounds3D((-2, -2, 0), (2, 2, 0.12)),
        0.02,
        scene_version="analytic-scene",
        calibration_version="analytic-calibration",
    )
    return ScanFreeMemory(
        memory,
        free,
        free,
        np.zeros_like(free),
        free.astype(np.uint32),
        ["analytic-mask"],
        {
            "extent_m": 2.0,
            "resolution_m": 0.02,
            "certified_height_m": 0.12,
            "floor_z_m": 0.0,
        },
    )


def test_exact_connector_fixes_raster_boundary_without_changing_grid_predicate():
    memory = analytic_memory()
    start, goal, radius = [-0.15, 0.25], [0.8, 1.3], 0.14
    grid = memory.planning_grid(radius)
    center = grid.point(grid.cell(goal))
    np.testing.assert_allclose(center, [0.81, 1.31], rtol=0, atol=1e-15)
    assert grid.free(grid.cell(goal))
    assert memory.segment_free(center, goal, radius)
    assert not grid.segment_free(center, goal)
    with pytest.raises(ValueError, match="endpoints cannot connect"):
        grid.route(start, goal)
    route = memory.certified_route(start, goal, radius, grid=grid)
    np.testing.assert_array_equal(route[0], start)
    np.testing.assert_array_equal(route[-1], goal)
    assert all(memory.segment_free(a, b, radius) for a, b in pairwise(route))
    assert not grid.segment_free(center, goal)


@pytest.mark.parametrize(
    "goal",
    [[0.78, 1.28], [0.779, 1.28], [0.63, 1.27], [1.86, 0.0]],
)
def test_tangent_intruding_unknown_and_map_boundary_endpoints_still_reject(goal):
    memory = analytic_memory()
    assert not memory.segment_free(goal, goal, 0.14)
    with pytest.raises(ValueError):
        memory.certified_route([0, 0], goal, 0.14)
    # A genuinely separated exact disk remains geometrically admissible.
    assert memory.segment_free([0.781, 1.28], [0.781, 1.28], 0.14)


def test_exactly_clear_but_conservatively_blocked_endpoint_cell_still_rejects():
    memory = analytic_memory()
    point = [0.781, 1.28]
    assert memory.segment_free(point, point, 0.14)
    grid = memory.planning_grid(0.14)
    assert not grid.free(grid.cell(point))
    with pytest.raises(ValueError, match="blocked after footprint inflation"):
        memory.certified_route([0, 0], point, 0.14, grid=grid)


def test_existing_successful_route_and_exact_distinct_endpoints_are_preserved():
    memory = analytic_memory()
    grid = memory.planning_grid(0.14)
    start, goal = [-0.3, -0.2], [0.5, -0.1]
    np.testing.assert_array_equal(
        memory.certified_route(start, goal, 0.14, grid=grid), grid.route(start, goal)
    )
    goal = np.array(start) + 1e-12
    route = memory.certified_route(start, goal, 0.14)
    np.testing.assert_array_equal(route[0], start)
    np.testing.assert_array_equal(route[-1], goal)
    np.testing.assert_array_equal(memory.certified_route(start, start, 0.14), [start])


def test_disconnected_conservative_grid_does_not_gain_a_metric_shortcut():
    memory = analytic_memory(wall=True)
    with pytest.raises(ValueError, match="disconnected"):
        memory.certified_route([-0.5, 0], [0.5, 0], 0.14)


@pytest.mark.parametrize("radius", [-0.1, np.inf, np.nan])
def test_invalid_clearance_fails_before_grid_search(radius):
    with pytest.raises(ValueError):
        analytic_memory().certified_route([0, 0], [0.5, 0], radius)


@pytest.mark.parametrize("point", [[np.nan, 0], [0, np.inf], [0], [0, 0, 0]])
def test_nonfinite_or_malformed_endpoints_reject(point):
    memory = analytic_memory()
    for start, goal in ((point, [0, 0]), ([0, 0], point)):
        with pytest.raises(ValueError):
            memory.certified_route(start, goal, 0.14)


def test_invalidated_memory_rejects_even_a_previously_cached_grid():
    memory = analytic_memory()
    grid = memory.planning_grid(0.14)
    memory.memory.invalidate_if_changed(
        scene_version="changed-scene",
        calibration_version="analytic-calibration",
        at_time=1,
    )
    with pytest.raises(ValueError, match="valid memory"):
        memory.certified_route([0, 0], [0.5, 0], 0.14, grid=grid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("extent", 3.0),
        ("resolution", 0.04),
        ("n", 100),
        ("inflation", 0.1),
        ("inflation", np.nan),
        ("blocked", np.zeros((100, 100), bool)),
        ("blocked", np.zeros((200, 200), np.uint8)),
        ("blocked", np.zeros((200, 200), bool)),
    ],
)
def test_cached_grid_requires_matching_geometry_clearance_and_conservative_mask(
    field, value
):
    memory = analytic_memory()
    grid = memory.planning_grid(0.14)
    setattr(grid, field, value)
    with pytest.raises(ValueError, match="Cached grid"):
        memory.certified_route([0, 0], [0.5, 0], 0.14, grid=grid)


def test_more_conservative_cache_is_allowed_without_reducing_metric_certificate():
    memory = analytic_memory()
    grid = memory.planning_grid(0.16)
    route = memory.certified_route([0, 0], [0.5, 0], 0.14, grid=grid)
    assert all(memory.segment_free(a, b, 0.14) for a, b in pairwise(route))


def test_every_route_edge_needs_metric_certificate(monkeypatch):
    memory = analytic_memory()
    monkeypatch.setattr(
        memory, "segment_free", lambda a, b, radius: np.array_equal(a, b)
    )
    with pytest.raises(ValueError, match="full requested clearance"):
        memory.certified_route([0, 0], [0.5, 0], 0.14)


def test_control_session_uses_metric_connector_with_full_requested_reserve():
    memory = analytic_memory()
    control = ControlSession(
        policy=lambda _: np.zeros(2),
        memory=memory,
        stop=lambda: None,
        map_version="analytic-map",
        calibration_version="analytic-calibration",
        demo_goal=[0.8, 1.3],
        planning_reserve=0.04,
        clock=lambda: 0.0,
    )
    route = control.route_with_clearance([-0.15, 0.25], [0.8, 1.3], 0.1)
    assert control.route_planning_details["route_certified"]
    assert control.route_planning_details["certificate_radius_m"] == 0.14
    assert not control.route_planning_details["exact_fallback_used"]
    assert all(memory.segment_free(a, b, 0.14) for a, b in pairwise(route))
