from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.camera import Calibration
from bb8_rl.mapping.contracts import Bounds3D, EvidenceSource, Provenance, SpaceState
from bb8_rl.mapping.room_memory import RoomMemory
from bb8_rl.planner import OccupancyGrid
from bb8_rl.visibility_planning import VisibilityPlanner


class ScannedFixture:
    """Independent planar clearance certificate and reconstructed 3D occluders."""

    def __init__(self, *, occluders=(), physical_boxes=(), uncertainty=0.0):
        self.extent, self.resolution = 1.0, 0.1
        self.physical_boxes = physical_boxes
        self.memory = RoomMemory(
            Bounds3D((-1.0, -1.0, 0.0), (1.0, 1.0, 1.5)),
            0.1,
            scene_version="reconstructed-room",
            calibration_version="registered-rig",
        )
        for i, box in enumerate(occluders):
            self.memory.observe_volume(
                Bounds3D(*box),
                SpaceState.OCCUPIED,
                EvidenceSource(
                    f"scan-{i}",
                    ("mapping-view",),
                    0.0,
                    Provenance.RGB_RECONSTRUCTION,
                    uncertainty,
                ),
            )
        self.clearance_calls = []

    def planning_grid(self, radius):
        return OccupancyGrid(self.extent, self.resolution, radius, self.physical_boxes)

    def segment_free(self, a, b, radius):
        self.clearance_calls.append(radius)
        return self.planning_grid(radius).segment_free(a, b)


def camera(x=0.0, y=0.0, z=2.0, *, extent=1.0):
    rotation = np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ np.array([x, y, z])
    return Calibration(
        np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]]),
        transform,
        (100, 100),
        extent,
        head_height=0.1,
        provenance="estimated test registration",
    )


def overhead_box():
    # Elevated surface blocks the camera while leaving body-plane clearance.
    return ((-0.15, -0.15, 0.5), (0.15, 0.15, 1.0))


def planner(memory, **kwargs):
    return VisibilityPlanner(
        memory, {"A": camera()}, head_radius_m=0.0, pixel_guard=0, **kwargs
    )


def test_clear_view_retains_direct_route_and_full_clearance():
    memory = ScannedFixture()
    candidate = planner(memory)
    start, goal, radius = (-0.65, -0.15), (0.65, -0.15), 0.10
    route = candidate.route(
        start, goal, grid=memory.planning_grid(radius), radius=radius
    )
    np.testing.assert_allclose(route, [start, goal])
    assert candidate.diagnostics["clearance_certified"]
    assert not candidate.diagnostics["visibility_certified"]
    assert set(memory.clearance_calls) == {radius}
    assert candidate.diagnostics["predicted_max_blind_seconds"] == 0
    assert "not certified" in " ".join(candidate.diagnostics["limitations"])


def test_scan_shadow_causes_detour_without_changing_clearance_grid():
    memory = ScannedFixture(occluders=[overhead_box()])
    candidate = planner(memory)
    grid = memory.planning_grid(0.1)
    before = grid.blocked.copy()
    start, goal = (-0.65, 0.05), (0.65, 0.05)
    assert len(grid.route(start, goal)) == 2
    route = candidate.route(start, goal, grid=grid, radius=0.1)
    assert len(route) > 2
    assert candidate.diagnostics["predicted_blind_distance_m"] == 0
    np.testing.assert_array_equal(grid.blocked, before)
    for a, b in pairwise(route):
        assert memory.segment_free(a, b, 0.1)
    np.testing.assert_allclose(route[0], start)
    np.testing.assert_allclose(route[-1], goal)


def test_additional_registered_camera_can_observe_otherwise_hidden_goal():
    memory = ScannedFixture(occluders=[overhead_box()])
    single = planner(memory)
    point = (0.05, 0.05)
    assert not single._point_visible(point)
    with pytest.raises(ValueError, match="blind_goal"):
        single.route((0.65, 0.05), point, grid=memory.planning_grid(0.1), radius=0.1)
    pair = VisibilityPlanner(
        memory, {"A": camera(), "B": camera(x=1.2)}, head_radius_m=0.0, pixel_guard=0
    )
    assert pair._point_visible(point)
    route = pair.route((0.65, 0.05), point, grid=memory.planning_grid(0.1), radius=0.1)
    assert pair.diagnostics["predicted_blind_distance_m"] == 0
    assert len(route) >= 2


def test_evidence_uncertainty_only_reduces_predicted_visibility():
    tight = planner(ScannedFixture(occluders=[overhead_box()], uncertainty=0.0))
    expanded = planner(ScannedFixture(occluders=[overhead_box()], uncertainty=0.1))
    assert np.all(~expanded.visible | tight.visible)
    assert expanded.visible.sum() < tight.visible.sum()


def test_unknown_scan_height_is_reported_as_limitation_not_ray_certificate():
    candidate = planner(ScannedFixture())
    assert candidate.diagnostics["scan_vertical_bounds_m"] == [0.0, 1.5]
    assert "omit obstacles above" in candidate.diagnostics["limitations"][0]
    assert candidate.diagnostics["kind"].endswith("heuristic")


def test_blind_distance_budget_cannot_be_bypassed_by_route_smoothing():
    memory = ScannedFixture(occluders=[overhead_box()])
    candidate = planner(memory, max_blind_distance_m=0.1)
    route = candidate.route(
        (-0.65, 0.05), (0.65, 0.05), grid=memory.planning_grid(0.1), radius=0.1
    )
    assert candidate.assess(route)["predicted_max_blind_distance_m"] <= 0.1 + 1e-10
    assert (
        candidate.assess([route[0], route[-1]])["predicted_max_blind_distance_m"] > 0.1
    )


def test_unavoidable_blind_crossing_is_allowed_only_with_sufficient_distance_budget():
    memory = ScannedFixture(occluders=[((-0.05, -1.0, 0.5), (0.05, 1.0, 1.0))])
    start, goal, grid = (-0.65, 0.05), (0.65, 0.05), memory.planning_grid(0.1)
    strict = planner(memory, max_blind_distance_m=0.1)
    with pytest.raises(ValueError, match="no_route"):
        strict.route(start, goal, grid=grid, radius=0.1)
    bounded = planner(memory, max_blind_distance_m=0.6)
    route = bounded.route(start, goal, grid=grid, radius=0.1)
    assert 0 < bounded.assess(route)["predicted_max_blind_distance_m"] <= 0.6
    assert bounded.diagnostics["predicted_max_blind_seconds"] > 0


def test_exposure_metrics_accumulate_across_waypoint_boundaries_and_reset_on_visibility():
    candidate = planner(
        ScannedFixture(), max_blind_distance_m=1.0, reference_speed_m_s=0.1
    )
    mask = np.ones((20, 20), bool)
    mask[:, 8:12] = False
    mask.flags.writeable = False
    candidate.visible = mask
    split = candidate.assess([[-0.45, 0.05], [-0.05, 0.05], [0.45, 0.05]])
    unsplit = candidate.assess([[-0.45, 0.05], [0.45, 0.05]])
    assert split == pytest.approx(unsplit)
    assert split["predicted_max_blind_distance_m"] == pytest.approx(0.4)
    assert split["predicted_max_blind_seconds"] == pytest.approx(4.0)
    twice = candidate.assess([[-0.45, 0.05], [0.45, 0.05], [-0.45, 0.05]])
    assert twice["predicted_blind_distance_m"] == pytest.approx(0.8)
    assert twice["predicted_max_blind_distance_m"] == pytest.approx(0.4)


def test_grid_boundary_exposure_includes_both_adjacent_cells():
    candidate = planner(ScannedFixture())
    mask = np.ones((20, 20), bool)
    mask[:, 9] = False
    candidate.visible = mask
    metrics = candidate.assess([[0.0, -0.35], [0.0, 0.35]])
    assert metrics["predicted_max_blind_distance_m"] == pytest.approx(0.7)


def test_float_boundary_just_below_integer_index_includes_the_right_neighbor():
    candidate = planner(ScannedFixture())
    mask = np.ones((20, 20), bool)
    mask[:, 12] = False
    candidate.visible = mask
    # (.2 + 1) / .1 is 11.999999999999998 on the selected runtime.
    metrics = candidate.assess([[0.2, -0.35], [0.2, 0.35]])
    assert metrics["predicted_max_blind_distance_m"] == pytest.approx(0.7)


def test_visibility_budget_never_relaxes_footprint_clearance():
    memory = ScannedFixture(physical_boxes=[(0.0, 0.0, 0.1, 2.0)])
    candidate = planner(memory, max_blind_distance_m=1.0)
    with pytest.raises(ValueError, match="no_route"):
        candidate.route(
            (-0.65, 0.05), (0.65, 0.05), grid=memory.planning_grid(0.1), radius=0.1
        )
    assert not candidate.diagnostics["clearance_certified"]


def test_untrusted_coarse_grid_still_cannot_bypass_final_segment_certificate():
    memory = ScannedFixture(physical_boxes=[(0.0, 0.0, 0.1, 2.0)])
    candidate = planner(memory)
    uninflated_or_stale_grid = OccupancyGrid(1.0, 0.1, 0.1, [])
    with pytest.raises(ValueError, match="uncertified_clearance_segment"):
        candidate.route(
            (-0.65, 0.05), (0.65, 0.05), grid=uninflated_or_stale_grid, radius=0.1
        )


@pytest.mark.parametrize(
    "change", ["map_version", "invalidated", "camera", "camera_set"]
)
def test_changed_map_or_camera_context_rejects_stale_visibility(change):
    memory = ScannedFixture()
    calibration = camera()
    cameras = {"A": calibration}
    candidate = VisibilityPlanner(memory, cameras)
    if change == "map_version":
        memory.memory.map_version += 1
    elif change == "invalidated":
        memory.memory.invalidate_if_changed(
            scene_version="changed", calibration_version="registered-rig", at_time=1.0
        )
    elif change == "camera":
        calibration.world_to_camera[0, 3] += 0.1
    else:
        cameras["B"] = camera(x=0.5)
    with pytest.raises(ValueError, match="context_changed"):
        candidate.route(
            (-0.65, 0.05), (0.65, 0.05), grid=memory.planning_grid(0.1), radius=0.1
        )


def test_search_budget_exhaustion_rejects_without_unrestricted_fallback():
    memory = ScannedFixture(occluders=[overhead_box()])
    candidate = planner(memory, max_expansions=1)
    with pytest.raises(ValueError, match="budget_exhausted"):
        candidate.route(
            (-0.65, 0.05), (0.65, 0.05), grid=memory.planning_grid(0.1), radius=0.1
        )
    assert candidate.diagnostics["status"] == "rejected"


def test_geometry_budget_rejects_without_dropping_occluders():
    with pytest.raises(ValueError, match="box budget"):
        planner(
            ScannedFixture(occluders=[overhead_box(), overhead_box()]),
            max_geometry_boxes=1,
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("max_blind_distance_m", -1.0),
        ("max_blind_distance_m", float("inf")),
        ("reference_speed_m_s", 0.0),
        ("head_radius_m", -0.1),
        ("max_expansions", 0),
        ("pixel_guard", -1),
    ],
)
def test_invalid_visibility_budgets_are_rejected(parameter, value):
    with pytest.raises(ValueError, match="bounded"):
        VisibilityPlanner(ScannedFixture(), {"A": camera()}, **{parameter: value})


def test_unsupported_low_camera_and_out_of_frame_goal_fail_closed():
    memory = ScannedFixture()
    candidate = VisibilityPlanner(memory, {"A": camera(z=0.1)})
    assert not candidate.visible.any()
    with pytest.raises(ValueError, match="blind_goal"):
        candidate.route(
            (-0.65, 0.05), (0.65, 0.05), grid=memory.planning_grid(0.1), radius=0.1
        )
    far = VisibilityPlanner(memory, {"A": camera(x=10.0)})
    assert not far.visible.any()


def test_only_declared_scan_memory_and_calibrations_are_read():
    # No project/world or evaluation truth attributes exist on this fixture.
    memory = ScannedFixture()
    minimal = SimpleNamespace(
        memory=memory.memory,
        extent=memory.extent,
        resolution=memory.resolution,
        segment_free=memory.segment_free,
    )
    candidate = VisibilityPlanner(minimal, {"A": camera()})
    candidate.route(
        (-0.65, 0.05), (0.65, 0.05), grid=memory.planning_grid(0.1), radius=0.1
    )
    assert candidate.diagnostics["status"] == "accepted"
