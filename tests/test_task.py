from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from bb8_rl.baseline import RouteController
from bb8_rl.control.contract import PlanarDriveParameters
from bb8_rl.env import NavigationEnv
from bb8_rl.evaluate import suite_cases, wilson
from bb8_rl.layouts import layout_grid, make_layout, sample_endpoints
from bb8_rl.planner import OccupancyGrid
from bb8_rl.task import SPLITS, TaskConfig, require_split_seed

TASK = Path(__file__).resolve().parents[1] / "configs/navigation/bb8-task.yaml"
SUITES = TASK.parent / "suites.json"


def test_frozen_heldout_seeds_are_disjoint_and_require_authored_fixtures():
    require_split_seed("heldout", 926101)
    require_split_seed("heldout", 926102)
    for name, (lower, upper) in SPLITS.items():
        if name != "heldout":
            assert upper <= SPLITS["heldout"][0]
            with pytest.raises(ValueError):
                require_split_seed(name, 926101)
            with pytest.raises(ValueError):
                require_split_seed("heldout", lower)
    with pytest.raises(ValueError, match="authored"):
        NavigationEnv(TASK, split="heldout")


def test_astar_detour_never_crosses_blocked_cells():
    grid = OccupancyGrid(1.5, 0.1, 0.147, [(0, 0, 0.3, 1.5)])
    route = grid.route((-0.95, 0.05), (0.95, 0.05))
    assert len(route) > 2
    assert all(grid.segment_free(a, b) for a, b in pairwise(route))
    assert np.linalg.norm(np.diff(route, axis=0), axis=1).sum() > 2.5


def test_corner_cutting_and_disconnected_goals_are_rejected():
    grid = OccupancyGrid(1.5, 1.0, 0.0, [])
    grid.blocked[:] = True
    grid.blocked[0, 0] = False
    grid.blocked[1, 1] = False
    with pytest.raises(ValueError, match="disconnected"):
        grid.route(grid.point((0, 0)), grid.point((1, 1)))
    with pytest.raises(ValueError, match="blocked"):
        grid.route((0, 0), (1, 1))


def test_segment_rejects_corner_grazing():
    grid = OccupancyGrid(1.5, 1.0, 0.0, [])
    grid.blocked[:] = False
    grid.blocked[1, 1] = True
    assert not grid.segment_free((-1, 0), (0, 1))
    assert grid.segment_free((-1, -1), (-1, 1))


@pytest.mark.parametrize(
    "split,seed", [("train", 101), ("validation", 10001), ("test", 20001)]
)
def test_sampling_is_seeded_free_and_connected(split, seed):
    layout = make_layout(seed, split)
    grid = layout_grid(
        layout, extent=1.5, resolution=0.1, inflation=0.147, sizes=[(0.3, 0.3, 0.3)] * 8
    )
    a = sample_endpoints(grid, np.random.default_rng(42), layout, 0.6, 2.5)
    b = sample_endpoints(grid, np.random.default_rng(42), layout, 0.6, 2.5)
    assert np.array_equal(a, b)
    assert all(grid.free(grid.cell(p)) for p in a)
    assert len(grid.route(*a)) >= 2


def test_splits_are_disjoint_and_nonempty_geometries_vary():
    domains = [
        {r["layout_seed"] for r in suite_cases(SUITES, s)}
        for s in ("train", "validation", "test")
    ]
    assert (
        not domains[0] & domains[1]
        and not domains[0] & domains[2]
        and not domains[1] & domains[2]
    )
    assert (
        make_layout(101, "train").positions
        != make_layout(10001, "validation").positions
    )
    with pytest.raises(ValueError):
        make_layout(1, "validation")
    with pytest.raises(ValueError):
        suite_cases(SUITES, "validation", 13)


def test_unknown_map_is_never_free_and_channels_are_exclusive():
    grid = OccupancyGrid(1.5, 0.1, 0.147, [])
    local = grid.local_map((1.45, 1.45), 21)
    assert np.all(local.sum(axis=0) == 1)
    assert local[2, -1, -1] == 1 and local[1, -1, -1] == 0


def test_her_scalar_batched_reward_preserves_failure_penalty():
    env = NavigationEnv(TASK)
    achieved = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0, 0.04], [0.2, 0, 0]])
    goals = np.zeros_like(achieved)
    info = np.array([{}, {}, {}, {"event_penalty": 10}], dtype=object)
    assert np.array_equal(env.compute_reward(achieved, goals, info), [0, -1, -1, -11])
    assert np.array_equal(env.compute_reward(achieved, achieved, info), [0, 0, 0, -10])
    assert env.compute_reward(achieved[3], achieved[3], info[3]) == -10
    assert env.compute_reward([4, 4, 0], [4, 4, 0], {"event_penalty": 10}) < 0


def test_controller_is_bounded_and_inverts_heading_offset():
    task = TaskConfig(world_config="unused")
    grid = OccupancyGrid(1.5, 0.1, 0.147, [])
    obs = {
        "achieved_goal": np.array([-0.5, 0, 0]),
        "desired_goal": np.array([0.5, 0, 0]),
        "observation": np.zeros(6),
    }
    policy = RouteController(task, PlanarDriveParameters(heading_offset=np.pi / 2))
    policy.reset(grid, obs)
    action = policy.action(obs)
    assert np.linalg.norm(action) <= 1 and abs(action[0]) < 1e-6 and action[1] < 0
    obs["achieved_goal"] = np.array([0.5, 0, 0])
    assert np.array_equal(policy.action(obs), [0, 0])


def test_confidence_interval_does_not_claim_zero_risk():
    lo, hi = wilson(0, 12)
    assert lo == 0 and hi > 0.2
    lo, hi = wilson(12, 12)
    assert lo < 0.8 and hi == 1


@pytest.mark.parametrize("problem", ["unmapped", "floating", "unparked"])
def test_task_rejects_geometry_that_disagrees_with_its_map(monkeypatch, problem):
    import bb8_rl.env as module
    from bb8_rl.task import load_task

    _, world_path = load_task(TASK)
    result = module.validate_world(world_path)
    project = result[1]
    slot = next(obj for obj in project.objects if obj.name == "obstacle_0")
    if problem == "unmapped":
        extra = slot.model_copy(deep=True)
        extra.name, extra.position = "unmapped_obstacle", (0, 0, 0.15)
        project.objects.append(extra)
    elif problem == "floating":
        slot.position = (*slot.position[:2], 2)
    else:
        slot.position = (0, 0, 0.15)
    monkeypatch.setattr(module, "validate_world", lambda _: result)
    with pytest.raises(ValueError, match="interior collider|Obstacle slots"):
        NavigationEnv(TASK)
