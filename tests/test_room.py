import json
from pathlib import Path

import numpy as np
import pytest

from bb8_rl.env import NavigationEnv
from bb8_rl.planner import OccupancyGrid
from bb8_rl.room import generate_room, project_points
from bb8_rl.world import validate_world

BASE = Path(__file__).resolve().parents[1] / "configs/navigation/bb8-state.yaml"


def test_seeded_room_has_reproducible_geometry_and_connected_endpoints(tmp_path):
    layouts = []
    for name, seed in (("a", 42), ("b", 42), ("c", 91)):
        folder = tmp_path / name
        generate_room(BASE, folder, seed=seed)
        config, project, body, _, camera = validate_world(folder / "room.yaml")
        layout = json.loads((folder / "layout.json").read_text())
        boxes = layout["obstacles_xy_width_depth_height"]
        assert len(boxes) == 7 and len(project.objects) == 13
        assert config.provenance == "synthetic" and config.arena_half_extent == 2
        assert camera.position[2] == 5.2 and camera.resolution == (1280, 960)
        grid = OccupancyGrid(2, 0.1, body.radius + 0.11, [b[:4] for b in boxes])
        route = grid.route(layout["start_xy"], layout["goal_xy"])
        assert np.array_equal(route, layout["route_xy"])
        layouts.append(layout)
        with NavigationEnv(folder / "task.yaml") as env:
            assert env.task.layout_mode == "authored"
            assert env.authored_positions == tuple(tuple(b[:2]) for b in boxes)
            assert env.sizes == [tuple(b[2:]) for b in boxes]
    assert layouts[0] == layouts[1] and layouts[0] != layouts[2]
    with pytest.raises(ValueError, match="already exists"):
        generate_room(BASE, tmp_path / "a")


def test_room_dimensions_and_empty_layout(tmp_path):
    generate_room(BASE, tmp_path / "empty", side=5.0, obstacles=0)
    config, project, *_ = validate_world(tmp_path / "empty/room.yaml")
    assert config.arena_half_extent == 2.5 and len(project.objects) == 6
    for side in (float("nan"), 2.0, 4.05):
        with pytest.raises(ValueError):
            generate_room(BASE, tmp_path / "invalid", side=side)


def test_pinhole_projection_uses_world_to_camera_and_rejects_behind_camera():
    intrinsics = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
    extrinsics = np.eye(4)
    extrinsics[2, 3] = 2
    projected = project_points(intrinsics, extrinsics, [[0, 0, 0], [1, 1, 0]])
    assert np.array_equal(projected, [[50, 40], [100, 90]])
    with pytest.raises(ValueError, match="in front"):
        project_points(intrinsics, extrinsics, [[0, 0, -3]])
