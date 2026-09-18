import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bb8_rl.room import preview_room

pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(
        os.getenv("RUN_GENESIS_INTEGRATION") != "1", reason="Opt-in real camera"
    ),
]


def test_real_room_camera_framing_visibility_and_floor_calibration(tmp_path):
    config = (
        Path(__file__).resolve().parents[1] / "projects/bb8/synthetic-room/room.yaml"
    )
    result = preview_room(config, tmp_path)
    assert result["status"] == "passed" and result["bb8_visible_pixels"] >= 8
    rgb = np.asarray(Image.open(tmp_path / "camera.png"))
    assert rgb.shape == (960, 1280, 3) and np.ptp(rgb) > 0
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert calibration["provenance"] == "synthetic_exact"
    assert calibration["numerical_roundtrip_error_m"] < 1e-5
    # The projected shell center must land within its visible rendered footprint.
    from bb8_rl.room import project_points
    from bb8_rl.world import validate_world

    _, _, body, _, _ = validate_world(config)
    pixel = project_points(
        calibration["intrinsics"], calibration["world_to_camera"], [body.position]
    )[0]
    x0, y0, x1, y1 = result["bb8_bbox_xyxy"]
    assert x0 - 2 <= pixel[0] <= x1 + 2 and y0 - 2 <= pixel[1] <= y1 + 2


def test_authored_room_task_keeps_its_real_geometry_across_resets():
    from gymnasium.utils.env_checker import check_env

    from bb8_rl.env import NavigationEnv

    task = Path(__file__).resolve().parents[1] / "projects/bb8/synthetic-room/task.yaml"
    with NavigationEnv(task) as env:
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=23, options={"layout_seed": 100})
        assert env.layout.kind == "authored"
        for name, position in zip(env.config.obstacle_names, env.authored_positions):
            assert np.allclose(
                env.world.scene.get_entity(name=name).get_pos().tolist()[:2], position
            )
        for _ in range(3):
            env.step([0.1, 0.1])
        repeated, _ = env.reset(seed=23, options={"layout_seed": 100})
        assert all(np.array_equal(first[k], repeated[k]) for k in first)
