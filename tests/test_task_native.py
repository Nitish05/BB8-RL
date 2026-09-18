import os
from pathlib import Path

import pytest

from bb8_rl.baseline import RouteController
from bb8_rl.env import NavigationEnv

pytestmark = [
    pytest.mark.native_gui,
    pytest.mark.skipif(os.getenv("RUN_NATIVE_GUI") != "1", reason="Opt-in task viewer"),
]


def test_obstacle_task_and_route_controller_in_native_viewer():
    path = Path(__file__).resolve().parents[1] / "configs/navigation/bb8-task.yaml"
    with NavigationEnv(path, render_mode="human") as env:
        obs, _ = env.reset(seed=5, options={"layout_seed": 1})
        controller = RouteController(env.task, env.config.drive.parameters())
        controller.reset(env.grid, obs)
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(controller.action(obs))
        assert not terminated and env.world.scene.viewer.is_alive()
