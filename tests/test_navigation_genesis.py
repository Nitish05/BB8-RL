import math
import os
from itertools import pairwise
from pathlib import Path

import pytest

from bb8_rl.world import NavigationWorld

CONFIG = Path(__file__).resolve().parents[1] / "configs/navigation/bb8-state.yaml"
pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(
        os.getenv("RUN_GENESIS_INTEGRATION") != "1",
        reason="Opt-in real Genesis fixture",
    ),
]


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_real_drive_expiry_contact_and_reset(backend):
    if backend == "metal" and os.getenv("RUN_GENESIS_METAL") != "1":
        pytest.skip("Set RUN_GENESIS_METAL=1 on a native Mac with Metal")
    with NavigationWorld(CONFIG, backend=backend) as world:
        for _ in range(20):
            world.drive(1, 0)
            state = world.step(10)
        assert state.position[0] > 0.1
        assert abs(state.position[1]) < 0.005
        assert abs(float(world.body.get_quat()[0])) < 0.99  # shell really rolled
        assert world.contacts == 0  # floor contact excluded
        # Heading stays in the floor frame after rolling and a rotated reset pose.
        for _ in range(20):
            world.drive(0, 1)
            state = world.step(10)
        assert state.position[1] > 0.1 and state.velocity[1] > 0.1
        state = world.step(500)  # no new command; expiry must brake
        assert state.command_expired
        assert math.hypot(*state.velocity) < 0.01
        assert world.head.get_quat().tolist() == pytest.approx([1, 0, 0, 0])
        assert world.head.get_pos().tolist()[:2] == pytest.approx(
            state.position, abs=1e-6
        )
        initial = world.reset()
        assert initial.position == (0, 0) and initial.velocity == (0, 0)
        assert world.steps == 0 and not world.frames and world.backend.heading == 0
        assert world.body.get_ang().tolist() == pytest.approx([0, 0, 0], abs=1e-7)
        assert math.hypot(*world.step(100).velocity) < 1e-5
        world.reset()
        # Wall collision must be resolved by Genesis, not by position clipping.
        for _ in range(160):
            world.drive(1, 0)
            state = world.step(10)
        assert world.contacts > 0
        assert state.position[0] <= 1.5 - world.body_spec.radius + 0.005
        world.reset()
        assert world.contacts == 0 and not world.boundary_failure


def test_render_history_reset_and_fixed_camera():
    import numpy as np

    with NavigationWorld(CONFIG, render=True) as world:
        world.step(40)
        before = world.frames[-1].rgb.copy()
        stamps = [f.timestamp for f in world.frames]
        assert 1 < len(stamps) <= 4
        assert all(a < b for a, b in pairwise(stamps))
        assert before.shape == (240, 320, 3) and before.dtype == np.uint8
        assert np.ptp(before) > 0
        # Viewer pose is a separate camera; fixed observation camera is never rebound.
        camera_position = world.camera.pos.copy()
        world.drive(1, 0)
        world.step(40)
        assert np.allclose(world.camera.pos, camera_position)
        world.reset()
        assert not world.frames


def test_finer_physics_step_preserves_response(tmp_path):
    import json

    from bb8_rl.world import validate_world

    config, project, *_ = validate_world(CONFIG)
    outcomes = []
    for dt, steps in ((0.005, 10), (0.0025, 20)):
        project.physics.time_step = dt
        (tmp_path / "world.json").write_text(project.model_dump_json())
        data = config.model_dump()
        data.update(project="world.json", action_steps=steps)
        task = tmp_path / "task.yaml"
        task.write_text(json.dumps(data))
        with NavigationWorld(task) as world:
            for _ in range(20):
                world.drive(0.8, 0.2)
                state = world.step(steps)
            driven = state.position
            world.stop()
            stopped = world.step(round(1.5 / dt))
            outcomes.append((*driven, *stopped.position, *stopped.velocity))
    assert outcomes[0] == pytest.approx(outcomes[1], abs=0.005)
