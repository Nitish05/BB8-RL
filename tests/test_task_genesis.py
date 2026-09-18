import os
from pathlib import Path

import numpy as np
import pytest

from bb8_rl.env import NavigationEnv
from bb8_rl.world import NavigationWorld

TASK = Path(__file__).resolve().parents[1] / "configs/navigation/bb8-task.yaml"
pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(
        os.getenv("RUN_GENESIS_INTEGRATION") != "1",
        reason="Opt-in real Genesis navigation task",
    ),
]


def test_gymnasium_checker_and_seeded_reset():
    from gymnasium.utils.env_checker import check_env

    with NavigationEnv(TASK) as env:
        check_env(env, skip_render_check=True)
        first, info = env.reset(seed=14, options={"layout_seed": 3})
        for _ in range(3):
            env.step(np.array([0.5, 0.1], dtype=np.float32))
        again, repeat = env.reset(seed=14, options={"layout_seed": 3})
        assert all(np.array_equal(first[k], again[k]) for k in first)
        assert repeat == info and env.world.contacts == 0


def test_arrival_dwell_is_not_training_termination_and_timeout_requires_reset():
    with NavigationEnv(TASK) as env:
        env.task.max_episode_steps = 12
        _obs, _ = env.reset(
            seed=1, options={"layout_seed": 0, "start": (0.0, 0.0), "goal": (0.0, 0.0)}
        )
        for _ in range(10):
            _obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        assert info["is_success"] and not terminated and not truncated and reward == 0
        for _ in range(2):
            terminal, _, terminated, truncated, info = env.step([0.0, 0.0])
        assert truncated and not terminated
        with pytest.raises(RuntimeError, match="reset"):
            env.step([0.0, 0.0])
        env.reset(
            seed=1,
            options={"layout_seed": 0, "start": (-0.65, 0.0), "goal": (0.65, 0.0)},
        )
        assert np.array_equal(terminal["desired_goal"], [0, 0, 0])


def test_goal_does_not_leak_into_flat_observation():
    with NavigationEnv(TASK) as env:
        a, _ = env.reset(
            seed=1, options={"layout_seed": 0, "start": (0.0, 0.0), "goal": (0.65, 0.0)}
        )
        b, _ = env.reset(
            seed=1,
            options={"layout_seed": 0, "start": (0.0, 0.0), "goal": (-0.65, 0.0)},
        )
        assert np.array_equal(a["observation"], b["observation"])
        assert not np.array_equal(a["desired_goal"], b["desired_goal"])


def test_repositioned_obstacle_contact_terminates_and_resets_clear_slots():
    with NavigationEnv(TASK) as env:
        env.reset(
            seed=5,
            options={"layout_seed": 1, "start": (-0.65, 0.05), "goal": (0.95, 0.05)},
        )
        for _ in range(150):
            terminal, reward, terminated, truncated, info = env.step([1.0, 0.0])
            if terminated or truncated:
                break
        assert (
            terminated
            and info["collision"]
            and reward <= -10
            and not info["is_success"]
        )
        assert terminal["achieved_goal"][0] < 0.2
        assert env.world.backend.applied_request == (0.0, 0.0)
        env.reset(
            seed=5,
            options={"layout_seed": 0, "start": (-0.65, 0.05), "goal": (0.95, 0.05)},
        )
        for _ in range(60):
            obs, _, terminated, _, info = env.step([1.0, 0.0])
            assert not terminated
        assert obs["achieved_goal"][0] > 0.05 and not info["collision"]


def test_second_engine_creation_does_not_destroy_existing_world():
    with NavigationEnv(TASK) as env:
        env.reset(seed=1)
        with pytest.raises(env.world.gs.GenesisException, match="already initialized"):
            NavigationWorld(env.world_path)
        obs, *_ = env.step([0.0, 0.0])
        assert np.isfinite(obs["observation"]).all()


def test_numerical_failure_stops_and_requires_reset(monkeypatch):
    with NavigationEnv(TASK) as env:
        env.reset(seed=1)

        def invalid_step():
            raise RuntimeError("Invalid numerical state")

        monkeypatch.setattr(env.world, "step", invalid_step)
        with pytest.raises(RuntimeError, match="numerical"):
            env.step([0.5, 0])
        assert env.world.backend.applied_request == (0.0, 0.0)
        with pytest.raises(RuntimeError, match="reset"):
            env.step([0, 0])
