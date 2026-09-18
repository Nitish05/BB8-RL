import os

import numpy as np
import pytest

from bb8_rl.guided import WaypointEnv

pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(
        os.getenv("RUN_GENESIS_INTEGRATION") != "1", reason="Native waypoint adapter"
    ),
]


def test_arrival_terminates_only_after_original_dwell_and_speed_gate():
    with WaypointEnv() as env:
        obs, _ = env.reset(
            seed=1, options={"layout_seed": 0, "start": (0, 0), "goal": (0, 0)}
        )
        assert obs.shape == (5,) and obs[-1] == 1
        for _ in range(9):
            _, _, terminal, truncated, info = env.step(np.zeros(2, np.float32))
            assert not terminal and not truncated and not info["is_success"]
        _, reward, terminal, truncated, info = env.step(np.zeros(2, np.float32))
        assert terminal and not truncated and info["is_success"] and reward > 25
        assert "episode" not in info  # Reserved by the SB3 Monitor wrapper.
        assert "training_episode" in info
        assert info["dwell_seconds"] >= 0.5 - 1e-9
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(np.zeros(2, np.float32))


def test_time_limit_is_truncation_and_progress_reward_is_goal_directed():
    with WaypointEnv() as env:
        env.core.task.max_episode_steps = 1
        env.reset(seed=1, options={"layout_seed": 0, "start": (0, 0), "goal": (0.8, 0)})
        _, reward, terminal, truncated, info = env.step(np.zeros(2, np.float32))
        assert truncated and not terminal and not info["is_success"]
        assert abs(reward + 0.01) < 1e-6
