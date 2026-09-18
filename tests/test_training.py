import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from bb8_rl.env import NavigationEnv
from bb8_rl.training import NavigationFeatures, TrainingConfig, read_checkpoint

TASK = Path(__file__).resolve().parents[1] / "projects/bb8/synthetic-room/task.yaml"


def test_goal_features_recompute_after_relabel_and_state_stays_independent():
    env = NavigationEnv(TASK)
    extractor = NavigationFeatures(
        env.observation_space, map_size=21, history_length=4, extent=2, max_speed=0.35
    )
    obs = {
        k: torch.zeros((1, *space.shape))
        for k, space in env.observation_space.spaces.items()
    }
    first = extractor(obs)
    obs["desired_goal"][0] = torch.tensor([1.0, -1.0, 0.35])
    second = extractor(obs)
    assert torch.equal(first[:, :-3], second[:, :-3])
    assert torch.allclose(second[:, -3:], torch.tensor([[0.5, -0.5, 1.0]]))


def test_actual_her_buffer_retains_collision_penalties_and_timeout_bootstrap():
    env = DummyVecEnv([lambda: NavigationEnv(TASK)])
    buffer = HerReplayBuffer(
        32,
        env.observation_space,
        env.action_space,
        env,
        n_sampled_goal=4,
        copy_info_dict=True,
        device="cpu",
    )
    obs = {
        k: np.zeros((1, *s.shape), dtype=np.float32)
        for k, s in env.observation_space.spaces.items()
    }
    obs["desired_goal"][0, 0] = 1
    buffer.add(
        obs,
        obs,
        np.zeros((1, 2)),
        np.array([-11]),
        np.array([True]),
        [{"event_penalty": 10, "TimeLimit.truncated": False}],
    )
    buffer.add(
        obs,
        obs,
        np.zeros((1, 2)),
        np.array([-1]),
        np.array([True]),
        [{"event_penalty": 0, "TimeLimit.truncated": True}],
    )
    samples = buffer.sample(256)
    rewards, dones = samples.rewards.flatten(), samples.dones.flatten()
    assert bool(torch.any(rewards == -10)) and bool(torch.any(rewards == 0))
    assert torch.equal(dones.bool(), rewards <= -10)
    env.close()


def test_checkpoint_checksum_failure_is_detected_before_loading(tmp_path):
    (tmp_path / "policy.pt").write_bytes(b"tampered")
    (tmp_path / "state.json").write_text(
        json.dumps({"schema_version": 1, "files": {"policy.pt": "bad"}})
    )
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(tmp_path, full=False)
    with pytest.raises(ValueError):
        TrainingConfig(task="unused", learning_rate=float("nan"))


def test_corrected_failure_penalty_removes_early_termination_advantage():
    original = NavigationEnv(TASK)
    corrected = NavigationEnv(TASK.with_name("task-penalty-v2.yaml"))
    gamma = 0.98
    continuing_return = -1 / (1 - gamma)
    for failure_penalty in (
        corrected.task.collision_penalty,
        corrected.task.boundary_penalty,
    ):
        failure_reward = float(
            corrected.compute_reward(
                [0, 0, 0], [1, 1, 0], {"event_penalty": failure_penalty}
            )
        )
        for length in (1, 10, 50, 200):
            failure_return = (
                -sum(gamma**i for i in range(length - 1))
                + gamma ** (length - 1) * failure_reward
            )
            assert failure_return < continuing_return
    old_failure = float(
        original.compute_reward(
            [0, 0, 0], [1, 1, 0], {"event_penalty": original.task.collision_penalty}
        )
    )
    assert (
        old_failure > continuing_return
    )  # Documents the original pilot's objective defect.
