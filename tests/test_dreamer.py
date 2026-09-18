"""Interface tests independent of optional JAX and native simulation."""

import json
from pathlib import Path

import numpy as np
import pytest

from bb8_rl.dreamer import (
    UPSTREAM_REVISION,
    DreamerConfig,
    DreamerEnv,
    FeatureAdapter,
    observation,
    read_checkpoint,
)
from bb8_rl.env import NavigationEnv
from bb8_rl.training import file_hash

ROOT = Path(__file__).resolve().parents[1]


def test_first_timeout_and_terminal_are_distinct():
    vector = np.zeros(180, np.float32)
    reset = observation(vector, first=True)
    timeout = observation(vector, -1, truncated=True)
    collision = observation(vector, -101, terminated=True)
    assert reset["is_first"] and not reset["is_last"] and not reset["is_terminal"]
    assert timeout["is_last"] and not timeout["is_terminal"]
    assert collision["is_last"] and collision["is_terminal"]
    assert timeout["reward"] == -1 and collision["reward"] == -101
    assert all(
        np.asarray(row[k]).dtype == bool
        for row in (reset, timeout, collision)
        for k in ("is_first", "is_last", "is_terminal")
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_adapter_preserves_last_observation_before_reset(terminal):
    class Fake:
        resets = 0

        def reset(self):
            self.resets += 1
            return np.array([self.resets], np.float32), {"reset": True}

        def step(self, action):
            np.testing.assert_equal(action, [1, -1])
            return (
                np.array([99], np.float32),
                -101,
                terminal,
                not terminal,
                {"end": True},
            )

    adapter = DreamerEnv.__new__(DreamerEnv)
    adapter.env, adapter.features, adapter.done = Fake(), lambda x: x, True
    first = adapter.step({"action": np.zeros(2), "reset": True})
    end = adapter.step({"action": np.array([2, -2]), "reset": False})
    assert first["is_first"] and adapter.env.resets == 1
    assert end["is_last"] and end["is_terminal"] == terminal
    assert end["vector"][0] == 99 and end["reward"] == -101
    again = adapter.step({"action": np.zeros(2), "reset": True})
    assert again["is_first"] and again["vector"][0] == 2


def test_features_preserve_goal_and_local_map_information():
    with NavigationEnv(
        ROOT / "projects/bb8/synthetic-room/task-penalty-v2.yaml"
    ) as env:
        encode = FeatureAdapter(env)
        obs = {
            k: np.zeros(s.shape, np.float32)
            for k, s in env.observation_space.spaces.items()
        }
        obs["observation"][30:] = 1
        obs["desired_goal"][:2] = [1, -1]
        features = encode(obs)
        assert features.shape == (180,)
        np.testing.assert_equal(features[30:177], 1)
        np.testing.assert_allclose(features[-3:], [0.5, -0.5, 0])
        obs["achieved_goal"][:2] = [0.5, -0.5]
        np.testing.assert_allclose(encode(obs)[-3:], [0.25, -0.25, 0])


def test_config_rejects_unusable_replay_and_update_ratio():
    for override in ({"train_ratio": 17}, {"learning_starts": 1}, {"buffer_size": 128}):
        with pytest.raises(ValueError):
            DreamerConfig(task="task.yaml", upstream="upstream", **override)


def test_checkpoint_checksum_and_algorithm_are_checked(tmp_path):
    payload = tmp_path / "agent.pkl"
    payload.write_bytes(b"test payload, not unpickled")
    state = {
        "schema_version": 1,
        "algorithm": "dreamerv3",
        "upstream_revision": UPSTREAM_REVISION,
        "agent_sha256": file_hash(payload),
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert read_checkpoint(tmp_path) == state
    payload.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(tmp_path)
