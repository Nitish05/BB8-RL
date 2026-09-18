import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from bb8_rl.env import NavigationEnv
from bb8_rl.training import load_policy, read_checkpoint, train

pytestmark = [
    pytest.mark.genesis,
    pytest.mark.skipif(
        os.getenv("RUN_GENESIS_INTEGRATION") != "1", reason="Opt-in real training"
    ),
]
ROOT = Path(__file__).resolve().parents[1]


def test_real_sac_her_checkpoint_replay_and_resume(tmp_path):
    task = yaml.safe_load((ROOT / "projects/bb8/synthetic-room/task.yaml").read_text())
    task.update(
        world_config=str(ROOT / "projects/bb8/synthetic-room/room.yaml"),
        max_episode_steps=20,
    )
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task))
    config = {
        "task": "task.yaml",
        "seed": 7,
        "total_steps": 40,
        "checkpoint_every": 40,
        "buffer_size": 128,
        "learning_starts": 20,
        "batch_size": 8,
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    first = train(path, tmp_path / "first")
    state = read_checkpoint(first["final_checkpoint"])
    assert first["steps"] >= 40 and first["updates"] > 0 and state["replay_size"] >= 40
    with NavigationEnv(tmp_path / "task.yaml") as env:
        obs, _ = env.reset(seed=3)
        policy, _ = load_policy(first["final_checkpoint"], env)
        again, _ = load_policy(first["final_checkpoint"], env)
        action, _ = policy.predict(obs, deterministic=True)
        assert np.array_equal(action, again.predict(obs, deterministic=True)[0])
        assert np.isfinite(action).all() and np.max(abs(action)) <= 1
    config["total_steps"] = 80
    path.write_text(yaml.safe_dump(config))
    second = train(path, tmp_path / "resumed", resume=first["final_checkpoint"])
    after = read_checkpoint(second["final_checkpoint"])
    assert second["steps"] > first["steps"] and second["updates"] > first["updates"]
    assert (
        after["replay_size"] > state["replay_size"]
        and after["next_episode"] > state["next_episode"]
    )
    manifest = json.loads((tmp_path / "resumed/manifest.json").read_text())
    assert (
        manifest["initial_steps"] == first["steps"]
        and manifest["initial_updates"] == first["updates"]
    )
    config.update(total_steps=120, gamma=0.5)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="matching"):
        train(path, tmp_path / "wrong-contract", resume=second["final_checkpoint"])


def test_sigterm_preserves_a_complete_checkpoint(tmp_path):
    task = yaml.safe_load((ROOT / "projects/bb8/synthetic-room/task.yaml").read_text())
    task.update(
        world_config=str(ROOT / "projects/bb8/synthetic-room/room.yaml"),
        max_episode_steps=20,
    )
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task))
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "task": "task.yaml",
                "total_steps": 10000,
                "buffer_size": 128,
                "batch_size": 8,
                "learning_starts": 20,
            }
        )
    )
    output = tmp_path / "run"
    with (
        (tmp_path / "process.log").open("w") as log,
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bb8_rl.cli",
                "train",
                "--config",
                str(path),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process,
    ):
        deadline = time.monotonic() + 60
        try:
            while not (output / "latest.json").exists():
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=60) == 130
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    state = read_checkpoint(manifest["final_checkpoint"])
    assert state["steps"] == manifest["steps"]
