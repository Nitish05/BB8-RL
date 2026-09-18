"""Verify real checkpoint tensors and seeded inference through native Genesis."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from bb8_rl.diagnostics import write_report
from bb8_rl.dreamer import DreamerPolicy
from bb8_rl.env import NavigationEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
with NavigationEnv(args.task, split="validation") as env:
    obs, _ = env.reset(seed=200001, options={"layout_seed": 10000})
    policy = DreamerPolicy(args.checkpoint, env)
    with (args.checkpoint / "agent.pkl").open("rb") as file:
        stored = pickle.load(file)
    loaded = policy.agent.save()
    assert stored["params"].keys() == loaded["params"].keys()
    assert all(
        np.array_equal(value, loaded["params"][key])
        for key, value in stored["params"].items()
    )
    policy.reset(200001)
    observations, first = [], []
    for _ in range(8):
        observations.append(obs)
        action, _ = policy.predict(obs)
        assert np.isfinite(action).all() and np.max(abs(action)) <= 1
        first.append(action.tolist())
        obs, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
    again = DreamerPolicy(args.checkpoint, env)
    again.reset(200001)
    second = [again.predict(obs)[0].tolist() for obs in observations]
    assert np.array_equal(first, second)
    result = {
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "parameters_and_optimizer_tensors_equal": True,
        "reloaded_seeded_action_trace_equal": True,
        "actions": first,
        "steps": policy.state["steps"],
        "updates": policy.state["updates"],
    }
    write_report(args.output, result)
    print(json.dumps(result))
