"""Hybrid navigation: map-derived waypoints and learned low-level control.

The planner never supplies drive actions to a learned policy at evaluation.
Teacher commands are used only for demonstration collection and training losses.
"""

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from .diagnostics import machine_report, write_report
from .env import NavigationEnv
from .training import file_hash

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "procedural": ROOT / "configs/navigation/bb8-task.yaml",
    "authored": ROOT / "projects/bb8/synthetic-room/task-penalty-v2.yaml",
}
FEATURE_VERSION = "waypoint-delta-velocity-final-v1"


def teacher_action(vector):
    """Training-only feedback labels for the declared synthetic drive parameters."""
    x = np.asarray(vector, dtype=np.float32)
    delta, velocity = x[..., :2], x[..., 2:4] * 0.35
    target = 1.6 * delta - 0.6 * velocity
    norm = np.linalg.norm(target, axis=-1, keepdims=True)
    magnitude = 0.05 + 0.95 * np.minimum(norm, 0.22) / 0.35
    action = target / np.maximum(norm, 1e-8) * magnitude
    stop = ((x[..., 4] > 0.5) & (np.linalg.norm(delta, axis=-1) < 0.035)) | (
        norm[..., 0] < 1e-8
    )
    return np.where(stop[..., None], 0, action).astype(np.float32)


class WaypointEnv(gym.Env):
    """Identical arrival gate, explicit waypoint observations and shaped reward."""

    def __init__(self, family="procedural", split="train", seed=17):
        self.core = NavigationEnv(TASKS[family], split=split, backend="cpu")
        self.core.task.max_episode_steps = 1200
        self.family, self.split, self.seed = family, split, seed
        self.episode = 0
        self.action_space = self.core.action_space
        self.observation_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)
        d = self.core.config.drive
        if (d.max_speed, d.dead_zone, d.heading_offset) != (0.35, 0.05, 0.0):
            raise ValueError(
                "Guided controller contract requires the trained drive calibration"
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self.seed * 1000000 + self.episode
        self.episode += 1
        self.raw, info = self.core.reset(seed=seed, options=options)
        self.route = self.core.grid.route(
            self.raw["achieved_goal"][:2], self.raw["desired_goal"][:2]
        )
        self.index = min(1, len(self.route) - 1)
        self.done = False
        return self.vector(), info

    def vector(self):
        delta = self.route[self.index] - self.raw["achieved_goal"][:2]
        velocity = self.raw["observation"][2:4]
        if (
            self.index < len(self.route) - 1
            and np.linalg.norm(delta) < 0.045
            and np.linalg.norm(velocity) < 0.06
        ):
            self.index += 1
            delta = self.route[self.index] - self.raw["achieved_goal"][:2]
        return np.asarray(
            [*delta, *(velocity / 0.35), self.index == len(self.route) - 1], np.float32
        )

    def remaining(self):
        return float(
            np.linalg.norm(self.route[self.index] - self.raw["achieved_goal"][:2])
            + np.linalg.norm(np.diff(self.route[self.index :], axis=0), axis=1).sum()
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset completed waypoint episode")
        before = self.remaining()
        self.raw, _, terminal, truncated, info = self.core.step(action)
        vector = self.vector()
        progress = before - self.remaining()
        success = info["is_success"]
        failure = info["collision"] or info["boundary_failure"]
        near = self.core.goal_matches(
            self.raw["achieved_goal"], self.raw["desired_goal"]
        )
        reward = 10.0 * progress - 0.01 - 0.01 * float(np.square(action).sum())
        reward += 0.2 * bool(near) + 25.0 * success - 50.0 * failure
        terminal = bool(terminal or success)
        truncated = bool(truncated and not terminal)
        self.done = terminal or truncated
        info.update(
            route_progress=progress,
            waypoint_index=self.index,
            observation_mode="oracle-waypoints",
            training_episode=self.episode - 1,
        )
        return vector, float(reward), terminal, truncated, info

    def close(self):
        self.core.close()


def collect_demonstrations(output, episodes=64, seed=17):
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve existing demonstrations; choose a fresh path")
    output.mkdir(parents=True)
    rows, transitions, results = [], [], []
    started = time.perf_counter()
    with WaypointEnv(seed=seed) as env:
        for episode in range(episodes):
            obs, info = env.reset(options={"layout_seed": episode})
            first, previous_reward = True, 0.0
            while True:
                action = teacher_action(obs)
                rows.append(
                    {
                        "vector": obs.copy(),
                        "action": action.copy(),
                        "teacher_action": action.copy(),
                        "reward": np.float32(previous_reward),
                        "is_first": np.bool_(first),
                        "is_last": np.bool_(False),
                        "is_terminal": np.bool_(False),
                    }
                )
                nxt, reward, terminal, truncated, info = env.step(action)
                transitions.append(
                    (obs.copy(), nxt.copy(), action.copy(), reward, terminal, truncated)
                )
                obs, previous_reward, first = nxt, reward, False
                if terminal or truncated:
                    rows.append(
                        {
                            "vector": obs.copy(),
                            "action": np.zeros(2, np.float32),
                            "teacher_action": teacher_action(obs),
                            "reward": np.float32(reward),
                            "is_first": np.bool_(False),
                            "is_last": np.bool_(True),
                            "is_terminal": np.bool_(terminal),
                        }
                    )
                    break
            results.append(
                {
                    "episode": episode,
                    "success": info["is_success"],
                    "collision": info["collision"],
                    "steps": env.core.steps,
                }
            )
            if episode % 8 == 0:
                print(
                    json.dumps(
                        {"demo_episodes": episode + 1, "transitions": len(transitions)}
                    ),
                    flush=True,
                )
        manifest = machine_report(env.core.world_path)
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    for index, key in enumerate(
        (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminated",
            "truncated",
        )
    ):
        arrays[key] = np.asarray([r[index] for r in transitions])
    np.savez_compressed(output / "demonstrations.npz", **arrays)
    manifest.update(
        status="complete",
        split="train",
        seed=seed,
        episodes=results,
        steps=len(transitions),
        elapsed_seconds=time.perf_counter() - started,
        dataset_sha256=file_hash(output / "demonstrations.npz"),
    )
    write_report(output / "manifest.json", manifest)
    return manifest


class GuidedSAC(SAC):
    """SAC's original losses plus a supervised gradient at the actor optimizer step."""

    def _setup_model(self):
        super()._setup_model()
        self.bc_weight = getattr(self, "bc_weight", 100.0)

        def add_teacher_gradient(optimizer, args, kwargs):
            batch = self.replay_buffer.sample(self.batch_size).observations
            target = torch.as_tensor(
                teacher_action(batch.detach().cpu().numpy()), device=batch.device
            )
            mean = self.actor(batch, deterministic=True)
            loss = torch.nn.functional.mse_loss(mean, target)
            (self.bc_weight * loss).backward()
            self.logger.record("train/imitation_mse", float(loss.detach()))

        self.actor.optimizer.register_step_pre_hook(add_teacher_gradient)


def train_sac(
    output,
    demonstrations,
    *,
    steps=20000,
    seed=17,
    pretrain_updates=4000,
    bc_weight=100.0,
):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose fresh output")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    core = WaypointEnv(seed=seed)
    vec = DummyVecEnv([lambda: core])
    manifest = machine_report(core.core.world_path)
    manifest.update(
        status="running",
        algorithm="SAC with teacher regularization and A* waypoints",
        seed=seed,
        requested_steps=steps,
        bc_weight=bc_weight,
        demonstration_sha256=file_hash(demonstrations),
        feature_version=FEATURE_VERSION,
    )
    write_report(output / "manifest.json", manifest)
    try:
        model = GuidedSAC(
            "MlpPolicy",
            vec,
            seed=seed,
            device="cpu",
            learning_rate=0.0003,
            batch_size=256,
            buffer_size=100000,
            learning_starts=0,
            train_freq=4,
            gradient_steps=1,
            gamma=0.995,
            ent_coef=0.0001,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=0,
        )
        model.bc_weight = bc_weight
        model.set_logger(configure(str(output / "logs"), ["csv"]))
        data = np.load(demonstrations)
        for i in range(len(data["observations"])):
            model.replay_buffer.add(
                data["observations"][i : i + 1],
                data["next_observations"][i : i + 1],
                data["actions"][i : i + 1],
                data["rewards"][i : i + 1],
                (data["terminated"][i : i + 1] | data["truncated"][i : i + 1]),
                [{"TimeLimit.truncated": bool(data["truncated"][i])}],
            )
        # Supplement real demonstrations with teacher-labeled recovery states.
        rng = np.random.default_rng(seed)
        angle = rng.uniform(-np.pi, np.pi, 100000)
        radius = np.exp(rng.uniform(np.log(0.002), np.log(4), len(angle)))
        synthetic = np.c_[
            radius * np.cos(angle),
            radius * np.sin(angle),
            rng.uniform(-1, 1, (len(angle), 2)),
            rng.integers(0, 2, len(angle)),
        ].astype(np.float32)
        inputs = torch.as_tensor(np.concatenate((data["observations"], synthetic)))
        targets = torch.as_tensor(teacher_action(inputs.numpy()))
        opt = torch.optim.Adam(model.actor.parameters(), lr=0.001)
        for _ in range(pretrain_updates):
            idx = torch.randint(len(inputs), (512,))
            loss = (
                (model.actor(inputs[idx], deterministic=True) - targets[idx])
                .square()
                .mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.nn.init.zeros_(model.actor.log_std.weight)
        torch.nn.init.constant_(model.actor.log_std.bias, -4)
        model.save(output / "bc-only.zip")
        manifest.update(
            pretrain_mse=float(loss.detach()),
            pretrain_updates=pretrain_updates,
            synthetic_teacher_states=len(synthetic),
            demonstration_steps=len(data["observations"]),
        )
        print(json.dumps({"pretrain_mse": manifest["pretrain_mse"]}), flush=True)
        # One learner chunk per public call; the environment carries across chunks.
        while model.num_timesteps < steps:
            model.learn(
                total_timesteps=min(1000, steps - model.num_timesteps),
                reset_num_timesteps=False,
                log_interval=1,
            )
            model.logger.dump(model.num_timesteps)
            print(
                json.dumps({"steps": model.num_timesteps, "updates": model._n_updates}),
                flush=True,
            )
        model.save(output / "model.zip")
        model.save_replay_buffer(output / "replay.pkl")
        manifest.update(
            status="complete",
            steps=model.num_timesteps,
            updates=model._n_updates,
            elapsed_seconds=time.perf_counter() - started,
            model_sha256=file_hash(output / "model.zip"),
            bc_model_sha256=file_hash(output / "bc-only.zip"),
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        vec.close()
        write_report(output / "manifest.json", manifest)
    return manifest
