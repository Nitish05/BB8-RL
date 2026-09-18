"""Local SAC/HER training with complete episode-boundary checkpoints."""

import gzip
import hashlib
import io
import json
import pickle
import random
import signal
import time
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac.policies import SACPolicy
from torch import nn
from torch.nn import functional as F

from .diagnostics import machine_report, write_report
from .env import NavigationEnv


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    task: str
    seed: int = Field(default=17, ge=0, le=100000)
    total_steps: int = Field(default=12000, ge=1, le=10000000)
    checkpoint_every: int = Field(default=6000, ge=1)
    buffer_size: int = Field(default=20000, ge=32, le=1000000)
    learning_starts: int = Field(default=1500, ge=1)
    batch_size: int = Field(default=128, ge=2)
    learning_rate: float = Field(default=0.0003, gt=0)
    gamma: float = Field(default=0.98, gt=0, le=1)
    tau: float = Field(default=0.005, gt=0, le=1)
    device: Literal["cpu", "mps"] = "cpu"
    physics_backend: Literal["cpu", "metal"] = "cpu"
    torch_threads: int = Field(default=2, ge=1, le=8)


def load_training(path):
    path = Path(path)
    cfg = TrainingConfig.model_validate(yaml.safe_load(path.read_text()))
    return cfg, (path.resolve().parent / cfg.task).resolve()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_contract(env):
    from .config import load_config

    _, project = load_config(env.world_path)
    return {
        "task": env.task.model_dump(),
        "world_config_sha256": file_hash(env.world_path),
        "project_sha256": file_hash(project),
        "features": "relative-goal-scaled-state-average-map-7-v1",
        "reward": "sparse-goal-v1",
    }


class NavigationFeatures(BaseFeaturesExtractor):
    """Goal differences are calculated after HER relabeling, not stored in replay."""

    def __init__(
        self, observation_space, *, map_size, history_length, extent, max_speed
    ):
        self.state_size = 6 + 6 * history_length
        self.map_size = map_size
        super().__init__(
            observation_space, features_dim=self.state_size + 3 * 7 * 7 + 3
        )
        scales = np.ones(self.state_size, dtype=np.float32)
        scales[:2], scales[2:4] = extent, max_speed
        for i in range(history_length):
            scales[6 + i * 6 : 8 + i * 6] = max_speed
        self.register_buffer("state_scales", torch.tensor(scales))
        self.register_buffer("goal_scales", torch.tensor([extent, extent, max_speed]))

    def forward(self, observations):
        flat = observations["observation"]
        state = flat[:, : self.state_size] / self.state_scales
        local = flat[:, self.state_size :].reshape(-1, 3, self.map_size, self.map_size)
        pooled = F.adaptive_avg_pool2d(local, (7, 7)).flatten(1)
        relative = (
            observations["desired_goal"] - observations["achieved_goal"]
        ) / self.goal_scales
        return torch.cat((state, pooled, relative), dim=1)


class TrainingEpisodes(gym.Wrapper):
    """Each episode has its own deterministic RNG; reset carries no solver history."""

    def __init__(self, env, seed, next_episode=0):
        super().__init__(env)
        self.training_seed = seed
        self.next_episode = next_episode
        self.current_episode = None

    def reset(self, *, seed=None, options=None):
        index = self.next_episode
        episode_seed = self.training_seed * 1000000 + index
        obs, info = self.env.reset(seed=episode_seed, options=options)
        self.current_episode = index
        self.next_episode += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["training_episode"] = self.current_episode
        return obs, reward, terminated, truncated, info


class EpisodeLog(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = Path(output)
        self.episode_return, self.length, self.arrived = 0.0, 0, False

    def _on_step(self):
        info = self.locals["infos"][0]
        self.episode_return += float(self.locals["rewards"][0])
        self.length += 1
        self.arrived |= info["is_success"]
        if self.locals["dones"][0]:
            record = {
                "step": self.num_timesteps,
                "episode": info["training_episode"],
                "return": self.episode_return,
                "length": self.length,
                "arrived_anytime": self.arrived,
                "success_at_end": info["is_success"],
                "collision": info["collision"],
                "boundary_failure": info["boundary_failure"],
                "timeout": info.get("TimeLimit.truncated", False),
            }
            with self.output.open("a") as f:
                f.write(json.dumps(record) + "\n")
            self.episode_return, self.length, self.arrived = 0.0, 0, False
        return True


def require_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; no silent fallback")
    if name not in ("cpu", "mps"):
        raise ValueError("Choose cpu or mps")


class TimedSAC(SAC):
    """Measure actual optimization separately from physics, startup and checkpoints."""

    def train(self, gradient_steps, batch_size=64):
        if self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.gradient_wall_seconds = (
            getattr(self, "gradient_wall_seconds", 0.0) + time.perf_counter() - started
        )
        if any(
            not np.isfinite(value)
            for key, value in self.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (float, np.floating))
        ):
            raise RuntimeError("Nonfinite training metric")


def make_model(env, cfg):
    core = env.envs[0].unwrapped
    return TimedSAC(
        "MultiInputPolicy",
        env,
        seed=cfg.seed,
        device=cfg.device,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        tau=cfg.tau,
        ent_coef="auto",
        train_freq=(1, "episode"),
        gradient_steps=-1,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
            "handle_timeout_termination": True,
        },
        policy_kwargs={
            "net_arch": [128, 128],
            "activation_fn": nn.ReLU,
            "features_extractor_class": NavigationFeatures,
            "features_extractor_kwargs": {
                "map_size": core.task.local_map_size,
                "history_length": core.task.history_length,
                "extent": core.config.arena_half_extent,
                "max_speed": core.config.drive.max_speed,
            },
        },
        verbose=0,
    )


def rng_state(model):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "action": model.action_space.np_random.bit_generator.state,
    }
    if model.device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, model):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    model.action_space.np_random.bit_generator.state = state["action"]
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def finite_model(model):
    tensors = list(model.policy.parameters())
    if model.log_ent_coef is not None:
        tensors.append(model.log_ent_coef)
    if not all(bool(torch.isfinite(p).all()) for p in tensors):
        raise RuntimeError("Nonfinite model parameters; checkpoint rejected")


def save_checkpoint(model, wrapper, output, cfg, contract, source_hash):
    finite_model(model)
    if wrapper.unwrapped.steps != 0:
        raise RuntimeError("Checkpoint requires the freshly reset next episode")
    # HER must contain no incomplete trajectory at this episode boundary.
    if np.any(model.replay_buffer._current_ep_start != model.replay_buffer.pos):
        raise RuntimeError("Checkpoint contains an incomplete HER episode")
    path = Path(output) / "checkpoints" / f"step-{model.num_timesteps:09d}"
    if path.exists():
        raise ValueError("Checkpoint already exists")
    staging = path.with_name("." + path.name + ".partial")
    staging.mkdir(parents=True)
    model.save(staging / "model.zip")
    model.policy.save(staging / "policy.pt")
    with (
        gzip.open(staging / "replay.pkl.gz", "wb", compresslevel=1) as raw,
        io.BufferedWriter(raw) as f,
    ):
        model.save_replay_buffer(f)
    with (staging / "rng.pkl").open("wb") as f:
        pickle.dump(rng_state(model), f, protocol=pickle.HIGHEST_PROTOCOL)
    state = {
        "schema_version": 1,
        "steps": model.num_timesteps,
        "updates": model._n_updates,
        "gradient_wall_seconds": getattr(model, "gradient_wall_seconds", 0.0),
        "next_episode": wrapper.current_episode,
        "config": cfg.model_dump(),
        "contract": contract,
        "source_sha256": source_hash,
        "replay_size": model.replay_buffer.size(),
        "resume_mode": "fresh episode boundary; no solver snapshot",
        "normalization": "fixed feature scaling, no running statistics",
        "files": {
            name: file_hash(staging / name)
            for name in ("model.zip", "policy.pt", "replay.pkl.gz", "rng.pkl")
        },
    }
    write_report(staging / "state.json", state)
    staging.rename(path)
    write_report(
        Path(output) / "latest.json",
        {"checkpoint": str(path.resolve()), "steps": model.num_timesteps},
    )
    return path


def read_checkpoint(path, *, full=True):
    path = Path(path)
    state = json.loads((path / "state.json").read_text())
    if state.get("schema_version") != 1:
        raise ValueError("Unknown checkpoint schema")
    for name in (
        ("model.zip", "policy.pt", "replay.pkl.gz", "rng.pkl")
        if full
        else ("policy.pt",)
    ):
        if file_hash(path / name) != state["files"][name]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return state


def load_policy(checkpoint, env, device="cpu"):
    require_device(device)
    state = read_checkpoint(checkpoint, full=False)
    if state["contract"] != task_contract(env):
        raise ValueError(
            "Checkpoint task/world/reward contract does not match evaluation"
        )
    policy = SACPolicy.load(Path(checkpoint) / "policy.pt", device=device)
    policy.set_training_mode(False)
    return policy, state


def train(config_path, output, *, resume=None):
    cfg, task_path = load_training(config_path)
    require_device(cfg.device)
    torch.set_num_threads(cfg.torch_threads)
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Run already exists; resume into a fresh output directory")
    core = NavigationEnv(task_path, split="train", backend=cfg.physics_backend)
    if (
        cfg.learning_starts < core.task.max_episode_steps
        or cfg.buffer_size <= 2 * core.task.max_episode_steps
    ):
        raise ValueError(
            "Warmup must cover a complete episode and replay must exceed two episode horizons"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = machine_report(core.world_path)
    manifest.update(
        status="running",
        algorithm="SAC + HER",
        training_config=cfg.model_dump(),
        task_path=str(task_path),
        contract=task_contract(core),
        resumed_from=str(Path(resume).resolve()) if resume else None,
        requested_total_steps=cfg.total_steps,
        learning_rate_schedule="constant",
    )
    source_hash = manifest["bb8_source_sha256"]
    previous = read_checkpoint(resume) if resume else None
    if previous:
        allowed = {"total_steps", "checkpoint_every", "task"}
        old = {k: v for k, v in previous["config"].items() if k not in allowed}
        new = {k: v for k, v in cfg.model_dump().items() if k not in allowed}
        if (
            old != new
            or previous["contract"] != manifest["contract"]
            or previous["source_sha256"] != source_hash
        ):
            raise ValueError(
                "Resume requires matching training settings, task/world contract and source"
            )
        if previous["steps"] >= cfg.total_steps:
            raise ValueError("Resume total_steps must exceed checkpoint steps")
    wrapper = TrainingEpisodes(
        core, cfg.seed, previous["next_episode"] if previous else 0
    )
    vec = DummyVecEnv([lambda: wrapper])
    model = None
    started = time.perf_counter()
    requested_stop = False
    old_handlers = {}

    def stop_handler(signum, frame):
        nonlocal requested_stop
        if requested_stop:
            raise KeyboardInterrupt()
        requested_stop = True
        print(
            "Cancellation requested; finishing this episode and its gradient updates.",
            flush=True,
        )

    write_report(output / "manifest.json", manifest)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop_handler)
        if previous:
            model = TimedSAC.load(
                Path(resume) / "model.zip", env=vec, device=cfg.device, force_reset=True
            )
            with (
                gzip.open(Path(resume) / "replay.pkl.gz", "rb") as raw,
                io.BufferedReader(raw) as f,
            ):
                model.load_replay_buffer(f, truncate_last_traj=False)
        else:
            model = make_model(vec, cfg)
        # Initialize Genesis before restoring policy RNG; engine init may seed libraries.
        # These SB3 fields establish the already-reset first observation for learn().
        model._last_obs = vec.reset()
        model._last_episode_starts = np.ones(1, dtype=bool)
        # Native engine initialization replaces OS signal handlers. Re-arm after
        # it completes so SIGTERM/SIGINT reach our episode-boundary cancellation.
        for sig in old_handlers:
            signal.signal(sig, stop_handler)
        if previous:
            with (Path(resume) / "rng.pkl").open("rb") as f:
                restore_rng(
                    pickle.load(f), model
                )  # trusted locally produced checkpoints only
        else:
            # Engine initialization is done: policy exploration receives the declared seed.
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            model.action_space.seed(cfg.seed)
        if model.device.type != cfg.device:
            raise RuntimeError("Learner device differs from requested device")
        model.set_logger(configure(str(output / "logs"), ["csv"]))
        callback = EpisodeLog(output / "episodes.jsonl")
        initial_steps = model.num_timesteps
        initial_gradient_seconds = getattr(model, "gradient_wall_seconds", 0.0)
        manifest.update(
            resolved_device=str(model.device),
            resolved_physics_backend=core.world.gs.backend.name,
            initial_steps=initial_steps,
            initial_updates=model._n_updates,
        )
        checkpoint = save_checkpoint(
            model, wrapper, output, cfg, manifest["contract"], source_hash
        )
        next_save = model.num_timesteps + cfg.checkpoint_every
        last_progress = started
        while model.num_timesteps < cfg.total_steps and not requested_stop:
            # Public learn API, one complete episode per call. All schedules are constant.
            model.learn(
                total_timesteps=1,
                reset_num_timesteps=False,
                callback=callback,
                log_interval=1,
            )
            finite_model(model)
            model.logger.dump(step=model.num_timesteps)
            if (
                model.num_timesteps >= next_save
                or model.num_timesteps >= cfg.total_steps
                or requested_stop
            ):
                checkpoint = save_checkpoint(
                    model, wrapper, output, cfg, manifest["contract"], source_hash
                )
                next_save = model.num_timesteps + cfg.checkpoint_every
            if (
                time.perf_counter() - last_progress > 10
                or model.num_timesteps >= cfg.total_steps
            ):
                print(
                    json.dumps(
                        {
                            "steps": model.num_timesteps,
                            "updates": model._n_updates,
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last_progress = time.perf_counter()
        if (
            requested_stop
            and json.loads((checkpoint / "state.json").read_text())["steps"]
            != model.num_timesteps
        ):
            checkpoint = save_checkpoint(
                model, wrapper, output, cfg, manifest["contract"], source_hash
            )
        manifest.update(
            status="cancelled" if requested_stop else "complete",
            steps=model.num_timesteps,
            updates=model._n_updates,
            gradient_seconds=getattr(model, "gradient_wall_seconds", 0.0)
            - initial_gradient_seconds,
            final_checkpoint=str(checkpoint.resolve()),
            replay_size=model.replay_buffer.size(),
            elapsed_seconds=time.perf_counter() - started,
            collected_steps=model.num_timesteps - initial_steps,
            episode_boundary_overshoot=max(0, model.num_timesteps - cfg.total_steps),
        )
    except BaseException as error:
        manifest.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        vec.close()
        write_report(output / "manifest.json", manifest)
    return {
        k: manifest[k]
        for k in ("status", "steps", "updates", "elapsed_seconds", "final_checkpoint")
    }
