"""Optional adapter to the pinned authors' DreamerV3 implementation.

Upstream owns the world model, imagination losses, replay and policy. BB8-RL owns
the environment contract, feature parity, interaction budget and evaluation.
"""

import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .diagnostics import machine_report, write_report
from .env import NavigationEnv
from .training import NavigationFeatures, TrainingEpisodes, file_hash, task_contract

UPSTREAM_REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"


class DreamerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    task: str
    upstream: str
    seed: int = Field(default=17, ge=0, le=100000)
    total_steps: int = Field(default=30000, ge=1, le=10000000)
    checkpoint_every: int = Field(default=15000, ge=1)
    buffer_size: int = Field(default=40000, ge=128)
    learning_starts: int = Field(default=1500, ge=1)
    batch_size: int = Field(default=4, ge=1)
    batch_length: int = Field(default=32, ge=2)
    train_ratio: int = Field(default=16, ge=1)
    horizon: int = Field(default=50, ge=2)
    physics_backend: Literal["cpu", "metal"] = "cpu"

    @model_validator(mode="after")
    def valid_replay(self):
        if self.batch_size * self.batch_length % self.train_ratio:
            raise ValueError("train_ratio must divide batch_size * batch_length")
        if self.buffer_size <= self.batch_size * (self.batch_length + 1):
            raise ValueError("Replay must hold more than one complete batch")
        if self.learning_starts < self.batch_size * (self.batch_length + 1):
            raise ValueError("Warmup must cover a full replay batch")
        return self


def load_config(path):
    path = Path(path).resolve()
    cfg = DreamerConfig.model_validate(yaml.safe_load(path.read_text()))
    return (
        cfg,
        (path.parent / cfg.task).resolve(),
        (path.parent / cfg.upstream).resolve(),
    )


def load_upstream(path):
    path = Path(os.getenv("BB8_DREAMER_ROOT", str(path))).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != UPSTREAM_REVISION or changed:
        raise ValueError("Dreamer requires the pinned, unmodified upstream checkout")
    sys.path.insert(0, str(path))
    import dreamerv3.agent

    if Path(dreamerv3.agent.__file__).resolve().parent.parent != path:
        raise RuntimeError("Another Dreamer installation was imported")
    return path


class FeatureAdapter:
    def __init__(self, env):
        self.encoder = NavigationFeatures(
            env.observation_space,
            map_size=env.task.local_map_size,
            history_length=env.task.history_length,
            extent=env.config.arena_half_extent,
            max_speed=env.config.drive.max_speed,
        ).eval()
        self.size = self.encoder.features_dim

    def __call__(self, observation):
        with torch.no_grad():
            tensors = {
                k: torch.as_tensor(v).unsqueeze(0) for k, v in observation.items()
            }
            return self.encoder(tensors)[0].numpy().copy()


def observation(
    features, reward=0.0, *, first=False, terminated=False, truncated=False
):
    return {
        "vector": np.asarray(features, np.float32),
        "reward": np.float32(reward),
        "is_first": np.bool_(first),
        "is_last": np.bool_(terminated or truncated),
        "is_terminal": np.bool_(terminated),
    }


class DreamerEnv:
    """Gymnasium-to-Embodied adapter. Reset rows are explicitly identifiable."""

    def __init__(self, core, seed):
        import elements

        self.env = TrainingEpisodes(core, seed)
        self.features = FeatureAdapter(core)
        self.done = True
        self.last_info = None
        self.obs_space = {
            "vector": elements.Space(np.float32, (self.features.size,)),
            "reward": elements.Space(np.float32),
            **{
                key: elements.Space(bool)
                for key in ("is_first", "is_last", "is_terminal")
            },
        }
        self.act_space = {
            "action": elements.Space(np.float32, (2,), -1, 1),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if action["reset"] or self.done:
            obs, self.last_info = self.env.reset()
            self.done = False
            return observation(self.features(obs), first=True)
        obs, reward, terminal, truncated, self.last_info = self.env.step(
            np.clip(action["action"], -1, 1).astype(np.float32)
        )
        self.done = terminal or truncated
        return observation(
            self.features(obs), reward, terminated=terminal, truncated=truncated
        )

    def close(self):
        self.env.close()


def agent_config(cfg, upstream, output):
    import elements
    from ruamel.yaml import YAML

    defaults = YAML(typ="safe").load((upstream / "dreamerv3/configs.yaml").read_text())
    config = elements.Config(defaults["defaults"]).update(defaults["size1m"])
    config = config.update(
        logdir=str(Path(output).resolve()),
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        report_length=cfg.batch_length,
        **{
            "jax.platform": "cpu",
            "jax.compute_dtype": "float32",
            "jax.prealloc": False,
            "agent.horizon": cfg.horizon,
            "agent.report": False,
        },
    )
    # Options absent from the upstream YAML are supported by its JAX Agent.
    config = elements.Config(
        dict(config),
        jax={
            **dict(config.jax),
            "profiler": False,
            "precompile": False,
        },
    )
    return config


def make_agent(config, env):
    import elements
    from dreamerv3.agent import Agent

    return Agent(
        env.obs_space,
        {"action": env.act_space["action"]},
        elements.Config(
            **dict(config.agent),
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=0,
            replicas=1,
        ),
    )


def save_checkpoint(agent, output, steps, cfg, config, manifest, next_episode):
    destination = Path(output) / "checkpoints" / f"step-{steps:09d}"
    if destination.exists():
        return destination
    staging = destination.with_suffix(".partial")
    staging.mkdir(parents=True)
    saved = agent.save()
    if not all(np.isfinite(x).all() for x in saved["params"].values()):
        raise RuntimeError("Nonfinite Dreamer parameters")
    with (staging / "agent.pkl").open("wb") as file:
        pickle.dump(saved, file, protocol=pickle.HIGHEST_PROTOCOL)
    state = {
        "schema_version": 1,
        "algorithm": "dreamerv3",
        "steps": steps,
        "updates": int(agent.n_updates),
        "next_episode": next_episode,
        "config": cfg.model_dump(),
        "agent_config": dict(config),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_path": manifest["upstream_path"],
        "source_sha256": manifest["bb8_source_sha256"],
        "contract": manifest["contract"],
        "agent_sha256": file_hash(staging / "agent.pkl"),
        "resume_supported": False,
        "evaluation_policy": "upstream sampled actions; case-seeded",
    }
    write_report(staging / "state.json", state)
    staging.rename(destination)
    write_report(
        Path(output) / "latest.json", {"checkpoint": str(destination.resolve())}
    )
    return destination


def read_checkpoint(path):
    path = Path(path)
    state = json.loads((path / "state.json").read_text())
    if (
        state.get("schema_version") != 1
        or state.get("algorithm") != "dreamerv3"
        or state.get("upstream_revision") != UPSTREAM_REVISION
        or file_hash(path / "agent.pkl") != state.get("agent_sha256")
    ):
        raise ValueError("Invalid Dreamer checkpoint or checksum mismatch")
    return state


class DreamerPolicy:
    def __init__(self, checkpoint, core):
        self.state = read_checkpoint(checkpoint)
        if self.state["contract"] != task_contract(core):
            raise ValueError("Dreamer checkpoint does not match task/world contract")
        load_upstream(self.state["upstream_path"])
        import elements

        self.features = FeatureAdapter(core)
        adapter = DreamerEnv(core, self.state["config"]["seed"])
        self.agent = make_agent(elements.Config(self.state["agent_config"]), adapter)
        with (Path(checkpoint) / "agent.pkl").open("rb") as file:
            self.agent.load(pickle.load(file))  # Trusted locally generated checkpoints.
        self.device = "cpu"
        self.reset(0)

    def reset(self, episode_seed):
        # Upstream combines its training seed with this action counter for its RNG.
        self.agent.n_actions.value = int(episode_seed) * 100000
        self.carry = self.agent.init_policy(1)
        self.first = True

    def predict(self, obs, deterministic=False):
        data = observation(self.features(obs), first=self.first)
        self.carry, actions, _ = self.agent.policy(
            self.carry, {k: np.asarray(v)[None] for k, v in data.items()}, mode="eval"
        )
        self.first = False
        return np.clip(actions["action"][0], -1, 1).astype(np.float32), None


def train(config_path, output):
    cfg, task_path, upstream = load_config(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a fresh Dreamer output directory")
    upstream = load_upstream(upstream)
    import embodied
    from dreamerv3.main import make_stream

    torch.set_num_threads(2)
    core = NavigationEnv(task_path, split="train", backend=cfg.physics_backend)
    env = DreamerEnv(core, cfg.seed)
    output.mkdir(parents=True)
    manifest = machine_report(core.world_path)
    manifest.update(
        status="running",
        algorithm="DreamerV3",
        training_config=cfg.model_dump(),
        task_path=str(task_path),
        contract=task_contract(core),
        upstream_path=str(upstream),
        upstream_revision=UPSTREAM_REVISION,
        requested_total_steps=cfg.total_steps,
        observation_mode="oracle",
        resume_supported=False,
    )
    write_report(output / "manifest.json", manifest)
    config = agent_config(cfg, upstream, output)
    config.save(str(output / "agent-config.yaml"))
    started = time.perf_counter()
    driver = replay = agent = None
    previous_handlers = {}
    stop = False
    steps = 0
    updates_seconds = 0.0
    episode_return, episode_length, arrived = 0.0, 0, False
    last = True
    checkpoint = None

    def stop_handler(signum, frame):
        nonlocal stop
        if stop:
            raise KeyboardInterrupt()
        stop = True
        print(
            "Cancellation requested; finishing episode and saving Dreamer.", flush=True
        )

    try:
        # Initialize native Genesis before installing our OS signal handlers.
        env.step({"reset": True, "action": np.zeros(2, np.float32)})
        env.env.next_episode = 0
        env.done = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stop_handler)
        agent = make_agent(config, env)
        manifest.update(
            initialization_seconds=time.perf_counter() - started,
            resolved_device="cpu",
            resolved_physics_backend=core.world.gs.backend.name,
            agent_parameter_elements=sum(
                v.size for k, v in agent.params.items() if not k.startswith("opt/")
            ),
        )
        replay = embodied.replay.Replay(
            length=cfg.batch_length + config.replay_context,
            capacity=cfg.buffer_size,
            directory=str(output / "replay"),
            online=True,
            save_wait=True,
            seed=cfg.seed,
        )
        driver = embodied.Driver([lambda: env], parallel=False)
        driver.on_step(replay.add)
        carry = agent.init_train(cfg.batch_size)
        stream = None
        interval = cfg.batch_size * cfg.batch_length // cfg.train_ratio
        next_save = cfg.checkpoint_every
        last_progress = time.perf_counter()

        def on_step(transition, worker):
            nonlocal steps, carry, stream, updates_seconds, last
            nonlocal episode_return, episode_length, arrived
            last = bool(transition["is_last"])
            if transition["is_first"]:
                return
            steps += 1
            episode_return += float(transition["reward"])
            episode_length += 1
            info = env.last_info
            arrived |= info["is_success"]
            if last:
                record = {
                    "step": steps,
                    "episode": info["training_episode"],
                    "return_sum": episode_return,
                    "length": episode_length,
                    "arrived_anytime": arrived,
                    "success_at_end": info["is_success"],
                    "collision": info["collision"],
                    "boundary_failure": info["boundary_failure"],
                    "timeout": not bool(transition["is_terminal"]),
                }
                with (output / "episodes.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                episode_return, episode_length, arrived = 0.0, 0, False
            if steps >= cfg.learning_starts and steps % interval == 0:
                if stream is None:
                    stream = iter(agent.stream(make_stream(config, replay, "train")))
                tick = time.perf_counter()
                carry, outs, metrics = agent.train(carry, next(stream))
                # Block for accurate CPU optimization timing, not just dispatch.
                import jax

                jax.block_until_ready(agent.params)
                updates_seconds += time.perf_counter() - tick
                if "replay" in outs:
                    replay.update(outs["replay"])
                scalar = {
                    k: float(v) for k, v in metrics.items() if np.asarray(v).ndim == 0
                }
                if not all(np.isfinite(v) for v in scalar.values()):
                    raise RuntimeError("Nonfinite Dreamer training metric")
                if scalar:
                    with (output / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(dict(step=steps, **scalar)) + "\n")

        driver.on_step(on_step)
        driver.reset(agent.init_policy)
        checkpoint = save_checkpoint(
            agent, output, steps, cfg, config, manifest, env.env.next_episode
        )
        while steps < cfg.total_steps and not stop or not last:
            driver(lambda *args: agent.policy(*args, mode="train"), steps=1)
            if last and (steps >= next_save or steps >= cfg.total_steps or stop):
                replay.save()
                checkpoint = save_checkpoint(
                    agent, output, steps, cfg, config, manifest, env.env.next_episode
                )
                next_save = steps + cfg.checkpoint_every
            if time.perf_counter() - last_progress >= 10:
                print(
                    json.dumps(
                        {
                            "steps": steps,
                            "updates": int(agent.n_updates),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last_progress = time.perf_counter()
        manifest.update(
            status="cancelled" if stop else "complete",
            steps=steps,
            updates=int(agent.n_updates),
            gradient_seconds=updates_seconds,
            final_checkpoint=str(checkpoint),
            replay_size=len(replay),
            elapsed_seconds=time.perf_counter() - started,
            episode_boundary_overshoot=max(0, steps - cfg.total_steps),
        )
    except BaseException as error:
        manifest.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error),
            steps=steps,
        )
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if driver:
            driver.close()
        else:
            env.close()
        if replay:
            replay.workers.shutdown(wait=True)
        write_report(output / "manifest.json", manifest)
    return {
        k: manifest[k]
        for k in ("status", "steps", "updates", "elapsed_seconds", "final_checkpoint")
    }
