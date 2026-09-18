"""DreamerV3 with training-only actor supervision and mean-action evaluation."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .diagnostics import machine_report, write_report
from .dreamer import DreamerConfig, agent_config, load_upstream, observation
from .guided import FEATURE_VERSION, ROOT, WaypointEnv, teacher_action
from .training import file_hash


def guided_agent(config, obs_space, act_space):
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.agent import Agent

    class SupervisedAgent(Agent):
        @property
        def ext_space(self):
            return {
                **super().ext_space,
                "teacher_action": elements.Space(np.float32, (2,), -1, 1),
            }

        def _apply_replay_context(self, carry, data):
            carry, obs, prevact, stepid = super()._apply_replay_context(carry, data)
            obs = {
                **obs,
                "_teacher": data["teacher_action"][:, self.config.replay_context :],
            }
            return carry, obs, prevact, stepid

        def loss(self, carry, obs, prevact, training):
            obs = dict(obs)
            target = obs.pop("_teacher")
            loss, (carry, entries, outs, metrics) = super().loss(
                carry, obs, prevact, training
            )
            predicted = self.pol(self.feat2tensor(outs["repfeat"]), 2)["action"].pred()
            bc = jnp.square(predicted - jax.lax.stop_gradient(target)).mean()
            metrics["loss/imitation_mse"] = bc
            return loss + self.config.bc_weight * bc, (carry, entries, outs, metrics)

        def policy(self, carry, obs, mode="train"):
            enc_carry, dyn_carry, dec_carry, prevact = carry
            kw = {"training": False, "single": True}
            reset = obs["is_first"]
            enc_carry, _, tokens = self.enc(enc_carry, obs, reset, **kw)
            dyn_carry, _, feat = self.dyn.observe(
                dyn_carry, tokens, prevact, reset, **kw
            )
            if dec_carry:
                dec_carry, _, _ = self.dec(dec_carry, feat, reset, **kw)
            distributions = self.pol(self.feat2tensor(feat), bdims=1)
            actions = {
                k: d.pred() if mode == "eval" else d.sample(nj.seed())
                for k, d in distributions.items()
            }
            return (enc_carry, dyn_carry, dec_carry, actions), actions, {}

    return SupervisedAgent(
        obs_space,
        act_space,
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


def spaces():
    import elements

    obs_space = {
        "vector": elements.Space(np.float32, (5,)),
        "reward": elements.Space(np.float32),
        **{k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")},
    }
    return obs_space, {"action": elements.Space(np.float32, (2,), -1, 1)}


class EmbodiedWaypoints:
    def __init__(self, seed):
        import elements

        self.env = WaypointEnv(seed=seed)
        self.obs_space, self.act_space = spaces()
        self.act_space = {**self.act_space, "reset": elements.Space(bool)}
        self.done = True

    def step(self, actions):
        if actions["reset"] or self.done:
            vector, self.info = self.env.reset()
            self.done = False
            return observation(vector, first=True)
        vector, reward, terminal, truncated, self.info = self.env.step(
            np.clip(actions["action"], -1, 1).astype(np.float32)
        )
        self.done = terminal or truncated
        return observation(vector, reward, terminated=terminal, truncated=truncated)

    def close(self):
        self.env.close()


def save_agent(agent, output, name, config, manifest):
    payload = agent.save()
    if not all(np.isfinite(v).all() for v in payload["params"].values()):
        raise RuntimeError("Nonfinite Dreamer parameters")
    path = Path(output) / f"{name}.pkl"
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    write_report(
        path.with_suffix(".json"),
        {
            "algorithm": "guided-dreamerv3",
            "agent_sha256": file_hash(path),
            "feature_version": FEATURE_VERSION,
            "config": dict(config),
            "upstream_path": manifest["upstream_path"],
            "source_sha256": manifest["bb8_source_sha256"],
            "updates": int(agent.n_updates),
            "resume_supported": False,
        },
    )


class GuidedDreamerPolicy:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        state = json.loads(checkpoint.with_suffix(".json").read_text())
        if state["feature_version"] != FEATURE_VERSION or state[
            "agent_sha256"
        ] != file_hash(checkpoint):
            raise ValueError("Invalid guided Dreamer checkpoint")
        load_upstream(state["upstream_path"])
        import elements

        self.agent = guided_agent(elements.Config(state["config"]), *spaces())
        with checkpoint.open("rb") as file:
            self.agent.load(pickle.load(file))
        self.reset(0)

    def reset(self, seed):
        self.agent.n_actions.value = int(seed) * 100000
        self.carry = self.agent.init_policy(1)
        self.first = True

    def action(self, vector):
        obs = observation(vector, first=self.first)
        self.carry, actions, _ = self.agent.policy(
            self.carry, {k: np.asarray(v)[None] for k, v in obs.items()}, mode="eval"
        )
        self.first = False
        return np.clip(actions["action"][0], -1, 1).astype(np.float32)


def train_dreamer(
    output,
    demonstrations,
    *,
    steps=10000,
    offline_updates=3000,
    seed=17,
    bc_weight=100.0,
):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a fresh output")
    upstream = load_upstream(ROOT / "work/m5/dreamerv3-upstream")
    import elements
    import embodied
    import jax
    from dreamerv3.main import make_stream

    torch.set_num_threads(2)
    output.mkdir(parents=True)
    started = time.perf_counter()
    env = EmbodiedWaypoints(seed)
    manifest = machine_report(env.env.core.world_path)
    manifest.update(
        status="running",
        algorithm="DreamerV3 with teacher regularization and A* waypoints",
        upstream_path=str(upstream),
        demonstration_sha256=file_hash(demonstrations),
        seed=seed,
        requested_steps=steps,
        requested_offline_updates=offline_updates,
        bc_weight=bc_weight,
        feature_version=FEATURE_VERSION,
    )
    write_report(output / "manifest.json", manifest)
    cfg = DreamerConfig(
        task="unused",
        upstream=str(upstream),
        batch_size=4,
        batch_length=32,
        horizon=200,
    )
    config = agent_config(cfg, upstream, output).update(
        {
            "seed": seed,
            "replay_context": 0,
            "agent.opt.lr": 0.0003,
            "agent.opt.warmup": 100,
            "agent.policy.minstd": 0.01,
            "agent.policy.maxstd": 0.1,
        }
    )
    config = elements.Config(
        dict(config), agent={**dict(config.agent), "bc_weight": bc_weight}
    )
    config.save(str(output / "agent-config.yaml"))
    replay = driver = None
    try:
        env.step({"reset": True, "action": np.zeros(2, np.float32)})
        env.env.episode = 0
        env.done = True
        agent = guided_agent(config, *spaces())
        replay = embodied.replay.Replay(
            length=cfg.batch_length,
            capacity=100000,
            directory=str(output / "replay"),
            online=False,
            save_wait=True,
            seed=seed,
        )
        data = np.load(demonstrations)
        keys = (*spaces()[0], "action", "teacher_action")
        for i in range(len(data["vector"])):
            replay.add({k: data[k][i] for k in keys})
        stream = iter(agent.stream(make_stream(config, replay, "train")))
        carry = agent.init_train(cfg.batch_size)

        def update(stage, step):
            nonlocal carry
            carry, _, metrics = agent.train(carry, next(stream))
            jax.block_until_ready(agent.params)
            values = {
                k: float(v) for k, v in metrics.items() if np.asarray(v).ndim == 0
            }
            if not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError("Nonfinite Dreamer metric")
            with (output / "metrics.jsonl").open("a") as file:
                file.write(
                    json.dumps(
                        {
                            "stage": stage,
                            "step": step,
                            "updates": int(agent.n_updates),
                            **values,
                        }
                    )
                    + "\n"
                )
            return values

        for i in range(offline_updates):
            metrics = update("offline", 0)
            if (i + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "offline_updates": i + 1,
                            "imitation_mse": metrics.get("loss/imitation_mse"),
                        }
                    ),
                    flush=True,
                )
        save_agent(agent, output, "offline", config, manifest)
        driver = embodied.Driver([lambda: env], parallel=False)
        driver.on_step(
            lambda tran, _: tran.update(teacher_action=teacher_action(tran["vector"]))
        )
        driver.on_step(replay.add)
        collected = 0

        def learn(tran, worker):
            nonlocal collected
            if tran["is_first"]:
                return
            collected += 1
            if collected % 8 == 0:
                update("online", collected)
            if tran["is_last"]:
                with (output / "episodes.jsonl").open("a") as file:
                    file.write(json.dumps({"step": collected, **env.info}) + "\n")
            if collected % 1000 == 0:
                print(
                    json.dumps({"steps": collected, "updates": int(agent.n_updates)}),
                    flush=True,
                )

        driver.on_step(learn)
        driver.reset(agent.init_policy)
        while collected < steps:
            driver(lambda *args: agent.policy(*args, mode="train"), steps=1)
        replay.save()
        save_agent(agent, output, "model", config, manifest)
        manifest.update(
            status="complete",
            steps=collected,
            updates=int(agent.n_updates),
            elapsed_seconds=time.perf_counter() - started,
            model_sha256=file_hash(output / "model.pkl"),
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        env.close()
        if replay:
            replay.workers.shutdown(wait=True)
        write_report(output / "manifest.json", manifest)
    return manifest
