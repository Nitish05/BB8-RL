"""Paired fixed-suite evaluation with per-episode evidence and honest oracle metrics."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .baseline import RouteController
from .diagnostics import machine_report, write_report
from .env import NavigationEnv
from .task import require_split_seed


def wilson(successes, count):
    if not count:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    delta = (
        z
        * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
        / (1 + z * z / count)
    )
    return [
        0.0 if successes == 0 else max(0.0, center - delta),
        1.0 if successes == count else min(1.0, center + delta),
    ]


def suite_cases(path, split, count=None):
    specification = json.loads(Path(path).read_text())
    if specification.get("schema_version") != 1 or split not in (
        "train",
        "validation",
        "test",
    ):
        raise ValueError("Invalid evaluation suite schema or split")
    s = specification[split]
    n = s["count"] if count is None else count
    if not 1 <= n <= s["count"]:
        raise ValueError(f"Choose 1–{s['count']} episodes for {split}")
    cases = [
        {
            "id": f"{split}-{i:03}",
            "layout_seed": s["first_layout_seed"] + i,
            "episode_seed": s["first_episode_seed"] + i,
        }
        for i in range(n)
    ]
    for case in cases:
        require_split_seed(split, case["layout_seed"])
    return cases


def summarize(records):
    summary = {}
    for name in dict.fromkeys(r["controller"] for r in records):
        rows = [r for r in records if r["controller"] == name]
        n = len(rows)
        wins = sum(r["success"] for r in rows)
        collisions = sum(r["collision"] for r in rows)
        good = [r for r in rows if r["success"]]
        summary[name] = {
            "episodes": n,
            "successes": wins,
            "success_rate": wins / n,
            "success_95ci": wilson(wins, n),
            "collision_episodes": collisions,
            "collision_rate": collisions / n,
            "collision_95ci": wilson(collisions, n),
            "boundary_failures": sum(r["boundary_failure"] for r in rows),
            "timeouts": sum(r["ending"] == "time_limit" for r in rows),
            "mean_final_error_m": float(np.mean([r["final_error_m"] for r in rows])),
            "mean_path_length_m": float(np.mean([r["path_length_m"] for r in rows])),
            "mean_completion_seconds_success_only": float(
                np.mean([r["sim_time"] for r in good])
            )
            if good
            else None,
            "mean_path_efficiency_success_only": float(
                np.mean(
                    [
                        min(1.0, r["route_length_m"] / max(r["path_length_m"], 1e-9))
                        for r in good
                    ]
                )
            )
            if good
            else None,
            "action_latency_p95_seconds": float(
                np.quantile([v for r in rows for v in r["action_latencies"]], 0.95)
            ),
        }
    return summary


def evaluate(
    task_path,
    suite_path,
    *,
    split="validation",
    count=None,
    controllers=("baseline", "random"),
    backend="cpu",
    output=Path("work/evaluation"),
    viewer=False,
    checkpoint=None,
    device="cpu",
):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError(
            "Evaluation output already exists; choose a new directory to preserve evidence"
        )
    if (
        not controllers
        or set(controllers) - {"baseline", "random", "sac-her", "dreamerv3"}
        or len(set(controllers)) != len(controllers)
    ):
        raise ValueError("Choose baseline, random, sac-her and/or dreamerv3 once each")
    learned_names = set(controllers) & {"sac-her", "dreamerv3"}
    if len(learned_names) > 1 or bool(learned_names) != (checkpoint is not None):
        raise ValueError("Supply one checkpoint for exactly one learned controller")
    cases = suite_cases(suite_path, split, count)
    output.mkdir(parents=True, exist_ok=True)
    env = NavigationEnv(
        task_path, split=split, backend=backend, render_mode="human" if viewer else None
    )
    manifest = machine_report(env.world_path)
    learned = None
    if checkpoint is not None:
        import torch

        from .training import file_hash, load_policy

        torch.set_num_threads(2)
        if "dreamerv3" in controllers:
            if device != "cpu":
                raise ValueError("This Dreamer adapter supports CPU only")
            from .dreamer import DreamerPolicy

            learned = DreamerPolicy(checkpoint, env)
            checkpoint_state = learned.state
            policy_file = "agent.pkl"
            manifest["evaluation_policy"] = "upstream sampled actions; case-seeded"
        else:
            learned, checkpoint_state = load_policy(checkpoint, env, device)
            policy_file = "policy.pt"
            manifest["evaluation_policy"] = "deterministic SAC action"
        manifest.update(
            checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_state=checkpoint_state,
            policy_sha256=file_hash(Path(checkpoint) / policy_file),
            learner_device=str(learned.device),
        )
    manifest.update(
        task=env.task.model_dump(),
        task_sha256=hashlib.sha256(Path(task_path).read_bytes()).hexdigest(),
        suite_sha256=hashlib.sha256(Path(suite_path).read_bytes()).hexdigest(),
        cases=cases,
        controllers=list(controllers),
        observation_mode="oracle",
        reward_version="sparse-goal-v1",
        status="running",
        requested_backend=backend,
    )
    write_report(output / "manifest.json", manifest)
    records = []
    try:
        for controller_name in controllers:
            for case in cases:
                obs, info = env.reset(
                    seed=case["episode_seed"],
                    options={"layout_seed": case["layout_seed"]},
                )
                if controller_name == "dreamerv3":
                    learned.reset(case["episode_seed"])
                policy = RouteController(env.task, env.config.drive.parameters())
                policy.reset(env.grid, obs)
                rng = np.random.default_rng(case["episode_seed"] + 1_000_000)
                trace = [[0.0, *obs["achieved_goal"].tolist()]]
                latencies = []
                total_reward = 0.0
                min_error = float("inf")
                position = obs["achieved_goal"][:2].copy()
                length = 0.0
                # Export the exact sampled geometry as an ordinary Studio project.
                project = env.world.project.model_copy(deep=True)
                slots = dict(zip(env.config.obstacle_names, env.layout.positions))
                for obj in project.objects:
                    if obj.name in slots:
                        obj.position = (*slots[obj.name], obj.position[2])
                    if obj.name in (env.config.body, env.config.head):
                        obj.position = (*[float(v) for v in position], obj.position[2])
                if controller_name == controllers[0]:
                    write_report(
                        output / "worlds" / f"{case['id']}.genesis.json",
                        project.model_dump(mode="json"),
                    )
                while True:
                    tick = time.perf_counter()
                    if controller_name == "baseline":
                        action = policy.action(obs)
                    elif controller_name == "sac-her":
                        action, _ = learned.predict(obs, deterministic=True)
                    elif controller_name == "dreamerv3":
                        action, _ = learned.predict(obs)
                    else:
                        action = rng.uniform(-1, 1, 2).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    latencies.append(time.perf_counter() - tick)
                    total_reward += reward
                    length += float(np.linalg.norm(obs["achieved_goal"][:2] - position))
                    position = obs["achieved_goal"][:2].copy()
                    error = float(np.linalg.norm(position - obs["desired_goal"][:2]))
                    min_error = min(min_error, error)
                    trace.append([info["sim_time"], *obs["achieved_goal"].tolist()])
                    if viewer and not env.world.scene.viewer.is_alive():
                        raise KeyboardInterrupt()
                    if info["is_success"] or terminated or truncated:
                        break
                ending = (
                    "arrival"
                    if info["is_success"]
                    else "collision"
                    if info["collision"]
                    else "boundary"
                    if info["boundary_failure"]
                    else "time_limit"
                )
                record = dict(
                    case,
                    controller=controller_name,
                    layout_kind=env.layout.kind,
                    success=info["is_success"],
                    collision=info["collision"],
                    boundary_failure=info["boundary_failure"],
                    ending=ending,
                    sim_time=info["sim_time"],
                    steps=env.steps,
                    final_error_m=error,
                    minimum_error_m=min_error,
                    final_speed_mps=float(obs["achieved_goal"][2]),
                    dwell_seconds=info["dwell_seconds"],
                    path_length_m=length,
                    return_sum=total_reward,
                    goal=obs["desired_goal"].tolist(),
                    start=trace[0][1:3],
                    route=policy.route.tolist(),
                    obstacles=env.layout.positions,
                    obstacle_sizes=env.sizes[: len(env.layout.positions)],
                    arena_half_extent=env.config.arena_half_extent,
                    route_length_m=float(
                        np.linalg.norm(np.diff(policy.route, axis=0), axis=1).sum()
                    ),
                    action_latencies=latencies,
                    trace=trace,
                )
                records.append(record)
                write_report(
                    output / "episodes" / f"{case['id']}-{controller_name}.json", record
                )
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "controller": controller_name,
                            "ending": ending,
                            "seconds": info["sim_time"],
                        }
                    ),
                    flush=True,
                )
        manifest["resolved_backend"] = env.world.gs.backend.name
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        env.close()
        write_report(output / "manifest.json", manifest)
        if records:
            write_report(output / "summary.json", summarize(records))
    summary = summarize(records)
    write_markdown(output, summary, split)
    plot_paths(output, records)
    return summary


def write_markdown(output, summary, split):
    lines = [
        f"# Synthetic navigation evaluation — {split}",
        "",
        "Oracle state/map; static synthetic layouts. Learned controllers use the recorded checkpoint and evaluation mode. No physical-robot result.",
        "",
        "| Controller | Episodes | Success (95% Wilson CI) | Contact episodes | Timeouts | Final error, mean | Successful arrival time, mean |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, s in summary.items():
        lo, hi = s["success_95ci"]
        t = s["mean_completion_seconds_success_only"]
        arrival = "—" if t is None else f"{t:.2f} s"
        lines.append(
            f"| {name} | {s['episodes']} | {s['success_rate']:.0%} ({lo:.0%}–{hi:.0%}) | {s['collision_episodes']} | {s['timeouts']} | {s['mean_final_error_m']:.3f} m | {arrival} |"
        )
    lines += [
        "",
        "Arrival requires position ≤10 cm and speed ≤3 cm/s continuously for 0.5 s with the default task settings. See manifest.json for resolved tolerances.",
        "Contacts exclude floor contact and are checked every physics step. Episode files retain every failure, trajectory, seed and latency sample.",
        "These small paired suites establish a baseline; they do not certify the final 200-episode acceptance target or zero collision risk.",
        "",
        "![Paths](paths.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def plot_paths(output, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cases = list(dict.fromkeys(r["id"] for r in records))
    columns = min(4, len(cases))
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
    )
    for ax, case in zip(axes.flat, cases):
        matches = [r for r in records if r["id"] == case]
        first = matches[0]
        for (x, y), size in zip(
            first["obstacles"],
            first.get("obstacle_sizes", [(0.3, 0.3, 0.3)] * len(first["obstacles"])),
        ):
            ax.add_patch(
                Rectangle(
                    (x - size[0] / 2, y - size[1] / 2),
                    size[0],
                    size[1],
                    color="#4c566a",
                )
            )
        for r in matches:
            points = np.asarray(r["trace"])
            ax.plot(
                points[:, 1],
                points[:, 2],
                label=f"{r['controller']}: {r['ending']}",
                lw=1.4,
            )
        ax.scatter(*first["start"], c="black", marker="o", s=20)
        ax.scatter(*first["goal"][:2], c="#bc5090", marker="*", s=90)
        extent = first.get("arena_half_extent", 1.5)
        ax.set(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            aspect="equal",
            title=f"{case} / {first['layout_kind']}",
            xlabel="x (m)",
            ylabel="y (m)",
        )
        ax.legend(fontsize=7, loc="upper right")
    for ax in list(axes.flat)[len(cases) :]:
        ax.set_visible(False)
    fig.suptitle("BB8-RL: oracle navigation trajectories by controller")
    fig.tight_layout()
    fig.savefig(output / "paths.png", dpi=150)
    plt.close(fig)
