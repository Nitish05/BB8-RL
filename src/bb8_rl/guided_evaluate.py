"""Unassisted learned drive evaluation with a frozen-map waypoint planner."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .diagnostics import machine_report, write_report
from .evaluate import plot_paths, summarize, write_markdown
from .guided import FEATURE_VERSION, GuidedSAC, WaypointEnv
from .training import file_hash


def evaluate(
    algorithm, checkpoint, output, *, count=12, split="validation", validation=None
):
    if algorithm not in ("sac", "dreamer", "bc") or split not in ("validation", "test"):
        raise ValueError("Unsupported algorithm or split")
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists():
        raise ValueError("Choose a fresh evaluation output")
    checksum = file_hash(checkpoint)
    if split == "test":
        if validation is None or count != 100:
            raise ValueError(
                "Final test requires passing 100-case validation and 100 cases per family"
            )
        accepted = json.loads((Path(validation) / "manifest.json").read_text())
        scores = json.loads((Path(validation) / "summary.json").read_text())[algorithm]
        if (
            accepted["checkpoint_sha256"] != checksum
            or accepted["algorithm"] != algorithm
            or accepted["status"] != "complete"
            or accepted["split"] != "validation"
            or scores["episodes"] < 100
            or scores["success_rate"] < 0.95
        ):
            raise ValueError(
                "Candidate has not passed the preregistered validation gate"
            )
    torch.set_num_threads(2)
    if algorithm == "dreamer":
        from .guided_dreamer import GuidedDreamerPolicy

        policy = GuidedDreamerPolicy(checkpoint)
    else:
        policy = GuidedSAC.load(checkpoint, device="cpu")
    output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "algorithm": algorithm,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "feature_version": FEATURE_VERSION,
        "count_per_family": count,
        "teacher_action_fallback": False,
        "observation_mode": "oracle-map-waypoints",
        "validation_gate": str(validation) if validation else None,
        "seed_start": 15000 if split == "validation" else 25000,
    }
    write_report(output / "manifest.json", manifest)
    rows = []
    started = time.perf_counter()
    try:
        for family in ("authored", "procedural"):
            with WaypointEnv(family=family, split=split) as env:
                if "machine" not in manifest:
                    manifest["machine"] = machine_report(env.core.world_path)
                for index in range(count):
                    seed = (15000 if split == "validation" else 25000) + index
                    layout_seed = (10000 if split == "validation" else 20000) + index
                    vector, info = env.reset(
                        seed=seed, options={"layout_seed": layout_seed}
                    )
                    if algorithm == "dreamer":
                        policy.reset(seed)
                    trace = [[0.0, *env.raw["achieved_goal"].tolist()]]
                    previous = env.raw["achieved_goal"][:2].copy()
                    length, reward_sum, minimum = 0.0, 0.0, float("inf")
                    latencies = []
                    while True:
                        tick = time.perf_counter()
                        action = (
                            policy.action(vector)
                            if algorithm == "dreamer"
                            else policy.predict(vector, deterministic=True)[0]
                        )
                        latencies.append(time.perf_counter() - tick)
                        if not np.isfinite(action).all() or np.max(abs(action)) > 1:
                            raise RuntimeError("Invalid learned action")
                        vector, reward, terminal, truncated, info = env.step(action)
                        position = env.raw["achieved_goal"][:2]
                        length += float(np.linalg.norm(position - previous))
                        previous = position.copy()
                        error = float(
                            np.linalg.norm(position - env.raw["desired_goal"][:2])
                        )
                        minimum = min(minimum, error)
                        reward_sum += reward
                        trace.append(
                            [info["sim_time"], *env.raw["achieved_goal"].tolist()]
                        )
                        if terminal or truncated:
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
                    row = {
                        "id": f"{family}-{split}-{index:03d}",
                        "controller": algorithm,
                        "family": family,
                        "episode_seed": seed,
                        "layout_seed": layout_seed,
                        "layout_kind": env.core.layout.kind,
                        "success": info["is_success"],
                        "collision": info["collision"],
                        "boundary_failure": info["boundary_failure"],
                        "ending": ending,
                        "steps": env.core.steps,
                        "sim_time": info["sim_time"],
                        "final_error_m": error,
                        "minimum_error_m": minimum,
                        "final_speed_mps": float(env.raw["achieved_goal"][2]),
                        "dwell_seconds": info["dwell_seconds"],
                        "path_length_m": length,
                        "return_sum": reward_sum,
                        "goal": env.raw["desired_goal"].tolist(),
                        "start": trace[0][1:3],
                        "route": env.route.tolist(),
                        "route_length_m": float(
                            np.linalg.norm(np.diff(env.route, axis=0), axis=1).sum()
                        ),
                        "obstacles": env.core.layout.positions,
                        "obstacle_sizes": env.core.sizes[
                            : len(env.core.layout.positions)
                        ],
                        "arena_half_extent": env.core.config.arena_half_extent,
                        "action_latencies": latencies,
                        "trace": trace,
                    }
                    rows.append(row)
                    write_report(output / "episodes" / f"{row['id']}.json", row)
                    print(
                        json.dumps(
                            {
                                "case": row["id"],
                                "ending": ending,
                                "successes": sum(r["success"] for r in rows),
                                "episodes": len(rows),
                            }
                        ),
                        flush=True,
                    )
        manifest.update(
            status="complete", elapsed_seconds=time.perf_counter() - started
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_report(output / "manifest.json", manifest)
        if rows:
            write_report(output / "summary.json", summarize(rows))
    summary = summarize(rows)
    write_markdown(output, summary, split)
    # Keep readable plots; every trajectory remains in its own episode JSON.
    plot_paths(output, rows[:6] + rows[count : count + 6])
    return summary
