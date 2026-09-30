"""Bounded native autonomy audit; only estimates supervise, truth scores afterward."""

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import queue
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def native_worker(config, commands, events, stop, log_path):
    with open(log_path, "a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        from bb8_rl.interactive_runtime import worker_main

        worker_main(config, commands, events, stop)


def send(channel, message):
    try:
        channel.put_nowait(message)
        return True
    except queue.Full:
        return False


def read_jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def score_exploration(arrivals, required_distinct):
    """Score destination diversity after independent arrival verification.

    Arrival episodes and unique intention IDs alone do not establish exploration:
    A -> B -> A must fail even though it contains three successful trips.
    """
    successful = [arrival for arrival in arrivals if arrival["independently_valid"]]
    seen, repeated = set(), []
    for arrival in successful:
        destination = tuple(arrival["goal"])
        if destination in seen:
            repeated.append(arrival)
        seen.add(destination)
    return {
        "valid_arrival_episodes": len(successful),
        "distinct_valid_destinations": len(seen),
        "distinct_valid_destination_goals": [list(goal) for goal in sorted(seen)],
        "repeated_successful_destinations": repeated,
        "required_distinct_destinations": required_distinct,
        "exploration_passed": len(seen) >= required_distinct and not repeated,
    }


def score_completed(output, required_distinct=3):
    """Read privileged records only after the native worker has ended."""
    spec = importlib.util.spec_from_file_location(
        "independent_interactive_audit", ROOT / "scripts/audit-interactive.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    rows = read_jsonl(output / "worker/rows.jsonl")
    report = scorer.score_rows(rows)
    write_json(output / "independent-audit.json", report)
    commands = read_jsonl(output / "worker/commands.jsonl")
    arrivals = []
    previous = None
    active = None
    intentions = {}
    for row in rows:
        intention = row.get("agency", {}).get("intention")
        if intention:
            active = intention["event_id"]
            intentions[active] = intention
        arrived = bool(row.get("controller_arrived"))
        goal = tuple(row["commanded_goal"]) if row.get("commanded_goal") else None
        key = (row.get("generation"), goal)
        if arrived and previous != key:
            arrivals.append({"time": row["time"], "goal": goal, "event_id": active})
        previous = key if arrived else None
    for arrival in arrivals:
        matching = [
            event
            for event in report["arrival_events"]
            if abs(event["time"] - arrival["time"]) < 1e-8
        ]
        arrival["independently_valid"] = bool(matching and matching[0]["passed"])
    stops = [
        command
        for command in commands
        if command.get("action") == "stop"
        and command.get("generation") == 2
        and command.get("outcome") == "accepted"
    ]
    stop_time = stops[0]["sim_time"] if stops else None
    post = [row for row in rows if stop_time is not None and row["time"] >= stop_time]
    end = post[-1]["after_step"]["time"] if post else None
    tail_samples = (
        [
            sample
            for row in post
            for sample in row["after_step"]["physics_trace_scoring_only"]
            if sample["time"] >= end - 0.5
        ]
        if end is not None
        else []
    )
    speeds = [math.hypot(*sample["velocity"]) for sample in tail_samples]
    commanded_zero = bool(post) and all(
        math.hypot(*row["action"]) <= 1e-7 for row in post
    )
    disabled = bool(post) and all(
        row.get("agency", {}).get("enabled") is False for row in post
    )
    post_duration = end - stop_time if end is not None else 0.0
    stop_report = {
        "accepted": bool(stops),
        "accepted_sim_time": stop_time,
        "observed_sim_seconds": post_duration,
        "rows": len(post),
        "all_requested_actions_zero": commanded_zero,
        "agency_disabled_entire_window": disabled,
        "final_half_second_max_physics_speed_m_s": max(speeds) if speeds else None,
        "physically_stationary_final_half_second": bool(speeds)
        and max(speeds) <= 0.001,
        "passed": bool(
            post_duration >= 1.0 - 1e-8
            and commanded_zero
            and disabled
            and speeds
            and max(speeds) <= 0.001
        ),
        "scope": "Zero request immediately; physical braking allowed before final 0.5s stationary check.",
    }
    manifest_path = output / "worker/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary = {
        "frames": len(rows),
        "simulation_seconds": rows[-1]["after_step"]["time"] if rows else 0,
        "unique_intentions": len(intentions),
        "unique_candidate_ids": sorted(
            {v["candidate_id"] for v in intentions.values()}
        ),
        "arrival_episodes": arrivals,
        **score_exploration(arrivals, required_distinct),
        "independent_valid_arrival_frames": report["valid_arrivals"],
        "premature_or_unobservable_arrival_frames": report[
            "premature_or_unobservable_arrivals"
        ],
        "collision": report["collision"],
        "boundary": report["boundary"],
        "integrity_pass": report["integrity_pass"],
        "audit_errors": report["errors"],
        "stop": stop_report,
        "worker_status": manifest.get("status"),
        "source_files_unchanged": manifest.get("source_files_unchanged"),
        "saved_rgb_images": len(list((output / "worker").glob("frame-*-rgb.png"))),
        "controller_status_counts": dict(
            Counter(row["controller_status"] for row in rows)
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def run_mode(args, mode):
    output = args.output / f"mode-{mode}"
    output.mkdir()
    memory = args.output / f"{mode}.sqlite3"
    if memory.exists():
        raise ValueError(
            "Audit requires a fresh, independent memory for each camera mode"
        )
    config = {
        "asset_dir": str(args.assets.resolve()),
        "recording_mode": "audit",
        "mode": mode,
        "run_dir": str(output / "worker"),
        "agency_memory": str(memory),
        "agency_mode": "coverage",
        "generation": 0,
        "max_steps": 3000,
        "save_frames": True,
        "save_every": 20,
        "scoring_labels": True,
    }
    write_json(output / "config.json", config)
    context = multiprocessing.get_context("spawn")
    commands, events, stop = context.Queue(32), context.Queue(32), context.Event()
    process = context.Process(
        target=native_worker,
        args=(config, commands, events, stop, str(output / "native.log")),
    )
    process.start()
    started, last_heartbeat, last_update = time.monotonic(), -math.inf, -math.inf
    enabled_at = stopped_at = stop_sent_wall = None
    enabled_sent = stop_sent = False
    arrival_count, previous_arrival = 0, None
    reported_destinations = set()
    errors, state = [], None
    ending = None
    try:
        while process.is_alive():
            now = time.monotonic()
            if now - last_heartbeat >= 0.2:
                send(
                    commands,
                    {"action": "heartbeat", "generation": 2 if stop_sent else 1},
                )
                last_heartbeat = now
            if now - started >= args.wall_seconds and not stop_sent:
                ending = "wall_limit"
                if send(commands, {"action": "stop", "generation": 2}):
                    stop_sent, stop_sent_wall = True, now
            if stop_sent and now - stop_sent_wall > 60:
                errors.append("stop_window_wall_timeout")
                break
            try:
                event = events.get(timeout=0.03)
            except queue.Empty:
                continue
            if event.get("type") == "error":
                errors.append(event)
            if event.get("type") != "state":
                continue
            state = event["state"]
            sim = state.get("sim_time", 0.0)
            if (
                not enabled_sent
                and not stop_sent
                and state.get("phase") == "idle"
                and state.get("localization_valid")
                and state.get("agency", {}).get("available")
                and send(
                    commands, {"action": "autonomy", "enabled": True, "generation": 1}
                )
            ):
                enabled_sent, enabled_at = True, sim
            goal = tuple(state["goal"]) if state.get("goal") else None
            arrival = (
                (state.get("generation"), goal)
                if state.get("phase") == "arrived"
                else None
            )
            if arrival is not None and arrival != previous_arrival:
                arrival_count += 1
                if goal is not None:
                    reported_destinations.add(goal)
            previous_arrival = arrival
            if (
                not stop_sent
                and enabled_sent
                and (
                    len(reported_destinations) >= args.arrivals
                    or sim - enabled_at >= args.sim_seconds
                )
            ):
                ending = (
                    "distinct_destination_limit"
                    if len(reported_destinations) >= args.arrivals
                    else "simulation_limit"
                )
                if send(commands, {"action": "stop", "generation": 2}):
                    stop_sent, stop_sent_wall = True, now
            if stop_sent and state.get("generation", 0) >= 2:
                if stopped_at is None:
                    stopped_at = sim
                if sim - stopped_at >= 1.0:
                    stop.set()
                    break
            if now - last_update >= 30:
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "wall_s": round(now - started, 1),
                            "sim_s": round(sim, 2),
                            "phase": state.get("phase"),
                            "arrivals_seen": arrival_count,
                            "distinct_destinations_seen": len(reported_destinations),
                            "agency": state.get("agency", {}).get("status"),
                        }
                    ),
                    flush=True,
                )
                last_update = now
    finally:
        stop.set()
        cleanup_start = time.monotonic()
        while process.is_alive() and time.monotonic() - cleanup_start < 30:
            try:
                events.get(timeout=0.1)
            except queue.Empty:
                pass
            process.join(timeout=0.1)
        if process.is_alive():
            errors.append("worker_cleanup_timeout")
            process.terminate()
            process.join(5)
        if process.is_alive():
            errors.append("worker_killed_after_cleanup_timeout")
            process.kill()
            process.join(5)
        for channel in (commands, events):
            channel.cancel_join_thread()
            channel.close()
    harness = {
        "mode": mode,
        "errors": errors,
        "exitcode": process.exitcode,
        "enabled_once": enabled_sent,
        "enabled_sim_time": enabled_at,
        "stop_sent": stop_sent,
        "ending": ending,
        "supervised_arrival_episodes": arrival_count,
        "supervised_distinct_reported_destinations": len(reported_destinations),
        "supervised_reported_destination_goals": [
            list(goal) for goal in sorted(reported_destinations)
        ],
        "wall_seconds": time.monotonic() - started,
        "last_state": state,
    }
    write_json(output / "harness.json", harness)
    try:
        summary = score_completed(output, required_distinct=args.arrivals)
    except Exception as exc:  # noqa: BLE001 — retain native failures in denominator
        summary = {"scoring_failed": f"{type(exc).__name__}: {exc}"}
        write_json(output / "summary.json", summary)
    summary.update(mode=mode, harness_errors=errors, exitcode=process.exitcode)
    summary["passed"] = bool(
        not errors
        and process.exitcode == 0
        and summary.get("integrity_pass")
        and summary.get("source_files_unchanged")
        and summary.get("exploration_passed")
        and summary.get("stop", {}).get("passed")
    )
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "mode": mode,
                "passed": summary["passed"],
                "valid_arrivals": summary.get("valid_arrival_episodes"),
                "distinct_valid_destinations": summary.get(
                    "distinct_valid_destinations"
                ),
                "repeated_successful_destinations": len(
                    summary.get("repeated_successful_destinations", [])
                ),
                "collision": summary.get("collision"),
                "errors": errors,
            }
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument("--modes", type=int, nargs="+", choices=(1, 3), default=[1, 3])
    parser.add_argument(
        "--arrivals",
        type=int,
        default=3,
        help="Required distinct independently verified destinations, with no successful revisits.",
    )
    parser.add_argument("--sim-seconds", type=float, default=120.0)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if (
        not 1 <= args.arrivals <= 3
        or not 0 < args.sim_seconds <= 120
        or not 0 < args.wall_seconds <= 600
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error(
            "Require ≤3 distinct destinations, ≤120 simulation seconds, ≤600 wall seconds, unique modes"
        )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness-source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        args.output / "protocol.json",
        {
            "modes": args.modes,
            "required_distinct_verified_destinations": args.arrivals,
            "supervision_stop_after_distinct_reported_destinations": args.arrivals,
            "successful_revisits_allowed": False,
            "maximum_active_sim_seconds": args.sim_seconds,
            "maximum_worker_wall_seconds_before_stop": args.wall_seconds,
            "stop_observation_sim_seconds": 1.0,
            "stop_observation_wall_grace_seconds": 60,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Synthetic native development run, self-selected goals; no truth fed to decisions.",
        },
    )
    summaries = [run_mode(args, mode) for mode in args.modes]
    write_json(
        args.output / "summary.json",
        {"runs": summaries, "all_passed": all(r["passed"] for r in summaries)},
    )


if __name__ == "__main__":
    main()
