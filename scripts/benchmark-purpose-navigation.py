"""Native virtual-station outcome audit; independent truth is scoring-only.

This experiment evaluates learning which simulated station changes a resource.
It does not evaluate personality, real battery charging, or visual station recognition.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import queue
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIONS = {"station-a": (0.0, 1.0), "station-b": (-0.5, 1.0)}
TARGET = 0.8
INITIAL_RESOURCE = 0.35
DWELL_SECONDS = 1.0
STATION_RADIUS = 0.14
STATION_SPEED = 0.03
TRAVEL_COST = 0.025
STATION_GAINS = {"station-a": 0.0, "station-b": 0.45}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def score_stop(rows, commands):
    stops = [
        c
        for c in commands
        if c.get("action") == "stop"
        and c.get("generation") == 2
        and c.get("outcome") == "accepted"
    ]
    stopped_at = stops[0]["sim_time"] if stops else None
    post = [row for row in rows if stopped_at is not None and row["time"] >= stopped_at]
    end = post[-1]["after_step"]["time"] if post else None
    tail = (
        [
            sample
            for row in post
            for sample in row["after_step"]["physics_trace_scoring_only"]
            if sample["time"] >= end - 0.5
        ]
        if end is not None
        else []
    )
    speeds = [math.hypot(*sample["velocity"]) for sample in tail]
    zero = bool(post) and all(math.hypot(*row["action"]) <= 1e-7 for row in post)
    disabled = bool(post) and all(
        row.get("agency", {}).get("enabled") is False for row in post
    )
    duration = end - stopped_at if end is not None else 0.0
    return {
        "accepted_sim_time": stopped_at,
        "observed_sim_seconds": duration,
        "requested_actions_zero": zero,
        "agency_disabled": disabled,
        "final_half_second_max_physics_speed_m_s": max(speeds) if speeds else None,
        "passed": bool(
            duration >= 1.0 - 1e-8
            and zero
            and disabled
            and speeds
            and max(speeds) <= 0.001
        ),
    }


def score_effects(rows, *, stations=STATIONS):
    """Reconstruct actual effects and their requested stationary physics dwell.

    Repeated delivery of an unchanged outcome is allowed. A reused request ID,
    changed outcome payload, or effect without a full new requested dwell fails.
    """
    requests, effects, errors, trace = {}, {}, [], []
    previous_request = None
    for row in rows:
        request = row.get("agency", {}).get("interaction_request")
        current = request.get("request_id") if request else None
        if current:
            canonical = {"request_id": current, "station_id": request["station_id"]}
            if current in requests:
                if canonical != requests[current]["request"]:
                    errors.append(f"request_payload_changed:{current}")
                if current != previous_request:
                    errors.append(f"request_id_reused:{current}")
            else:
                requests[current] = {"request": canonical, "time": row["time"]}
        previous_request = current
        trace.extend(row["after_step"]["physics_trace_scoring_only"])
        telemetry = row.get("after_step", {}).get("interaction_telemetry") or {}
        outcome = telemetry.get("outcome")
        if not outcome:
            continue
        key = outcome["request_id"]
        canonical = {
            name: outcome[name]
            for name in (
                "request_id",
                "station_id",
                "before",
                "after",
                "timestamp",
                "source",
            )
        }
        if key in effects:
            if canonical != effects[key]["outcome"]:
                errors.append(f"effect_payload_changed:{key}")
            continue
        requested = requests.get(key)
        station = stations.get(outcome["station_id"])
        effect_time = float(outcome["timestamp"])
        samples = [
            sample
            for sample in trace
            if effect_time - DWELL_SECONDS + 1e-8 < sample["time"] <= effect_time + 1e-8
        ]
        requested_dwell = (
            bool(requested) and effect_time - requested["time"] >= DWELL_SECONDS - 1e-8
        )
        matching_station = (
            bool(requested)
            and requested["request"]["station_id"] == outcome["station_id"]
        )
        physical_dwell = (
            bool(station and samples)
            and len(samples) >= 200
            and all(
                math.dist(sample["position"], station) <= STATION_RADIUS + 1e-8
                and math.hypot(*sample["velocity"]) <= STATION_SPEED + 1e-8
                for sample in samples
            )
        )
        source_ok = outcome["source"] == "simulated_station_telemetry"
        passed = bool(
            requested_dwell and matching_station and physical_dwell and source_ok
        )
        effects[key] = {
            "outcome": canonical,
            "request_time": requested["time"] if requested else None,
            "actual_delta": outcome["after"] - outcome["before"],
            "requested_dwell": requested_dwell,
            "matching_station": matching_station,
            "physics_dwell": physical_dwell,
            "physics_samples": len(samples),
            "passed": passed,
        }
        if not passed:
            errors.append(f"invalid_effect:{key}")
    return {
        "requests": requests,
        "effects": list(effects.values()),
        "errors": errors,
        "passed": bool(effects) and not errors,
    }


def score_satisfied(rows):
    qualifying = []
    for row in rows:
        agency = row.get("agency", {})
        if (
            agency.get("status") == "satisfied"
            and agency.get("resource", -1) >= TARGET - 1e-8
        ):
            qualifying.append(row)
        elif qualifying:
            break
    duration = (
        qualifying[-1]["after_step"]["time"] - qualifying[0]["time"]
        if qualifying
        else 0.0
    )
    no_motion = bool(qualifying) and all(
        math.hypot(*row["action"]) <= 1e-7
        and row.get("commanded_goal") is None
        and all(
            math.hypot(*sample["velocity"]) <= STATION_SPEED + 1e-8
            for sample in row["after_step"]["physics_trace_scoring_only"]
        )
        for row in qualifying
    )
    return {
        "observed_sim_seconds": duration,
        "resource": qualifying[-1]["agency"]["resource"] if qualifying else None,
        "no_goal_or_motion": no_motion,
        "passed": bool(duration >= 1.0 - 1e-8 and no_motion),
    }


def score_resource_accounting(rows):
    """Verify world-side physics-tick travel cost and one gain per request."""
    seen, errors = set(), []
    for row in rows:
        before = row.get("interaction_telemetry") or {}
        after = row["after_step"].get("interaction_telemetry") or {}
        if "resource" not in before or "resource" not in after:
            errors.append(f"missing_resource:{row['time']}")
            continue
        expected = before["resource"]
        previous = row["truth_before_scoring_only"]["position"]
        outcome = after.get("outcome")
        for sample in row["after_step"]["physics_trace_scoring_only"]:
            expected = max(
                0.0, expected - TRAVEL_COST * math.dist(previous, sample["position"])
            )
            previous = sample["position"]
            if (
                outcome
                and outcome["request_id"] not in seen
                and abs(sample["time"] - outcome["timestamp"]) <= 1e-8
            ):
                seen.add(outcome["request_id"])
                if abs(outcome["before"] - expected) > 1e-8:
                    errors.append(f"incorrect_effect_before:{outcome['request_id']}")
                expected = min(1.0, expected + STATION_GAINS[outcome["station_id"]])
                if abs(outcome["after"] - expected) > 1e-8:
                    errors.append(f"incorrect_effect_after:{outcome['request_id']}")
        if outcome and outcome["request_id"] not in seen:
            errors.append(f"effect_timestamp_outside_physics:{outcome['request_id']}")
        if (
            not math.isfinite(after["resource"])
            or abs(after["resource"] - expected) > 1e-8
        ):
            errors.append(f"incorrect_resource_transition:{row['time']}")
    return {
        "unique_applied_effects": len(seen),
        "errors": errors,
        "passed": bool(rows) and not errors,
        "scope": "Travel cost and effect timing are checked against every recorded 5ms physics tick.",
    }


def score_learning(rows, effect_report):
    """Outcome counts must follow sensor evidence, never arrival or a proposal."""
    seen, errors = set(), []
    initial = rows[0].get("agency", {}) if rows else {}
    baseline = initial.get("episodes", 0)
    for row in rows:
        outcome = (row.get("interaction_telemetry") or {}).get("outcome")
        if outcome:
            seen.add(outcome["request_id"])
        agency = row.get("agency", {})
        count = agency.get("episodes", -1)
        if not baseline <= count <= baseline + len(seen):
            errors.append(f"learning_without_delivered_effect:{row['time']}")
        if sum(p["outcomes"] for p in agency.get("preferences", [])) != count:
            errors.append(f"inconsistent_learned_counts:{row['time']}")
    latest = rows[-1].get("agency", {}) if rows else {}
    effects = effect_report["effects"]
    counts = {station: 0 for station in STATIONS}
    for effect in effects:
        counts[effect["outcome"]["station_id"]] += 1
    preferences = {p["candidate_id"]: p for p in latest.get("preferences", [])}
    previous = {p["candidate_id"]: p for p in initial.get("preferences", [])}
    counts_match = latest.get("episodes") == baseline + len(effects) and all(
        preferences.get(station, {}).get("outcomes", 0)
        == previous.get(station, {}).get("outcomes", 0) + count
        for station, count in counts.items()
    )
    ineffective = any(
        e["outcome"]["station_id"] == "station-a" and abs(e["actual_delta"]) <= 1e-8
        for e in effects
    )
    effective = any(
        e["outcome"]["station_id"] == "station-b" and e["actual_delta"] >= 0.04 - 1e-8
        for e in effects
    )
    learned_difference = (
        set(preferences) >= set(STATIONS)
        and preferences["station-b"]["expected_gain"]
        > preferences["station-a"]["expected_gain"]
    )
    return {
        "episodes": latest.get("episodes"),
        "initial_episodes": baseline,
        "observed_effects_by_station": counts,
        "counts_match_unique_effects": counts_match,
        "ineffective_station_observed": ineffective,
        "useful_station_observed": effective,
        "useful_station_has_higher_learned_expected_gain": learned_difference,
        "no_learning_before_effect_telemetry": not errors,
        "errors": errors,
        "passed": bool(
            baseline == 0
            and effects
            and counts_match
            and ineffective
            and effective
            and learned_difference
            and not errors
        ),
    }


def score_restart(rows, learned_snapshot, commands):
    first = rows[0].get("agency", {}) if rows else {}
    last = rows[-1].get("agency", {}) if rows else {}
    keys = (
        "agent_id",
        "map_version",
        "episodes",
        "decisions",
        "change_epoch",
        "preferences",
    )
    memory_equal = bool(first and learned_snapshot) and all(
        learned_snapshot.get(key) == first.get(key) == last.get(key) for key in keys
    )
    reset_resource = first.get("resource")
    resource_reset = (
        isinstance(reset_resource, (int, float))
        and abs(reset_resource - INITIAL_RESOURCE) <= 1e-8
    )
    disabled = bool(rows) and all(
        row.get("agency", {}).get("enabled") is False
        and row.get("agency", {}).get("intention") is None
        and row.get("agency", {}).get("interaction_request") is None
        and row.get("commanded_goal") is None
        and math.hypot(*row["action"]) <= 1e-7
        for row in rows
    )
    no_enable = not any(
        command.get("action") == "autonomy" and command.get("enabled")
        for command in commands
    )
    stops = [
        command
        for command in commands
        if command.get("action") == "stop" and command.get("outcome") == "accepted"
    ]
    before_stop = [row for row in rows if stops and row["time"] < stops[0]["sim_time"]]
    duration = (
        before_stop[-1]["after_step"]["time"] - before_stop[0]["time"]
        if before_stop
        else 0.0
    )
    speeds = [
        math.hypot(*sample["velocity"])
        for row in before_stop
        for sample in row["after_step"]["physics_trace_scoring_only"]
    ]
    stationary = bool(speeds) and max(speeds) <= 0.001
    return {
        "persistent_outcomes_preferences_exact": memory_equal,
        "episode_resource_reset_to_035": resource_reset,
        "disabled_no_goal_request_or_action": disabled,
        "no_enable_command_sent": no_enable,
        "pre_stop_observation_seconds": duration,
        "pre_stop_max_physics_speed_m_s": max(speeds) if speeds else None,
        "passed": bool(
            memory_equal
            and resource_reset
            and disabled
            and no_enable
            and stationary
            and duration >= 1.0 - 1e-8
        ),
    }


def score_completed(output):
    rows = read_jsonl(output / "worker/rows.jsonl")
    commands = read_jsonl(output / "worker/commands.jsonl")
    auditor = load_script("audit-interactive.py", "purpose_independent_physics_audit")
    physics = auditor.score_rows(rows)
    manifest_path = output / "worker/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    provenance = auditor.verify_manifest(output / "worker", manifest)
    provenance.update(auditor.verify_recording(rows, manifest))
    physics["provenance"] = provenance
    physics["images"] = auditor.verify_images(output / "worker", rows)
    physics["integrity_pass"] = (
        physics["integrity_pass"]
        and all(provenance.values())
        and not physics["images"]["failures"]
    )
    write_json(output / "independent-audit.json", physics)
    effects = score_effects(rows)
    result = {
        "frames": len(rows),
        "simulation_seconds": rows[-1]["after_step"]["time"] if rows else 0.0,
        "collision": physics["collision"],
        "boundary": physics["boundary"],
        "integrity_pass": physics["integrity_pass"],
        "valid_arrival_frames": physics["valid_arrivals"],
        "invalid_arrival_frames": physics["premature_or_unobservable_arrivals"],
        "audit_errors": physics["errors"],
        "effects": effects,
        "resource_accounting": score_resource_accounting(rows),
        "learning": score_learning(rows, effects),
        "satisfied": score_satisfied(rows),
        "stop": score_stop(rows, commands),
        "source_files_unchanged": manifest.get("source_files_unchanged"),
        "worker_status": manifest.get("status"),
        "first_agency": rows[0].get("agency") if rows else None,
        "last_agency": rows[-1].get("agency") if rows else None,
    }
    write_json(output / "audit.json", result)
    return result


def run_phase(args, mode, output, memory, *, learning):
    output.mkdir()
    config = {
        "asset_dir": str(args.assets.resolve()),
        "recording_mode": "audit",
        "mode": mode,
        "run_dir": str(output / "worker"),
        "agency_memory": str(memory),
        "agency_mode": "purpose",
        "generation": 0,
        "max_steps": 3000,
        "save_frames": learning,
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
    ready_at = satisfied_at = stopped_at = stop_sent_wall = None
    enabled_sent = stop_sent = False
    errors, state, ending = [], None, None
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
            sim, agency = state.get("sim_time", 0.0), state.get("agency", {})
            if (
                ready_at is None
                and state.get("phase") == "idle"
                and state.get("localization_valid")
                and agency.get("available")
            ):
                ready_at = sim
            if learning and ready_at is not None and not enabled_sent and not stop_sent:
                enabled_sent = send(
                    commands, {"action": "autonomy", "enabled": True, "generation": 1}
                )
            if (
                learning
                and agency.get("status") == "satisfied"
                and agency.get("resource", -1) >= TARGET - 1e-8
            ):
                satisfied_at = sim if satisfied_at is None else satisfied_at
            else:
                satisfied_at = None
            phase_complete = (
                satisfied_at is not None and sim - satisfied_at >= 1.0
                if learning
                else ready_at is not None and sim - ready_at >= 1.0
            )
            if (
                not stop_sent
                and ready_at is not None
                and (phase_complete or sim - ready_at >= args.sim_seconds)
            ):
                ending = (
                    ("satisfied_idle_window" if learning else "disabled_restart_window")
                    if phase_complete
                    else "simulation_limit"
                )
                if send(commands, {"action": "stop", "generation": 2}):
                    stop_sent, stop_sent_wall = True, now
            if stop_sent and state.get("generation", 0) >= 2:
                stopped_at = sim if stopped_at is None else stopped_at
                if sim - stopped_at >= 1.0:
                    stop.set()
                    break
            if now - last_update >= 30:
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "phase": "learning" if learning else "restart",
                            "wall_s": round(now - started, 1),
                            "sim_s": round(sim, 2),
                            "status": agency.get("status"),
                            "resource": agency.get("resource"),
                            "episodes": agency.get("episodes"),
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
        "learning": learning,
        "errors": errors,
        "exitcode": process.exitcode,
        "enabled_once": enabled_sent,
        "stop_sent": stop_sent,
        "ending": ending,
        "wall_seconds": time.monotonic() - started,
        "last_state": state,
    }
    write_json(output / "harness.json", harness)
    try:
        audit = score_completed(output)
    except Exception as error:  # noqa: BLE001 — preserve failed native attempts
        audit = {"scoring_failed": f"{type(error).__name__}: {error}"}
        write_json(output / "audit.json", audit)
    return {"harness": harness, "audit": audit}


def accepted_physics(result):
    harness, audit = result["harness"], result["audit"]
    return bool(
        not harness["errors"]
        and harness["exitcode"] == 0
        and audit.get("worker_status") == "complete"
        and audit.get("source_files_unchanged")
        and audit.get("integrity_pass")
        and audit.get("resource_accounting", {}).get("passed")
        and audit.get("stop", {}).get("passed")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument(
        "--modes", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3]
    )
    parser.add_argument("--sim-seconds", type=float, default=120.0)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if (
        not 0 < args.sim_seconds <= 120
        or not 0 < args.wall_seconds <= 600
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error(
            "Require ≤120 simulation seconds, ≤600 wall seconds, unique camera modes"
        )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness-source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        args.output / "protocol.json",
        {
            "modes": args.modes,
            "initial_virtual_resource": INITIAL_RESOURCE,
            "target_virtual_resource": TARGET,
            "station_radius_m": STATION_RADIUS,
            "station_speed_m_s": STATION_SPEED,
            "station_dwell_s": DWELL_SECONDS,
            "station_gains": STATION_GAINS,
            "station_positions": STATIONS,
            "travel_cost_per_metre": TRAVEL_COST,
            "maximum_active_sim_seconds": args.sim_seconds,
            "maximum_worker_wall_seconds_before_stop": args.wall_seconds,
            "satisfied_idle_observation_seconds": 1.0,
            "disabled_restart_observation_seconds": 1.0,
            "stop_observation_seconds": 1.0,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Synthetic native station telemetry; camera-based navigation; no personality or real battery claim.",
        },
    )
    runs = []
    for mode in args.modes:
        output = args.output / f"mode-{mode}"
        output.mkdir()
        memory = output / "purpose.sqlite3"
        if memory.exists():
            raise ValueError("Initial purpose memory must be fresh")
        learning = run_phase(args, mode, output / "learning", memory, learning=True)
        restart = run_phase(args, mode, output / "restart", memory, learning=False)
        restart_rows = read_jsonl(output / "restart/worker/rows.jsonl")
        restart_commands = read_jsonl(output / "restart/worker/commands.jsonl")
        try:
            durability = score_restart(
                restart_rows, learning["audit"].get("last_agency"), restart_commands
            )
        except Exception as error:  # noqa: BLE001 — retain a failed restart audit
            durability = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        checks = {
            "learning_physics_and_stop": accepted_physics(learning),
            "station_effects_independently_valid": learning["audit"]
            .get("effects", {})
            .get("passed", False),
            "learned_from_consequences_not_arrivals": learning["audit"]
            .get("learning", {})
            .get("passed", False),
            "resource_need_satisfied_and_idle": learning["audit"]
            .get("satisfied", {})
            .get("passed", False),
            "restart_physics_and_stop": accepted_physics(restart),
            "outcomes_durable_restart_disabled": durability["passed"],
        }
        result = {
            "mode": mode,
            "learning": learning,
            "restart": restart,
            "durability": durability,
            "checks": checks,
            "passed": all(checks.values()),
        }
        write_json(output / "summary.json", result)
        print(
            json.dumps({"mode": mode, "passed": result["passed"], "checks": checks}),
            flush=True,
        )
        runs.append(result)
    passed = all(run["passed"] for run in runs)
    write_json(args.output / "summary.json", {"runs": runs, "all_passed": passed})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
