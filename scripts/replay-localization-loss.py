#!/usr/bin/env python3
"""Replay native RGB/command inputs, then synthetic prolonged-loss bookkeeping.

No Genesis, camera model, policy, or truth state is loaded. The appended zero
acknowledgements are an explicit counterfactual, not evidence of native rest.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bb8_rl.interactive_runtime import ControlSession, jsonable

ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = (
    "measurement",
    "prior_applied_command",
    "prior_acknowledged_command_intervals",
)
SAMPLE_SECONDS = (0, 1, 10, 60, 120, 240)
RECOVERY_ANCHOR_KEYS = (
    "recovery_anchor_pose",
    "recovery_anchor_time",
    "recovery_anchor_source",
)
FRAME_SECONDS = 0.05
TICK_SECONDS = 0.005
GOAL_ATTEMPT_SECONDS = 2.0


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (
        json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n"
    ).encode()


class NoNavigationMemory:
    """Permissive point predicate solely for a rejected-goal API probe.

    There is no geometry or navigation experiment. Any route/policy call fails.
    """

    def planning_grid(self, radius):
        return self

    def segment_free(self, start, end, radius):
        return bool(np.isfinite([*start, *end, radius]).all() and radius >= 0)

    def route(self, start, goal):
        raise AssertionError("The idle replay must never plan a route")


def no_policy(vector):
    raise AssertionError("The idle replay must never call a drive policy")


def zero_timeline(start, end):
    boundaries = np.linspace(start, end, 11)
    return [
        {"start": float(a), "end": float(b), "command": [0.0, 0.0]}
        for a, b in pairwise(boundaries)
    ]


def measurement(record):
    item = record["measurement"]
    return SimpleNamespace(
        timestamp=float(item["timestamp"]),
        status=item["status"],
        xy=None if item["xy"] is None else np.asarray(item["xy"], float),
        covariance=None
        if item["covariance"] is None
        else np.asarray(item["covariance"], float),
    )


def replay(source, output, *, duration_seconds=240.0, max_prefix_rows=200):
    """Freeze one causal native prefix, then append labelled synthetic inputs."""
    source, output = Path(source), Path(output)
    steps = round(duration_seconds / FRAME_SECONDS)
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds < GOAL_ATTEMPT_SECONDS
        or not math.isclose(steps * FRAME_SECONDS, duration_seconds, abs_tol=1e-9)
        or not 1 <= max_prefix_rows <= 10000
    ):
        raise ValueError("Need a whole number of 50 ms frames and at least two seconds")
    if output.exists():
        raise ValueError("Preserve replay artifacts; choose a fresh output directory")
    stops = []
    session = ControlSession(
        policy=no_policy,
        memory=NoNavigationMemory(),
        stop=lambda: stops.append(True),
        map_version="offline-no-navigation-map",
        calibration_version="native-measurement-frame",
        demo_goal=[0.5, 0.0],
        clock=lambda: 0.0,
    )
    prefix_bytes, prefix_inputs, prefix_outputs = [], [], []
    consecutive_visible = 0
    with source.open("rb") as stream:
        for index in range(max_prefix_rows):
            line = stream.readline()
            if not line:
                break
            try:
                # All other native fields, including scoring truth, are discarded.
                source_row = json.loads(line)
                record = {key: source_row[key] for key in INPUT_KEYS}
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    "Need complete native command/measurement rows"
                ) from error
            observed = measurement(record)
            action = session.decide(
                observed.timestamp,
                observed,
                record["prior_applied_command"],
                record["prior_acknowledged_command_intervals"],
            )
            if np.any(action):
                raise AssertionError("Idle prefix issued a nonzero request")
            state = jsonable(session.state())
            prefix_bytes.append(line)
            prefix_inputs.append(record)
            prefix_outputs.append(
                {
                    "kind": "recorded_native_input_replayed_offline",
                    "index": index,
                    "input": record,
                    "action": action.tolist(),
                    "state": state,
                }
            )
            consecutive_visible = (
                consecutive_visible + 1 if session.idle_belief.measured else 0
            )
            zero_ack = np.array_equal(record["prior_applied_command"], [0, 0]) and all(
                np.array_equal(part["command"], [0, 0])
                for part in record["prior_acknowledged_command_intervals"]
            )
            if (
                state["localization_valid"]
                and session.idle_belief.measured
                and consecutive_visible >= session.parameters.minimum_visual_samples
                and session.idle_belief.velocity_radius
                <= session.parameters.max_initial_velocity_radius
                and zero_ack
            ):
                break
        else:
            raise ValueError("No stable zero-ack visible prefix within row limit")
    if not prefix_outputs or not (
        session.idle_belief.measured
        and consecutive_visible >= session.parameters.minimum_visual_samples
        and session.idle_belief.velocity_radius
        <= session.parameters.max_initial_velocity_radius
        and zero_ack
        and session.state()["localization_valid"]
    ):
        raise ValueError("Native prefix is incomplete or not yet stably localized")

    output.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/bb8_rl/interactive_runtime.py",
        ROOT / "src/bb8_rl/occluded_control.py",
        ROOT / "src/bb8_rl/braking_prediction.py",
    ]
    (output / "source").mkdir()
    source_hashes = {}
    for path in sources:
        data = path.read_bytes()
        (output / "source" / path.name).write_bytes(data)
        source_hashes[str(path.relative_to(ROOT))] = digest(data)
    allowed_bytes = b"".join(encoded(item) for item in prefix_inputs)
    (output / "prefix-inputs.jsonl").write_bytes(allowed_bytes)
    anchor = session.capture_time
    protocol = {
        "schema_version": 1,
        "scope": "Offline counterfactual bookkeeping after a native RGB/command prefix; NOT 240 seconds of native or physical observation.",
        "source_rows_path": str(source.resolve()),
        "source_prefix_rows": len(prefix_inputs),
        "source_native_raw_prefix_sha256": digest(b"".join(prefix_bytes)),
        "allowed_prefix_sha256": digest(allowed_bytes),
        "allowed_input_fields": list(INPUT_KEYS),
        "truth_to_replay": False,
        "policy_loaded": False,
        "geometry": "No navigation; permissive point predicate for a goal that must be rejected before route planning.",
        "prefix_selection": "Earliest accepted visible zero-ACK frame with existing consecutive minimum visual samples and velocity-radius initialization gate.",
        "prefix_end_sim_time": anchor,
        "duration_seconds": duration_seconds,
        "synthetic_frame_seconds": FRAME_SECONDS,
        "synthetic_ack_tick_seconds": TICK_SECONDS,
        "synthetic_inputs": "All frames missing; all complete acknowledged intervals and endpoint commands zero. This is assumed counterfactual input, not measured actuator history.",
        "goal_attempt_offset_seconds": GOAL_ATTEMPT_SECONDS,
        "sample_offset_seconds": [t for t in SAMPLE_SECONDS if t <= duration_seconds],
        "recovery_anchor_assertion": "Freeze pose, timestamp and source on the first lost frame; require exact equality throughout the synthetic loss.",
        "parameters": asdict(session.parameters),
        "dynamics": asdict(session.dynamics),
        "source_sha256": source_hashes,
    }
    (output / "protocol.json").write_bytes(encoded(protocol))
    samples = [{"offset_seconds": 0, "state": prefix_outputs[-1]["state"]}]
    first_lost, goal_attempt, recovery_anchor = None, None, None
    checks = {
        "all_actions_zero": True,
        "expired_pose_and_radius_unavailable": True,
        "no_goal_or_route_after_loss": True,
        "advisory_finite_and_non_authoritative": True,
        "raw_radius_monotone_during_missing": True,
        "recovery_anchor_valid_and_fixed_after_loss": True,
        "existing_prediction_limits_respected": True,
    }
    previous_radius = session.idle_belief.position_radius
    with (output / "replay.jsonl").open("wb") as stream:
        for row in prefix_outputs:
            stream.write(encoded(row))
        for index in range(1, steps + 1):
            timestamp = anchor + index * FRAME_SECONDS
            observed = SimpleNamespace(
                timestamp=timestamp, status="missing", xy=None, covariance=None
            )
            timeline = zero_timeline(anchor + (index - 1) * FRAME_SECONDS, timestamp)
            action = session.decide(timestamp, observed, [0, 0], timeline)
            state = jsonable(session.state())
            offset = index * FRAME_SECONDS
            if state["localization_status"] == "lost" and first_lost is None:
                first_lost = offset
                recovery_anchor = {key: state[key] for key in RECOVERY_ANCHOR_KEYS}
            if index == round(GOAL_ATTEMPT_SECONDS / FRAME_SECONDS):
                outcome = session.receive(
                    [{"action": "goal", "generation": 1, "x": 0.5, "y": 0.0}],
                    sim_time=timestamp,
                )[0]
                goal_attempt = {"offset_seconds": offset, "record": outcome}
            raw = state["raw_position_radius"]
            advisory = state["braking_prediction"]
            checks["all_actions_zero"] &= not bool(np.any(action))
            checks["raw_radius_monotone_during_missing"] &= raw >= previous_radius
            checks["advisory_finite_and_non_authoritative"] &= (
                advisory["valid"]
                and not advisory["motion_authority"]
                and not advisory["observed"]
                and math.isfinite(advisory["radius_limit_m"])
                and advisory["radius_m"] <= advisory["radius_limit_m"] + 1e-12
            )
            if (
                offset > session.parameters.max_occlusion_seconds + 1e-9
                or raw > session.parameters.max_position_radius
            ):
                checks["existing_prediction_limits_respected"] &= (
                    state["localization_status"] == "lost"
                    and not state["localization_valid"]
                    and state["pose"] is None
                    and state["position_radius"] is None
                )
            if first_lost is not None:
                checks["recovery_anchor_valid_and_fixed_after_loss"] &= (
                    recovery_anchor["recovery_anchor_pose"] is not None
                    and len(recovery_anchor["recovery_anchor_pose"]) == 2
                    and np.isfinite(recovery_anchor["recovery_anchor_pose"]).all()
                    and recovery_anchor["recovery_anchor_time"] is not None
                    and math.isfinite(recovery_anchor["recovery_anchor_time"])
                    and recovery_anchor["recovery_anchor_time"] <= anchor + first_lost
                    and bool(recovery_anchor["recovery_anchor_source"])
                    and all(
                        state[key] == recovery_anchor[key]
                        for key in RECOVERY_ANCHOR_KEYS
                    )
                )
                checks["expired_pose_and_radius_unavailable"] &= (
                    not state["localization_valid"]
                    and state["pose"] is None
                    and state["position_radius"] is None
                )
                checks["no_goal_or_route_after_loss"] &= (
                    session.goal is None
                    and session.pending_goal is None
                    and session.controller is None
                )
            if index in {round(t / FRAME_SECONDS) for t in SAMPLE_SECONDS if t}:
                samples.append({"offset_seconds": offset, "state": state})
            stream.write(
                encoded(
                    {
                        "kind": "synthetic_missing_rgb_zero_ack_extension",
                        "offset_seconds": offset,
                        "input": {
                            "measurement": {
                                "timestamp": timestamp,
                                "status": "missing",
                                "xy": None,
                                "covariance": None,
                            },
                            "prior_applied_command": [0, 0],
                            "prior_acknowledged_command_intervals": timeline,
                        },
                        "action": action.tolist(),
                        "state": state,
                    }
                )
            )
            previous_radius = raw
    checks["loss_detected"] = first_lost is not None
    checks["new_goal_rejected_while_lost"] = (
        goal_attempt is not None and goal_attempt["record"]["outcome"] == "rejected"
    )
    checks["sources_unchanged"] = all(
        digest((ROOT / path).read_bytes()) == sha for path, sha in source_hashes.items()
    )
    report = {
        "status": "complete",
        "passed": all(checks.values()),
        "scope": protocol["scope"],
        "full_240_second_bookkeeping": duration_seconds >= 240,
        "prefix_rows": len(prefix_inputs),
        "synthetic_frames": steps,
        "first_lost_offset_seconds": first_lost,
        "recovery_anchor": recovery_anchor,
        "goal_attempt": goal_attempt,
        "stop_calls": len(stops),
        "checks": checks,
        "samples": samples,
        "protocol_sha256": digest((output / "protocol.json").read_bytes()),
        "artifact_sha256": {
            name: digest((output / name).read_bytes())
            for name in ("prefix-inputs.jsonl", "replay.jsonl")
        },
    }
    (output / "report.json").write_bytes(encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=240.0)
    parser.add_argument("--max-prefix-rows", type=int, default=200)
    args = parser.parse_args()
    report = replay(
        args.source,
        args.output,
        duration_seconds=args.duration_seconds,
        max_prefix_rows=args.max_prefix_rows,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "prefix_rows": report["prefix_rows"],
                "first_lost_offset_seconds": report["first_lost_offset_seconds"],
                "report": str(args.output / "report.json"),
            }
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
