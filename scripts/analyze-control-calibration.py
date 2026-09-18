"""Score historical camera-control residuals and acknowledged zero-target braking.

Offline scoring only: this module neither changes model bounds nor fits a controller.
Frames are correlated; reported quantiles are descriptive, not confidence limits.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    f"work/m710/native-v2/{name}/worker"
    for name in ("route-mode1", "route-mode3", "demo-mode1")
] + [
    f"work/m79/native-bundle2/{name}/worker"
    for name in (
        "goal-mode1", "goal-mode2", "goal-mode3", "stop-moving",
        "handover-mode2", "all-lost-mode2", "invalid-memory",
    )
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    if not np.isfinite(a).all():
        raise ValueError("Non-finite scoring metric")
    return {
        "n": len(a), "min": float(a.min()), "p05": float(np.quantile(a, .05)),
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p95": float(np.quantile(a, .95)), "max": float(a.max()),
    }


def norm(vector):
    return float(np.linalg.norm(vector))


def conditional_stop_bound(speed, threshold=.03, tau=.35, deceleration=.5):
    """Current controller's declared zero-target tail, used only for comparison."""
    speed = max(float(speed), threshold)
    knee = tau * deceleration
    saturated_time = max(0., speed - max(knee, threshold)) / deceleration
    tail_start = max(threshold, min(speed, knee))
    time_to_threshold = saturated_time + tau * math.log(tail_start / threshold)
    saturated_distance = max(0., speed**2 - max(knee, threshold)**2) / (2 * deceleration)
    distance_to_threshold = saturated_distance + tau * (tail_start - threshold)
    return time_to_threshold, distance_to_threshold


def braking_episodes(ticks, threshold=.03, dead_zone=.05):
    """Keep zero-command bouts, explicitly right-censor interrupted decelerations.

    Each tick supplies pre/post physics state and the command acknowledged BEFORE
    its integration. Endpoint applied-command samples cannot replace that command.
    Initial idle is excluded; a preceding nonzero target and speed > threshold
    are required. A gap is an error rather than an assumed continuation.
    """
    episodes, active, preceding_nonzero = [], None, False
    previous_end = None

    def finish(reason):
        nonlocal active
        if active is None:
            return
        active["ended_by"] = reason
        active["right_censored_at_speed_threshold"] = active["time_to_threshold_s"] is None
        active["zero_tail_tau_s"] = stats(active.pop("tau_samples"))
        bound_t, bound_d = conditional_stop_bound(active["initial_speed_m_s"], threshold)
        active["conditional_bound_time_to_threshold_s"] = bound_t
        active["conditional_bound_distance_to_threshold_m"] = bound_d
        if not active["right_censored_at_speed_threshold"]:
            active["time_bound_excess_s"] = active["time_to_threshold_s"] - bound_t
            active["distance_bound_excess_m"] = active["distance_to_threshold_m"] - bound_d
        episodes.append(active)
        active = None

    for tick in ticks:
        if previous_end is not None and abs(tick["start"] - previous_end) > 1e-7:
            raise ValueError("Physics/command timeline has a gap or overlap")
        previous_end = tick["end"]
        zero = norm(tick["command"]) <= dead_zone
        initial_speed = norm(tick["velocity_before"])
        speed = norm(tick["velocity_after"])
        if not zero:
            finish("nonzero_target_resumed")
            preceding_nonzero = True
            continue
        if active is None and preceding_nonzero and initial_speed > threshold:
            active = {
                "start": tick["start"], "initial_speed_m_s": initial_speed,
                "duration_s": 0., "path_length_m": 0.,
                "time_to_threshold_s": None, "distance_to_threshold_m": None,
                "tau_samples": [], "ticks": 0,
            }
        preceding_nonzero = False
        if active is None:
            continue
        active["duration_s"] = tick["end"] - active["start"]
        active["path_length_m"] += norm(np.subtract(tick["position_after"], tick["position_before"]))
        active["final_speed_m_s"] = speed
        active["ticks"] += 1
        if active["time_to_threshold_s"] is None and speed <= threshold:
            active["time_to_threshold_s"] = active["duration_s"]
            active["distance_to_threshold_m"] = active["path_length_m"]
        if initial_speed > speed > threshold:
            active["tau_samples"].append((tick["end"] - tick["start"]) / math.log(initial_speed / speed))
    finish("recording_ended")
    return episodes


def physics_ticks(rows):
    ticks = []
    for row in rows:
        after = row["after_step"]
        trace = after["physics_trace_scoring_only"]
        commands = after["acknowledged_command_intervals"]
        if len(trace) != len(commands):
            raise ValueError("Physics/command tick count mismatch")
        position = row["truth_before_scoring_only"]["position"]
        velocity = row["truth_before_scoring_only"]["velocity"]
        for state, command in zip(trace, commands, strict=True):
            if abs(state["time"] - command["end"]) > 1e-7:
                raise ValueError("Physics/command tick time mismatch")
            if set(command) != {"start", "end", "command"}:
                raise ValueError("Command timeline must contain command-only telemetry")
            ticks.append({
                "start": command["start"], "end": command["end"],
                "command": command["command"], "position_before": position,
                "velocity_before": velocity, "position_after": state["position"],
                "velocity_after": state["velocity"],
            })
            position, velocity = state["position"], state["velocity"]
    return ticks


def residual_records(rows, run_id):
    records = []
    for row in rows:
        truth = row["truth_before_scoring_only"]
        speed = norm(truth["velocity"])
        measurement = row["measurement"]
        controller = row.get("controller") or {}
        record = {
            "run": run_id, "mode": row["mode"], "time": row["time"],
            "motion": "moving" if speed > .03 else "stationary_or_slow",
            "true_speed_m_s": speed, "pose_source": controller.get("pose_source", "none"),
            "sources": "+".join(row["source_ids"]) or "none",
            "measurement_status": measurement["status"],
            "xy_bin": tuple(math.floor(float(x) / .20) for x in truth["position"]),
        }
        if measurement["xy"] is not None and measurement["status"] == "visible":
            if abs(measurement["timestamp"] - row["capture_time"]) > 1e-7:
                raise ValueError("RGB residual requires a synchronized measurement")
            record["rgb_error_m"] = norm(np.subtract(measurement["xy"], truth["position"]))
        if controller.get("state_is_current") and abs(controller["timestamp"] - row["capture_time"]) < 1e-7:
            record["position_error_m"] = norm(np.subtract(controller["xy"], truth["position"]))
            record["velocity_error_m_s"] = norm(np.subtract(controller["velocity"], truth["velocity"]))
            record["speed_bias_m_s"] = controller["speed_m_s"] - speed
            record["position_radius_m"] = controller["position_radius_m"]
            record["velocity_radius_m_s"] = controller["velocity_radius_m_s"]
        records.append(record)
    return records


def residual_summary(records):
    out = {
        "frames": len(records),
        "measurement_status_counts": dict(Counter(r["measurement_status"] for r in records)),
        "source_set_counts": dict(Counter(r["sources"] for r in records)),
    }
    for key in ("rgb_error_m", "position_error_m", "velocity_error_m_s", "speed_bias_m_s",
                "position_radius_m", "velocity_radius_m_s", "true_speed_m_s"):
        out[key] = stats([r[key] for r in records if key in r])
    for error, radius, label in (("position_error_m", "position_radius_m", "position"),
                                 ("velocity_error_m_s", "velocity_radius_m_s", "velocity")):
        valid = [r for r in records if error in r]
        out[label + "_radius_coverage"] = {
            "n": len(valid), "inside": sum(r[error] <= r[radius] + 1e-9 for r in valid),
            "maximum_excess": max((r[error] - r[radius] for r in valid), default=None),
        }
    # One average per run / 20cm position bin; waiting frames cannot dominate
    # another part of the route. Report alongside moving-only results.
    bins = defaultdict(list)
    for r in records:
        if r["motion"] == "moving" and "rgb_error_m" in r:
            bins[(r["run"], r["xy_bin"])].append(r["rgb_error_m"])
    out["equal_weight_moving_run_position_bin_mean_rgb_error_m"] = stats([np.mean(v) for v in bins.values()])
    return out


def certificate_summary(rows):
    statuses = [r["controller_status"] for r in rows]
    guards = {"stopping_envelope_not_certified", "proactive_envelope_not_certified", "route_not_certified"}
    out = {
        "status_counts": dict(Counter(statuses)),
        "guard_frames": sum(s in guards for s in statuses),
        "guard_episodes": sum(s in guards and (i == 0 or statuses[i - 1] not in guards) for i, s in enumerate(statuses)),
        "guard_while_physics_moving_frames": 0,
        "lowest_candidate_history_or_momentum_dominated_frames": 0,
    }
    metrics = defaultdict(list)
    for row in rows:
        c = row.get("controller") or {}
        if c.get("speed_scale") is not None:
            metrics["selected_speed_scale"].append(c["speed_scale"])
        if row["controller_status"] not in guards:
            continue
        out["guard_while_physics_moving_frames"] += norm(row["truth_before_scoring_only"]["velocity"]) > .03
        trials = c.get("envelope_trials", [])
        if not trials:
            continue
        low = trials[-1]
        for key in ("envelope_radius_m", "reaction_path_length_m", "stopping_distance_m",
                    "candidate_target_speed_m_s", "initial_speed_bound_m_s", "retained_target_speed_m_s"):
            metrics["guard_lowest_trial_" + key].append(low[key])
        metrics["guard_position_radius_m"].append(c["position_radius_m"])
        out["lowest_candidate_history_or_momentum_dominated_frames"] += (
            max(low["initial_speed_bound_m_s"], low["acknowledged_target_speed_m_s"],
                low["retained_target_speed_m_s"]) > low["candidate_target_speed_m_s"] + 1e-8
        )
    out.update({k: stats(v) for k, v in metrics.items()})
    out["interpretation"] = "Rejected commands are counterfactual; observed braking does not prove their rejection unnecessary."
    return out


def analyze_run(path):
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    manifest = json.loads((path / "manifest.json").read_text())
    run_id = str(path.relative_to(ROOT))
    source_checks = {
        name: digest(path / "source" / name) == expected
        for name, expected in manifest["source_sha256"].items()
    }
    records = residual_records(rows, run_id)
    episodes = braking_episodes(physics_ticks(rows))
    complete = [e for e in episodes if not e["right_censored_at_speed_threshold"]]
    report = {
        "run": run_id, "mode": manifest["mode"],
        "manifest_sha256": digest(path / "manifest.json"), "rows_sha256": digest(path / "rows.jsonl"),
        "source_sha256": manifest["source_sha256"], "source_snapshots_match": source_checks,
        "asset_sha256": manifest["asset_sha256"],
        "truth_to_controller": manifest["truth_to_controller"],
        "segmentation_to_controller": manifest["segmentation_to_controller"],
        "all": residual_summary(records),
        "by_motion": {m: residual_summary([r for r in records if r["motion"] == m]) for m in ("moving", "stationary_or_slow")},
        "by_pose_source": {s: residual_summary([r for r in records if r["pose_source"] == s]) for s in sorted({r["pose_source"] for r in records})},
        "certificate": certificate_summary(rows),
        "braking": {
            "episodes": episodes, "moving_zero_bouts": len(episodes),
            "complete_to_3cm_s": len(complete), "censored_before_3cm_s": len(episodes) - len(complete),
            "complete_time_to_threshold_s": stats([e["time_to_threshold_s"] for e in complete]),
            "complete_distance_to_threshold_m": stats([e["distance_to_threshold_m"] for e in complete]),
            "complete_time_bound_excess_s": stats([e["time_bound_excess_s"] for e in complete]),
            "complete_distance_bound_excess_m": stats([e["distance_bound_excess_m"] for e in complete]),
            "equal_weight_episode_median_zero_tail_tau_s": stats([e["zero_tail_tau_s"]["median"] for e in episodes if e["zero_tail_tau_s"]["n"]]),
        },
    }
    return report, records


def main(args):
    if args.output.exists():
        raise ValueError("Keep prior evidence; choose a fresh output")
    reports, records = [], []
    for path in args.runs or [ROOT / p for p in DEFAULT_RUNS]:
        report, rec = analyze_run(path.resolve())
        reports.append(report)
        records.extend(rec)
    result = {
        "scope": "Historical synthetic development calibration diagnostics; no controller or bound changes; no independent room validation.",
        "script_sha256": digest(__file__), "runs": reports,
        "definitions": {
            "moving": "True capture speed >0.03m/s; stationary/slow strata retained separately.",
            "spatial_balance": "Equal-weight means per run and 20cm truth-position bin among moving RGB frames, scoring only.",
            "braking": "Command-only acknowledged pre-integration ticks define zero-target bouts; require preceding nonzero target and initial true speed >0.03m/s. Interrupted bouts remain censored.",
            "threshold_sampling": "First post-physics tick <=0.03m/s; time upper-rounded by at most0.005s. Distances are cumulative XY path.",
            "conditional_bound": "Fixed pre-existing tau_upper=.35s and deceleration_lower=.5m/s² with no adverse sustained braking force; comparison starts from scorer-only true speed, not controller certainty.",
        },
        "by_mode": {},
        "recommendation": [
            "Do not shrink localization, dynamics or map bounds from these correlated old-room residuals.",
            "Prioritize earlier command speed reduction and route clearance consistency; lower candidate speed cannot remove acknowledged momentum or unexpired commands.",
            "Treat2cm/3cm clearance margins as experimental explicit controls, not evidence that map error shrank. Select on frozen tuning routes only and require untouched validation/control outcomes.",
            "Preserve V8 physical settling before the original visible .5s dwell; nominal speed alone underestimated the prior tail.",
        ],
    }
    for mode in sorted({r["mode"] for r in records}):
        selected = [r for r in records if r["mode"] == mode]
        result["by_mode"][str(mode)] = {
            "all": residual_summary(selected),
            "moving": residual_summary([r for r in selected if r["motion"] == "moving"]),
            "stationary_or_slow": residual_summary([r for r in selected if r["motion"] != "moving"]),
            "by_pose_source": {s: residual_summary([r for r in selected if r["pose_source"] == s]) for s in sorted({r["pose_source"] for r in selected})},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(reports), "frames": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
