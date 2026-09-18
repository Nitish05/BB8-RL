"""Independent scoring of completed interactive native logs; never drives a robot.

The fixed arrival gate is 10 cm, 3 cm/s and 0.5 s continuous physics dwell.
Renderer labels and physics records are read only here, after recorded decisions.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(value, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite two-vector")
    return array


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Timestamp must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Timestamp must be finite")
    return value


def _stats(values):
    return {
        "samples": len(values),
        "maximum": max(values) if values else None,
        "p95": float(np.quantile(values, 0.95)) if values else None,
    }


def score_rows(rows, *, tick_seconds=0.005):
    """Pure, strict frame/physics audit with no estimator or simulator imports."""
    errors, arrivals, hidden_times, reacquisitions = [], [], [], []
    previous_end, previous_goal, previous_generation = None, None, None
    dwell, physics_samples, moving_frames, nonzero_frames = 0.0, 0, 0, 0
    collision, boundary, all_hidden_false_accepts = False, False, 0
    predicted_errors, predicted_covered, hidden_nonzero = [], 0, 0
    missing_since, source_changes, last_sources = None, [], None
    last_visible_time = None
    for index, row in enumerate(rows):
        t = _time(row["time"])
        capture = _time(row["capture_time"])
        if abs(t - capture) > 1e-8:
            errors.append({"frame": index, "reason": "capture_time_mismatch"})
        if previous_end is not None and abs(t - previous_end) > 1e-8:
            errors.append({"frame": index, "reason": "physics_capture_gap"})
            dwell = 0.0
        goal = row.get("commanded_goal")
        goal = tuple(_vector(goal, "goal")) if goal is not None else None
        generation = row.get("generation")
        if goal != previous_goal or generation != previous_generation:
            dwell = 0.0
        previous_goal, previous_generation = goal, generation
        action = _vector(row["action"], "action")
        nonzero = np.linalg.norm(action) > 1e-7
        nonzero_frames += int(nonzero)
        if np.linalg.norm(action) > 1 + 1e-6:
            errors.append({"frame": index, "reason": "unnormalized_action"})
        if goal is None and nonzero:
            errors.append({"frame": index, "reason": "motion_without_goal"})
        truth = row["truth_before_scoring_only"]
        xy = _vector(truth["position"], "truth position")
        velocity = _vector(truth["velocity"], "truth velocity")
        speed = float(np.linalg.norm(velocity))
        moving_frames += int(speed > 0.03)
        distance = float(np.linalg.norm(xy - goal)) if goal is not None else None
        if goal is None or distance > 0.10 + 1e-10 or speed > 0.03 + 1e-10:
            dwell = 0.0
        sources = row.get("source_ids", [])
        views = row.get("views", {})
        if len(sources) != len(set(sources)) or not set(sources) <= set(views):
            errors.append({"frame": index, "reason": "invalid_source_ids"})
        mode = row.get("mode")
        if isinstance(mode, bool) or mode not in (1, 2, 3) or len(views) != mode:
            errors.append({"frame": index, "reason": "mode_view_count_mismatch"})
        labels = row.get("head_pixels_scoring_only", {})
        labels_available = (
            set(labels) == set(views)
            and bool(views)
            and all(n is not None for n in labels.values())
        )
        if labels_available and any(
            isinstance(n, bool) or not isinstance(n, int) or n < 0
            for n in labels.values()
        ):
            errors.append({"frame": index, "reason": "malformed_head_count"})
            labels_available = False
        all_hidden = labels_available and all(n == 0 for n in labels.values())
        if all_hidden:
            hidden_times.append(t)
            hidden_nonzero += int(nonzero)
        controller = row.get("controller") or {}
        measurement = row.get("measurement", {})
        rgb_measured = (
            measurement.get("xy") is not None
            and measurement.get("status") == "visible"
            and abs(_time(measurement.get("timestamp", -1)) - capture) <= 1e-8
            and bool(sources)
        )
        measured = (
            rgb_measured
            and controller.get("pose_source") == "measured"
            and controller.get("input_accepted") is True
            and controller.get("state_is_current") is True
        )
        if all_hidden and rgb_measured:
            all_hidden_false_accepts += 1
        visible = (
            measured
            and labels_available
            and any(labels.get(source, 0) >= 3 for source in sources)
        )
        if rgb_measured:
            last_visible_time = t
            if missing_since is not None:
                reacquisitions.append({"time": t, "missing_since": missing_since})
                missing_since = None
            if last_sources is not None and set(sources) != set(last_sources):
                source_changes.append(
                    {"time": t, "before": last_sources, "after": sources}
                )
            last_sources = list(sources)
        elif missing_since is None:
            missing_since = t
        if nonzero and (
            last_visible_time is None or t - last_visible_time > 1.0 + 1e-9
        ):
            errors.append(
                {"frame": index, "reason": "motion_beyond_one_second_rgb_loss"}
            )
        if (
            controller.get("pose_source") == "predicted"
            and controller.get("state_is_current") is True
        ):
            estimate = _vector(controller["xy"], "predicted position")
            error = float(np.linalg.norm(estimate - xy))
            radius = float(controller["position_radius_m"])
            if not math.isfinite(radius) or radius < 0:
                raise ValueError("Malformed uncertainty radius")
            predicted_errors.append(error)
            predicted_covered += int(error <= radius + 1e-9)
        if row.get("controller_arrived"):
            passed = bool(visible and goal is not None and dwell >= 0.5 - 1e-9)
            arrivals.append(
                {
                    "time": t,
                    "goal": goal,
                    "distance_m": distance,
                    "speed_m_s": speed,
                    "true_dwell_at_capture_s": dwell,
                    "visible_fresh_rgb": visible,
                    "passed": passed,
                }
            )
        after = row["after_step"]
        collision |= bool(after.get("collision_scoring_only", False))
        boundary |= bool(after.get("boundary_scoring_only", False))
        end = _time(after["time"])
        sample_time = t
        trace = after.get("physics_trace_scoring_only", [])
        if not trace:
            errors.append({"frame": index, "reason": "missing_physics_trace"})
        for sample in trace:
            stamp = _time(sample["time"])
            dt = stamp - sample_time
            if abs(dt - tick_seconds) > 1e-8 or stamp > end + 1e-8:
                errors.append({"frame": index, "reason": "noncontiguous_physics_tick"})
                dwell = 0.0
            sample_xy = _vector(sample["position"], "physics position")
            sample_velocity = _vector(sample["velocity"], "physics velocity")
            near = goal is not None and np.linalg.norm(sample_xy - goal) <= 0.10 + 1e-10
            slow = np.linalg.norm(sample_velocity) <= 0.03 + 1e-10
            dwell = dwell + max(0, dt) if near and slow else 0.0
            sample_time = stamp
            physics_samples += 1
        if abs(sample_time - end) > 1e-8 or end <= t:
            errors.append({"frame": index, "reason": "incomplete_physics_trace"})
            dwell = 0.0
        previous_end = end
    failed_arrivals = sum(not item["passed"] for item in arrivals)
    return {
        "frames": len(rows),
        "physics_samples": physics_samples,
        "errors": errors,
        "collision": collision,
        "boundary": boundary,
        "nonzero_requested_frames": nonzero_frames,
        "moving_frames": moving_frames,
        "arrival_events": arrivals,
        "valid_arrivals": sum(a["passed"] for a in arrivals),
        "premature_or_unobservable_arrivals": failed_arrivals,
        "all_views_native_hidden_captures": len(hidden_times),
        "all_views_native_hidden_nonzero_requested_frames": hidden_nonzero,
        "all_views_native_hidden_times": hidden_times,
        "all_views_native_hidden_false_accepts": all_hidden_false_accepts,
        "reacquisitions": reacquisitions,
        "accepted_source_changes": source_changes,
        "predicted_error_m": _stats(predicted_errors),
        "predicted_radius_coverage": {
            "covered": predicted_covered,
            "samples": len(predicted_errors),
        },
        "integrity_pass": bool(rows)
        and not errors
        and not collision
        and not boundary
        and failed_arrivals == 0
        and all_hidden_false_accepts == 0,
        "scope": "Recorded synthetic development; visibility sampled at capture times, uncertainty coverage descriptive only.",
    }


def _inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact path escapes run directory")
    return path


def verify_images(directory, rows):
    checks, unavailable = [], 0
    for index, row in enumerate(rows):
        saved = row.get("frames", row.get("images", {}))
        if not saved:
            unavailable += 1
            continue
        if set(saved) != set(row.get("views", {})):
            checks.append({"frame": index, "kind": "camera_set", "pass": False})
        for camera, record in saved.items():
            for kind in ("rgb", "head_mask"):
                if f"{kind}_path" not in record:
                    unavailable += 1
                    continue
                path = _inside(directory, record[f"{kind}_path"])
                valid = path.is_file() and digest(path) == record[f"{kind}_sha256"]
                if valid and kind == "head_mask":
                    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    valid = mask is not None and int(np.count_nonzero(mask)) == row[
                        "head_pixels_scoring_only"
                    ].get(camera)
                checks.append(
                    {
                        "frame": index,
                        "camera": camera,
                        "kind": kind,
                        "pass": bool(valid),
                    }
                )
    return {
        "verified_artifacts": len(checks),
        "unavailable_frames": unavailable,
        "failures": [c for c in checks if not c["pass"]],
    }


def verify_manifest(directory, manifest):
    checks = {
        "complete": manifest.get("status") == "complete",
        "truth_excluded_declaration": manifest.get("truth_to_controller") is False,
        "segmentation_excluded_declaration": manifest.get("segmentation_to_controller")
        is False,
        "source_unchanged": manifest.get("source_files_unchanged") is True,
    }
    for name, expected in manifest.get("source_sha256", {}).items():
        snapshot = _inside(directory, "source/" + Path(name).name)
        checks[f"source:{name}"] = snapshot.is_file() and digest(snapshot) == expected
    checks["source_snapshots_present"] = bool(manifest.get("source_sha256"))
    for name, item in manifest.get("assets", {}).items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = _inside(directory, str(path))
        checks[f"asset:{name}"] = path.is_file() and digest(path) == item["sha256"]
    if manifest.get("asset_sha256"):
        root = Path(manifest["asset_dir"]).resolve()
        for name, expected in manifest["asset_sha256"].items():
            path = _inside(root, name)
            checks[f"asset:{name}"] = path.is_file() and digest(path) == expected
    checks["asset_hashes_present"] = bool(
        manifest.get("assets") or manifest.get("asset_sha256")
    )
    # These statements are provenance assertions, not mathematical proof of isolation.
    # Source review and saved snapshots are necessary to assess their implementation.
    return checks


def score_commands(rows, commands):
    """Accepted stop/rejected goal must cancel motion until a new accepted goal.

    Events are applied at the logged simulation boundary, before that decision.
    Supervisor resets/mode replacements span separate worker files and require a
    separate suite-level check; this function does not infer them from file names.
    """
    errors, stops = [], []
    accepted_goals = [
        c
        for c in commands
        if c.get("action") in ("goal", "demo")
        and c.get("outcome") == "accepted_pending_visible_route"
    ]
    for command in commands:
        cancellation = (
            command.get("action") == "stop" and command.get("outcome") == "accepted"
        ) or (
            command.get("action") in ("goal", "demo")
            and command.get("outcome") == "rejected"
        )
        if not cancellation:
            continue
        start = _time(command["sim_time"])
        generation = command["generation"]
        end = min(
            (
                float(c["sim_time"])
                for c in accepted_goals
                if c["generation"] > generation and c["sim_time"] >= start
            ),
            default=math.inf,
        )
        selected = [r for r in rows if start - 1e-9 <= r["time"] < end - 1e-9]
        bad = [
            r["time"]
            for r in selected
            if r.get("commanded_goal") is not None
            or np.linalg.norm(_vector(r["action"], "action")) > 1e-7
        ]
        before = [r for r in rows if r["time"] < start - 1e-9]
        moving_before = any(
            np.linalg.norm(
                _vector(r["truth_before_scoring_only"]["velocity"], "velocity")
            )
            > 0.03
            for r in before
        )
        final_speed = (
            float(
                np.linalg.norm(
                    _vector(
                        selected[-1]["after_step"]["physics_trace_scoring_only"][-1][
                            "velocity"
                        ],
                        "velocity",
                    )
                )
            )
            if selected and selected[-1]["after_step"]["physics_trace_scoring_only"]
            else None
        )
        if bad:
            errors.append(
                {
                    "action": command["action"],
                    "time": start,
                    "resumed_or_uncancelled_times": bad,
                }
            )
        stops.append(
            {
                "action": command["action"],
                "time": start,
                "recorded_after_frames": len(selected),
                "motion_preceded_event": moving_before,
                "zero_commands_and_cancelled_goal": bool(selected) and not bad,
                "final_true_speed_m_s": final_speed,
            }
        )
    return {
        "commands": len(commands),
        "cancellations": stops,
        "errors": errors,
        "pass": not errors,
        "limit": "A zero-command cancellation is not demonstrated braking unless preceded by motion and followed to physical stop.",
    }


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh output; failed trials must remain intact")
    directory = args.run.resolve()
    rows_path = directory / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = score_rows(rows)
    spec = importlib.util.spec_from_file_location(
        "occluded_audit", Path(__file__).with_name("audit-occluded-control.py")
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    result["command_interval_audit"] = legacy.audit_command_intervals(
        rows, required=True
    )
    result["provenance"] = verify_manifest(directory, manifest)
    result["images"] = verify_images(directory, rows)
    result["images"]["required"] = manifest.get("config", {}).get("save_frames") is True
    commands_path = directory / "commands.jsonl"
    commands = (
        [json.loads(line) for line in commands_path.read_text().splitlines()]
        if commands_path.exists()
        else []
    )
    result["commands"] = score_commands(rows, commands)
    result["commands_sha256"] = (
        digest(commands_path) if commands_path.exists() else None
    )
    result.update(
        schema="bb8.interactive-audit.v1",
        run=str(directory),
        rows_sha256=digest(rows_path),
        manifest_sha256=digest(manifest_path),
        scorer_source_sha256=digest(Path(__file__)),
    )
    result["audit_pass"] = (
        result["integrity_pass"]
        and all(result["provenance"].values())
        and result["command_interval_audit"]["pass"]
        and not result["images"]["failures"]
        and (
            not result["images"]["required"]
            or result["images"]["unavailable_frames"] == 0
        )
        and result["commands"]["pass"]
    )
    args.output.mkdir(parents=True)
    (args.output / "scorer-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "audit_pass",
                    "frames",
                    "valid_arrivals",
                    "premature_or_unobservable_arrivals",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
