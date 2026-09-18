"""Independent, scoring-only audit of frozen M7.8 native experiments.

Static geometry, renderer labels and physics state are used only after the
controller has emitted its action. This script never modifies an input or emits
a map/controller input. Six correlated synthetic cases are not reliability proof.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.mapping.contracts import Provenance, SpaceState
from bb8_rl.mapping.scan_free_space import ScanFreeMemory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solid_intersections(low, high, boxes):
    """Positive-volume cell/solid overlap, not center or corner sampling.

    Exact face contact is reported separately from a false FREE interior. This
    audit does not license contact: runtime closed capsule queries block it.
    """
    low, high = np.asarray(low, float), np.asarray(high, float)
    if low.shape != high.shape or low.ndim != 2 or low.shape[1] != 3:
        raise ValueError("Expected Nx3 finite cell bounds")
    if not np.isfinite(np.r_[low.ravel(), high.ravel()]).all() or np.any(high <= low):
        raise ValueError("Expected finite positive cell bounds")
    overlap, contact = np.zeros(len(low), bool), np.zeros(len(low), bool)
    for box in boxes:
        center, size = (
            np.asarray(box["position"], float),
            np.asarray(box["size"], float),
        )
        if (
            center.shape != (3,)
            or size.shape != (3,)
            or np.any(size <= 0)
            or not np.isfinite(np.r_[center, size]).all()
            or not np.allclose(box.get("euler", [0, 0, 0]), 0, atol=1e-10, rtol=0)
        ):
            raise ValueError("Scorer requires finite axis-aligned positive boxes")
        widths = np.minimum(high, center + size / 2) - np.maximum(
            low, center - size / 2
        )
        overlap |= (widths > 1e-10).all(axis=1)
        contact |= (widths >= -1e-10).all(axis=1)
    return overlap, contact & ~overlap


def audit_memory(directory, scene):
    memory = ScanFreeMemory.load(directory)
    boxes = [b for b in scene["objects"] if b["format"] == "box" and b.get("fixed")]
    voxels = memory.memory._cells  # Scoring-only inspection, never controller input.
    indices = np.argwhere(voxels == memory.memory._FREE)
    low = np.array(memory.memory.bounds.minimum) + indices * memory.memory.resolution_m
    high = low + memory.memory.resolution_m
    overlap, contact = solid_intersections(low, high, boxes)
    outside = (
        (low[:, :2] < -memory.extent - 1e-9).any(axis=1)
        | (high[:, :2] > memory.extent + 1e-9).any(axis=1)
        | (low[:, 2] < -1e-9)
    )
    y, x = np.indices(memory.free_mask.shape)
    cell_low = np.c_[
        (-memory.extent + x.ravel() * memory.resolution),
        (-memory.extent + y.ravel() * memory.resolution),
        np.full(x.size, memory.floor_z),
    ]
    cell_high = cell_low + [memory.resolution, memory.resolution, memory.body_height_m]
    prism_bad, prism_contact = solid_intersections(cell_low, cell_high, boxes)
    prism_bad = prism_bad.reshape(x.shape)
    counts = np.array(
        [int(value).bit_count() for value in memory.support_bits.ravel()]
    ).reshape(x.shape)
    ids = set(memory.view_ids)
    excluded = {f"scan-{i:03d}" for i in (14, 17, 20, 23)}
    free_sources = [
        v.source for v in memory.memory.evidence if v.state is SpaceState.FREE
    ]
    provenance_checks = {}
    occupied_source = memory.metadata.get("occupied_source")
    if occupied_source:
        original_path = Path(occupied_source)
        provenance_checks["occupied_source_hash"] = original_path.is_file() and digest(
            original_path
        ) == memory.metadata.get("occupied_source_sha256")
        if original_path.is_file():
            original = json.loads(original_path.read_text())
            saved = memory.memory.to_dict()
            provenance_checks["occupied_volumes_preserved"] = all(
                record in saved["volumes"] for record in original["volumes"]
            )
            provenance_checks["occupied_objects_preserved"] = all(
                record in saved["objects"] for record in original["objects"]
            )
    scan_directory = memory.metadata.get("scan_dir")
    if scan_directory:
        for view, hashes in memory.metadata["input_sha256"].items():
            for kind, extension in (("rgb", ".png"), ("calibration", ".json")):
                path = Path(scan_directory) / (view + extension)
                provenance_checks[f"{view}:{kind}"] = (
                    path.is_file() and digest(path) == hashes[kind]
                )
    sources_valid = all(
        s.provenance is Provenance.RGB_RECONSTRUCTION
        and len(s.view_ids) >= 2
        and set(s.view_ids) <= ids
        for s in free_sources
    )
    bad_support = memory.free_mask & (counts < 2)
    bit_overflow = bool(np.any(memory.support_bits.astype(np.uint64) >> len(ids)))
    good = (
        memory.memory.valid
        and not overlap.any()
        and not outside.any()
        and not (prism_bad & memory.free_mask).any()
        and not bad_support.any()
        and not bit_overflow
        and not ids & excluded
        and sources_valid
        and len(ids) == 20
        and bool(memory.free_mask.any())
        and all(provenance_checks.values())
    )
    return {
        "status": "pass" if good else "fail",
        "directory": str(Path(directory).resolve()),
        "manifest_sha256": digest(Path(directory) / "manifest.json"),
        "meaning": "Full positive-volume intersection with static boxes; no oracle correction, no physical/general guarantee.",
        "free_voxels": len(indices),
        "occupied_voxels": int((voxels == 2).sum()),
        "retained_unknown_voxels": int((voxels == 0).sum()),
        "false_free_solid_intersection_voxels": int(overlap.sum()),
        "free_solid_face_contact_voxels": int(contact.sum()),
        "free_voxels_outside_room_or_below_floor": int(outside.sum()),
        "free_prisms": int(memory.free_mask.sum()),
        "candidate_prisms": int(memory.candidate_free_mask.sum()),
        "candidate_solid_intersections": int(
            (prism_bad & memory.candidate_free_mask).sum()
        ),
        "free_solid_intersection_prisms": int((prism_bad & memory.free_mask).sum()),
        "free_solid_face_contact_prisms": int(
            (prism_contact.reshape(x.shape) & memory.free_mask).sum()
        ),
        "retained_unknown_prisms": int(
            (~memory.free_mask & ~memory.occupied_mask).sum()
        ),
        "free_without_two_view_support": int(bad_support.sum()),
        "support_bits_outside_view_ids": bit_overflow,
        "heldout_query_in_map": sorted(ids & excluded),
        "free_sources_rgb_only": sources_valid,
        "input_provenance_checks": provenance_checks,
        "source_rgb_hashes_checked": bool(scan_directory),
        "body_height_m": memory.body_height_m,
        "mapping_input_view_ids": list(memory.view_ids),
        "failing_voxel_indices_first20": indices[overlap | outside][:20].tolist(),
    }


def summary(values):
    a = np.asarray(values, float)
    return {
        "samples": len(a),
        "median": float(np.median(a)) if len(a) else None,
        "p95": float(np.percentile(a, 95)) if len(a) else None,
        "maximum": float(a.max()) if len(a) else None,
    }


def is_current_measurement(row):
    m, c = row["measurement"], row["controller"]
    return (
        m["status"] == "visible"
        and m.get("xy") is not None
        and abs(m["timestamp"] - row["time"]) <= 1e-9
        and c.get("pose_source") == "measured"
        and c.get("last_visual_time") is not None
        and abs(c["last_visual_time"] - row["time"]) <= 1e-9
    )


def hidden_tick_metrics(rows, epsilon=1e-6):
    """Actuation/motion evidence adjacent to native zero-head captures.

    Command snapshots are per physics tick; visibility is observed only at RGB
    capture times. The stricter subset has zero pixels at both interval ends.
    Neither asserts an unobserved intermediate renderer result.
    """

    def collect(indices):
        before_nonzero = after_nonzero = command_samples = ticks = 0
        acknowledged_nonzero = acknowledged_samples = 0
        distance = duration = 0.0
        speeds = []
        for index in indices:
            row = rows[index]
            previous_position = np.asarray(
                row["truth_before_scoring_only"]["position"][:2]
            )
            previous_time = row["time"]
            acknowledged = row["after_step"].get("acknowledged_command_intervals", [])
            for tick_index, sample in enumerate(
                row["after_step"]["physics_trace_scoring_only"]
            ):
                position = np.asarray(sample["position"][:2])
                distance += float(np.linalg.norm(position - previous_position))
                duration += sample["time"] - previous_time
                speeds.append(float(np.linalg.norm(sample["velocity"][:2])))
                previous_position, previous_time = position, sample["time"]
                ticks += 1
                if tick_index < len(acknowledged):
                    acknowledged_samples += 1
                    acknowledged_nonzero += int(
                        np.linalg.norm(acknowledged[tick_index]["command"]) > epsilon
                    )
                if all(
                    k in sample
                    for k in ("command_active_before_tick", "command_active_after_tick")
                ):
                    command_samples += 1
                    before_nonzero += int(
                        np.linalg.norm(sample["command_active_before_tick"]) > epsilon
                    )
                    after_nonzero += int(
                        np.linalg.norm(sample["command_active_after_tick"]) > epsilon
                    )
        return {
            "intervals": len(indices),
            "physics_ticks": ticks,
            "ticks_with_command_snapshots": command_samples,
            "ticks_with_exact_acknowledged_intervals": acknowledged_samples,
            "nonzero_exact_acknowledged_intervals": acknowledged_nonzero,
            "nonzero_active_before_tick_samples": before_nonzero,
            "nonzero_active_after_tick_samples": after_nonzero,
            "physical_path_distance_m": distance,
            "duration_seconds": duration,
            "true_speed_m_s": summary(speeds),
        }

    hidden = [
        i
        for i, row in enumerate(rows)
        if row["head_pixels_scoring_only"] == 0 and not row["dropout_injected"]
    ]
    both_ends = [i for i in hidden if i + 1 in hidden]
    return {
        "after_zero_head_capture": collect(hidden),
        "bounded_by_two_zero_head_captures": collect(both_ends),
        "limit": "20Hz renderer visibility; 5ms motion/command samples. Intermediate visibility is not rendered. Terminal/boundary stop can clear the after-tick command.",
    }


def audit_command_intervals(rows, *, required=False, tick_seconds=0.005):
    """Require a causal, complete, command-only partition; never inspect state."""
    present = any(
        "prior_acknowledged_command_intervals" in row
        or "acknowledged_command_intervals" in row["after_step"]
        for row in rows
    )
    if not required and not present:
        return {
            "required": False,
            "present": False,
            "pass": True,
            "intervals": 0,
            "note": "Legacy endpoint acknowledgements only; exact intervals unavailable.",
        }
    errors, interval_count = [], 0
    previous = []
    for index, row in enumerate(rows):
        prior = row.get("prior_acknowledged_command_intervals")
        intervals = row["after_step"].get("acknowledged_command_intervals")
        if prior != previous:
            errors.append(
                {"frame": index, "reason": "prior_not_previous_completed_partition"}
            )
        if index and (
            not isinstance(previous, list)
            or not previous
            or not isinstance(previous[-1], dict)
            or not isinstance(previous[-1].get("end"), (float, int))
            or abs(previous[-1]["end"] - row["time"]) > 1e-8
        ):
            errors.append(
                {"frame": index, "reason": "prior_partition_does_not_end_at_capture"}
            )
        if not isinstance(intervals, list) or not intervals:
            errors.append({"frame": index, "reason": "missing_completed_partition"})
            previous = intervals
            continue
        expected_start = row["time"]
        trace = row["after_step"]["physics_trace_scoring_only"]
        if len(intervals) != len(trace):
            errors.append({"frame": index, "reason": "interval_physics_count"})
        for tick_index, interval in enumerate(intervals):
            if not isinstance(interval, dict) or set(interval) != {
                "start",
                "end",
                "command",
            }:
                errors.append(
                    {
                        "frame": index,
                        "tick": tick_index,
                        "reason": "non_command_fields_or_schema",
                    }
                )
                continue
            try:
                a, b = float(interval["start"]), float(interval["end"])
                command = np.asarray(interval["command"], float)
                valid = (
                    np.isfinite([a, b]).all()
                    and command.shape == (2,)
                    and np.isfinite(command).all()
                    and np.linalg.norm(command) <= 1 + 1e-6
                    and abs(a - expected_start) <= 1e-8
                    and abs(b - a - tick_seconds) <= 1e-8
                    and b <= row["after_step"]["time"] + 1e-8
                    and tick_index < len(trace)
                    and abs(b - trace[tick_index]["time"]) <= 1e-8
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(
                    {
                        "frame": index,
                        "tick": tick_index,
                        "reason": "invalid_time_or_normalization",
                    }
                )
            expected_start = interval["end"]
            interval_count += 1
        if (
            not isinstance(expected_start, (float, int))
            or abs(expected_start - row["after_step"]["time"]) > 1e-8
        ):
            errors.append({"frame": index, "reason": "incomplete_partition_end"})
        previous = intervals
    if required and not rows:
        errors.append({"reason": "missing_frames"})
    return {
        "required": required,
        "present": present,
        "pass": not errors,
        "tick_seconds": tick_seconds,
        "intervals": interval_count,
        "errors": errors,
        "allowed_interval_fields": ["start", "end", "command"],
        "causality": "Next capture receives exactly the previous completed partition; no physics state fields.",
    }


def score_episode(rows, case, protocol, max_blind_seconds=1.0):
    """Pure audit, useful for regression tests as well as saved native logs."""
    if not rows:
        return {
            "id": case["id"],
            "status": "missing",
            "intent_pass": False,
            "frames": 0,
        }
    arrival = protocol["arrival"]
    goal = np.asarray(case["goal"])
    epsilon = protocol["nonzero_action_epsilon"]
    dwell, last_time = 0.0, float(rows[0]["time"])
    errors, hidden_errors, coverage, hidden_coverage = [], [], [], []
    geometric_errors, geometric_coverage = [], []
    sync_errors, dwell_discrepancies, collision, boundary = [], [], False, False
    geometric, hidden_nonzero, hidden_applied, hidden_false_observed = [], [], [], []
    injected_nonzero, expired, invalid, reacquisitions, arrival_events = (
        [],
        [],
        [],
        [],
        [],
    )
    missing_since, hidden_since, previous = None, None, None
    physics_count = 0
    for index, row in enumerate(rows):
        t, c = float(row["time"]), row["controller"]
        if (
            row["step"] != index
            or abs(row["capture_time"] - t) > 1e-9
            or abs(last_time - t) > 1e-7
        ):
            sync_errors.append(index)
        action = np.asarray(row["action"], float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Malformed action")
        nonzero = np.linalg.norm(action) > epsilon
        measured = is_current_measurement(row)
        injected = bool(row["dropout_injected"])
        hidden = row["head_pixels_scoring_only"] == 0
        expected_dropout = any(
            a - 1e-9 <= t < b - 1e-9
            for a, b in case.get("dropout_intervals_seconds", [])
        )
        # The native runner intentionally uses direct float comparisons at interval edges.
        if injected != expected_dropout and not any(
            abs(t - v) < 1e-7
            for interval in case.get("dropout_intervals_seconds", [])
            for v in interval
        ):
            sync_errors.append(index)
        if hidden and not injected:
            geometric.append(index)
            if nonzero:
                hidden_nonzero.append(index)
            if np.linalg.norm(row["after_step"]["applied_command"]) > epsilon:
                hidden_applied.append(index)
            if measured:
                hidden_false_observed.append(index)
            if hidden_since is None:
                hidden_since = t
        if injected and nonzero:
            injected_nonzero.append(index)
        if not measured and missing_since is None:
            missing_since = t
        if measured and missing_since is not None:
            residual = c.get("innovation_m")
            reacquisitions.append(
                {
                    "time": t,
                    "missing_started": missing_since,
                    "missing_duration_seconds": t - missing_since,
                    "after_geometric_hidden": hidden_since is not None,
                    "innovation_m": residual,
                }
            )
            missing_since = None
            hidden_since = None
        truth = row["truth_before_scoring_only"]
        position, velocity = (
            np.asarray(truth["position"]),
            np.asarray(truth["velocity"]),
        )
        if not np.isfinite(np.r_[position, velocity]).all():
            raise ValueError("Nonfinite truth")
        if c.get("xy") is not None and c.get("state_is_current", True):
            error = float(np.linalg.norm(np.asarray(c["xy"]) - position[:2]))
            errors.append(error)
            radius = c.get("position_radius_m")
            if radius is not None:
                coverage.append(error <= radius + 1e-9)
            if c.get("pose_source") == "predicted":
                hidden_errors.append(error)
                if radius is not None:
                    hidden_coverage.append(error <= radius + 1e-9)
                if hidden and not injected:
                    geometric_errors.append(error)
                    if radius is not None:
                        geometric_coverage.append(error <= radius + 1e-9)
        if (
            not measured
            and c.get("last_visual_time") is not None
            and t - c["last_visual_time"] > max_blind_seconds + 1e-9
        ):
            expired.append(
                {
                    "time": t,
                    "zero_command": not nonzero,
                    "speed_m_s": float(np.linalg.norm(velocity[:2])),
                }
            )
        change = case.get("invalidate_memory_at_seconds")
        if change is not None and t >= change - 1e-9:
            invalid.append(
                {
                    "time": t,
                    "zero_command": not nonzero,
                    "speed_m_s": float(np.linalg.norm(velocity[:2])),
                }
            )
        before_dwell = dwell
        after = row["after_step"]
        collision |= bool(after["collision_scoring_only"])
        boundary |= bool(after["boundary_scoring_only"])
        trace = after["physics_trace_scoring_only"]
        if not trace:
            sync_errors.append(index)
        for sample in trace:
            dt = sample["time"] - last_time
            if not 0 < dt <= protocol["action_period_seconds"] + 1e-8:
                sync_errors.append(index)
            delta = np.linalg.norm(np.asarray(sample["position"])[:2] - goal)
            speed = np.linalg.norm(np.asarray(sample["velocity"])[:2])
            dwell = (
                dwell + dt
                if delta <= arrival["position_tolerance_m"]
                and speed <= arrival["speed_tolerance_m_s"]
                else 0.0
            )
            last_time = sample["time"]
            physics_count += 1
        if after["collision_scoring_only"] or after["boundary_scoring_only"]:
            dwell = 0.0
        if abs(last_time - after["time"]) > 1e-8:
            sync_errors.append(index)
        if abs(dwell - after["dwell_seconds_scoring_only"]) > 1e-7:
            dwell_discrepancies.append(index)
        independent_success = dwell + 1e-9 >= arrival["dwell_seconds"]
        if bool(after["success_scoring_only"]) != independent_success:
            dwell_discrepancies.append(index)
        if row["controller_arrived"]:
            visible = (
                measured
                and not injected
                and row["head_pixels_scoring_only"]
                >= arrival["visible_head_min_pixels"]
            )
            arrival_events.append(
                {
                    "time": t,
                    "visible_fresh_rgb": bool(visible),
                    "distance_m": float(np.linalg.norm(position[:2] - goal)),
                    "speed_m_s": float(np.linalg.norm(velocity[:2])),
                    "independent_dwell_at_capture_seconds": before_dwell,
                    "passed": bool(
                        visible and before_dwell + 1e-9 >= arrival["dwell_seconds"]
                    ),
                }
            )
        previous = row
    integrity = not sync_errors and not dwell_discrepancies
    safe = not collision and not boundary and not hidden_false_observed
    arrived = any(e["passed"] for e in arrival_events)
    reacquired_geometric = any(e["after_geometric_hidden"] for e in reacquisitions)
    minimum_end = case.get("minimum_end_time_seconds", 0)
    complete_window = previous["after_step"]["time"] + 1e-8 >= minimum_end
    kind = case["kind"]
    event_times = [a for a, _ in case.get("dropout_intervals_seconds", [])]
    if case.get("invalidate_memory_at_seconds") is not None:
        event_times.append(case["invalidate_memory_at_seconds"])
    trigger = min(event_times) if event_times else None
    before_event = [
        r for r in rows if trigger is not None and r["time"] < trigger - 1e-9
    ]
    moving_before_event = [
        r
        for r in before_event
        if np.linalg.norm(r["truth_before_scoring_only"]["velocity"][:2])
        > arrival["speed_tolerance_m_s"]
    ]
    commands_before_event = [
        r for r in before_event if np.linalg.norm(r["action"]) > epsilon
    ]
    motion_before_event = bool(moving_before_event and commands_before_event)
    if kind == "geometric_occlusion":
        intent = bool(
            hidden_nonzero and hidden_applied and reacquired_geometric and arrived
        )
    elif kind == "injected_dropout":
        end = max(b for _, b in case["dropout_intervals_seconds"])
        intent = bool(
            injected_nonzero
            and any(e["time"] >= end - 1e-8 for e in reacquisitions)
            and arrived
        )
    elif kind == "long_dropout":
        end = max(b for _, b in case["dropout_intervals_seconds"])
        intent = bool(
            expired
            and motion_before_event
            and all(e["zero_command"] for e in expired)
            and expired[-1]["speed_m_s"] <= arrival["speed_tolerance_m_s"]
            and any(e["time"] >= end - 1e-8 for e in reacquisitions)
            and complete_window
        )
    elif kind == "invalid_memory":
        intent = bool(
            invalid
            and motion_before_event
            and all(e["zero_command"] for e in invalid)
            and invalid[-1]["speed_m_s"] <= arrival["speed_tolerance_m_s"]
            and complete_window
        )
    else:
        intent = arrived
    return {
        "id": case["id"],
        "kind": kind,
        "status": "complete",
        "frames": len(rows),
        "nonzero_requested_frames": sum(
            int(np.linalg.norm(r["action"]) > epsilon) for r in rows
        ),
        "motion_before_test_event": motion_before_event,
        "before_event_moving_frames": len(moving_before_event),
        "before_event_nonzero_requested_frames": len(commands_before_event),
        "intent_pass": bool(intent and safe and integrity),
        "collision": collision,
        "boundary": boundary,
        "geometry_hidden_frames": len(geometric),
        "geometry_hidden_nonzero_requested_frames": len(hidden_nonzero),
        "geometry_hidden_nonzero_applied_frames": len(hidden_applied),
        "geometric_hidden_physics": hidden_tick_metrics(rows, epsilon),
        "geometry_hidden_falsely_measured_frames": len(hidden_false_observed),
        "geometry_hidden_first_time": rows[geometric[0]]["time"] if geometric else None,
        "injected_dropout_nonzero_requested_frames": len(injected_nonzero),
        "visible_arrival": arrived,
        "arrival_events": arrival_events,
        "reacquisitions": reacquisitions,
        "position_error_m": summary(errors),
        "predicted_position_error_m": summary(hidden_errors),
        "geometric_hidden_predicted_position_error_m": summary(geometric_errors),
        "position_radius_coverage": {
            "covered": sum(coverage),
            "samples": len(coverage),
        },
        "predicted_position_radius_coverage": {
            "covered": sum(hidden_coverage),
            "samples": len(hidden_coverage),
        },
        "geometric_hidden_predicted_position_radius_coverage": {
            "covered": sum(geometric_coverage),
            "samples": len(geometric_coverage),
        },
        "expired_blind_frames": len(expired),
        "expired_blind_nonzero_commands": sum(not e["zero_command"] for e in expired),
        "last_expired_blind_speed_m_s": expired[-1]["speed_m_s"] if expired else None,
        "invalid_memory_frames": len(invalid),
        "invalid_memory_nonzero_commands": sum(not e["zero_command"] for e in invalid),
        "final_invalid_memory_speed_m_s": invalid[-1]["speed_m_s"] if invalid else None,
        "required_recording_window_complete": complete_window,
        "physics_trace_samples": physics_count,
        "timestamp_errors": sorted(set(sync_errors)),
        "independent_dwell_discrepancies": sorted(set(dwell_discrepancies)),
        "end_time": previous["after_step"]["time"],
        "hidden_evidence_limit": "Zero head pixels at capture instants, not continuous visibility between frames; applied commands are logged after the following step.",
    }


def verify_images(directory, rows):
    failures = []
    for i, row in enumerate(rows):
        for kind in ("rgb", "head_mask"):
            path = (directory / row[f"{kind}_path"]).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or not path.is_file()
                or digest(path) != row[f"{kind}_sha256"]
            ):
                failures.append({"frame": i, "kind": kind, "reason": "path/hash"})
        mask = cv2.imread(str(directory / row["head_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if (
            mask is None
            or int(np.count_nonzero(mask)) != row["head_pixels_scoring_only"]
        ):
            failures.append({"frame": i, "kind": "head_mask", "reason": "pixel_count"})
    return {"frames": len(rows), "artifacts": 2 * len(rows), "failures": failures}


def verify_native_fixture(directory, scene):
    """Compare saved reset-time authored obstacles; walls are in scene snapshot."""
    path = directory / "native-static-obstacles.json"
    drive = directory / "native-drive-parameters.json"
    if not path.is_file() or not drive.is_file():
        return {"pass": False, "reason": "missing_native_fixture_snapshot"}
    native = json.loads(path.read_text())
    expected = {
        b["name"]: b
        for b in scene["objects"]
        if b["format"] == "box"
        and b.get("fixed")
        and b["name"].startswith("room_obstacle_")
    }
    good = (
        len(native["names"]) == len(native["positions"]) == len(native["sizes"])
        and len(native["names"]) == len(set(native["names"]))
        and set(native["names"]) == set(expected)
    )
    if good:
        for name, position, size in zip(
            native["names"], native["positions"], native["sizes"], strict=True
        ):
            good &= bool(
                np.allclose(
                    position, expected[name]["position"][:2], atol=1e-10, rtol=0
                )
                and np.allclose(size, expected[name]["size"], atol=1e-10, rtol=0)
            )
    return {
        "pass": bool(good),
        "static_obstacles_sha256": digest(path),
        "drive_parameters_sha256": digest(drive),
        "drive_parameters": json.loads(drive.read_text()),
        "limit": "Saved authored reset configuration, supported by separate original-scene RGB replay; not live per-body pose telemetry.",
    }


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh audit output; never overwrite an earlier result")
    protocol = json.loads(args.protocol.read_text())
    if digest(protocol["scene"]) != protocol["scene_sha256"]:
        raise ValueError("Frozen scoring scene changed")
    scene = json.loads(Path(protocol["scene"]).read_text())
    report = {
        "schema": "bb8.occluded-control-score.v1",
        "scorer_source_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(args.protocol),
        "denominator": len(protocol["cases"]),
        "memory": audit_memory(args.memory, scene),
        "scope": "Scoring-only bounded synthetic development. No truth is used to create inputs or repair the map.",
        "uncertainty_limit": "Envelope coverage is descriptive under assumed dynamics; six correlated cases do not calibrate confidence.",
        "runs": [],
        "cases": [],
    }
    found = {}
    for directory in args.episodes:
        manifest = json.loads((directory / "manifest.json").read_text())
        checks = {
            "protocol_hash": manifest["protocol_sha256"] == report["protocol_sha256"],
            "static_scene_hash": manifest["static_scene_sha256"]
            == protocol["scene_sha256"],
            "registration_hash": manifest["registration_sha256"]
            == protocol["registration_report_sha256"],
            "map_hash": manifest["map_version"] == report["memory"]["manifest_sha256"],
            "complete": manifest["status"] == "complete",
            "source_unchanged": manifest.get("source_files_unchanged") is True,
            "policy_unchanged": manifest.get("policy_unchanged") is True,
            "vision_unchanged": manifest.get("vision_unchanged") is True,
            "truth_excluded": manifest.get("truth_to_controller") is False,
            "segmentation_excluded": manifest.get("segmentation_to_controller")
            is False,
            "single_protocol_camera": manifest.get("camera_count") == 1
            and manifest.get("camera_query_index")
            == int(protocol["camera_view_id"].removeprefix("scan-")),
        }
        saved_calibration = json.loads((directory / "calibration.json").read_text())
        registration = json.loads(Path(protocol["registration_report"]).read_text())
        expected_registration = next(
            r
            for r in registration["queries"]
            if r["query_index"] == manifest["camera_query_index"]
        )
        checks["estimated_camera_pose_matches_registration"] = bool(
            np.allclose(
                saved_calibration["estimated_world_to_camera"],
                expected_registration["world_to_camera"],
                rtol=0,
                atol=1e-12,
            )
        )
        for name, expected in manifest["source_sha256"].items():
            path = directory / "source" / Path(name).name
            checks[f"source_snapshot:{name}"] = (
                path.is_file() and digest(path) == expected
            )
        for path_key, hash_key in (
            ("policy_path", "policy_sha256"),
            ("vision_checkpoint_path", "vision_sha256"),
        ):
            path = Path(manifest[path_key]) if manifest.get(path_key) else None
            checks[f"checkpoint_rehash:{path_key}"] = (
                path is not None
                and path.is_file()
                and digest(path) == manifest[hash_key]
            )
        snapshot = directory / manifest.get(
            "static_scene_snapshot", "static-scene.json"
        )
        checks["static_scene_snapshot"] = (
            snapshot.is_file() and digest(snapshot) == protocol["scene_sha256"]
        )
        report["runs"].append(
            {
                "directory": str(directory.resolve()),
                "manifest_sha256": digest(directory / "manifest.json"),
                "checks": checks,
            }
        )
        for case_id in manifest["requested_cases"]:
            if case_id in found:
                raise ValueError(
                    "Duplicate case trials require separate audit; no best-run selection"
                )
            found[case_id] = (directory, manifest, all(checks.values()))
    for case in protocol["cases"]:
        if case["id"] not in found:
            result = score_episode([], case, protocol)
        else:
            directory, manifest, provenance_ok = found[case["id"]]
            target = directory / case["id"]
            rows = (
                [
                    json.loads(line)
                    for line in (target / "rows.jsonl").read_text().splitlines()
                ]
                if (target / "rows.jsonl").exists()
                else []
            )
            result = score_episode(
                rows,
                case,
                protocol,
                manifest["controller_parameters"]["max_occlusion_seconds"],
            )
            result["images"] = verify_images(target, rows)
            result["rows_sha256"] = digest(target / "rows.jsonl") if rows else None
            interval_required = any(
                "complete timestamped acknowledged command intervals" in value
                for value in manifest.get("control_inputs", [])
            )
            result["command_interval_audit"] = audit_command_intervals(
                rows, required=interval_required
            )
            result["native_fixture"] = verify_native_fixture(target, scene)
            result["provenance_pass"] = provenance_ok
            result["intent_pass"] &= (
                provenance_ok
                and result["command_interval_audit"]["pass"]
                and result["native_fixture"]["pass"]
                and not result["images"]["failures"]
                and report["memory"]["status"] == "pass"
            )
        report["cases"].append(result)
    report["intent_pass_count"] = sum(c["intent_pass"] for c in report["cases"])
    args.output.mkdir(parents=True)
    (args.output / "scorer-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "memory": report["memory"]["status"],
                "intent_pass": report["intent_pass_count"],
                "denominator": report["denominator"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
