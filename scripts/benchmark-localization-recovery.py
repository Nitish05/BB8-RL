"""Run one frozen M7.13 native case and independently audit localization recovery.

No simulation imports occur when importing the pure audit helpers. Native cases
run serially under the caller's coordination. Truth is read only after decisions.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import queue
import shutil
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
INVALID_STATES = {"lost", "reacquiring", "uninitialized"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def norm(value):
    if len(value) != 2 or not all(math.isfinite(v) for v in value):
        raise ValueError("Expected a finite two-vector")
    return math.hypot(*value)


def zero(value):
    return norm(value) == 0.0


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def queue_heartbeat(channel, generation):
    """Routine keepalive may retry; never evict a goal/Stop during startup."""
    try:
        channel.put_nowait({"action": "heartbeat", "generation": generation})
        return True
    except queue.Full:
        return False


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), ROOT / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motion_before(rows, when):
    recent = [r for r in rows if when - 0.25 - 1e-8 <= r["time"] < when - 1e-8]
    return (
        bool(recent)
        and any(norm(r["truth_before_scoring_only"]["velocity"]) > 0.03 for r in recent)
        and any(
            norm(c["command"]) > 0.05
            for r in recent
            for c in r["after_step"]["acknowledged_command_intervals"]
        )
    )


def braking_checks(rows):
    """Capture-time containment uses truth only after recorded controller decisions.

    A model violation is reported separately from navigation/lifecycle success.
    It cannot be used to tune these frozen runs or shrink the authoritative bound.
    """
    failures, covered, samples, speed_covered = [], 0, 0, 0
    maximum_excess, maximum_speed_excess = 0.0, 0.0
    anchors, assumptions = set(), set()
    for row in rows:
        record = row.get("localization", {}).get("braking_prediction")
        if not isinstance(record, dict):
            failures.append(
                {"time": row["time"], "reason": "missing_braking_diagnostic"}
            )
            continue
        if (
            record.get("motion_authority") is not False
            or record.get("observed") is not False
        ):
            failures.append(
                {
                    "time": row["time"],
                    "reason": "diagnostic_claims_authority_or_observation",
                }
            )
        if not record.get("valid"):
            continue
        try:
            anchor = record["anchor_xy"]
            radius, limit = record["radius_m"], record["radius_limit_m"]
            upper = record["speed_upper_m_s"]
            if (
                not all(math.isfinite(v) and v >= 0 for v in (radius, limit, upper))
                or radius > limit + 1e-10
                or record["anchor_time"] > row["time"]
            ):
                raise ValueError("Invalid region dimensions/time")
            if not zero(row["action"]) or not zero(row["prior_applied_command"]):
                raise ValueError(
                    "Valid braking region with nonzero issued/acknowledged command"
                )
            if not row["prior_acknowledged_command_intervals"]:
                raise ValueError(
                    "Valid braking region without acknowledged interval evidence"
                )
            if not all(
                zero(c["command"]) for c in row["prior_acknowledged_command_intervals"]
            ):
                raise ValueError("Valid region across nonzero acknowledged interval")
            if not record.get("assumptions"):
                raise ValueError("Missing conditional assumptions")
            assumptions.update(record["assumptions"])
            anchors.add((record["anchor_time"], *anchor))
            truth = row["truth_before_scoring_only"]
            distance = norm([truth["position"][i] - anchor[i] for i in range(2)])
            excess = distance - radius
            speed_excess = norm(truth["velocity"]) - upper
            covered += excess <= 1e-9
            speed_covered += speed_excess <= 1e-9
            samples += 1
            maximum_excess = max(maximum_excess, excess)
            maximum_speed_excess = max(maximum_speed_excess, speed_excess)
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"time": row["time"], "reason": str(error)})
    return {
        "contract_pass": not failures,
        "failures": failures,
        "capture_samples": samples,
        "distinct_anchors": len(anchors),
        "position_covered_samples": covered,
        "speed_covered_samples": speed_covered,
        "position_coverage_pass": samples > 0 and covered == samples,
        "speed_coverage_pass": samples > 0 and speed_covered == samples,
        "maximum_position_excess_m": maximum_excess,
        "maximum_speed_excess_m_s": maximum_speed_excess,
        "tail_assumptions": sorted(assumptions),
        "scope": "Post-decision capture-time scoring; assumption-dependent advisory only. No motion authority.",
    }


def score_localization(rows, commands, case, arrival_audit):
    """Independent lifecycle, command and intent checks; no estimator imports."""
    failures, counts = [], Counter()
    if not rows:
        return {"pass": False, "failures": ["no_rows"]}
    for row in rows:
        loc = row.get("localization") or {}
        status = loc.get("localization_status")
        counts[status or "missing"] += 1
        if status not in INVALID_STATES | {"measured", "predicted"}:
            failures.append(
                {
                    "time": row["time"],
                    "reason": "missing_or_invalid_localization_status",
                }
            )
            continue
        if status in INVALID_STATES:
            if (
                loc.get("localization_valid") is not False
                or loc.get("pose") is not None
                or loc.get("position_radius") is not None
                or not zero(row["action"])
            ):
                failures.append(
                    {
                        "time": row["time"],
                        "reason": "unusable_state_exposes_pose_or_motion",
                    }
                )
            if status in {"lost", "reacquiring"} and (
                row.get("commanded_goal") is not None
                or loc.get("requires_new_goal") is not True
                or row.get("controller_arrived")
            ):
                failures.append(
                    {"time": row["time"], "reason": "lost_goal_not_cancelled"}
                )
        else:
            try:
                norm(loc["pose"])
                radius = loc["position_radius"]
                if (
                    loc.get("localization_valid") is not True
                    or not math.isfinite(radius)
                    or not 0 <= radius <= 0.08 + 1e-9
                ):
                    raise ValueError("Invalid usable pose/radius")
            except (TypeError, ValueError, KeyError):
                failures.append(
                    {
                        "time": row["time"],
                        "reason": "usable_state_invalid_pose_or_radius",
                    }
                )
        if status == "measured" and (
            row["measurement"]["status"] != "visible"
            or not row["source_ids"]
            or abs(row["measurement"]["timestamp"] - row["time"]) > 1e-9
        ):
            failures.append(
                {"time": row["time"], "reason": "measured_without_fresh_rgb"}
            )
    intent = case["intent"]
    minimum = case.get("minimum_end_s", 0.0)
    checks = {
        "minimum_window_complete": rows[-1]["after_step"]["time"] >= minimum - 1e-7
    }
    valid_arrivals = [e for e in arrival_audit["arrival_events"] if e["passed"]]
    if intent in {"stop", "invalid"}:
        if intent == "stop":
            events = [
                c
                for c in commands
                if c.get("action") == "stop" and c.get("outcome") == "accepted"
            ]
            when = events[0]["sim_time"] if events else None
        else:
            when = case["invalidate_memory_at"]
        after = [] if when is None else [r for r in rows if r["time"] >= when - 1e-7]
        checks.update(
            motion_before_event=when is not None and motion_before(rows, when),
            zero_after_event=bool(after)
            and all(zero(r["action"]) and r["commanded_goal"] is None for r in after),
            stopped_physically=norm(rows[-1]["after_step"]["velocity_scoring_only"])
            <= 0.03,
        )
    else:
        checks["valid_visible_arrival"] = bool(valid_arrivals)
    if "camera_dropouts" in case:
        windows = list(case["camera_dropouts"].values())
        start, end = windows[0][0]
        during = [r for r in rows if start - 1e-7 <= r["time"] < end - 1e-7]
        checks.update(
            motion_before_event=motion_before(rows, start),
            full_dropout_observed=len(during) >= round((end - start) / 0.05),
        )
        if intent == "one_camera_loss":
            checks["unaffected_camera_contributes"] = bool(during) and all(
                "A" not in r["source_ids"]
                and "B" in r["source_ids"]
                and r["measurement"]["status"] == "visible"
                for r in during
            )
            checks["no_unnecessary_loss"] = all(
                r["localization"]["localization_status"] not in {"lost", "reacquiring"}
                for r in during
            )
        else:
            checks["all_views_missing"] = bool(during) and all(
                not r["source_ids"]
                and r["measurement"]["xy"] is None
                and r["measurement"]["status"] == "missing"
                for r in during
            )
        if intent == "short_loss":
            checks["short_loss_retains_goal"] = bool(during) and all(
                r["commanded_goal"] is not None
                and r["localization"]["localization_status"]
                not in {"lost", "reacquiring"}
                for r in during
            )
        if intent == "recovery":
            lost = [
                r for r in during if r["localization"]["localization_status"] == "lost"
            ]
            recovered = [
                r
                for r in rows
                if r["time"] >= end - 1e-7
                and r["localization"]["localization_status"] == "measured"
                and r["localization"]["requires_new_goal"]
            ]
            recovery_time = recovered[0]["time"] if recovered else None
            rejected = [
                c
                for c in commands
                if c.get("action") == "goal" and c.get("generation") == 2
            ]
            renewed = [
                c
                for c in commands
                if c.get("action") == "goal"
                and c.get("generation") == 3
                and c.get("outcome") == "accepted_pending_visible_route"
            ]
            renewed_time = renewed[0]["sim_time"] if renewed else None
            hold = [
                r
                for r in rows
                if lost
                and r["time"] >= lost[0]["time"]
                and (renewed_time is None or r["time"] < renewed_time - 1e-7)
            ]
            recovering = [
                r
                for r in rows
                if r["localization"]["localization_status"] == "reacquiring"
            ]
            checks.update(
                actual_loss=bool(lost),
                loss_goal_rejected=len(rejected) == 1
                and rejected[0].get("outcome") == "rejected",
                rejected_goal_in_loss_window=(
                    len(rejected) == 1
                    and start <= rejected[0].get("sim_time", -1) < end
                ),
                multiple_recovery_frames=len(recovering) >= 2,
                stable_recovery=bool(recovered),
                old_goal_never_resumed=bool(hold)
                and all(
                    zero(r["action"])
                    and r["commanded_goal"] is None
                    and r["localization"]["requires_new_goal"]
                    for r in hold
                ),
                explicit_new_generation=len(renewed) == 1,
                recovery_hold_complete=(
                    renewed_time is not None
                    and recovery_time is not None
                    and renewed_time
                    >= max(
                        case["new_goal_not_before"],
                        recovery_time + case["recovery_hold_s"],
                    )
                    - 1e-7
                ),
                arrival_after_explicit_goal=renewed_time is not None
                and any(e["time"] >= renewed_time for e in valid_arrivals),
            )
    braking = braking_checks(rows)
    return {
        "pass": not failures and all(checks.values()) and braking["contract_pass"],
        "failures": failures,
        "checks": checks,
        "status_counts": dict(counts),
        "conditional_braking": braking,
        "conditional_coverage_is_separate": True,
    }


def audit_run(directory, output=None):
    directory = directory.resolve()
    output = output or directory / "localization-audit"
    output.mkdir(parents=True, exist_ok=False)
    audit_dir = output / "original-arrival-audit"
    scorer = load_script("audit-interactive.py")
    scorer.main(SimpleNamespace(run=directory / "worker", output=audit_dir))
    original = json.loads((audit_dir / "report.json").read_text())
    metadata = json.loads((directory / "case.json").read_text())
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    rows = read_rows(directory / "worker/rows.jsonl")
    commands = read_rows(directory / "worker/commands.jsonl")
    harness = json.loads((directory / "harness.json").read_text())
    manifest = json.loads((directory / "worker/manifest.json").read_text())
    loc = score_localization(rows, commands, metadata["case"], original)
    provenance = {
        "protocol_hash": digest(protocol_path) == metadata["protocol_sha256"],
        "case_matches_protocol": protocol["cases"].get(metadata["case_id"])
        == metadata["case"],
        "harness_snapshot_hash": digest(directory / "benchmark-source.py")
        == metadata["harness_sha256"],
        "baseline_bundle_hash": digest(directory / "baseline-bundle.json")
        == protocol["bundle_manifest_sha256"],
        "map_unchanged": digest(directory / "assets/memory/manifest.json")
        == protocol["map_manifest_sha256"],
        "memory_unchanged": digest(directory / "assets/memory/room-memory.json")
        == protocol["room_memory_sha256"],
        "worker_completed": harness["exitcode"] == 0 and not harness["errors"],
        "worker_config_matches_case": manifest.get("config") == metadata["config"],
        "profile_frozen": metadata["profile"]["name"] == protocol["control_profile"]
        and manifest.get("control_profile") == metadata["profile"],
        "mode_frozen": metadata["config"]["mode"] == metadata["case"]["mode"],
        "dropout_and_invalidation_frozen": all(
            metadata["config"].get(key) == metadata["case"].get(key)
            for key in ("camera_dropouts", "invalidate_memory_at")
        ),
        "images_complete": original["images"]["required"]
        and not original["images"]["unavailable_frames"]
        and not original["images"]["failures"],
    }
    baseline_bundle = json.loads((directory / "baseline-bundle.json").read_text())
    provenance["all_frozen_assets_unchanged"] = all(
        name == "demo.json" or digest(directory / "assets" / name) == expected
        for name, expected in baseline_bundle["sha256"].items()
    )
    baseline_demo = json.loads((directory / "baseline-demo.json").read_text())
    demo = json.loads((directory / "assets/demo.json").read_text())
    expected_demo = dict(
        baseline_demo, start=metadata["case"]["start"], goal=metadata["case"]["goal"]
    )
    provenance["only_authored_demo_endpoints_changed"] = (
        digest(directory / "baseline-demo.json")
        == baseline_bundle["sha256"]["demo.json"]
        and demo == expected_demo
    )
    hard_fail = (
        not original["audit_pass"]
        or original["collision"]
        or original["boundary"]
        or original["premature_or_unobservable_arrivals"] > 0
        or not all(provenance.values())
    )
    report = {
        "schema": "bb8.localization-recovery-audit.v1",
        "case": metadata["case_id"],
        "scope": "Frozen M7.13 intent; separate from prior twelve-case family.",
        "intended_pass": not hard_fail and loc["pass"],
        "original_audit_pass": original["audit_pass"],
        "provenance": provenance,
        "valid_arrivals": original["valid_arrivals"],
        "arrival_events": original["arrival_events"],
        "collision": original["collision"],
        "boundary": original["boundary"],
        "premature_arrivals": original["premature_or_unobservable_arrivals"],
        "localization": loc,
        "frames": len(rows),
        "physics_samples": original["physics_samples"],
        "recorded_simulation_s": rows[-1]["after_step"]["time"] if rows else 0,
        "input_sha256": {
            name: digest(directory / name)
            for name in (
                "case.json",
                "protocol.json",
                "harness.json",
                "worker/manifest.json",
                "worker/rows.jsonl",
                "worker/commands.jsonl",
                "benchmark-source.py",
            )
        },
        "original_audit_sha256": digest(audit_dir / "report.json"),
        "scorer_source_sha256": digest(__file__),
    }
    (output / "scorer-source.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "case": report["case"],
                "intended_pass": report["intended_pass"],
                "valid_arrivals": report["valid_arrivals"],
                "report": str(output / "report.json"),
            }
        ),
        flush=True,
    )
    return report


def run(args):
    from bb8_rl.control_profiles import get_profile
    from bb8_rl.demo_assets import validate_assets
    from bb8_rl.interactive_runtime import worker_main

    protocol = json.loads(args.protocol.read_text())
    case = protocol["cases"][args.case]
    for key, name in (
        ("bundle_manifest_sha256", "bundle.json"),
        ("map_manifest_sha256", "memory/manifest.json"),
        ("room_memory_sha256", "memory/room-memory.json"),
    ):
        if digest(args.assets / name) != protocol[key]:
            raise ValueError(f"Assets differ from frozen protocol: {name}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source, target in (
        (Path(__file__), "benchmark-source.py"),
        (args.protocol, "protocol.json"),
        (args.assets / "demo.json", "baseline-demo.json"),
        (args.assets / "bundle.json", "baseline-bundle.json"),
    ):
        (output / target).write_bytes(source.read_bytes())
    assets = output / "assets"
    shutil.copytree(args.assets, assets)
    demo_path = assets / "demo.json"
    demo = json.loads(demo_path.read_text())
    demo.update(start=case["start"], goal=case["goal"])
    write_json(demo_path, demo)
    bundle_path = assets / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["sha256"]["demo.json"] = digest(demo_path)
    write_json(bundle_path, bundle)
    validate_assets(assets)
    profile = get_profile(protocol["control_profile"])
    config = {
        "asset_dir": str(assets),
        "run_dir": str(output / "worker"),
        "mode": case["mode"],
        "max_steps": round(case["duration_s"] / 0.05),
        "control_profile": profile.name,
        "generation": 0,
        "save_frames": args.save_frames,
        "scoring_labels": True,
        **{
            key: case[key]
            for key in ("camera_dropouts", "invalidate_memory_at")
            if key in case
        },
    }
    write_json(
        output / "case.json",
        {
            "case_id": args.case,
            "case": case,
            "config": config,
            "profile": profile.record(),
            "protocol_sha256": digest(args.protocol),
            "harness_sha256": digest(__file__),
        },
    )
    context = multiprocessing.get_context("spawn")
    commands, events, stop = context.Queue(32), context.Queue(16), context.Event()
    process = context.Process(target=worker_main, args=(config, commands, events, stop))
    commands.put(
        {"action": "goal", "generation": 1, "x": case["goal"][0], "y": case["goal"][1]}
    )
    commands.put({"action": "heartbeat", "generation": 1})
    process.start()
    started, last_heartbeat = time.monotonic(), 0.0
    sent_stop = sent_rejected = sent_new = False
    recovery_time, last_state = None, None
    errors, issued = [], []
    try:
        while process.is_alive():
            now = time.monotonic()
            if now - started > 1200:
                errors.append("worker_wall_timeout")
                break
            if now - last_heartbeat >= 0.3:
                queue_heartbeat(commands, generation=3)
                last_heartbeat = now
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("type") == "error":
                errors.append(event)
            if event.get("type") != "state":
                continue
            last_state = state = event["state"]
            t = state.get("sim_time", 0)
            if case["intent"] == "recovery":
                if not sent_rejected and t >= case["rejected_goal_at"]:
                    request = {
                        "action": "goal",
                        "generation": 2,
                        "x": case["goal"][0],
                        "y": case["goal"][1],
                    }
                    commands.put(request, timeout=1)
                    issued.append(
                        dict(
                            request,
                            supervisor_observed_sim_time=t,
                            purpose="goal_during_loss",
                        )
                    )
                    sent_rejected = True
                if (
                    recovery_time is None
                    and state.get("localization_status") == "measured"
                    and state.get("requires_new_goal")
                    and t >= 7.0
                ):
                    recovery_time = t
                if (
                    not sent_new
                    and recovery_time is not None
                    and t
                    >= max(
                        case["new_goal_not_before"],
                        recovery_time + case["recovery_hold_s"],
                    )
                ):
                    request = {
                        "action": "goal",
                        "generation": 3,
                        "x": case["goal"][0],
                        "y": case["goal"][1],
                    }
                    commands.put(request, timeout=1)
                    issued.append(
                        dict(
                            request,
                            supervisor_observed_sim_time=t,
                            purpose="explicit_goal_after_recovery",
                        )
                    )
                    sent_new = True
            if "stop_at" in case and not sent_stop and t >= case["stop_at"]:
                commands.put({"action": "stop", "generation": 2}, timeout=1)
                commands.put(
                    {
                        "action": "goal",
                        "generation": 1,
                        "x": case["goal"][0],
                        "y": case["goal"][1],
                    },
                    timeout=1,
                )
                sent_stop = True
            if (
                state.get("phase") == "arrived"
                and case["intent"] not in {"stop", "invalid"}
                and t >= case.get("minimum_end_s", 0)
                and (case["intent"] != "recovery" or sent_new)
            ):
                stop.set()
    except (queue.Full, KeyboardInterrupt) as error:
        errors.append(type(error).__name__)
    finally:
        stop.set()
        process.join(30)
        if process.is_alive():
            errors.append("worker_cleanup_timeout")
            process.terminate()
            process.join(10)
        for channel in (commands, events):
            channel.close()
        write_json(
            output / "harness.json",
            {
                "exitcode": process.exitcode,
                "errors": errors,
                "last_state": last_state,
                "issued_scheduled_commands": issued,
                "recovery_first_observed_s": recovery_time,
                "wall_seconds": time.monotonic() - started,
            },
        )
    if errors or process.exitcode != 0:
        raise RuntimeError(f"Worker failed; retained evidence in {output}")
    audit_run(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/interactive/localization-recovery-cases.json",
    )
    parser.add_argument("--case")
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--audit-only",
        type=Path,
        help="Rescore a completed case; --output must be a fresh audit directory",
    )
    args = parser.parse_args()
    if args.audit_only:
        audit_run(args.audit_only, args.output)
    elif not args.case or not args.output:
        parser.error("Native execution requires --case and --output")
    else:
        run(args)
