"""Independent tuning ranking and frozen-selection validation of native runs.

Uses the original interactive audit for arrival and artifact checks, rechecks
its input hashes, and separately verifies benchmark/control intent. Never runs
physics or changes the controller. Validation outcomes cannot select a profile.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDS = {"route_not_certified", "stopping_envelope_not_certified", "proactive_envelope_not_certified"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def portable(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return "/".join(Path(path).parts[-2:])


def norm(vector):
    value = math.hypot(*vector)
    if not math.isfinite(value):
        raise ValueError("Non-finite vector in evidence")
    return value


def nonzero(row):
    return norm(row["action"]) > 1e-8


def acknowledged_motion_before(rows, time):
    recent = [r for r in rows if time - .25 - 1e-7 <= r["time"] < time - 1e-7]
    moving = any(norm(r["truth_before_scoring_only"]["velocity"]) > .03 for r in recent)
    commanded = any(
        norm(c["command"]) > .05
        for r in recent for c in r["after_step"]["acknowledged_command_intervals"]
    )
    return {"recent_physical_motion": moving, "recent_nonzero_acknowledged_target": commanded,
            "pass": moving and commanded}


def blind_motion_violations(rows):
    """A newly visible but rejected observation cannot refresh last acceptance."""
    last_seen, failures = None, []
    for row in rows:
        c = row.get("controller") or {}
        m = row["measurement"]
        if (c.get("pose_source") == "measured" and c.get("input_accepted")
                and c.get("state_is_current") and m["status"] == "visible"
                and m["xy"] is not None and row["source_ids"]
                and abs(m["timestamp"] - row["time"]) <= 1e-7):
            last_seen = row["time"]
        if nonzero(row) and (last_seen is None or row["time"] - last_seen > 1.0 + 1e-7):
            failures.append(row["time"])
    return failures


def control_checks(case, rows, commands, audit):
    """Control success requires motion before the fixed event and full follow-up."""
    result = {}
    if not rows:
        return {"pass": False, "reason": "no_rows"}
    intent = case.get("intent", "arrival")
    if intent in ("stop", "invalid"):
        if intent == "stop":
            events = [c for c in commands if c.get("action") == "stop" and c.get("outcome") == "accepted"]
            event_time = events[0]["sim_time"] if events else None
        else:
            event_time = case["invalidate_memory_at"]
        if event_time is None:
            return {"pass": False, "reason": "accepted_stop_missing"}
        after = [r for r in rows if r["time"] >= event_time - 1e-7]
        result = {
            "event_time_s": event_time,
            "motion_before_event": acknowledged_motion_before(rows, event_time),
            "after_event_frames": len(after),
            "all_postevent_zero_and_goal_cleared": bool(after) and all(not nonzero(r) and r["commanded_goal"] is None for r in after),
            "final_true_speed_m_s": norm(rows[-1]["after_step"]["velocity_scoring_only"]),
            "complete_observation_window": rows[-1]["after_step"]["time"] >= case["duration_s"] - 1e-7,
        }
        result["pass"] = (
            result["motion_before_event"]["pass"] and result["all_postevent_zero_and_goal_cleared"]
            and result["final_true_speed_m_s"] <= .03 and result["complete_observation_window"]
        )
    elif case.get("camera_dropouts"):
        windows = list(case["camera_dropouts"].values())
        if not windows or any(w != windows[0] for w in windows) or len(windows[0]) != 1:
            raise ValueError("Control reporter requires one common all-camera loss window")
        start, end = windows[0][0]
        missing = [r for r in rows if start - 1e-7 <= r["time"] < end - 1e-7]
        expired = [r for r in rows if start + 1.0 - 1e-7 <= r["time"] < end - 1e-7]
        result = {
            "loss_window_s": [start, end], "motion_before_event": acknowledged_motion_before(rows, start),
            "loss_captures": len(missing), "expired_loss_captures": len(expired),
            "all_loss_frames_missing": bool(missing) and all(r["measurement"]["status"] != "visible" and r["measurement"]["xy"] is None and not r["source_ids"] for r in missing),
            "all_expired_loss_requests_zero": bool(expired) and all(not nonzero(r) for r in expired),
            "all_cameras_omitted": set(case["camera_dropouts"]) == set(rows[0]["views"]),
            "complete_loss_window": len(missing) >= round((end - start) / .05),
            "recovery_valid_arrival": any(e["passed"] and e["time"] >= end for e in audit["arrival_events"]),
            "observed_through_minimum_7s": rows[-1]["after_step"]["time"] >= max(7., end + 2.) - 1e-7,
            "last_expired_loss_true_speed_m_s": norm(expired[-1]["after_step"]["velocity_scoring_only"]) if expired else None,
        }
        result["pass"] = result["motion_before_event"]["pass"] and all(result[key] for key in (
            "all_loss_frames_missing", "all_expired_loss_requests_zero", "all_cameras_omitted",
            "complete_loss_window", "recovery_valid_arrival", "observed_through_minimum_7s",
        )) and result["last_expired_loss_true_speed_m_s"] <= .03
    else:
        result = {"pass": audit["valid_arrivals"] > 0, "kind": "arrival"}
    return result


def inspect_run(path, protocol, validation=False):
    files = {
        "case": path / "case.json", "summary": path / "summary.json",
        "audit": path / "audit/report.json", "manifest": path / "worker/manifest.json",
        "rows": path / "worker/rows.jsonl", "commands": path / "worker/commands.jsonl",
        "scorer": path / "audit/scorer-source.py", "harness": path / "harness.json",
    }
    missing = [name for name, p in files.items() if not p.is_file()]
    metadata = json.loads(files["case"].read_text())
    if missing:
        return {"run": portable(path), "case": metadata["case_id"], "profile": metadata["profile"],
                "hard_failures": ["incomplete_run:" + ",".join(missing)], "intended_pass": False,
                "valid_arrival": False, "recorded_simulation_s": None, "guarded_stop_episodes": 0,
                "input_sha256": {name: digest(p) for name, p in files.items() if p.is_file()}}
    summary, audit, manifest, harness = [json.loads(files[k].read_text()) for k in ("summary", "audit", "manifest", "harness")]
    rows = [json.loads(line) for line in files["rows"].read_text().splitlines()]
    commands = [json.loads(line) for line in files["commands"].read_text().splitlines()]
    hashes = {name: digest(p) for name, p in files.items()}
    failures = []
    case_id, profile = metadata["case_id"], metadata["profile"]
    expected = protocol["cases"].get(case_id)
    if expected is None or metadata["case"] != expected:
        failures.append("case_differs_from_frozen_protocol")
    if summary["case"] != case_id or summary["profile"] != profile["name"]:
        failures.append("summary_identity_mismatch")
    if manifest["config"].get("control_profile") != profile["name"]:
        failures.append("manifest_profile_mismatch")
    for label, actual, recorded in (
        ("report", hashes["audit"], summary["report_sha256"]),
        ("rows", hashes["rows"], audit["rows_sha256"]),
        ("manifest", hashes["manifest"], audit["manifest_sha256"]),
        ("commands", hashes["commands"], audit["commands_sha256"]),
        ("scorer", hashes["scorer"], audit["scorer_source_sha256"]),
    ):
        if actual != recorded:
            failures.append(label + "_hash_mismatch")
    if harness.get("exitcode") != 0 or harness.get("errors"):
        failures.append("harness_failure")
    if not audit["audit_pass"] or not audit["integrity_pass"] or audit.get("errors"):
        failures.append("independent_audit_failure")
    if not all(audit["provenance"].values()):
        failures.append("corrupted_or_incomplete_provenance")
    if not audit["command_interval_audit"]["pass"] or not audit["commands"]["pass"]:
        failures.append("command_provenance_failure")
    if audit["collision"] or any(r["after_step"]["collision_scoring_only"] for r in rows):
        failures.append("collision")
    if audit["boundary"] or any(r["after_step"]["boundary_scoring_only"] for r in rows):
        failures.append("boundary")
    if audit["premature_or_unobservable_arrivals"] or any(not e["passed"] for e in audit["arrival_events"]):
        failures.append("premature_or_unobservable_arrival")
    blind = blind_motion_violations(rows)
    if blind:
        failures.append("motion_beyond_1s_visual_loss_bound")
    if validation and (not audit["images"]["required"] or audit["images"]["unavailable_frames"] or audit["images"]["failures"]):
        failures.append("validation_image_evidence_missing_or_failed")
    checks = control_checks(metadata["case"], rows, commands, audit)
    statuses = [r["controller_status"] for r in rows]
    episodes = sum(s in GUARDS and (i == 0 or statuses[i - 1] not in GUARDS) for i, s in enumerate(statuses))
    valid_arrival = audit["valid_arrivals"] > 0 and not failures
    return {
        "run": portable(path), "case": case_id, "profile": profile, "mode": metadata["case"]["mode"],
        "intent": metadata["case"].get("intent", "arrival"), "hard_failures": failures,
        "audit_pass": audit["audit_pass"], "valid_arrival": valid_arrival,
        "intended_pass": not failures and checks["pass"], "control_checks": checks,
        "recorded_simulation_s": rows[-1]["after_step"]["time"] if rows else 0.,
        "first_valid_arrival": next((e for e in audit["arrival_events"] if e["passed"]), None),
        "guarded_stop_episodes": episodes, "status_counts": dict(Counter(statuses)),
        "blind_motion_violation_times": blind, "frames": len(rows),
        "physics_samples": audit["physics_samples"], "verified_image_artifacts": audit["images"]["verified_artifacts"],
        "unavailable_image_frames": audit["images"]["unavailable_frames"],
        "input_sha256": hashes, "recorded_protocol_sha256": metadata["protocol_sha256"],
        "source_sha256": manifest["source_sha256"], "asset_sha256": manifest["asset_sha256"],
    }


def rank_tuning(runs, expected_cases):
    groups = defaultdict(list)
    for run in runs:
        groups[run["profile"]["name"]].append(run)
    ranks = []
    for name, cases in groups.items():
        counts = Counter(r["case"] for r in cases)
        failures = [f"{r['case']}:{reason}" for r in cases for reason in r["hard_failures"]]
        if set(counts) != set(expected_cases):
            failures.append("missing_or_unexpected_tuning_cases")
        if any(n != 1 for n in counts.values()):
            failures.append("duplicate_tuning_case_no_best_of_repeats")
        if len({json_digest(r["profile"]) for r in cases}) != 1:
            failures.append("profile_settings_changed_between_cases")
        duration = sum(r["recorded_simulation_s"] or 0. for r in cases)
        ranks.append({
            "profile": name, "settings": cases[0]["profile"], "eligible": not failures,
            "exclusions": failures, "attempted_cases": len(cases), "expected_cases": len(expected_cases),
            "valid_goals": sum(r["valid_arrival"] for r in cases),
            "recorded_simulation_s": duration,
            "guarded_stop_episodes": sum(r["guarded_stop_episodes"] for r in cases),
        })
    ranks.sort(key=lambda r: (not r["eligible"], -r["valid_goals"], r["recorded_simulation_s"], r["guarded_stop_episodes"], r["profile"]))
    return ranks


def discover(directories):
    return sorted({p.parent.resolve() for directory in directories for p in directory.glob("*/case.json")})


def check_frozen_selection(frozen, derived, tuning, validation):
    """Accept the pre-validation orchestration freeze without rewriting it."""
    if frozen["schema"] == "bb8.m711.selection-decision.v1":
        result_records, hash_checks = [], []
        for relative, expected in frozen["tuning_result_sha256"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Frozen result reference escapes repository")
            hash_checks.append(path.is_file() and digest(path) == expected)
            if path.is_file():
                result_records.extend(json.loads(path.read_text()))
        pinned = {(r["profile"], r["case"]): r["report_sha256"] for r in result_records}
        actual = {(r["profile"]["name"], r["case"]): r["input_sha256"].get("audit") for r in tuning}
        ranks_match = len(frozen["ranked_tuning_profiles"]) == len(derived["ranking"])
        for a, b in zip(frozen["ranked_tuning_profiles"], derived["ranking"], strict=False):
            ranks_match &= (
                a["profile"] == b["profile"] and a["eligible"] == b["eligible"]
                and a["valid_goals"] == b["valid_goals"]
                and abs(a["total_simulation_s"] - b["recorded_simulation_s"]) < 1e-7
                and a["guarded_stop_episodes"] == b["guarded_stop_episodes"]
            )
        profile, settings = frozen["selected_profile"], frozen["profile"]
        checks = {
            "frozen_tuning_result_hashes_match": bool(hash_checks) and all(hash_checks),
            "all_tuning_audits_pinned_by_frozen_results": pinned == actual and len(pinned) == len(result_records) == len(tuning),
            "complete_tuning_ranking_matches": bool(ranks_match),
            "selection_was_tuning_only": frozen["validation_inspected"] is False,
        }
    else:
        profile, settings = frozen["profile"], frozen["settings"]
        checks = {
            "tuning_inputs_unchanged": frozen["tuning_inputs_sha256"] == derived["tuning_inputs_sha256"],
            "selection_was_tuning_only": frozen["selected_from"] == "tuning_only",
        }
    checks.update({
        "protocol_unchanged": frozen["protocol_sha256"] == derived["protocol_sha256"],
        "tuning_winner_unchanged": profile == derived["profile"] and settings == derived["settings"],
        "validation_uses_frozen_profile": all(r["profile"] == settings for r in validation),
    })
    return profile, checks


def main(args):
    if args.output.exists():
        raise ValueError("Keep old evidence; choose a fresh report")
    if args.freeze_selection and (args.validation or args.selection):
        raise ValueError("Freeze selection using tuning only, before validation")
    if args.validation and not args.selection:
        raise ValueError("Validation requires a selection previously frozen from tuning")
    protocol = json.loads(args.protocol.read_text())
    tuning = [inspect_run(p, protocol) for p in discover(args.tuning)]
    validation = [inspect_run(p, protocol, True) for p in discover(args.validation)]
    for run in tuning + validation:
        if run.get("recorded_protocol_sha256") != digest(args.protocol):
            run["hard_failures"].append("protocol_hash_mismatch")
            run["valid_arrival"] = run["intended_pass"] = False
    expected = sorted(k for k in protocol["cases"] if k.startswith("tuning-"))
    ranking = rank_tuning(tuning, expected)
    winner = next((r for r in ranking if r["eligible"] and r["valid_goals"]), None)
    tuning_identity = {r["run"]: r["input_sha256"] for r in tuning}
    selection = {
        "schema": "bb8.control-profile-selection.v1", "selected_from": "tuning_only",
        "profile": winner["profile"] if winner else None,
        "settings": winner["settings"] if winner else None,
        "protocol_sha256": digest(args.protocol), "tuning_inputs_sha256": json_digest(tuning_identity),
        "ranking": ranking, "scorer_sha256": digest(__file__),
    }
    selection_checks = {}
    selected_profile = selection["profile"]
    if args.selection:
        frozen = json.loads(args.selection.read_text())
        selected_profile, selection_checks = check_frozen_selection(frozen, selection, tuning, validation)
        selection = frozen
    if args.freeze_selection:
        args.freeze_selection.parent.mkdir(parents=True, exist_ok=True)
        with args.freeze_selection.open("x") as stream:
            stream.write(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    validation_counts = Counter(r["case"] for r in validation)
    expected_validation = sorted(set(protocol["cases"]) - set(expected))
    complete_validation = bool(validation) and set(validation_counts) == set(expected_validation) and all(n == 1 for n in validation_counts.values())
    report = {
        "schema": "bb8.control-benchmark-summary.v1", "scope": "Synthetic development tuning and separately frozen validation; no real-time or unseen-room claim.",
        "selection_rule": "Exclude hard failures/incomplete or duplicate case sets; rank valid goals descending, total recorded simulation time ascending, guard episodes ascending. Never rank validation outcomes.",
        "protocol_sha256": digest(args.protocol), "scorer_sha256": digest(__file__),
        "selection_file_sha256": digest(args.selection) if args.selection else None,
        "tuning": tuning, "ranking": ranking, "selection": selection,
        "selection_checks": selection_checks, "validation": validation,
        "validation_totals": {
            "expected_cases": len(expected_validation), "recorded_cases": len(validation),
            "intended_passes": sum(r["intended_pass"] for r in validation),
            "complete_unique_case_set": complete_validation,
            "all_pass": complete_validation and all(selection_checks.values()) and all(r["intended_pass"] for r in validation),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": portable(args.output), "tuning_runs": len(tuning), "selected_profile": selected_profile, "selection_checks": selection_checks, "validation": report["validation_totals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--freeze-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
