"""Frozen navigation request inventory and serialized native runs with authenticated family assets.

The 24 held-out requests per variant need new independently scanned assets. This
runner validates independent provisioning and refuses the development map. All scoring is post-decision;
importing this file does not import or start a simulator or a trained model.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
import queue
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/interactive/navigation-heldout-v1.json"
FROZEN_SHA256 = "82ccfbe9e65c8a65b1ef454b69317087994a4b6257a56cf4a07af9f6ddd875fe"
DEVELOPMENT_CASES = (
    "short_arrival",
    "difficult_arrival",
    "occupied_rejection",
    "camera_change",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_protocol(path=PROTOCOL):
    if digest(path) != FROZEN_SHA256:
        raise ValueError("Evaluation specification differs from its pre-tuning freeze")
    return json.loads(path.read_text())


def development_case(protocol, name, mode):
    if name not in DEVELOPMENT_CASES or type(mode) is not int or mode not in (1, 2, 3):
        raise ValueError("Unknown development case or camera mode")
    development = protocol["development"]
    goal_name = "difficult_arrival" if name == "camera_change" else name
    case = {
        "id": f"{name}-m{mode}",
        "family": "existing_development_bundle",
        "mode": mode,
        "start": development["start"],
        "goal": development["goals"][goal_name],
        "intent": name,
        # An occupied user request must not become an invalid environment-reset
        # fixture. Initialization uses the original safe destination in that case.
        "reset_goal": development["goals"]["short_arrival"]
        if name == "occupied_rejection"
        else development["goals"][goal_name],
        "duration_s": 8.0
        if name == "camera_change"
        else 5.0
        if name == "occupied_rejection"
        else 30.0,
        "minimum_end_s": 2.0,
    }
    if name == "camera_change":
        case["scope"] = (
            "Startup camera change and latched lockout; does not establish stopping after active motion."
        )
        case["evaluation_camera_changes"] = [
            {
                key: value
                for key, value in development["camera_change"].items()
                if key != "scope"
            }
        ]
    return case


def heldout_case(protocol, name, mode=None):
    if name not in protocol["cases"]:
        raise ValueError("Unknown frozen held-out case")
    case = dict(protocol["cases"][name], id=name)
    if mode is not None and (type(mode) is not int or mode != case["mode"]):
        raise ValueError("Camera mode differs from frozen held-out case")
    family = protocol["families"][case["family"]]
    layout = protocol["layouts"][family["layout"]]
    case["reset_goal"] = layout["arrival_goal"]
    return case


def select_case(protocol, suite, name, mode=None):
    if suite == "heldout":
        return heldout_case(protocol, name, mode)
    if suite == "development":
        return development_case(protocol, name, 1 if mode is None else mode)
    raise ValueError("Unknown evaluation suite")


def configured_demo(baseline, suite, case):
    result = dict(baseline, start=case["start"], goal=case["reset_goal"])
    if suite == "heldout":
        result.update(seed=case["seed"], layout_seed=case["layout_seed"])
    return result


def provisioner():
    spec = importlib.util.spec_from_file_location(
        "heldout_runner_provisioning", ROOT / "scripts/provision-navigation-heldout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_inventory(protocol, suite):
    if suite == "heldout":
        cases = [dict(case, id=name) for name, case in protocol["cases"].items()]
    elif suite == "development":
        cases = [
            development_case(protocol, name, mode)
            for name in DEVELOPMENT_CASES
            for mode in (1, 2, 3)
        ]
    else:
        raise ValueError("Unknown evaluation suite")
    return [
        {
            "request_id": f"{suite}/{case['id']}/{variant}",
            "suite": suite,
            "case": case,
            "variant": variant,
            "flags": flags,
            "status": "unexecuted",
            "reason": "requires_new_rgb_scan_and_estimated_registration"
            if suite == "heldout"
            else "not_yet_run",
        }
        for case in cases
        for variant, flags in protocol["variants"].items()
    ]


def summarize(inventory, reports):
    """Every frozen request stays in the denominator, including missing runs."""
    requested = {item["request_id"]: item for item in inventory}
    actual = {}
    for report in reports:
        key = report["request_id"]
        if key not in requested:
            raise ValueError("Report is outside this frozen request inventory")
        if key in actual:
            raise ValueError(
                "Duplicate attempt: preserve it in a separate complete batch"
            )
        if report.get("protocol_sha256") != FROZEN_SHA256:
            raise ValueError("Report does not match the frozen protocol")
        actual[key] = report
    counts = Counter(
        requested=len(requested),
        executed=len(actual),
        unexecuted=len(requested) - len(actual),
        contacts=0,
        arrivals=0,
        localization_loss_episodes=0,
        rejected_requests=0,
        unexpected_rejections=0,
        manual_interventions=0,
        scene_invalidations=0,
        worker_errors=0,
        audit_errors=0,
        unscored_requests=0,
        passed_requests=0,
    )
    for report in actual.values():
        for metric in (
            "contacts",
            "arrivals",
            "localization_loss_episodes",
            "rejected_requests",
            "unexpected_rejections",
            "manual_interventions",
            "scene_invalidations",
            "worker_errors",
            "audit_errors",
        ):
            counts[metric] += report.get("metrics", {}).get(metric, 0)
        counts["passed_requests"] += int(report.get("intended_pass") is True)
        counts["unscored_requests"] += int(report.get("evidence_complete") is False)
    return {
        "schema": "bb8.navigation-visibility-summary.v1",
        "protocol_sha256": FROZEN_SHA256,
        "counts": dict(counts),
        "denominators": {
            "all_frozen_requests": len(requested),
            "executed_requests": len(actual),
        },
        "complete": len(actual) == len(requested),
        "all_requested_passed": bool(requested)
        and len(actual) == len(requested)
        and all(report.get("intended_pass") is True for report in actual.values()),
        "requests": [
            dict(
                item,
                status="executed",
                reason="recorded_attempt",
                report=actual[item["request_id"]],
            )
            if item["request_id"] in actual
            else item
            for item in inventory
        ],
        "scope": "Development and held-out denominators are separate; unexecuted requests are not successes. Contacts/arrivals/rejections count affected requests; loss episodes count transitions.",
    }


def _zero(vector):
    return (
        isinstance(vector, (list, tuple))
        and len(vector) == 2
        and all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and abs(value) <= 1e-9
            for value in vector
        )
    )


def _same_goal(value, expected):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and abs(actual - target) <= 1e-9
            for actual, target in zip(value, expected, strict=True)
        )
    )


def camera_events_match(events, expected):
    """The renderer must record each prescribed disturbance at its capture tick."""
    actual = [
        event for event in events if event.get("type") == "evaluation_camera_change"
    ]
    return len(actual) == len(expected) and all(
        set(event) == {"type", "time", *change}
        and all(event[key] == value for key, value in change.items())
        and type(event["time"]) in (int, float)
        and math.isfinite(event["time"])
        and change["at_time"] - 1e-9 <= event["time"] <= change["at_time"] + 0.05 + 1e-9
        for event, change in zip(actual, expected, strict=True)
    )


def terminal_rejection(case, state, first_rejected_at):
    """Held-out request is terminal; observe revocation before ending its window."""
    if (
        case["intent"] == "camera_change"
        or state.get("generation") != 1
        or state.get("phase") != "goal_rejected"
        or state.get("commanded_goal") is not None
    ):
        return None
    now = state.get("sim_time")
    if type(now) not in (int, float) or not math.isfinite(now):
        return None
    first = now if first_rejected_at is None else first_rejected_at
    if now + 1e-7 < max(case["minimum_end_s"], first + 0.5):
        return None
    return {
        "reason": "initial_goal_terminal_rejection",
        "first_observed_s": first,
        "stopped_after_s": now,
        "zero_authority_hold_s": 0.5,
    }


def terminal_stop_valid(rows, commands, case, terminal):
    if terminal is None:
        return True
    if (
        not isinstance(terminal, dict)
        or terminal.get("reason") != "initial_goal_terminal_rejection"
        or terminal.get("zero_authority_hold_s") != 0.5
        or case["intent"] == "camera_change"
    ):
        return False
    first, last = terminal.get("first_observed_s"), terminal.get("stopped_after_s")
    if not all(type(t) in (int, float) and math.isfinite(t) for t in (first, last)):
        return False
    initial = [
        c for c in commands if c.get("action") == "goal" and c.get("generation") == 1
    ]
    after = [r for r in rows if r["time"] >= first - 1e-7]
    return (
        len(initial) == 1
        and initial[0].get("outcome") in {"rejected", "accepted_pending_visible_route"}
        and not any(
            c.get("action") in {"goal", "demo", "autonomy"}
            and c.get("generation", 0) > 1
            for c in commands
        )
        and last + 1e-7 >= max(case["minimum_end_s"], first + 0.5)
        and bool(after)
        and after[0]["time"] <= first + 0.05 + 1e-7
        and after[-1]["after_step"]["time"] + 1e-7 >= last
        and all(
            r.get("controller_status") == "goal_rejected"
            and r.get("commanded_goal") is None
            and _zero(r.get("action"))
            for r in after
        )
    )


def cleanup_worker(process, errors):
    """Return only after bounded cleanup; caller treats forced cleanup as failure."""
    process.join(30)
    if process.is_alive():
        errors.append("worker_cleanup_timeout")
        process.terminate()
        process.join(10)
    if process.is_alive():
        errors.append("worker_kill_required")
        process.kill()
        process.join(10)
    alive = process.is_alive()
    if alive:
        errors.append("worker_still_alive")
    return not alive


def score_navigation(rows, commands, case, flags, arrival_audit, harness, events=()):
    """Pure post-decision metrics supplement the independent physics/RGB auditor."""
    statuses = [row.get("localization", {}).get("localization_status") for row in rows]
    losses = sum(
        status in {"lost", "reacquiring"}
        and (index == 0 or statuses[index - 1] not in {"lost", "reacquiring"})
        for index, status in enumerate(statuses)
    )
    goal_commands = [
        command
        for command in commands
        if command.get("action") == "goal" and command.get("generation") == 1
    ]
    rejected = any(
        command.get("outcome") == "rejected" for command in goal_commands
    ) or any(row.get("controller_status") == "goal_rejected" for row in rows)
    invalid = [
        index
        for index, row in enumerate(rows)
        if row.get("scene_validity", {}).get("invalidated") is True
    ]
    after_invalid = rows[invalid[0] :] if invalid else []
    latch_retained = bool(after_invalid) and all(
        row.get("scene_validity", {}).get("invalidated") is True
        and row["scene_validity"].get("navigation_allowed") is False
        for row in after_invalid
    )
    zero_after_invalid = bool(after_invalid) and all(
        _zero(row.get("action"))
        and row.get("commanded_goal") is None
        and row.get("localization", {}).get("requires_new_goal") is True
        for row in after_invalid
    )
    metrics = {
        "contacts": int(bool(arrival_audit.get("collision"))),
        "arrivals": int(arrival_audit.get("valid_arrivals", 0) > 0),
        "localization_loss_episodes": losses,
        "rejected_requests": int(rejected),
        "unexpected_rejections": int(
            rejected and case["intent"] not in {"occupied_rejection", "camera_change"}
        ),
        "manual_interventions": len(harness.get("manual_interventions", [])),
        "scene_invalidations": int(bool(invalid)),
        "worker_errors": int(
            harness.get("exitcode") != 0 or bool(harness.get("errors"))
        ),
    }
    checks = {
        "rows_present": bool(rows),
        "terminal_stop_valid": terminal_stop_valid(
            rows, commands, case, harness.get("terminal_stop")
        ),
        "independent_audit_pass": arrival_audit.get("audit_pass") is True,
        "no_contacts": not metrics["contacts"],
        "no_boundary": not arrival_audit.get("boundary"),
        "no_invalid_arrivals": arrival_audit.get(
            "premature_or_unobservable_arrivals", 0
        )
        == 0,
        "worker_completed": not metrics["worker_errors"],
        "one_initial_request_observed": len(goal_commands) == 1,
        "initial_request_matches_frozen_goal": len(goal_commands) == 1
        and _same_goal(
            [goal_commands[0].get("x"), goal_commands[0].get("y")], case["goal"]
        ),
        "recorded_goals_match_frozen_goal": all(
            row.get("commanded_goal") is None
            or _same_goal(row["commanded_goal"], case["goal"])
            for row in rows
        ),
        "arrival_goals_match_frozen_goal": all(
            _same_goal(event.get("goal"), case["goal"])
            for event in arrival_audit.get("arrival_events", [])
        ),
        "scheduled_camera_changes_recorded": camera_events_match(
            events, case.get("evaluation_camera_changes", [])
        ),
        "minimum_window_complete": bool(rows)
        and rows[-1]["after_step"]["time"] >= case["minimum_end_s"] - 1e-7,
    }
    if case["intent"] == "occupied_rejection":
        checks.update(
            occupied_request_rejected=rejected,
            no_motion_for_rejected_goal=bool(rows)
            and all(_zero(row.get("action")) for row in rows),
            no_arrival_for_rejected_goal=not metrics["arrivals"],
        )
    elif case["intent"] == "camera_change":
        checks["full_camera_change_window"] = (
            bool(rows)
            and rows[-1]["after_step"]["time"] >= case["duration_s"] - 0.05 - 1e-7
        )
        if flags["scene_validity"]:
            changes = case["evaluation_camera_changes"]
            first = after_invalid[0] if after_invalid else {}
            guard = first.get("scene_validity", {})
            camera = guard.get("camera_id")
            diagnostic = guard.get("cameras", {}).get(camera, {})
            confirmations = guard.get("parameters", {}).get("confirmation_samples")
            reason = guard.get("reason")
            image_change_reported = (
                bool(invalid)
                and reason in {"camera_or_global_scene_change", "scene_content_changed"}
                and camera in {change["camera_id"] for change in changes}
                and guard.get("timestamp") == first["time"] == guard.get("checked_at")
                and guard.get("capture_times", {}).get(camera) == first["time"]
                and diagnostic.get("reason") == reason
                and type(confirmations) is int
                and confirmations >= 2
                and type(diagnostic.get("consecutive_changes")) is int
                and diagnostic["consecutive_changes"] >= confirmations
            )
            checks.update(
                rgb_change_invalidated=image_change_reported
                and rows[invalid[0]]["time"] >= changes[0]["at_time"],
                invalidation_latched=latch_retained,
                zero_after_latched_invalidation=zero_after_invalid,
                fresh_goal_after_invalidation_rejected=any(
                    command.get("action") == "goal"
                    and command.get("generation") == 2
                    and command.get("outcome") == "rejected"
                    and _same_goal([command.get("x"), command.get("y")], case["goal"])
                    and type(command.get("sim_time")) in (int, float)
                    and bool(invalid)
                    and command["sim_time"] >= rows[invalid[0]]["time"]
                    for command in commands
                ),
            )
    else:
        checks["valid_arrival"] = bool(metrics["arrivals"])
    return {
        "metrics": metrics,
        "checks": checks,
        "intended_pass": all(checks.values()),
        "status_counts": dict(Counter(statuses)),
        "first_invalidation_s": rows[invalid[0]]["time"] if invalid else None,
        "zero_motion_after_latched_invalidation": zero_after_invalid
        if invalid
        else None,
        "validity_latch_retained": latch_retained if invalid else None,
        "recorded_simulation_s": rows[-1]["after_step"]["time"] if rows else 0,
        "arrival_events": arrival_audit.get("arrival_events", []),
        "terminal_stop": harness.get("terminal_stop"),
    }


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_run(directory, output=None):
    output = output or directory / "visibility-audit"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((directory / "case.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "visibility_independent_audit", ROOT / "scripts/audit-interactive.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.main(
        SimpleNamespace(run=directory / "worker", output=output / "physics-rgb-audit")
    )
    original_path = output / "physics-rgb-audit/report.json"
    original = json.loads(original_path.read_text())
    rows = _read_rows(directory / "worker/rows.jsonl")
    commands = _read_rows(directory / "worker/commands.jsonl")
    event_path = directory / "worker/events.jsonl"
    events = _read_rows(event_path) if event_path.is_file() else []
    harness = json.loads((directory / "harness.json").read_text())
    manifest = json.loads((directory / "worker/manifest.json").read_text())
    protocol = load_protocol(directory / "protocol.json")
    expected_case = select_case(
        protocol,
        metadata["suite"],
        metadata["case"]["id"]
        if metadata["suite"] == "heldout"
        else metadata["case"]["intent"],
        metadata["case"]["mode"],
    )
    baseline_bundle = json.loads((directory / "baseline-bundle.json").read_text())
    baseline_demo = json.loads((directory / "baseline-demo.json").read_text())
    actual_demo = json.loads((directory / "assets/demo.json").read_text())
    result = score_navigation(
        rows, commands, metadata["case"], metadata["flags"], original, harness, events
    )
    effective = manifest.get("navigation_safeguards", {})
    provenance = {
        "protocol_frozen": digest(directory / "protocol.json")
        == FROZEN_SHA256
        == metadata["protocol_sha256"],
        "harness_snapshot": digest(directory / "benchmark-source.py")
        == metadata["harness_sha256"],
        "config_matches": manifest.get("config") == metadata["config"],
        "flags_effective": all(
            effective.get(key) is value for key, value in metadata["flags"].items()
        ),
        "identity_unchanged": manifest.get("run_identity_verification", {}).get(
            "unchanged"
        )
        is True,
        "case_matches_protocol": metadata["case"] == expected_case,
        "request_id_matches_protocol": metadata["request_id"]
        == f"{metadata['suite']}/{expected_case['id']}/{metadata['variant']}",
        "flags_match_protocol": metadata["flags"]
        == protocol["variants"].get(metadata["variant"]),
        "baseline_bundle_frozen": digest(directory / "baseline-bundle.json")
        == (
            metadata.get("provisioned_bundle_sha256")
            if metadata["suite"] == "heldout"
            else protocol["baseline_assets"]["bundle.json"]
        ),
        "only_demo_endpoints_changed": digest(directory / "baseline-demo.json")
        == baseline_bundle["sha256"]["demo.json"]
        and actual_demo
        == configured_demo(baseline_demo, metadata["suite"], expected_case)
        and all(
            name == "demo.json" or digest(directory / "assets" / name) == expected
            for name, expected in baseline_bundle["sha256"].items()
        ),
    }
    if metadata["suite"] == "heldout":
        provisioner().validate_bundle(directory / "assets", expected_case["family"])
        loaded = json.loads((directory / "worker/loaded-setup.json").read_text())
        provenance.update(
            independently_provisioned_family=True,
            evaluation_split=metadata["config"].get("evaluation_split") == "heldout",
            exact_frozen_seeds=actual_demo.get("seed") == expected_case["seed"]
            and actual_demo.get("layout_seed") == expected_case["layout_seed"]
            and loaded.get("layout", {}).get("seed") == expected_case["layout_seed"],
            exact_loaded_start_goal=loaded.get("reset_start") == expected_case["start"]
            and loaded.get("reset_goal") == expected_case["reset_goal"],
            authored_layout=loaded.get("task", {}).get("layout_mode") == "authored",
            provisioning_receipt=digest(directory / "assets/heldout-provisioning.json")
            == metadata.get("provisioning_sha256"),
        )
    result.update(
        schema="bb8.navigation-visibility-run.v1",
        request_id=metadata["request_id"],
        suite=metadata["suite"],
        case=metadata["case"]["id"],
        variant=metadata["variant"],
        mode=metadata["case"]["mode"],
        scope=metadata["case"].get(
            "scope",
            "Synthetic held-out request on a new RGB-scanned family; not a physical reliability claim."
            if metadata["suite"] == "heldout"
            else "Synthetic development request; excluded from held-out evaluation.",
        ),
        flags=metadata["flags"],
        protocol_sha256=FROZEN_SHA256,
        provenance=provenance,
        independent_audit_sha256=digest(original_path),
        input_sha256={
            name: digest(directory / name)
            for name in (
                "case.json",
                "harness.json",
                "worker/manifest.json",
                "worker/rows.jsonl",
                "worker/commands.jsonl",
            )
        },
    )
    if event_path.is_file():
        result["input_sha256"]["worker/events.jsonl"] = digest(event_path)
    result["intended_pass"] &= all(provenance.values())
    write_json(output / "report.json", result)
    return result


def preflight_native(protocol, suite, assets, family=None):
    if suite == "heldout":
        if (
            family not in protocol["families"]
            or not (assets / "heldout-provisioning.json").is_file()
        ):
            raise ValueError(
                "Held-out native assets are unprovisioned; the development map is inadmissible"
            )
        provisioner().validate_bundle(assets, family)
        return
    if suite != "development":
        raise ValueError("Unknown evaluation suite")
    for name, expected in protocol["baseline_assets"].items():
        if digest(assets / name) != expected:
            raise ValueError(f"Development baseline differs from frozen input: {name}")
    from bb8_rl.demo_assets import validate_assets

    validate_assets(assets)


def run(args):
    protocol = load_protocol(args.protocol)
    case = select_case(protocol, args.suite, args.case, args.mode)
    preflight_native(protocol, args.suite, args.assets, case["family"])
    flags = protocol["variants"][args.variant]
    # The lock is a coordination aid for this harness. Other native tools must
    # still obey the project-wide single-native-run rule.
    with (Path(tempfile.gettempdir()) / "bb8-navigation-visibility.lock").open(
        "a"
    ) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another visibility evaluation is already running"
            ) from error
        return _run_locked(args, protocol, case, flags)


def _run_locked(args, protocol, case, flags):
    from bb8_rl.demo_assets import validate_assets
    from bb8_rl.interactive_runtime import worker_main

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(args.protocol.read_bytes())
    (output / "benchmark-source.py").write_bytes(Path(__file__).read_bytes())
    for name in ("bundle", "demo"):
        (output / f"baseline-{name}.json").write_bytes(
            (args.assets / f"{name}.json").read_bytes()
        )
    assets = output / "assets"
    shutil.copytree(args.assets, assets)
    demo_path = assets / "demo.json"
    demo = json.loads(demo_path.read_text())
    demo = configured_demo(demo, args.suite, case)
    write_json(demo_path, demo)
    bundle = json.loads((assets / "bundle.json").read_text())
    bundle["sha256"]["demo.json"] = digest(demo_path)
    write_json(assets / "bundle.json", bundle)
    validate_assets(assets)
    config = {
        "asset_dir": str(assets),
        "run_dir": str(output / "worker"),
        "mode": case["mode"],
        "recording_mode": "audit",
        "save_frames": True,
        "save_every": 1,
        "scoring_labels": True,
        "generation": 0,
        "max_steps": round(case["duration_s"] / 0.05),
        "control_profile": protocol["control_profile"],
        "agency_memory": str(output / "unused-purpose-memory.sqlite3"),
        **flags,
    }
    if "evaluation_camera_changes" in case:
        config["evaluation_camera_changes"] = case["evaluation_camera_changes"]
    if args.suite == "heldout":
        config["evaluation_split"] = "heldout"
    request_id = f"{args.suite}/{case['id']}/{args.variant}"
    write_json(
        output / "case.json",
        {
            "request_id": request_id,
            "suite": args.suite,
            "case": case,
            "variant": args.variant,
            "flags": flags,
            "config": config,
            "protocol_sha256": FROZEN_SHA256,
            "harness_sha256": digest(__file__),
            **(
                {
                    "termination_policy": "duration_s is a maximum; definitive initial rejection may stop after minimum_end_s and a 0.5 s revoked-authority hold; never resubmit",
                    "provisioned_bundle_sha256": digest(args.assets / "bundle.json"),
                    "provisioning_sha256": digest(
                        args.assets / "heldout-provisioning.json"
                    ),
                }
                if args.suite == "heldout"
                else {}
            ),
        },
    )
    context = multiprocessing.get_context("spawn")
    commands, events, stop = context.Queue(32), context.Queue(16), context.Event()
    process = context.Process(target=worker_main, args=(config, commands, events, stop))
    commands.put({"action": "heartbeat", "generation": 0})
    process.start()
    started, heartbeat, generation = time.monotonic(), 0.0, 0
    issued, errors, last_state = [], [], None
    initial_sent = probe_sent = False
    invalidated_at = None
    first_rejected_at = terminal_stop = None
    try:
        while process.is_alive():
            now = time.monotonic()
            if now - started > args.wall_timeout:
                errors.append("worker_wall_timeout")
                break
            if now - heartbeat >= 0.3:
                try:
                    commands.put_nowait(
                        {"action": "heartbeat", "generation": generation}
                    )
                    heartbeat = now
                except queue.Full:
                    pass
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("type") == "error":
                errors.append(event)
            if event.get("type") != "state":
                continue
            last_state = state = event["state"]
            timestamp = state.get("sim_time", 0.0)
            validity = state.get("scene_validity", {})
            if not initial_sent and (
                case["intent"] == "camera_change"
                or (
                    state.get("localization_status") == "measured"
                    and state.get("localization_valid") is True
                    and (
                        not flags["scene_validity"]
                        or validity.get("navigation_allowed") is True
                    )
                )
            ):
                generation = 1
                command = {
                    "action": "goal",
                    "generation": generation,
                    "x": case["goal"][0],
                    "y": case["goal"][1],
                }
                commands.put(command, timeout=1)
                issued.append(
                    dict(
                        command,
                        supervisor_observed_sim_time=timestamp,
                        reason="startup_request_expected_to_be_rejected_during_guard_warmup"
                        if case["intent"] == "camera_change"
                        else "frozen_initial_request",
                    )
                )
                initial_sent = True
            if validity.get("invalidated") is True:
                invalidated_at = timestamp if invalidated_at is None else invalidated_at
                if (
                    case["intent"] == "camera_change"
                    and not probe_sent
                    and timestamp >= invalidated_at + 0.25
                ):
                    generation = 2
                    command = {
                        "action": "goal",
                        "generation": generation,
                        "x": case["goal"][0],
                        "y": case["goal"][1],
                    }
                    commands.put(command, timeout=1)
                    issued.append(
                        dict(
                            command,
                            supervisor_observed_sim_time=timestamp,
                            reason="explicit_request_must_not_clear_latched_invalidity",
                        )
                    )
                    probe_sent = True
            if args.suite == "heldout" and initial_sent:
                if (
                    state.get("generation") == 1
                    and state.get("phase") == "goal_rejected"
                    and state.get("commanded_goal") is None
                ):
                    first_rejected_at = (
                        timestamp if first_rejected_at is None else first_rejected_at
                    )
                    terminal_stop = terminal_rejection(case, state, first_rejected_at)
                    if terminal_stop is not None:
                        stop.set()
                else:
                    first_rejected_at = None
            if (
                initial_sent
                and case["intent"] in {"arrival", "short_arrival", "difficult_arrival"}
                and state.get("phase") == "arrived"
                and timestamp >= case["minimum_end_s"]
            ):
                stop.set()
    except (queue.Full, KeyboardInterrupt) as error:
        errors.append(type(error).__name__)
    finally:
        stop.set()
        worker_confirmed_dead = cleanup_worker(process, errors)
        for channel in (commands, events):
            channel.close()
        write_json(
            output / "harness.json",
            {
                "exitcode": process.exitcode,
                "errors": errors,
                "last_state": last_state,
                "issued_commands": issued,
                "manual_interventions": [],
                "initial_request_sent": initial_sent,
                "terminal_stop": terminal_stop,
                "worker_confirmed_dead": worker_confirmed_dead,
                "wall_seconds": time.monotonic() - started,
            },
        )
    if errors or process.exitcode != 0:
        failure = {
            "request_id": request_id,
            "protocol_sha256": FROZEN_SHA256,
            "suite": args.suite,
            "variant": args.variant,
            "mode": case["mode"],
            "intended_pass": False,
            "metrics": {"worker_errors": 1},
            "errors": errors,
            "evidence_complete": False,
        }
        write_json(output / "summary.json", failure)
        return failure
    try:
        result = audit_run(output)
    except Exception as error:  # noqa: BLE001 — failed audits remain in the denominator
        result = {
            "request_id": request_id,
            "protocol_sha256": FROZEN_SHA256,
            "suite": args.suite,
            "variant": args.variant,
            "mode": case["mode"],
            "intended_pass": False,
            "metrics": {"audit_errors": 1},
            "errors": [f"{type(error).__name__}: {error}"],
            "evidence_complete": False,
        }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "audit", "summarize"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--suite", choices=("development", "heldout"), default="development"
    )
    parser.add_argument(
        "--case", help="Development name or exact frozen held-out case ID"
    )
    parser.add_argument("--mode", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--variant",
        choices=("clearance_only", "visibility_and_validity"),
        default="visibility_and_validity",
    )
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reports", type=Path, nargs="*", default=[])
    parser.add_argument("--wall-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.command == "run":
        if not args.case or not 1 <= args.wall_timeout <= 1800:
            parser.error(
                "run requires --case and bounded --wall-timeout (1..1800 seconds)"
            )
        result = run(args)
    elif args.command == "audit":
        if not args.run_dir:
            parser.error("audit requires --run-dir")
        result = audit_run(args.run_dir, args.output)
    else:
        protocol = load_protocol(args.protocol)
        result = summarize(
            request_inventory(protocol, args.suite),
            [json.loads(path.read_text()) for path in args.reports]
            if args.command == "summarize"
            else [],
        )
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
        write_json(args.output / "summary.json", result)
    print(
        json.dumps(
            result.get(
                "counts",
                {
                    key: result.get(key)
                    for key in ("request_id", "intended_pass", "metrics")
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
