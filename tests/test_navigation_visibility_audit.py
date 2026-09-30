"""Independent frozen-inventory and post-decision navigation audit contracts."""

import builtins
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "navigation_visibility_audit", ROOT / "scripts/benchmark-navigation-visibility.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_heldout_specification_is_frozen_and_covers_all_requested_conditions():
    protocol = audit.load_protocol()
    assert audit.digest(audit.PROTOCOL) == audit.FROZEN_SHA256
    assert len(protocol["cases"]) == 24
    assert len(protocol["families"]) == 4
    assert (
        len(protocol["layouts"])
        == len(protocol["camera_sets"])
        == len(protocol["appearances"])
        == 2
    )
    for mode in (1, 2, 3):
        cases = [case for case in protocol["cases"].values() if case["mode"] == mode]
        assert len(cases) == 8
        assert sum(case["intent"] == "arrival" for case in cases) == 4
        assert sum(case["intent"] == "occupied_rejection" for case in cases) == 4
    assert protocol["development"]["excluded_from_heldout"]
    assert protocol["development"]["goals"]["difficult_arrival"] == [0.5, 1.5]


def test_changed_protocol_requires_a_new_freeze(tmp_path):
    path = tmp_path / "modified.json"
    protocol = audit.load_protocol()
    next(iter(protocol["cases"].values()))["goal"] = [0, 1]
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="pre-tuning freeze"):
        audit.load_protocol(path)


def test_import_and_unprovisioned_heldout_gate_never_import_native_or_weights(
    monkeypatch, tmp_path
):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "genesis",
            "genesis_studio",
            "torch",
            "stable_baselines3",
            "gymnasium",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    with pytest.raises(ValueError, match="unprovisioned"):
        module.preflight_native(
            module.load_protocol(), "heldout", tmp_path / "does-not-exist"
        )


def test_unexecuted_requests_remain_in_every_frozen_denominator():
    inventory = audit.request_inventory(audit.load_protocol(), "heldout")
    summary = audit.summarize(inventory, [])
    assert len(inventory) == 48
    assert summary["counts"]["requested"] == summary["counts"]["unexecuted"] == 48
    assert summary["counts"]["executed"] == summary["counts"]["passed_requests"] == 0
    assert not summary["complete"] and not summary["all_requested_passed"]
    assert all(
        item["reason"] == "requires_new_rgb_scan_and_estimated_registration"
        for item in summary["requests"]
    )


def test_partial_failures_and_contacts_are_not_removed_from_summary():
    inventory = audit.request_inventory(audit.load_protocol(), "heldout")
    reports = [
        {
            "request_id": inventory[0]["request_id"],
            "protocol_sha256": audit.FROZEN_SHA256,
            "intended_pass": True,
            "metrics": {"arrivals": 1},
        },
        {
            "request_id": inventory[1]["request_id"],
            "protocol_sha256": audit.FROZEN_SHA256,
            "intended_pass": False,
            "metrics": {"contacts": 1, "worker_errors": 1},
        },
    ]
    summary = audit.summarize(inventory, reports)
    assert summary["denominators"] == {
        "all_frozen_requests": 48,
        "executed_requests": 2,
    }
    assert summary["counts"]["passed_requests"] == 1
    assert summary["counts"]["contacts"] == summary["counts"]["worker_errors"] == 1
    assert summary["counts"]["unexecuted"] == 46
    assert not summary["all_requested_passed"]
    with pytest.raises(ValueError, match="Duplicate attempt"):
        audit.summarize(inventory, [reports[0], reports[0]])
    with pytest.raises(ValueError, match="outside"):
        audit.summarize(inventory, [dict(reports[0], request_id="development/other")])
    with pytest.raises(ValueError, match="frozen protocol"):
        audit.summarize(inventory, [dict(reports[0], protocol_sha256="0" * 64)])


def test_development_rejection_keeps_safe_reset_separate_from_user_goal():
    protocol = audit.load_protocol()
    case = audit.development_case(protocol, "occupied_rejection", 1)
    assert case["goal"] != case["reset_goal"]
    assert case["reset_goal"] == protocol["development"]["goals"]["short_arrival"]
    camera = audit.development_case(protocol, "camera_change", 3)
    assert set(camera["evaluation_camera_changes"][0]) == {
        "at_time",
        "camera_id",
        "position",
        "lookat",
    }
    assert camera["evaluation_camera_changes"][0]["at_time"] == 1.0
    assert len(audit.request_inventory(protocol, "development")) == 24


@pytest.fixture
def camera_run():
    case = audit.development_case(audit.load_protocol(), "camera_change", 1)
    rows = []
    for timestamp in (0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 4.0, 7.95):
        invalid = timestamp >= 1.25
        rows.append(
            {
                "time": timestamp,
                "after_step": {"time": timestamp + 0.05},
                "action": [0.0, 0.0] if invalid else [0.1, 0.0],
                "commanded_goal": None if invalid else case["goal"],
                "localization": {
                    "localization_status": "measured",
                    "requires_new_goal": invalid,
                },
                "scene_validity": {
                    "invalidated": invalid,
                    "navigation_allowed": not invalid,
                    "timestamp": 1.25 if invalid else timestamp,
                    "checked_at": 1.25 if invalid else timestamp,
                    "capture_times": {"A": 1.25 if invalid else timestamp},
                    "camera_id": "A" if invalid else None,
                    "reason": "camera_or_global_scene_change" if invalid else None,
                    "parameters": {"confirmation_samples": 2},
                    "cameras": {
                        "A": {
                            "reason": "camera_or_global_scene_change"
                            if invalid
                            else None,
                            "consecutive_changes": 2 if invalid else 0,
                        }
                    },
                },
            }
        )
    commands = [
        {
            "action": "goal",
            "generation": 1,
            "outcome": "accepted_pending_visible_route",
            "x": case["goal"][0],
            "y": case["goal"][1],
            "sim_time": 0.0,
        },
        {
            "action": "goal",
            "generation": 2,
            "outcome": "rejected",
            "x": case["goal"][0],
            "y": case["goal"][1],
            "sim_time": 1.5,
        },
    ]
    original = {
        "audit_pass": True,
        "collision": False,
        "boundary": False,
        "valid_arrivals": 0,
        "premature_or_unobservable_arrivals": 0,
        "arrival_events": [],
    }
    harness = {"exitcode": 0, "errors": [], "manual_interventions": []}
    return (
        rows,
        commands,
        case,
        {"scene_validity": True, "visibility_planning": True},
        original,
        harness,
        [
            {
                "type": "evaluation_camera_change",
                "time": 1.0,
                **case["evaluation_camera_changes"][0],
            }
        ],
    )


def test_camera_change_requires_rgb_invalidation_latch_and_zero_future_motion(
    camera_run,
):
    result = audit.score_navigation(*camera_run)
    assert result["intended_pass"]
    assert result["first_invalidation_s"] == 1.25
    assert result["metrics"]["scene_invalidations"] == 1
    assert result["zero_motion_after_latched_invalidation"] is True


@pytest.mark.parametrize(
    "defect",
    [
        "motion",
        "goal",
        "authority",
        "unlatch",
        "renewed_goal_accepted",
        "too_early",
        "short_window",
        "no_request",
        "contact",
    ],
)
def test_camera_audit_rejects_invalid_safety_evidence(camera_run, defect):
    rows, commands, case, flags, original, harness, events = copy.deepcopy(camera_run)
    if defect == "motion":
        rows[-1]["action"] = [0.001, 0]
    elif defect == "goal":
        rows[-1]["commanded_goal"] = [0.5, 1.5]
    elif defect == "authority":
        rows[-1]["localization"]["requires_new_goal"] = False
    elif defect == "unlatch":
        rows[-1]["scene_validity"]["invalidated"] = False
    elif defect == "renewed_goal_accepted":
        commands[1]["outcome"] = "accepted_pending_visible_route"
    elif defect == "too_early":
        rows[0]["scene_validity"]["invalidated"] = True
    elif defect == "short_window":
        rows.pop()
    elif defect == "no_request":
        commands = commands[1:]
    else:
        original["collision"] = True
    assert not audit.score_navigation(
        rows, commands, case, flags, original, harness, events
    )["intended_pass"]


def test_occupied_goal_requires_rejection_and_never_motion(camera_run):
    rows, commands, _, flags, original, harness, _ = copy.deepcopy(camera_run)
    case = audit.development_case(audit.load_protocol(), "occupied_rejection", 1)
    for row in rows:
        row["action"] = [0.0, 0.0]
        row["commanded_goal"] = None
    commands = [
        {
            "action": "goal",
            "generation": 1,
            "outcome": "rejected",
            "x": case["goal"][0],
            "y": case["goal"][1],
            "sim_time": 0.0,
        }
    ]
    assert audit.score_navigation(rows, commands, case, flags, original, harness)[
        "intended_pass"
    ]
    rows[-1]["action"] = [0.01, 0]
    assert not audit.score_navigation(rows, commands, case, flags, original, harness)[
        "intended_pass"
    ]


def test_arrival_intent_cannot_pass_from_a_clean_stop_without_arrival(camera_run):
    rows, commands, _, flags, original, harness, _ = copy.deepcopy(camera_run)
    case = audit.development_case(audit.load_protocol(), "difficult_arrival", 1)
    result = audit.score_navigation(rows, commands, case, flags, original, harness)
    assert not result["intended_pass"] and result["metrics"]["arrivals"] == 0
    original["valid_arrivals"] = 1
    original["arrival_events"] = [{"goal": case["goal"], "passed": True, "time": 2.0}]
    assert audit.score_navigation(rows, commands, case, flags, original, harness)[
        "intended_pass"
    ]


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_initial_goal",
        "wrong_recorded_goal",
        "wrong_arrival_goal",
        "stale_failclosed",
        "unavailable_rgb",
        "wrong_camera",
        "missing_diagnostics",
        "insufficient_confirmation",
        "stale_guard",
        "early_probe",
        "wrong_probe",
        "missing_event",
        "wrong_event_pose",
        "early_event",
        "late_event",
        "duplicate_event",
    ],
)
def test_camera_evaluation_rejects_false_detection_and_request_evidence(
    camera_run, defect
):
    rows, commands, case, flags, original, harness, events = copy.deepcopy(camera_run)
    guard = rows[3]["scene_validity"]
    if defect == "wrong_initial_goal":
        commands[0]["x"] = 0.0
    elif defect == "wrong_recorded_goal":
        rows[0]["commanded_goal"] = [0.0, 1.0]
    elif defect == "wrong_arrival_goal":
        original["valid_arrivals"] = 1
        original["arrival_events"] = [{"goal": [0.0, 1.0], "passed": True}]
    elif defect in {"stale_failclosed", "unavailable_rgb"}:
        reason = (
            "invalid_or_stale_scene_evidence"
            if defect == "stale_failclosed"
            else "Scene evidence unavailable: missing frame"
        )
        guard["reason"] = guard["cameras"]["A"]["reason"] = reason
    elif defect == "wrong_camera":
        guard["camera_id"] = "B"
    elif defect == "missing_diagnostics":
        guard.pop("cameras")
    elif defect == "insufficient_confirmation":
        guard["cameras"]["A"]["consecutive_changes"] = 1
    elif defect == "stale_guard":
        guard["timestamp"] = 1.0
    elif defect == "early_probe":
        commands[1]["sim_time"] = 0.5
    elif defect == "wrong_probe":
        commands[1]["x"] = 0.0
    elif defect == "missing_event":
        events = []
    elif defect == "wrong_event_pose":
        events[0]["position"] = [0, 0, 3]
    elif defect == "early_event":
        events[0]["time"] = 0.5
    elif defect == "late_event":
        events[0]["time"] = 7.0
    else:
        events *= 2
    result = audit.score_navigation(
        rows, commands, case, flags, original, harness, events
    )
    assert not result["intended_pass"]


def test_physics_valid_arrival_at_another_target_does_not_pass_frozen_request(
    camera_run,
):
    rows, commands, _, flags, original, harness, _ = copy.deepcopy(camera_run)
    case = audit.development_case(audit.load_protocol(), "difficult_arrival", 1)
    other_goal = [0.24745992166936648, 1.3086782891625797]
    commands[0].update(x=other_goal[0], y=other_goal[1])
    for row in rows:
        if row["commanded_goal"] is not None:
            row["commanded_goal"] = other_goal
    original["valid_arrivals"] = 1
    original["arrival_events"] = [{"goal": other_goal, "passed": True}]
    result = audit.score_navigation(rows, commands, case, flags, original, harness)
    assert not result["intended_pass"]
    assert not result["checks"]["initial_request_matches_frozen_goal"]
    assert not result["checks"]["arrival_goals_match_frozen_goal"]


def test_development_comparison_uses_same_thirty_second_arrival_window():
    inventory = audit.request_inventory(audit.load_protocol(), "development")
    arrivals = [
        item for item in inventory if item["case"]["intent"].endswith("arrival")
    ]
    assert len(arrivals) == 12
    assert all(item["case"]["duration_s"] == 30.0 for item in arrivals)


@pytest.mark.parametrize("name", tuple(audit.load_protocol()["cases"]))
def test_heldout_runner_preserves_exact_seed_geometry_and_request(name):
    protocol = audit.load_protocol()
    original = copy.deepcopy(protocol)
    case = audit.select_case(protocol, "heldout", name)
    frozen = protocol["cases"][name]
    assert {key: case[key] for key in frozen} == frozen
    assert case["id"] == name
    family = protocol["families"][case["family"]]
    assert case["reset_goal"] == protocol["layouts"][family["layout"]]["arrival_goal"]
    baseline = {
        "seed": 42,
        "layout_seed": 12345,
        "cameras": {"fixture": "retained"},
        "start": [0, 0],
        "goal": [0, 1],
    }
    demo = audit.configured_demo(baseline, "heldout", case)
    assert demo["seed"] == case["seed"]
    assert demo["layout_seed"] == case["layout_seed"]
    assert demo["start"] == case["start"]
    assert demo["goal"] == case["reset_goal"]
    assert demo["cameras"] == baseline["cameras"]
    assert baseline["seed"] == 42
    assert protocol == original
    with pytest.raises(ValueError, match="mode differs"):
        audit.select_case(protocol, "heldout", name, (case["mode"] % 3) + 1)


def test_new_family_is_mandatory_before_native_import(monkeypatch, tmp_path):
    protocol = audit.load_protocol()
    with pytest.raises(ValueError, match="unprovisioned"):
        audit.preflight_native(protocol, "heldout", tmp_path, "north_oblique_cool")
    with pytest.raises(ValueError, match="Unknown frozen"):
        audit.select_case(protocol, "heldout", "short_arrival", 1)


def test_executed_summary_clears_stale_unexecuted_reason():
    protocol = audit.load_protocol()
    inventory = audit.request_inventory(protocol, "heldout")
    summary = audit.summarize(
        inventory,
        [
            {
                "request_id": inventory[0]["request_id"],
                "protocol_sha256": audit.FROZEN_SHA256,
                "intended_pass": False,
                "metrics": {},
            }
        ],
    )
    assert summary["requests"][0]["reason"] == "recorded_attempt"
    assert (
        summary["requests"][1]["reason"]
        == "requires_new_rgb_scan_and_estimated_registration"
    )


def test_terminal_rejection_preserves_minimum_window_and_revocation_hold():
    case = audit.heldout_case(audit.load_protocol(), "north_oblique_cool-m1-arrival")
    state = {
        "generation": 1,
        "phase": "goal_rejected",
        "commanded_goal": None,
        "sim_time": 1.9,
    }
    assert audit.terminal_rejection(case, state, 1.0) is None
    state["sim_time"] = 2.0
    terminal = audit.terminal_rejection(case, state, 1.0)
    assert terminal["reason"] == "initial_goal_terminal_rejection"
    assert audit.terminal_rejection(case, state, 1.8) is None
    for changed in (
        {"generation": 0},
        {"phase": "braking"},
        {"commanded_goal": case["goal"]},
    ):
        assert audit.terminal_rejection(case, dict(state, **changed), 1.0) is None
    rows = [
        {
            "time": t,
            "after_step": {"time": t + 0.05},
            "action": [0, 0],
            "controller_status": "goal_rejected",
            "commanded_goal": None,
        }
        for t in (1.0, 1.5, 2.0)
    ]
    commands = [{"action": "goal", "generation": 1, "outcome": "rejected"}]
    assert audit.terminal_stop_valid(rows, commands, case, terminal)
    rows[-1]["action"] = [0.1, 0]
    assert not audit.terminal_stop_valid(rows, commands, case, terminal)
    rows[-1]["action"] = [0, 0]
    assert not audit.terminal_stop_valid(
        rows, commands + [{"action": "goal", "generation": 2}], case, terminal
    )
    assert not audit.terminal_stop_valid(rows, [], case, terminal)
    assert not audit.terminal_stop_valid(
        rows, commands, case, dict(terminal, stopped_after_s=1.3)
    )


@pytest.mark.parametrize(
    "alive_checks,expected,dead",
    (
        ([False, False, False], [], True),
        ([True, False, False], ["worker_cleanup_timeout"], True),
        ([True, True, False], ["worker_cleanup_timeout", "worker_kill_required"], True),
        (
            [True, True, True],
            ["worker_cleanup_timeout", "worker_kill_required", "worker_still_alive"],
            False,
        ),
    ),
)
def test_cleanup_confirms_death_and_retains_all_forced_interventions(
    alive_checks, expected, dead
):
    class FakeProcess:
        def __init__(self):
            self.states = iter(alive_checks)
            self.calls = []

        def join(self, timeout):
            self.calls.append(("join", timeout))

        def is_alive(self):
            return next(self.states)

        def terminate(self):
            self.calls.append(("terminate",))

        def kill(self):
            self.calls.append(("kill",))

    process, errors = FakeProcess(), []
    assert audit.cleanup_worker(process, errors) is dead
    assert errors == expected
    assert (("kill",) in process.calls) == ("worker_kill_required" in errors)
