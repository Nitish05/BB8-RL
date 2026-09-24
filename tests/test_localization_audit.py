"""Adversarial evidence tests for the independent M7.13 native auditor."""

import copy
import importlib.util
import queue
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "localization_benchmark",
    Path(__file__).resolve().parents[1] / "scripts/benchmark-localization-recovery.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_heartbeat_saturation_preserves_initial_goal_and_retries():
    commands = queue.Queue(maxsize=1)
    goal = {"action": "goal", "generation": 1, "x": 0.2, "y": 0.3}
    commands.put_nowait(goal)
    for _ in range(100):
        assert not AUDIT.queue_heartbeat(commands, generation=3)
    assert commands.get_nowait() is goal
    assert AUDIT.queue_heartbeat(commands, generation=3)
    assert commands.get_nowait() == {"action": "heartbeat", "generation": 3}


def test_heartbeat_does_not_suppress_broken_queue_errors():
    class BrokenQueue:
        def put_nowait(self, item):
            raise ValueError("Queue is closed")

    with pytest.raises(ValueError, match="closed"):
        AUDIT.queue_heartbeat(BrokenQueue(), generation=3)


def region(valid=False):
    return {
        "valid": valid,
        "motion_authority": False,
        "observed": False,
        "anchor_xy": [0.0, 0.0],
        "anchor_time": 0.0,
        "radius_m": 0.1,
        "radius_limit_m": 0.2,
        "speed_upper_m_s": 0.03,
        "assumptions": ["Unforced dissipative zero-target braking."],
    }


def row(t=0.0, status="measured", generation=1):
    valid = status in {"measured", "predicted"}
    visible = status in {"measured", "reacquiring"}
    return {
        "time": t,
        "generation": generation,
        "action": [0.0, 0.0],
        "commanded_goal": [0.0, 0.0] if valid else None,
        "controller_arrived": False,
        "source_ids": ["A", "B"] if visible else [],
        "measurement": {
            "timestamp": t,
            "xy": [0.0, 0.0] if visible else None,
            "status": "visible" if visible else "missing",
        },
        "prior_applied_command": [0.0, 0.0],
        "prior_acknowledged_command_intervals": [
            {"start": t - 0.05, "end": t, "command": [0.0, 0.0]}
        ]
        if t >= 0.05
        else [],
        "truth_before_scoring_only": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
        "after_step": {
            "time": t + 0.05,
            "velocity_scoring_only": [0.0, 0.0],
            "acknowledged_command_intervals": [],
        },
        "localization": {
            "localization_status": status,
            "localization_valid": valid,
            "pose": [0.0, 0.0] if valid else None,
            "position_radius": 0.02 if valid else None,
            "requires_new_goal": not valid,
            "braking_prediction": region(),
        },
    }


def score(rows):
    return AUDIT.score_localization(
        rows,
        [],
        {"intent": "arrival", "minimum_end_s": 0},
        {"arrival_events": [{"time": rows[-1]["time"], "passed": True}]},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pose", [0.0, 0.0]),
        ("position_radius", 0.01),
        ("localization_valid", True),
    ],
)
def test_expired_prediction_cannot_be_presented_as_current(field, value):
    r = row(status="lost")
    r["localization"][field] = value
    result = score([r])
    assert not result["pass"]
    assert result["failures"][0]["reason"] == "unusable_state_exposes_pose_or_motion"


def test_reacquisition_cannot_authorize_even_tiny_nonzero_request():
    r = row(status="reacquiring")
    r["action"] = [1e-12, 0.0]
    assert not score([r])["pass"]


def test_raw_huge_uncertainty_is_allowed_when_current_pose_unavailable():
    r = row(status="lost")
    r["localization"]["raw_position_radius"] = 7.192
    assert score([r])["pass"]


def test_freshness_not_claimed_from_status_label_alone():
    r = row(t=5.0)
    r["measurement"]["timestamp"] = 1.0
    assert not score([r])["pass"]


def test_stationary_stop_without_prior_motion_is_not_demonstrated_braking():
    rows = [row(t=1.9), row(t=2.0), row(t=5.45)]
    for r in rows[1:]:
        r["commanded_goal"] = None
    result = AUDIT.score_localization(
        rows,
        [{"action": "stop", "outcome": "accepted", "sim_time": 2.0}],
        {"intent": "stop", "minimum_end_s": 5.5},
        {"arrival_events": []},
    )
    assert not result["pass"]
    assert not result["checks"]["motion_before_event"]


def test_conditional_coverage_failure_is_preserved_separately():
    r = row(t=1.0)
    r["localization"]["braking_prediction"] = region(True)
    r["truth_before_scoring_only"]["position"] = [0.11, 0.0]
    result = score([r])
    assert result["pass"]
    assert not result["conditional_braking"]["position_coverage_pass"]
    assert result["conditional_braking"]["maximum_position_excess_m"] == pytest.approx(
        0.01
    )


def test_conditional_region_cannot_be_motion_authority():
    r = row()
    r["localization"]["braking_prediction"]["motion_authority"] = True
    assert not score([r])["pass"]


def test_zero_endpoint_does_not_hide_nonzero_interval():
    r = row(t=1.0)
    r["localization"]["braking_prediction"] = region(True)
    r["prior_acknowledged_command_intervals"] = [
        {"start": 0.95, "end": 1.0, "command": [0.1, 0.0]}
    ]
    assert not score([r])["pass"]


def test_braking_region_without_command_evidence_is_rejected():
    r = row(t=1.0)
    r["localization"]["braking_prediction"] = region(True)
    r["prior_acknowledged_command_intervals"] = []
    assert not score([r])["pass"]


def recovery_evidence():
    rows = [row(t=1.9)]
    rows[0]["truth_before_scoring_only"]["velocity"] = [0.1, 0.0]
    rows[0]["after_step"]["acknowledged_command_intervals"] = [{"command": [0.3, 0.0]}]
    for index in range(100):
        r = row(t=2.0 + index * 0.05, status="lost")
        rows.append(r)
    rows.extend([row(t=7.0, status="reacquiring"), row(t=7.05, status="reacquiring")])
    for t in (7.1, 8.0, 8.95):
        r = row(t=t)
        r["commanded_goal"] = None
        r["localization"]["requires_new_goal"] = True
        rows.append(r)
    rows.append(row(t=11.0, generation=3))
    commands = [
        {"action": "goal", "generation": 2, "sim_time": 4.0, "outcome": "rejected"},
        {
            "action": "goal",
            "generation": 3,
            "sim_time": 9.0,
            "outcome": "accepted_pending_visible_route",
        },
    ]
    case = {
        "intent": "recovery",
        "minimum_end_s": 11.0,
        "new_goal_not_before": 9.0,
        "recovery_hold_s": 1.0,
        "camera_dropouts": {"A": [[2.0, 7.0]], "B": [[2.0, 7.0]]},
    }
    return rows, commands, case, {"arrival_events": [{"time": 11.0, "passed": True}]}


def test_recovery_requires_new_goal_and_preserves_hold():
    evidence = recovery_evidence()
    assert AUDIT.score_localization(*evidence)["pass"]
    altered = copy.deepcopy(evidence)
    altered[0][-2]["action"] = [0.1, 0.0]
    altered[0][-2]["commanded_goal"] = [0.0, 0.0]
    result = AUDIT.score_localization(*altered)
    assert not result["pass"]
    assert not result["checks"]["old_goal_never_resumed"]


def test_failed_goal_rejection_is_not_hidden_by_later_valid_arrival():
    evidence = recovery_evidence()
    evidence[1][0]["outcome"] = "accepted_pending_visible_route"
    result = AUDIT.score_localization(*evidence)
    assert not result["pass"]
    assert not result["checks"]["loss_goal_rejected"]
