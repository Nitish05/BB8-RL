"""Adversarial checks for the independent interactive native scorer."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "interactive_audit", Path(__file__).parents[1] / "scripts/audit-interactive.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def rows(count=12, *, mode=1):
    result = []
    ids = list("ABC"[:mode])
    for i in range(count):
        t = i * 0.05
        trace = [
            {"time": t + (j + 1) * 0.005, "position": [0, 0], "velocity": [0, 0]}
            for j in range(10)
        ]
        result.append(
            {
                "time": t,
                "capture_time": t,
                "generation": 1,
                "commanded_goal": [0, 0],
                "action": [0, 0],
                "mode": mode,
                "source_ids": ids,
                "views": {k: {"status": "visible"} for k in ids},
                "head_pixels_scoring_only": {k: 12 for k in ids},
                "measurement": {"status": "visible", "xy": [0, 0], "timestamp": t},
                "controller": {
                    "pose_source": "measured",
                    "input_accepted": True,
                    "state_is_current": True,
                },
                "controller_arrived": i == count - 1,
                "truth_before_scoring_only": {"position": [0, 0], "velocity": [0, 0]},
                "after_step": {
                    "time": t + 0.05,
                    "physics_trace_scoring_only": trace,
                    "collision_scoring_only": False,
                    "boundary_scoring_only": False,
                },
            }
        )
    return result


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_arrival_requires_original_true_gate_in_each_mode(mode):
    score = audit.score_rows(rows(mode=mode))
    assert score["integrity_pass"]
    assert score["valid_arrivals"] == 1
    assert score["arrival_events"][0]["true_dwell_at_capture_s"] == pytest.approx(0.55)


def test_later_physics_cannot_credit_early_arrival():
    log = rows(count=10)
    score = audit.score_rows(log)
    assert score["arrival_events"][0]["true_dwell_at_capture_s"] == pytest.approx(0.45)
    assert not score["integrity_pass"]
    assert score["valid_arrivals"] == 0


def test_single_fast_physics_tick_resets_dwell_even_if_captures_are_slow():
    log = rows()
    log[7]["after_step"]["physics_trace_scoring_only"][4]["velocity"] = [0.04, 0]
    assert audit.score_rows(log)["valid_arrivals"] == 0


@pytest.mark.parametrize(
    "field,value", [("commanded_goal", [0.01, 0]), ("generation", 2)]
)
def test_new_goal_or_generation_cannot_inherit_old_dwell(field, value):
    log = rows()
    for row in log[7:]:
        row[field] = value
    assert audit.score_rows(log)["valid_arrivals"] == 0


def test_no_motion_after_goal_cancel():
    log = rows(1)
    log[0]["commanded_goal"] = None
    log[0]["action"] = [0.1, 0]
    log[0]["controller_arrived"] = False
    assert {"frame": 0, "reason": "motion_without_goal"} in audit.score_rows(log)[
        "errors"
    ]


def test_arrival_visibility_is_from_an_accepted_source_not_an_unrelated_camera():
    log = rows(mode=2)
    log[-1]["source_ids"] = ["A"]
    log[-1]["head_pixels_scoring_only"] = {"A": 0, "B": 20}
    assert audit.score_rows(log)["valid_arrivals"] == 0
    log[-1]["source_ids"] = ["B"]
    assert audit.score_rows(log)["valid_arrivals"] == 1


def test_all_native_hidden_false_measurement_fails():
    log = rows(mode=3)
    log[5]["head_pixels_scoring_only"] = dict.fromkeys("ABC", 0)
    score = audit.score_rows(log)
    assert score["all_views_native_hidden_false_accepts"] == 1
    assert not score["integrity_pass"]


def test_missing_observation_is_not_geometric_invisibility():
    log = rows(2)
    log[0]["measurement"]["status"] = "missing"
    log[0]["controller"].update(
        pose_source="predicted", xy=[0.02, 0], position_radius_m=0.03
    )
    log[-1]["controller_arrived"] = False
    score = audit.score_rows(log)
    assert score["all_views_native_hidden_captures"] == 0
    assert score["predicted_radius_coverage"] == {"covered": 1, "samples": 1}
    assert score["reacquisitions"] == [{"time": 0.05, "missing_since": 0.0}]


def test_stale_pose_never_counts_as_visible_arrival():
    log = rows()
    log[-1]["controller"]["state_is_current"] = False
    assert audit.score_rows(log)["valid_arrivals"] == 0


def test_missing_physics_cannot_inflate_dwell():
    log = rows()
    log[8]["after_step"]["physics_trace_scoring_only"].pop(3)
    score = audit.score_rows(log)
    assert any(e["reason"] == "noncontiguous_physics_tick" for e in score["errors"])
    assert not score["integrity_pass"]


def test_collision_invalidates_otherwise_valid_arrival():
    log = rows()
    log[3]["after_step"]["collision_scoring_only"] = True
    assert not audit.score_rows(log)["integrity_pass"]


def test_nan_and_bad_action_shape_rejected():
    log = rows(1)
    log[0]["action"] = [float("nan"), 0]
    with pytest.raises(ValueError):
        audit.score_rows(log)
    log[0]["action"] = [0, 0, 0]
    with pytest.raises(ValueError):
        audit.score_rows(log)


def test_scoring_never_mutates_estimates_or_inputs():
    log = rows()
    original = copy.deepcopy(log)
    audit.score_rows(log)
    assert log == original


def test_external_artifact_reference_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        audit._inside(tmp_path, "../other.png")


def test_source_assertions_alone_do_not_prove_provenance(tmp_path):
    checks = audit.verify_manifest(
        tmp_path,
        {
            "status": "complete",
            "source_files_unchanged": True,
            "truth_to_controller": False,
            "segmentation_to_controller": False,
        },
    )
    assert not checks["source_snapshots_present"]
    assert not checks["asset_hashes_present"]


def test_accepted_stop_cannot_resume_old_goal():
    log = rows()
    event = {"action": "stop", "generation": 2, "outcome": "accepted", "sim_time": 0.2}
    assert not audit.score_commands(log, [event])["pass"]
    for row in log[4:]:
        row["commanded_goal"] = None
    assert audit.score_commands(log, [event])["pass"]


def test_newer_explicit_goal_can_resume_after_stop():
    log = rows()
    for row in log[4:8]:
        row["commanded_goal"] = None
    events = [
        {"action": "stop", "generation": 2, "outcome": "accepted", "sim_time": 0.2},
        {
            "action": "goal",
            "generation": 3,
            "outcome": "accepted_pending_visible_route",
            "sim_time": 0.4,
        },
    ]
    assert audit.score_commands(log, events)["pass"]
    events[1]["outcome"] = "stale_generation"
    assert not audit.score_commands(log, events)["pass"]


def test_no_stationary_stop_as_braking_claim():
    log = rows()
    for row in log[4:]:
        row["commanded_goal"] = None
    event = {"action": "stop", "generation": 2, "outcome": "accepted", "sim_time": 0.2}
    result = audit.score_commands(log, [event])
    assert result["pass"]
    assert not result["cancellations"][0]["motion_preceded_event"]


def test_motion_after_long_rgb_loss_rejected_even_if_prediction_is_current():
    log = rows(24)
    for row in log[1:]:
        row["measurement"].update(status="missing", xy=None)
        row["source_ids"] = []
        row["controller"].update(
            pose_source="predicted", xy=[0, 0], position_radius_m=0.04
        )
        row["action"] = [0.1, 0]
        row["controller_arrived"] = False
    score = audit.score_rows(log)
    assert any(
        e["reason"] == "motion_beyond_one_second_rgb_loss" for e in score["errors"]
    )


def test_idle_and_disabled_scoring_labels_are_explicitly_unverified():
    log = rows(2)
    for row in log:
        row["controller"] = None
        row["commanded_goal"] = None
        row["controller_arrived"] = False
        row["head_pixels_scoring_only"] = {"A": None}
    result = audit.score_rows(log)
    assert result["integrity_pass"]
    assert result["valid_arrivals"] == 0
