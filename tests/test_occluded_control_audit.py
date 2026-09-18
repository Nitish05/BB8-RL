"""Counterexamples for the independent native scorer; no Genesis required."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "occlusion_audit", ROOT / "scripts/audit-occluded-control.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

PROTOCOL = {
    "arrival": {
        "position_tolerance_m": 0.1,
        "speed_tolerance_m_s": 0.03,
        "dwell_seconds": 0.5,
        "visible_head_min_pixels": 3,
    },
    "nonzero_action_epsilon": 1e-6,
    "action_period_seconds": 0.05,
}
CASE = {"id": "fixture", "kind": "visible_baseline", "goal": [0.0, 0.0]}


def frames(count=12):
    rows = []
    for i in range(count):
        t = i * 0.05
        rows.append(
            {
                "step": i,
                "time": t,
                "capture_time": t,
                "head_pixels_scoring_only": 10,
                "dropout_injected": False,
                "action": [0.0, 0.0],
                "prior_applied_command": [0.0, 0.0],
                "measurement": {"status": "visible", "xy": [0.0, 0.0], "timestamp": t},
                "controller": {
                    "pose_source": "measured",
                    "xy": [0.0, 0.0],
                    "last_visual_time": t,
                    "position_radius_m": 0.02,
                },
                "controller_arrived": i == count - 1,
                "truth_before_scoring_only": {
                    "position": [0.0, 0.0, 0.055],
                    "velocity": [0.0, 0.0, 0.0],
                },
                "after_step": {
                    "time": (i + 1) * 0.05,
                    "applied_command": [0.0, 0.0],
                    "collision_scoring_only": False,
                    "boundary_scoring_only": False,
                    "dwell_seconds_scoring_only": (i + 1) * 0.05,
                    "success_scoring_only": i >= 9,
                    "physics_trace_scoring_only": [
                        {
                            "time": t + (j + 1) * 0.005,
                            "position": [0.0, 0.0, 0.055],
                            "velocity": [0.0, 0.0, 0.0],
                        }
                        for j in range(10)
                    ],
                },
            }
        )
    return rows


def hidden(row, *, injected=False, measured=False):
    row["head_pixels_scoring_only"] = 10 if injected else 0
    row["dropout_injected"] = injected
    row["action"] = [0.2, 0.0]
    row["after_step"]["applied_command"] = [0.2, 0.0]
    if not measured:
        row["measurement"].update(status="missing", xy=None)
        row["controller"].update(pose_source="predicted", last_visual_time=0.0)


def test_cell_center_clear_but_solid_crosses_cell_is_false_free():
    box = {"position": [0.98, 0.5, 0.5], "size": [0.1, 1.0, 1.0]}
    overlap, contact = audit.solid_intersections(
        [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], [box]
    )
    assert overlap.tolist() == [True] and contact.tolist() == [False]


def test_zero_volume_face_contact_separate_from_solid_overlap():
    box = {"position": [1.5, 0.5, 0.5], "size": [1.0, 1.0, 1.0]}
    overlap, contact = audit.solid_intersections(
        [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], [box]
    )
    assert overlap.tolist() == [False] and contact.tolist() == [True]


def test_rotated_box_fails_instead_of_optimistic_axis_aligned_scoring():
    with pytest.raises(ValueError, match="axis-aligned"):
        audit.solid_intersections(
            [[0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0]],
            [
                {
                    "position": [1.0, 1.0, 1.0],
                    "size": [1.0, 1.0, 1.0],
                    "euler": [0.0, 0.0, 0.1],
                }
            ],
        )


def test_visible_arrival_reconstructed_from_physics_samples():
    report = audit.score_episode(frames(), CASE, PROTOCOL)
    assert report["intent_pass"] and report["visible_arrival"]
    assert report["physics_trace_samples"] == 120
    assert report["independent_dwell_discrepancies"] == []


def test_reported_success_cannot_override_a_single_fast_physics_sample():
    rows = frames()
    rows[8]["after_step"]["physics_trace_scoring_only"][4]["velocity"] = [
        0.04,
        0.0,
        0.0,
    ]
    result = audit.score_episode(rows, CASE, PROTOCOL)
    assert not result["intent_pass"] and not result["visible_arrival"]
    assert result["independent_dwell_discrepancies"]


def test_arrival_must_already_have_dwelled_at_visible_capture_time():
    rows = frames(10)
    result = audit.score_episode(rows, CASE, PROTOCOL)
    assert not result["visible_arrival"]  # after-step .50 does not certify capture .45


def test_injected_blackout_never_receives_geometric_occlusion_credit():
    rows = frames()
    hidden(rows[3], injected=True)
    case = dict(
        CASE, kind="geometric_occlusion", dropout_intervals_seconds=[[0.15, 0.2]]
    )
    result = audit.score_episode(rows, case, PROTOCOL)
    assert result["geometry_hidden_frames"] == 0 and not result["intent_pass"]


def test_actual_zero_pixel_occlusion_motion_reacquisition_visible_arrival():
    rows = frames()
    hidden(rows[3])
    result = audit.score_episode(rows, dict(CASE, kind="geometric_occlusion"), PROTOCOL)
    assert (
        result["intent_pass"]
        and result["geometry_hidden_nonzero_requested_frames"] == 1
    )
    assert any(e["after_geometric_hidden"] for e in result["reacquisitions"])


def test_detector_accepting_zero_pixel_head_is_exposed_as_false_accept():
    rows = frames()
    hidden(rows[3], measured=True)
    result = audit.score_episode(rows, dict(CASE, kind="geometric_occlusion"), PROTOCOL)
    assert result["geometry_hidden_falsely_measured_frames"] == 1
    assert not result["intent_pass"]


def test_prediction_radius_coverage_does_not_use_larger_braking_envelope():
    rows = frames()
    hidden(rows[3])
    rows[3]["controller"].update(
        xy=[0.04, 0.0], position_radius_m=0.02, envelope_radius_m=0.20
    )
    result = audit.score_episode(rows, CASE, PROTOCOL)
    assert result["predicted_position_radius_coverage"] == {"covered": 0, "samples": 1}
    assert result["predicted_position_error_m"]["maximum"] == pytest.approx(0.04)


def test_invalid_memory_requires_zero_request_and_complete_observation_window():
    rows = frames()
    case = dict(
        CASE,
        kind="invalid_memory",
        invalidate_memory_at_seconds=0.2,
        minimum_end_time_seconds=0.6,
    )
    assert not audit.score_episode(rows, case, PROTOCOL)["intent_pass"]
    rows[2]["action"] = [0.2, 0.0]
    rows[2]["truth_before_scoring_only"]["velocity"] = [0.04, 0.0]
    assert audit.score_episode(rows, case, PROTOCOL)["intent_pass"]
    rows[8]["action"] = [0.01, 0.0]
    assert not audit.score_episode(rows, case, PROTOCOL)["intent_pass"]
    assert not audit.score_episode(rows[:4], case, PROTOCOL)["intent_pass"]


def test_missing_case_is_failure_and_input_is_not_modified():
    assert audit.score_episode([], CASE, PROTOCOL)["status"] == "missing"
    rows = frames()
    before = copy.deepcopy(rows)
    audit.score_episode(rows, CASE, PROTOCOL)
    assert rows == before


def test_drifted_capture_timestamp_fails_integrity():
    rows = frames()
    rows[3]["capture_time"] += 0.005
    assert not audit.score_episode(rows, CASE, PROTOCOL)["intent_pass"]


def test_stale_diagnostics_not_compared_to_current_truth_as_prediction():
    rows = frames()
    rows[3]["controller"].update(
        xy=[9.0, 9.0], state_is_current=False, pose_source="stale"
    )
    result = audit.score_episode(rows, CASE, PROTOCOL)
    assert result["position_error_m"]["samples"] == 11
    assert result["position_error_m"]["maximum"] == 0.0


def test_long_blind_zero_command_then_reacquisition_has_separate_intent():
    rows = frames(45)
    case = dict(
        CASE,
        kind="long_dropout",
        dropout_intervals_seconds=[[0.2, 1.8]],
        minimum_end_time_seconds=2.0,
    )
    for i in range(4, 36):
        hidden(rows[i], injected=True)
        rows[i]["controller"]["last_visual_time"] = 0.15
        if rows[i]["time"] > 1.15 + 1e-9:
            rows[i]["action"] = [0.0, 0.0]
    assert not audit.score_episode(rows, case, PROTOCOL)["intent_pass"]
    rows[2]["action"] = [0.2, 0.0]
    rows[2]["truth_before_scoring_only"]["velocity"] = [0.04, 0.0]
    result = audit.score_episode(rows, case, PROTOCOL)
    assert result["intent_pass"] and result["expired_blind_frames"]
    rows[30]["action"] = [0.01, 0.0]
    assert not audit.score_episode(rows, case, PROTOCOL)["intent_pass"]


def test_hidden_tick_commands_survive_terminal_endpoint_stop_and_keep_visibility_limit():
    rows = frames()
    for i in (3, 4):
        hidden(rows[i])
        for j, tick in enumerate(rows[i]["after_step"]["physics_trace_scoring_only"]):
            tick["command_active_before_tick"] = [0.2, 0.0]
            tick["command_active_after_tick"] = [0.2, 0.0]
            tick["position"] = [0.001 * (j + 1), 0.0]
            tick["velocity"] = [0.2, 0.0]
        rows[i]["after_step"]["applied_command"] = [0.0, 0.0]
    result = audit.hidden_tick_metrics(rows)
    first = result["after_zero_head_capture"]
    bounded = result["bounded_by_two_zero_head_captures"]
    assert first["physics_ticks"] == first["nonzero_active_after_tick_samples"] == 20
    assert first["physical_path_distance_m"] == pytest.approx(0.02)
    assert bounded["physics_ticks"] == 10
    assert bounded["physical_path_distance_m"] == pytest.approx(0.01)
    assert bounded["true_speed_m_s"]["maximum"] == pytest.approx(0.2)


def test_injected_dropout_is_excluded_from_native_hidden_tick_metrics():
    rows = frames()
    hidden(rows[3], injected=True)
    assert (
        audit.hidden_tick_metrics(rows)["after_zero_head_capture"]["physics_ticks"] == 0
    )


def acknowledgements(rows):
    previous = []
    for row in rows:
        row["prior_acknowledged_command_intervals"] = copy.deepcopy(previous)
        previous = [
            {
                "start": row["time"] + j * 0.005,
                "end": tick["time"],
                "command": [0.1, 0.0],
            }
            for j, tick in enumerate(row["after_step"]["physics_trace_scoring_only"])
        ]
        row["after_step"]["acknowledged_command_intervals"] = copy.deepcopy(previous)
    return rows


def test_exact_command_partitions_are_causal_and_legacy_absence_explicit():
    result = audit.audit_command_intervals(acknowledgements(frames()), required=True)
    assert result["pass"] and result["intervals"] == 120
    assert audit.audit_command_intervals(frames())["pass"]
    assert not audit.audit_command_intervals(frames(), required=True)["pass"]


@pytest.mark.parametrize(
    "change",
    [
        {"velocity": [0.1, 0.0]},
        {"start": 0.001},
        {"end": 0.006},
        {"command": [0.8, 0.8]},
        {"command": [float("nan"), 0.0]},
    ],
)
def test_command_interval_rejects_state_leak_gaps_future_and_bad_commands(change):
    rows = acknowledgements(frames())
    rows[0]["after_step"]["acknowledged_command_intervals"][0].update(change)
    assert not audit.audit_command_intervals(rows, required=True)["pass"]


def test_prior_commands_must_equal_last_completed_partition_without_future_ticks():
    rows = acknowledgements(frames())
    rows[1]["prior_acknowledged_command_intervals"] = copy.deepcopy(
        rows[1]["after_step"]["acknowledged_command_intervals"]
    )
    assert not audit.audit_command_intervals(rows, required=True)["pass"]


def test_complete_acknowledged_command_metrics_do_not_use_expired_endpoint_value():
    rows = acknowledgements(frames())
    hidden(rows[3])
    rows[3]["after_step"]["applied_command"] = [0.0, 0.0]
    result = audit.hidden_tick_metrics(rows)["after_zero_head_capture"]
    assert result["nonzero_exact_acknowledged_intervals"] == 10
