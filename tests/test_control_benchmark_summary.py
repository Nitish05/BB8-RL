import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "benchmark_summary", Path(__file__).parents[1] / "scripts/summarize-control-benchmark.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(t, speed=.1, action=.2, visible=True, accepted=True, goal=True):
    return {
        "time": t, "action": [action, 0], "commanded_goal": [1, 1] if goal else None,
        "controller": {"pose_source": "measured" if visible else "predicted",
                       "input_accepted": accepted, "state_is_current": True},
        "measurement": {"status": "visible" if visible else "missing",
                        "xy": [0, 0] if visible else None, "timestamp": t},
        "source_ids": ["A"] if visible else [], "views": {"A": {}, "B": {}},
        "truth_before_scoring_only": {"velocity": [speed, 0]},
        "after_step": {"time": t + .05, "velocity_scoring_only": [speed, 0],
                       "acknowledged_command_intervals": [{"command": [action, 0]}]},
    }


def run(profile, case, arrived=True, errors=(), duration=10):
    return {"profile": {"name": profile}, "case": case, "valid_arrival": arrived,
            "hard_failures": list(errors), "recorded_simulation_s": duration,
            "guarded_stop_episodes": 0}


def test_arrival_count_precedes_speed_and_audit_pass_is_not_success():
    result = MODULE.rank_tuning([
        run("slow", "a", duration=50), run("slow", "b", duration=50),
        run("fast_failed", "a"), run("fast_failed", "b", arrived=False),
    ], ["a", "b"])
    assert result[0]["profile"] == "slow"
    assert not MODULE.control_checks({}, [row(0)], [], {"valid_arrivals": 0})["pass"]


def test_hard_failure_or_missing_case_cannot_win_ranking():
    ranked = MODULE.rank_tuning([
        run("clean", "a"), run("clean", "b"),
        run("collision", "a", errors=["collision"], duration=1), run("collision", "b", duration=1),
        run("incomplete", "a", duration=1),
    ], ["a", "b"])
    assert ranked[0]["profile"] == "clean"
    assert all(not r["eligible"] for r in ranked[1:])


def test_duplicate_attempt_is_not_best_of_repeats():
    ranked = MODULE.rank_tuning([run("p", "a"), run("p", "a"), run("p", "b")], ["a", "b"])
    assert not ranked[0]["eligible"]
    assert "duplicate_tuning_case_no_best_of_repeats" in ranked[0]["exclusions"]


def test_rejected_visible_measurement_does_not_refresh_blind_timer():
    rows = [row(0), row(.5, visible=False), row(1.05, accepted=False)]
    assert MODULE.blind_motion_violations(rows) == [1.05]
    rows[-1]["action"] = [0, 0]
    assert MODULE.blind_motion_violations(rows) == []


def test_idle_stop_not_counted_as_demonstrated_braking():
    case = {"intent": "stop", "duration_s": 2.05}
    commands = [{"action": "stop", "outcome": "accepted", "sim_time": 1.}]
    rows = [row(.95, speed=0, action=0), row(1, speed=0, action=0, goal=False), row(2, speed=0, action=0, goal=False)]
    assert not MODULE.control_checks(case, rows, commands, {})["pass"]
    rows[0] = row(.95)
    assert MODULE.control_checks(case, rows, commands, {})["pass"]
    rows[-1]["commanded_goal"] = [1, 1]
    assert not MODULE.control_checks(case, rows, commands, {})["pass"]


def test_dropout_requires_full_missing_window_recovery_and_minimum_followup():
    case = {"camera_dropouts": {"A": [[2, 5]], "B": [[2, 5]]}}
    rows = [row(i * .05, action=0 if 3 <= i * .05 < 5 else .2,
                speed=0 if 3 <= i * .05 < 5 else .1,
                visible=not 2 <= i * .05 < 5) for i in range(140)]
    audit = {"arrival_events": [{"passed": True, "time": 6.9}]}
    assert MODULE.control_checks(case, rows, [], audit)["pass"]
    assert not MODULE.control_checks(case, rows[:-2], [], audit)["pass"]
    rows[65]["action"] = [.2, 0]
    assert not MODULE.control_checks(case, rows, [], audit)["pass"]
