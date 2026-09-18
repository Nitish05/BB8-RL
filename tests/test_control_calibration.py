"""Independent braking analysis must preserve censoring and causal commands."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "control_calibration", Path(__file__).parents[1] / "scripts/analyze-control-calibration.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def tick(start, command, speed_before, speed_after):
    return {
        "start": start, "end": start + .005, "command": [command, 0],
        "position_before": [0, 0], "position_after": [.005 * (speed_before + speed_after) / 2, 0],
        "velocity_before": [speed_before, 0], "velocity_after": [speed_after, 0],
    }


def test_initial_idle_is_not_a_braking_trial():
    assert MODULE.braking_episodes([tick(0, 0, .02, .019)]) == []


def test_interrupted_braking_is_censored_and_retained():
    episodes = MODULE.braking_episodes([
        tick(0, .4, .08, .08), tick(.005, 0, .08, .07),
        tick(.010, .4, .07, .08), tick(.015, 0, .08, .02),
    ])
    assert len(episodes) == 2
    assert episodes[0]["right_censored_at_speed_threshold"]
    assert episodes[0]["ended_by"] == "nonzero_target_resumed"
    assert episodes[0]["time_to_threshold_s"] is None
    assert not episodes[1]["right_censored_at_speed_threshold"]
    assert episodes[1]["time_to_threshold_s"] == pytest.approx(.005)


def test_braking_distance_is_path_length_and_dead_zone_counts_as_zero():
    episodes = MODULE.braking_episodes([
        tick(0, .4, .05, .05), tick(.005, .04, .05, .04), tick(.010, 0, .04, .02),
    ])
    assert episodes[0]["distance_to_threshold_m"] == pytest.approx(.000225 + .00015)


def test_timeline_gap_cannot_be_assumed_stationary():
    with pytest.raises(ValueError, match="gap"):
        MODULE.braking_episodes([tick(0, .4, .08, .08), tick(.010, 0, .08, .02)])


def test_preexisting_conditional_tail_bound_at_low_and_high_speed():
    t, d = MODULE.conditional_stop_bound(.06)
    assert t == pytest.approx(.35 * MODULE.math.log(2))
    assert d == pytest.approx(.0105)
    t, d = MODULE.conditional_stop_bound(.35)
    assert t == pytest.approx(.35 + .35 * MODULE.math.log(.175 / .03))
    assert d == pytest.approx((.35**2 - .175**2) / 1. + .35 * (.175 - .03))
