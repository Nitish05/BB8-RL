"""Exploration requires different verified destinations, not just more trips."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "agency_navigation_audit",
    Path(__file__).resolve().parents[1] / "scripts/benchmark-agency-navigation.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

A, B, C = (0.5, 1.5), (0.0, 1.0), (-0.5, 0.5)


def arrivals(goals, *, invalid=()):
    return [
        {
            "time": index * 8.0,
            "goal": list(goal),
            "event_id": f"different-intention-{index}",
            "independently_valid": index not in invalid,
        }
        for index, goal in enumerate(goals)
    ]


def test_two_point_shuttle_cannot_pass_three_destination_exploration():
    result = AUDIT.score_exploration(arrivals([A, B, A]), required_distinct=3)
    assert result["valid_arrival_episodes"] == 3
    assert result["distinct_valid_destinations"] == 2
    assert len(result["repeated_successful_destinations"]) == 1
    assert result["repeated_successful_destinations"][0]["goal"] == list(A)
    assert not result["exploration_passed"]


def test_three_distinct_verified_destinations_pass():
    result = AUDIT.score_exploration(arrivals([A, B, C]), required_distinct=3)
    assert result["valid_arrival_episodes"] == 3
    assert result["distinct_valid_destinations"] == 3
    assert result["repeated_successful_destinations"] == []
    assert result["exploration_passed"]


def test_revisit_fails_even_when_distinct_destination_target_is_later_reached():
    result = AUDIT.score_exploration(arrivals([A, B, A, C]), required_distinct=3)
    assert result["distinct_valid_destinations"] == 3
    assert not result["exploration_passed"]


def test_unverified_reported_arrival_cannot_supply_missing_destination():
    result = AUDIT.score_exploration(
        arrivals([A, B, C], invalid=(2,)), required_distinct=3
    )
    assert result["valid_arrival_episodes"] == 2
    assert result["distinct_valid_destinations"] == 2
    assert not result["exploration_passed"]


@pytest.mark.parametrize("goals", [[], [A], [A, B]])
def test_incomplete_exploration_fails(goals):
    result = AUDIT.score_exploration(arrivals(goals), required_distinct=3)
    assert not result["exploration_passed"]


def test_invalid_attempt_does_not_count_as_successful_revisit():
    result = AUDIT.score_exploration(
        arrivals([A, A, B, C], invalid=(0,)), required_distinct=3
    )
    assert result["valid_arrival_episodes"] == 3
    assert result["repeated_successful_destinations"] == []
    assert result["exploration_passed"]
