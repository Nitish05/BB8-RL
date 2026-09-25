import math

import pytest

from bb8_rl.interaction_environment import (
    DEFAULT_STATIONS,
    SOURCE,
    InteractionEnvironment,
)
from bb8_rl.purpose import Station

A, B = DEFAULT_STATIONS


def sample(environment, now, station=B, request_id="request-1", velocity=(0, 0)):
    return environment.advance(
        now=now,
        position=station.xy,
        velocity=velocity,
        request={"request_id": request_id, "station_id": station.id},
    )


def test_idle_time_does_not_consume_resource_or_interact():
    environment = InteractionEnvironment()
    first = environment.advance(now=0, position=B.xy, velocity=(0, 0))
    later = environment.advance(now=100_000, position=B.xy, velocity=(0, 0))
    assert first == {"resource": 0.35, "source": SOURCE, "timestamp": 0.0}
    assert later["resource"] == 0.35
    assert "outcome" not in later


def test_request_requires_full_dwell_then_delivers_one_effect():
    environment = InteractionEnvironment()
    assert "outcome" not in sample(environment, 0)
    assert "outcome" not in sample(environment, 0.99)
    arrived = sample(environment, 1)
    assert arrived["resource"] == pytest.approx(0.8)
    assert arrived["outcome"] == {
        "request_id": "request-1",
        "station_id": B.id,
        "before": 0.35,
        "after": pytest.approx(0.8),
        "timestamp": 1.0,
        "source": SOURCE,
    }
    assert sample(environment, 1) == arrived
    replay = sample(environment, 50)
    assert replay["outcome"] == arrived["outcome"]
    assert replay["resource"] == arrived["resource"]
    assert replay["timestamp"] == 50


def test_zero_effect_station_still_reports_a_completed_interaction():
    environment = InteractionEnvironment()
    sample(environment, 0, A)
    outcome = sample(environment, 1, A)["outcome"]
    assert outcome["before"] == outcome["after"] == 0.35
    assert outcome["station_id"] == A.id


def test_actual_travel_cost_without_time_or_command_drain():
    environment = InteractionEnvironment(initial_resource=0.5)
    environment.advance(now=0, position=(0, 0), velocity=(0, 0))
    moved = environment.advance(now=10, position=(3, 4), velocity=(0, 0))
    assert moved["resource"] == pytest.approx(0.5 - 5 * 0.025)
    still = environment.advance(now=20, position=(3, 4), velocity=(0.02, 0))
    assert still["resource"] == moved["resource"]
    returned = environment.advance(now=30, position=(0, 0), velocity=(0, 0))
    assert returned["resource"] == pytest.approx(0.25)


@pytest.mark.parametrize("interruption", ["none", "outside", "fast", "changed"])
def test_interruption_restarts_the_full_dwell(interruption):
    environment = InteractionEnvironment()
    sample(environment, 0)
    request = {"request_id": "request-1", "station_id": B.id}
    point, velocity = B.xy, (0, 0)
    if interruption == "none":
        request = None
    elif interruption == "outside":
        point = (B.xy[0] + 0.2, B.xy[1])
    elif interruption == "fast":
        velocity = (0.04, 0)
    else:
        request["request_id"] = "request-2"
    assert "outcome" not in environment.advance(
        now=0.5, position=point, velocity=velocity, request=request
    )
    assert "outcome" not in sample(environment, 0.75)
    assert "outcome" not in sample(environment, 1.5)
    assert "outcome" in sample(environment, 1.75)


def test_fast_displacement_inside_zone_cannot_be_a_stationary_dwell():
    environment = InteractionEnvironment()
    sample(environment, 0)
    shifted = Station(B.id, (B.xy[0] + 0.1, B.xy[1]), B.label)
    assert "outcome" not in sample(environment, 1, shifted)
    assert "outcome" in sample(environment, 2, shifted)


def test_completed_replay_away_from_station_returns_original_receipt_only():
    environment = InteractionEnvironment()
    sample(environment, 0)
    result = sample(environment, 1)
    # Mutating a caller-owned telemetry receipt must not alter future delivery.
    result["outcome"]["after"] = 999
    moved = environment.advance(
        now=2,
        position=(B.xy[0] + 1, B.xy[1]),
        velocity=(1, 0),
        request={"request_id": "request-1", "station_id": B.id},
    )
    assert moved["resource"] == pytest.approx(0.775)
    assert moved["outcome"]["after"] == pytest.approx(0.8)
    assert moved["outcome"]["timestamp"] == 1


def test_request_id_binding_survives_cancellation_and_completion():
    environment = InteractionEnvironment()
    sample(environment, 0)
    environment.advance(now=0.25, position=B.xy, velocity=(0, 0))
    with pytest.raises(ValueError, match="different station"):
        sample(environment, 0.5, A)
    sample(environment, 0.5)
    sample(environment, 1.5)
    with pytest.raises(ValueError, match="different station"):
        sample(environment, 2, A)


def test_old_receipts_are_not_evicted_after_many_new_requests():
    environment = InteractionEnvironment(effects={A.id: 0, B.id: 0.01})
    sample(environment, 0)
    original = sample(environment, 1)["outcome"]
    for index in range(150):
        sample(environment, 2 + index * 2, request_id=f"later-{index}")
        sample(environment, 3 + index * 2, request_id=f"later-{index}")
    replay = sample(environment, 305)
    assert replay["outcome"] == original
    assert replay["resource"] == 1


def test_injected_effects_and_resource_saturation():
    environment = InteractionEnvironment(initial_resource=0.9, effects={A.id: 0.5})
    sample(environment, 0, A)
    assert sample(environment, 1, A)["resource"] == 1
    # An explicitly supplied effect mapping gives other stations zero effect.
    sample(environment, 2, B, request_id="other")
    result = sample(environment, 3, B, request_id="other")
    assert result["outcome"]["before"] == result["outcome"]["after"]
    drained = environment.advance(now=4, position=(100, 100), velocity=(0, 0))
    assert drained["resource"] == 0


@pytest.mark.parametrize(
    "invalid_request",
    [
        {},
        {"request_id": "x"},
        {"request_id": "x", "station_id": B.id, "effect": 1},
        {"request_id": "x", "station_id": "missing"},
        {"request_id": "", "station_id": B.id},
        {"request_id": True, "station_id": B.id},
    ],
)
def test_invalid_request_rejects_before_mutating_resource_or_dwell(invalid_request):
    environment = InteractionEnvironment()
    sample(environment, 0)
    with pytest.raises(ValueError):
        environment.advance(
            now=0.5, position=(100, 100), velocity=(0, 0), request=invalid_request
        )
    result = sample(environment, 1)
    assert result["resource"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("now", -1),
        ("now", math.nan),
        ("now", math.inf),
        ("now", 10**400),
        ("now", True),
        ("position", [0, math.inf]),
        ("position", [0, 0, 0]),
        ("position", "00"),
        ("velocity", [math.nan, 0]),
        ("velocity", [False, 0]),
    ],
)
def test_invalid_mechanics_reject_without_state_change(field, value):
    environment = InteractionEnvironment()
    valid = {"now": 0, "position": B.xy, "velocity": (0, 0)}
    with pytest.raises((TypeError, ValueError)):
        environment.advance(**{**valid, field: value})
    assert environment.advance(**valid)["resource"] == 0.35


def test_time_reversal_and_same_time_conflicting_mechanics_are_rejected():
    environment = InteractionEnvironment()
    sample(environment, 1)
    with pytest.raises(ValueError, match="nondecreasing"):
        sample(environment, 0.5)
    with pytest.raises(ValueError, match="conflicting mechanics"):
        environment.advance(now=1, position=(0, 0), velocity=(0, 0))
    assert sample(environment, 2)["resource"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_resource": -0.1},
        {"initial_resource": 1.1},
        {"initial_resource": math.nan},
        {"effects": {"missing": 0.1}},
        {"effects": {A.id: math.inf}},
        {"effects": {A.id: 2}},
        {"stations": (A, A)},
        {"stations": ({"id": "a"},)},
        {"stations": None},
    ],
)
def test_invalid_environment_configuration(kwargs):
    with pytest.raises(ValueError):
        InteractionEnvironment(**kwargs)


def test_telemetry_contains_no_privileged_geometry_or_effect_parameters():
    environment = InteractionEnvironment()
    initial = sample(environment, 0)
    complete = sample(environment, 1)
    assert set(initial) == {"resource", "source", "timestamp"}
    assert set(complete) == {"resource", "source", "timestamp", "outcome"}
    assert set(complete["outcome"]) == {
        "request_id",
        "station_id",
        "before",
        "after",
        "timestamp",
        "source",
    }
