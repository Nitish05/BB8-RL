import copy
import json
import math
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.braking_prediction import ConditionalBrakingRegion
from bb8_rl.occluded_control import (
    CommandBelief,
    CommandDynamics,
    OccludedController,
    OcclusionParameters,
)


def snapshot(timestamp=0.0, speed=0.2, velocity_radius=0.03, radius=0.025):
    return {
        "timestamp": timestamp,
        "xy": np.array([0.4, -0.2]),
        "velocity": np.array([speed, 0.0]),
        "position_radius": radius,
        "velocity_radius": velocity_radius,
        "map_version": "map-1",
        "calibration_version": "cal-1",
    }


def intervals(start, end, tick=0.005):
    count = round((end - start) / tick)
    bounds = np.linspace(start, end, count + 1)
    return [
        {"start": float(a), "end": float(b), "command": [0.0, 0.0]}
        for a, b in pairwise(bounds)
    ]


def update(region, before=None, end=0.05, **overrides):
    before = snapshot() if before is None else before
    arguments = {
        "previous_snapshot": before,
        "timestamp": end,
        "acknowledged_intervals": intervals(before["timestamp"], end),
        "endpoint_command": [0.0, 0.0],
        "map_version": "map-1",
        "calibration_version": "cal-1",
    }
    arguments.update(overrides)
    return region.update(**arguments)


def continue_to(region, end, *, tick=0.005):
    cursor = region.record["anchor_time"] + region.record["elapsed_seconds"]
    while cursor < end - 1e-12:
        nxt = min(end, cursor + 0.05)
        result = update(
            region,
            snapshot(cursor, speed=0.0, velocity_radius=0.0, radius=0.001),
            nxt,
            acknowledged_intervals=intervals(cursor, nxt, tick),
        )
        assert result["valid"]
        cursor = nxt
    return region.record


@pytest.mark.parametrize("speed", [0.0, 0.08, 0.175, 0.3, 0.6])
def test_analytic_tail_fixed_anchor_and_finite_limit(speed):
    region = ConditionalBrakingRegion()
    before = snapshot(speed=speed, velocity_radius=0.0)
    initial = update(region, before)
    assert initial["valid"]
    result = continue_to(region, 2.0)
    a, tau, t = 0.5, 0.35, 2.0
    threshold = a * tau
    saturated = max(0.0, (speed - threshold) / a)
    tail_start = min(speed, threshold)
    prior_distance = (speed**2 - tail_start**2) / (2 * a)
    distance = prior_distance + tail_start * tau * (
        1 - math.exp(-(t - saturated) / tau)
    )
    expected_limit = (
        before["position_radius"] + prior_distance + tail_start * tau + speed * 0.005
    )
    assert result["radius_m"] == pytest.approx(
        before["position_radius"] + distance + speed * 0.005
    )
    assert result["radius_limit_m"] == pytest.approx(expected_limit)
    assert result["speed_upper_m_s"] == pytest.approx(
        tail_start * math.exp(-(t - 0.005 - saturated) / tau)
    )
    assert result["anchor_time"] == 0.0
    assert result["anchor_xy"] == [0.4, -0.2]
    assert result["radius_limit_m"] == initial["radius_limit_m"]
    assert result["status"] == "settling_conditional"
    assert not result["motion_authority"] and not result["observed"]


def test_saturated_phase_and_phase_boundary_are_continuous():
    region = ConditionalBrakingRegion()
    result = update(region, snapshot(speed=0.4, velocity_radius=0.0), end=0.05)
    assert result["radius_m"] == pytest.approx(
        0.025 + 0.4 * 0.05 - 0.25 * 0.05**2 + 0.4 * 0.005
    )
    assert result["speed_upper_m_s"] == pytest.approx(0.4 - 0.5 * 0.045)
    ts = (0.4 - 0.175) / 0.5
    at = region._tail(0.4, ts)
    left, right = region._tail(0.4, ts - 1e-8), region._tail(0.4, ts + 1e-8)
    np.testing.assert_allclose(left, at, atol=1e-8)
    np.testing.assert_allclose(right, at, atol=1e-8)


@pytest.mark.parametrize("tick", [0.005, 0.0025])
def test_independent_discrete_braking_paths_fit_padded_disk(tick):
    # Forward Euler braking is at least as fast as the assumed exact tail;
    # left-endpoint position integration is intentionally the slower convention.
    for heading in (0.0, 0.73, 2.2):
        region = ConditionalBrakingRegion()
        state = snapshot(speed=0.42, velocity_radius=0.06, radius=0.04)
        state["velocity"] = 0.42 * np.array([math.cos(heading), math.sin(heading)])
        direction = np.array([math.cos(heading), math.sin(heading)])
        position = state["xy"] + 0.04 * direction
        speed, cursor, previous_radius = 0.48, 0.0, 0.0
        for frame in range(1, 41):
            end = frame * 0.05
            timeline = intervals(cursor, end, tick)
            before = dict(state, timestamp=cursor)
            result = update(region, before, end, acknowledged_intervals=timeline)
            for record in timeline:
                dt = record["end"] - record["start"]
                position += speed * dt * direction
                speed = max(0, speed - min(0.5, speed / 0.35) * dt)
            assert np.linalg.norm(position - state["xy"]) <= result["radius_m"] + 1e-12
            assert speed <= result["speed_upper_m_s"] + 1e-12
            assert previous_radius <= result["radius_m"] <= result["radius_limit_m"]
            previous_radius, cursor = result["radius_m"], end


def test_pre_interval_snapshot_is_copied_and_uncertainty_not_erased():
    before = snapshot(speed=0.1, velocity_radius=0.2)
    saved = copy.deepcopy(before)
    region = ConditionalBrakingRegion()
    result = update(region, before)
    assert result["anchor_speed_upper_m_s"] == pytest.approx(0.3)
    np.testing.assert_array_equal(before["xy"], saved["xy"])
    np.testing.assert_array_equal(before["velocity"], saved["velocity"])
    before["xy"][:] = 999
    before["velocity"][:] = 999
    result["anchor_xy"][0] = -999
    assert region.record["anchor_xy"] == [0.4, -0.2]
    smaller = update(
        ConditionalBrakingRegion(), snapshot(speed=0.1, velocity_radius=0.0)
    )
    assert region.record["radius_limit_m"] > smaller["radius_limit_m"]
    json.dumps(region.record, allow_nan=False)


@pytest.mark.parametrize(
    "problem",
    ["gap", "overlap", "missing_end", "long_tick", "nonzero", "nan", "extra_truth"],
)
def test_bad_acknowledged_timeline_invalidates_existing_region(problem):
    region = ConditionalBrakingRegion()
    assert update(region)["valid"]
    timeline = intervals(0.05, 0.1)
    if problem == "gap":
        timeline[3]["start"] += 0.001
    elif problem == "overlap":
        timeline[3]["start"] -= 0.001
    elif problem == "missing_end":
        timeline.pop()
    elif problem == "long_tick":
        timeline = [{"start": 0.05, "end": 0.1, "command": [0, 0]}]
    elif problem == "nonzero":
        timeline[4]["command"] = [1e-8, 0]
    elif problem == "nan":
        timeline[4]["end"] = float("nan")
    else:
        timeline[4]["truth_velocity"] = [0, 0]
    result = update(region, snapshot(0.05), 0.1, acknowledged_intervals=timeline)
    assert not result["valid"]
    assert result["anchor_xy"] is None and result["radius_limit_m"] is None
    assert not result["motion_authority"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "override",
    [
        {"endpoint_command": [0.2, 0]},
        {"endpoint_command": [0, float("inf")]},
        {"pending_nonzero": True},
        {"inputs_valid": False},
        {"map_version": "map-2"},
        {"calibration_version": "cal-2"},
        {"acknowledged_intervals": []},
    ],
)
def test_endpoint_pending_context_and_missing_history_fail_closed(override):
    region = ConditionalBrakingRegion()
    result = update(region, **override)
    assert not result["valid"] and not result["observed"]
    assert result["radius_m"] is None


def test_invalid_snapshot_and_post_prediction_timestamp_cannot_anchor():
    for key, value in (
        ("position_radius", math.inf),
        ("velocity_radius", -0.1),
        ("timestamp", 0.05),
        ("xy", [0, math.nan]),
    ):
        before = snapshot()
        before[key] = value
        assert not update(ConditionalBrakingRegion(), before)["valid"]


def test_invalidation_reanchor_requires_new_complete_interval_and_valid_snapshot():
    region = ConditionalBrakingRegion()
    assert update(region)["valid"]
    assert not region.invalidate("new_nonzero_request")["valid"]
    assert not update(region)["valid"]  # Consumed timeline is never made fresh.
    assert not update(region, snapshot(0.05), 0.1, inputs_valid=False)["valid"]
    result = update(region, snapshot(0.1, speed=0.3), 0.15)
    assert result["valid"] and result["anchor_time"] == 0.1
    changed = snapshot(0.15)
    changed["map_version"] = "map-2"
    assert not update(region, changed, 0.2, map_version="map-2")["valid"]
    changed["timestamp"] = 0.2
    assert update(region, changed, 0.25, map_version="map-2")["valid"]


def test_skipping_a_capture_breaks_anchor_continuity():
    region = ConditionalBrakingRegion()
    update(region)
    result = update(region, snapshot(0.1), 0.15)
    assert not result["valid"] and result["reason"] == "timeline_gap"


def test_240_seconds_bounded_advisory_leaves_raw_predictor_unchanged_and_cannot_bound_disturbance():
    dynamics, parameters = CommandDynamics(), OcclusionParameters()
    source, reference = (
        CommandBelief(dynamics, parameters),
        CommandBelief(dynamics, parameters),
    )
    measurement = SimpleNamespace(
        status="visible", xy=np.array([0.0, 0.0]), covariance=np.eye(2) * 0.005**2
    )
    for belief in (source, reference):
        belief.predict(0.0, [0, 0], applied_command_intervals=[])
        belief.observe(measurement)
    region = ConditionalBrakingRegion()
    for index in range(1, 4801):
        start, end = (index - 1) * 0.05, index * 0.05
        timeline = intervals(start, end)
        before = snapshot(start)
        before.update(
            xy=source.xy,
            velocity=source.velocity,
            position_radius=source.position_radius,
            velocity_radius=source.velocity_radius,
        )
        result = update(region, before, end, acknowledged_intervals=timeline)
        assert (
            result["valid"]
            and not result["motion_authority"]
            and not result["observed"]
        )
        for belief in (source, reference):
            belief.predict(end, [0, 0], applied_command_intervals=timeline)
    np.testing.assert_array_equal(source.xy, reference.xy)
    np.testing.assert_array_equal(source.velocity, reference.velocity)
    assert source.position_radius == reference.position_radius
    assert source.velocity_radius == reference.velocity_radius
    assert source.position_radius > 7.0
    assert result["radius_m"] <= result["radius_limit_m"] < 0.2
    # A sustained q-force balanced by drag produces q*tau steady drift. It is
    # permitted by the authoritative model and violates the advisory assumption.
    adversarial_displacement = 0.12 * 0.25 * (240 - 0.25 * (1 - math.exp(-240 / 0.25)))
    assert adversarial_displacement > result["radius_limit_m"]


def test_advisory_cannot_change_controller_actions_guards_or_arrival():
    def controller():
        return OccludedController(
            lambda vector: np.array([0.4, 0.0]),
            [0.6, 0.0],
            SimpleNamespace(route=lambda start, goal: np.array([start, goal])),
            lambda start, end, radius: radius < 0.7,
            map_version="map-1",
            calibration_version="cal-1",
        )

    decorated, reference = controller(), controller()
    region, statuses = ConditionalBrakingRegion(), set()
    for index in range(75):
        time = index * 0.05
        timeline = [] if not index else intervals((index - 1) * 0.05, time)
        belief = decorated.belief
        if belief.xy is not None:
            before = snapshot(belief.timestamp)
            before.update(
                xy=belief.xy,
                velocity=belief.velocity,
                position_radius=belief.position_radius,
                velocity_radius=belief.velocity_radius,
            )
            update(
                region,
                before,
                time,
                acknowledged_intervals=timeline,
                pending_nonzero=any(
                    end >= time and speed > 0
                    for _, end, speed in decorated.issued_history
                ),
            )
        visible = index < 35
        measurement = SimpleNamespace(
            timestamp=time,
            status="visible" if visible else "missing",
            xy=np.zeros(2) if visible else None,
            covariance=np.eye(2) * 0.002**2 if visible else None,
        )
        actions = [
            control.action(
                time,
                measurement,
                applied_command=[0, 0],
                applied_command_intervals=timeline,
                map_version="map-1",
                calibration_version="cal-1",
            )
            for control in (decorated, reference)
        ]
        if np.any(actions[0]):
            region.invalidate("new_nonzero_request")
        np.testing.assert_array_equal(*actions)
        assert decorated.diagnostics == reference.diagnostics
        assert not region.record["motion_authority"]
        statuses.add(decorated.status)
    assert "tracking_visible" in statuses
    assert "occlusion_timeout" in statuses
    assert not decorated.arrived


def test_disabled_and_invalid_configuration():
    disabled = ConditionalBrakingRegion(enabled=False)
    assert not update(disabled)["enabled"]
    assert disabled.record["reason"] == "disabled"
    for settings in (
        {"max_tick_seconds": 0},
        {"response_time_upper": math.nan},
        {"braking_acceleration_lower": -1},
        {"max_tick_seconds": 0.2},
    ):
        with pytest.raises(ValueError):
            ConditionalBrakingRegion(**settings)
