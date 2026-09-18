import json
from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.occluded_control import (
    CommandBelief,
    CommandDynamics,
    OccludedController,
    OcclusionParameters,
)


def measurement(timestamp, xy=(0, 0), sigma=0.002, status="visible"):
    return SimpleNamespace(
        timestamp=timestamp,
        xy=None if status != "visible" else np.asarray(xy, float),
        covariance=None if status != "visible" else np.eye(2) * sigma**2,
        status=status,
    )


class StraightGrid:
    def route(self, start, goal):
        return np.array([start, goal])


class Corridor:
    def __init__(self, half_width=0.7, unknown_x=None):
        self.half_width, self.unknown_x = half_width, unknown_x
        self.calls = []

    def __call__(self, a, b, radius):
        self.calls.append((a.copy(), b.copy(), radius))
        return bool(
            max(abs(a[1]), abs(b[1])) + radius < self.half_width
            and (self.unknown_x is None or max(a[0], b[0]) + radius < self.unknown_x)
        )


def controller(**kwargs):
    certificate = kwargs.pop("segment_free", Corridor())
    return OccludedController(
        lambda vector: np.array([0.8, 0.0]),
        kwargs.pop("goal_xy", (0.6, 0)),
        StraightGrid(),
        certificate,
        map_version="scan-v1",
        calibration_version="camera-v1",
        **kwargs,
    )


def step(control, timestamp, item=None, applied=(0, 0), **kwargs):
    return control.action(
        timestamp,
        measurement(timestamp) if item is None else item,
        applied_command=applied,
        map_version=kwargs.pop("map_version", "scan-v1"),
        calibration_version=kwargs.pop("calibration_version", "camera-v1"),
        **kwargs,
    )


def warm(control, count=30):
    for index in range(count):
        step(control, index * 0.05)
    return (count - 1) * 0.05


def test_hidden_motion_in_proven_corridor_uses_applied_not_issued_command():
    control = controller()
    last = warm(control)
    before = control.belief.xy.copy()
    action = step(
        control,
        last + 0.05,
        measurement(last + 0.05, status="missing"),
        applied=(0.4, 0),
    )
    assert action[0] > 0
    assert control.status == "tracking_hidden"
    assert control.belief.xy[0] > before[0]
    assert control.belief.last_visual_time == last
    assert control.diagnostics["pose_source"] == "predicted"
    assert control.diagnostics["motion_authorized"]
    assert (
        np.linalg.norm(control.dynamics.target(action))
        <= control.parameters.speed_cap + 1e-7
    )
    assert (
        control.envelope_radius
        > control.parameters.robot_radius + control.belief.position_radius
    )
    json.dumps(control.diagnostics, allow_nan=False)

    stopped = controller()
    last = warm(stopped)
    # Proposed outputs in warm() were nonzero, but applied requests were zero.
    step(
        stopped, last + 0.05, measurement(last + 0.05, status="missing"), applied=(0, 0)
    )
    np.testing.assert_array_equal(stopped.belief.xy, np.zeros(2))


def test_unknown_inside_stopping_radius_brakes_even_when_route_is_clear():
    corridor = Corridor(half_width=0.10)
    control, last = stationary_ready(half_width=0.10)
    control.segment_free = corridor
    control.policy = lambda vector: np.array([0.8, 0])
    step(control, last + 0.05)
    assert control.status == "stopping_envelope_not_certified"
    assert np.all(control.issued_action == 0)
    assert not control.diagnostics["motion_authorized"]
    assert any(radius < 0.1 for _, _, radius in corridor.calls)
    assert control.envelope_radius > 0.10


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"memory_valid": False}, "memory_or_calibration_invalid"),
        ({"map_version": "scan-v2"}, "memory_or_calibration_invalid"),
        ({"calibration_version": "camera-v2"}, "memory_or_calibration_invalid"),
        ({"now": 0.8}, "stale_or_invalid_frame"),
    ],
)
def test_memory_versions_and_stale_frames_fail_closed(change, reason):
    control = controller()
    last = warm(control)
    result = step(control, last + 0.05, **change)
    assert np.all(result == 0)
    assert control.status == reason
    assert control.belief.last_visual_time == last
    assert not control.belief.measured
    assert control.diagnostics["pose_source"] == "stale"
    assert not control.diagnostics["state_is_current"]
    assert not control.diagnostics["input_accepted"]


def test_long_occlusion_and_excess_uncertainty_brake():
    control = controller(parameters=OcclusionParameters(max_occlusion_seconds=0.15))
    last = warm(control)
    for dt in (0.05, 0.10, 0.15, 0.20):
        step(control, last + dt, measurement(last + dt, status="missing"))
    assert control.status == "occlusion_timeout"
    assert np.all(control.issued_action == 0)
    control = controller(parameters=OcclusionParameters(max_position_radius=0.02))
    last = warm(control)
    for dt in np.arange(0.05, 0.6, 0.05):
        step(control, last + dt, measurement(last + dt, status="missing"))
        if control.status == "uncertain_prediction":
            break
    assert control.status == "uncertain_prediction"
    assert np.all(control.issued_action == 0)


def test_reacquisition_gate_rejects_identity_jump_then_accepts_consistent_rgb():
    control = controller()
    last = warm(control)
    step(control, last + 0.05, measurement(last + 0.05, status="missing"))
    step(control, last + 0.10, measurement(last + 0.10, xy=(0.5, 0)))
    assert control.status == "reacquisition_rejected"
    assert control.belief.last_visual_time == last
    assert not control.belief.measured
    assert np.all(control.issued_action == 0)
    step(control, last + 0.15, measurement(last + 0.15, xy=(0.002, 0)))
    assert control.belief.measured
    assert control.belief.last_visual_time == pytest.approx(last + 0.15)
    assert control.status == "tracking_visible"


def test_arrival_needs_unbroken_visible_dwell_never_hidden_prediction():
    control = controller(goal_xy=(0.03, 0))
    for index in range(60):
        last = index * 0.05
        step(control, last)
        if control.velocity_initialized:
            break
    assert control.velocity_initialized
    assert not control.arrived
    step(control, last + 0.05, measurement(last + 0.05, status="missing"))
    assert not control.arrived
    assert control.visible_since is None
    # The hidden policy request may still be pending after visibility returns.
    for index in range(8):
        step(control, last + 0.1 + index * 0.05)
        if control.visible_since is not None:
            break
    start = control.visible_since
    assert start is not None
    for index in range(1, 10):
        step(control, start + index * 0.05)
        assert not control.arrived
    step(control, start + 0.5)
    assert control.arrived
    assert control.status == "arrived_visible"
    assert control.belief.measured
    assert np.all(control.issued_action == 0)
    step(control, start + 0.55, measurement(start + 0.55, status="missing"))
    assert not control.arrived


def test_no_route_created_during_missing_frames_and_ambiguity_brakes():
    control = controller()
    for index in range(4):
        step(control, index * 0.05, measurement(index * 0.05, status="missing"))
    assert control.route is None
    assert np.all(control.issued_action == 0)
    control = controller()
    last = warm(control)
    step(control, last + 0.05, measurement(last + 0.05, status="ambiguous"))
    assert control.status == "ambiguous"
    assert not control.belief.measured
    assert np.all(control.issued_action == 0)


def test_repeated_frame_and_large_clock_gap_never_refresh_rgb_history():
    control = controller()
    last = warm(control)
    step(control, last)
    assert control.status == "repeated_frame"
    assert not control.belief.measured
    step(control, last + 0.5)
    assert control.status == "control_interval_exceeded"
    assert control.belief.last_visual_time == last
    assert control.route is None


def test_prediction_radius_grows_with_command_transition_uncertainty():
    p = OcclusionParameters()
    with_latency = CommandBelief(CommandDynamics(), p)
    no_latency = CommandBelief(CommandDynamics(latency_mismatch=0), p)
    for belief in (with_latency, no_latency):
        belief.predict(0, (0, 0))
        belief.observe(measurement(0))
        belief.predict(0.05, (0.4, 0))
    np.testing.assert_allclose(with_latency.xy, no_latency.xy)
    assert with_latency.position_radius > no_latency.position_radius
    assert with_latency.velocity_radius > no_latency.velocity_radius


def test_known_zero_command_reduces_unknown_initial_speed_with_no_truth_velocity():
    control = controller()
    initial = None
    for index in range(20):
        timestamp = index * 0.05
        step(control, timestamp, measurement(timestamp, sigma=0.015))
        if index == 0:
            initial = control.belief.velocity_radius
    assert initial == 0.35
    assert control.belief.velocity_radius < 0.1
    assert control.belief.position_radius == pytest.approx(0.045)
    np.testing.assert_array_equal(control.belief.velocity, np.zeros(2))


def test_invalid_applied_command_and_missing_pose_payload_brake():
    control = controller()
    last = warm(control)
    step(control, last + 0.05, applied=(np.nan, 0))
    assert control.status == "invalid_applied_command"
    assert np.all(control.issued_action == 0)
    malformed = measurement(last + 0.05, status="missing")
    malformed.xy = np.array([1, 1])
    step(control, last + 0.05, malformed)
    assert control.status == "invalid_missing_measurement"


def stationary_ready(*, half_width=0.7):
    control = controller(segment_free=Corridor(half_width=half_width))
    control.policy = lambda vector: np.zeros(2)
    last = warm(control, count=30)
    return control, last


def test_speed_backoff_moves_in_narrow_corridor_without_changing_heading():
    control, last = stationary_ready(half_width=0.13)
    control.policy = lambda vector: np.array([0.8, 0.0])
    result = step(control, last + 0.05)
    assert result[0] > 0 and result[1] == 0
    assert 0 < control.speed_scale < 1
    assert control.envelope_trials[0]["certified"] is False
    assert control.envelope_trials[-1]["certified"] is True
    assert control.envelope_radius < 0.13
    # Only the actually returned request enters history, not rejected trials.
    assert control.issued_history[-1][2] == pytest.approx(0.15 * control.speed_scale)


def test_unacknowledged_target_is_retained_for_its_entire_command_lifetime():
    control, last = stationary_ready()
    control.policy = lambda vector: np.array([0.8, 0.0])
    issued_at = last + 0.05
    step(control, issued_at)
    control.segment_free = Corridor(half_width=0.13)
    control.policy = lambda vector: np.array([0.15, 0.0])
    # The command is never acknowledged, but it can still be pending. Do not
    # infer cancellation or expire it at the shorter 50 ms latency window.
    for age in (0.05, 0.10, 0.15):
        result = step(control, issued_at + age, applied=(0, 0))
        assert np.all(result == 0)
        assert control.envelope_trials[-1][
            "retained_target_speed_m_s"
        ] == pytest.approx(0.15)
        assert control.status == "stopping_envelope_not_certified"
    result = step(control, issued_at + 0.20, applied=(0, 0))
    assert result[0] > 0
    assert control.envelope_trials[-1]["retained_target_speed_m_s"] == 0


def test_equal_acknowledgement_does_not_remove_possibly_queued_duplicate():
    control, last = stationary_ready()
    control.policy = lambda vector: np.array([0.8, 0.0])
    issued = step(control, last + 0.05)
    control.policy = lambda vector: np.zeros(2)
    step(control, last + 0.10, applied=issued)
    assert control.envelope_trials[0]["retained_target_speed_m_s"] == pytest.approx(
        0.15
    )
    assert any(speed > 0.14 for _, _, speed in control.issued_history)


def test_commands_expiring_during_frame_age_still_constrain_reaction():
    control, last = stationary_ready()
    control.policy = lambda vector: np.array([0.8, 0.0])
    issued_at = last + 0.05
    step(control, issued_at)
    control.policy = lambda vector: np.zeros(2)
    step(control, issued_at + 0.10, now=issued_at + 0.19)
    # Expiry=issued_at+.15 is before now, but after the state capture at +.10.
    assert control.envelope_trials[0]["retained_target_speed_m_s"] == pytest.approx(
        0.15
    )


def test_unknown_history_uses_universal_target_bound():
    control = controller()
    warm(control, count=3)
    assert not control.command_history_complete
    assert control.status == "warming_up"
    assert not control.velocity_initialized
    control, last = stationary_ready()
    step(control, last + 0.05, command_history_valid=False)
    assert not control.command_history_complete
    assert control.envelope_trials[0]["target_speed_bound_m_s"] == 0.35


def test_backoff_cannot_erase_current_momentum():
    control, last = stationary_ready(half_width=0.15)
    control.belief.velocity = np.array([0.22, 0.0])
    control.belief.velocity_radius = 0.02
    control.policy = lambda vector: np.array([0.8, 0.0])
    result = step(control, last + 0.05, measurement(last + 0.05, status="missing"))
    assert np.all(result == 0)
    assert control.status == "stopping_envelope_not_certified"
    assert len(control.envelope_trials) == 4
    for trial in control.envelope_trials:
        assert trial["reaction_speed_bound_m_s"] >= trial["initial_speed_bound_m_s"]
        assert not trial["certified"]


def test_target_speed_scaling_preserves_diagonal_heading_and_dead_zone():
    control = controller()
    command = np.array([0.24, -0.32])
    original = control.dynamics.target(command)
    for factor in (1, 0.75, 0.5, 0.25):
        scaled = control._scaled_command(command, factor)
        np.testing.assert_allclose(
            control.dynamics.target(scaled), original * factor, atol=1e-12
        )
    np.testing.assert_array_equal(control._scaled_command([0.01, 0], 0.5), [0, 0])


def test_reaction_bound_encloses_switched_targets_and_acceleration_mismatch():
    control, last = stationary_ready()
    control.belief.velocity = np.array([0.06, 0.02])
    control.belief.velocity_radius = 0.03
    target = np.array([0.4, 0])
    now = last + 0.05
    control.issued_history.append((last, last + 0.15, 0.15))
    _, _, bounds = control._reaction_envelope(target, [0, -0.3], now, now)
    rng = np.random.default_rng(71)
    duration = bounds["reaction_seconds"]
    dt = duration / 100
    for _ in range(20):
        angle = rng.uniform(-np.pi, np.pi)
        velocity = control.belief.velocity + 0.03 * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        position = np.zeros(2)
        distance = 0.0
        for index in range(100):
            angle = rng.uniform(-np.pi, np.pi)
            speed = rng.uniform(0, 0.15)
            magnitude = 0.05 + 0.95 * speed / 0.35
            command = magnitude * np.array([np.cos(angle), np.sin(angle)])
            old_velocity = velocity.copy()
            position, velocity = control.dynamics.advance(
                position, velocity, command, dt
            )
            # Adversarial acceleration has the declared maximum magnitude.
            disturbance = rng.normal(size=2)
            disturbance *= control.dynamics.acceleration_uncertainty / np.linalg.norm(
                disturbance
            )
            velocity += disturbance * dt
            distance += (
                (np.linalg.norm(old_velocity) + np.linalg.norm(velocity)) * dt / 2
            )
            assert (
                np.linalg.norm(velocity) <= bounds["reaction_speed_bound_m_s"] + 1e-10
            )
        assert distance <= bounds["reaction_path_length_m"] + 1e-10


def acknowledged(start, end, command=(0, 0)):
    return {"start": start, "end": end, "command": list(command)}


@pytest.mark.parametrize("first,last", [((0.4, 0), (0, 0)), ((0, 0), (0.4, 0))])
def test_known_switch_timeline_recovers_piecewise_prediction_not_endpoint_guess(
    first, last
):
    d, p = CommandDynamics(), OcclusionParameters()
    known = CommandBelief(d, p)
    scalar = CommandBelief(d, p)
    for belief in (known, scalar):
        belief.predict(0, (0, 0))
        belief.observe(measurement(0))
    intervals = [acknowledged(0, 0.025, first), acknowledged(0.025, 0.05, last)]
    known.predict(0.05, last, applied_command_intervals=intervals)
    scalar.predict(0.05, last)
    xy, velocity = d.advance([0, 0], [0, 0], first, 0.025)
    xy, velocity = d.advance(xy, velocity, last, 0.025)
    np.testing.assert_allclose(known.xy, xy, atol=1e-14)
    np.testing.assert_allclose(known.velocity, velocity, atol=1e-14)
    assert not np.allclose(scalar.xy, known.xy, atol=1e-6)
    assert known.last_prediction_mode == "acknowledged_command_timeline"
    assert known.last_prediction_segments == 2
    assert known.last_visual_time == 0
    assert not known.measured


def test_known_timeline_removes_timing_only_and_keeps_model_uncertainty():
    p = OcclusionParameters()
    known = CommandBelief(CommandDynamics(), p)
    reference = CommandBelief(CommandDynamics(latency_mismatch=0), p)
    no_model_error = CommandBelief(CommandDynamics(acceleration_uncertainty=0), p)
    intervals = [acknowledged(0, 0.025, (0.4, 0)), acknowledged(0.025, 0.05, (0, 0))]
    for belief in (known, reference, no_model_error):
        belief.predict(0, (0, 0))
        belief.observe(measurement(0))
    known.predict(0.05, (0, 0), applied_command_intervals=intervals)
    no_model_error.predict(0.05, (0, 0), applied_command_intervals=intervals)
    for record in intervals:
        reference.predict(record["end"], record["command"])
    assert known.position_radius == pytest.approx(reference.position_radius)
    assert known.velocity_radius == pytest.approx(reference.velocity_radius)
    assert known.position_radius > no_model_error.position_radius
    assert known.velocity_radius > no_model_error.velocity_radius


@pytest.mark.parametrize(
    "malformation",
    [
        "empty",
        "missing_start",
        "missing_end",
        "gap",
        "overlap",
        "backwards",
        "nan",
        "invalid_command",
        "truth_payload",
        "wrong_container",
    ],
)
def test_invalid_acknowledgement_partition_brakes_without_partial_prediction(
    malformation,
):
    control, last = stationary_ready()
    end = last + 0.05
    intervals = [acknowledged(last, end)]
    if malformation == "empty":
        intervals = []
    elif malformation == "missing_start":
        intervals[0]["start"] += 0.005
    elif malformation == "missing_end":
        intervals[0]["end"] -= 0.005
    elif malformation == "gap":
        intervals = [acknowledged(last, last + 0.02), acknowledged(last + 0.025, end)]
    elif malformation == "overlap":
        intervals = [acknowledged(last, last + 0.03), acknowledged(last + 0.025, end)]
    elif malformation == "backwards":
        intervals = [
            acknowledged(last, last + 0.03),
            acknowledged(last + 0.03, last + 0.02),
        ]
    elif malformation == "nan":
        intervals[0]["end"] = float("nan")
    elif malformation == "invalid_command":
        intervals[0]["command"] = [1.1, 0]
    elif malformation == "truth_payload":
        intervals[0]["position"] = [0.1, 0.2]
    elif malformation == "wrong_container":
        intervals = intervals[0]
    old_xy = control.belief.xy.copy()
    old_radius = control.belief.position_radius
    result = step(control, end, applied_command_intervals=intervals)
    assert np.all(result == 0)
    assert control.status == "invalid_applied_command_intervals"
    assert control.belief.timestamp == last
    assert control.belief.last_visual_time == last
    assert control.belief.position_radius == old_radius
    np.testing.assert_array_equal(control.belief.xy, old_xy)
    assert control.diagnostics["pose_source"] == "stale"


def test_timeline_requires_bounded_full_interval_and_empty_initial_history():
    belief = CommandBelief(CommandDynamics(), OcclusionParameters())
    with pytest.raises(ValueError, match="no preceding"):
        belief.predict(0, (0, 0), applied_command_intervals=[acknowledged(-0.05, 0)])
    belief.predict(0, (0, 0), applied_command_intervals=[])
    belief.observe(measurement(0))
    with pytest.raises(ValueError, match="bounded"):
        belief.predict(0.2, (0, 0), applied_command_intervals=[acknowledged(0, 0.2)])
    assert belief.timestamp == 0


def test_known_past_timeline_never_clears_unexpired_future_issued_targets():
    control, last = stationary_ready()
    control.policy = lambda vector: np.array([0.8, 0])
    step(
        control,
        last + 0.05,
        applied_command_intervals=[acknowledged(last, last + 0.05)],
    )
    control.segment_free = Corridor(half_width=0.13)
    control.policy = lambda vector: np.zeros(2)
    step(
        control,
        last + 0.10,
        applied_command_intervals=[acknowledged(last + 0.05, last + 0.10)],
    )
    assert control.belief.last_prediction_mode == "acknowledged_command_timeline"
    assert control.envelope_trials[0]["retained_target_speed_m_s"] == pytest.approx(
        0.15
    )
    assert control.status == "stopping_envelope_not_certified"
    assert np.all(control.issued_action == 0)


def test_half_open_timeline_allows_endpoint_expiry_without_erasing_past_motion():
    belief = CommandBelief(CommandDynamics(), OcclusionParameters())
    belief.predict(0, (0, 0), applied_command_intervals=[])
    belief.observe(measurement(0))
    command = (0.4, 0)
    belief.predict(
        0.05,
        (0, 0),
        applied_command_intervals=[acknowledged(0, 0.05, command)],
    )
    xy, velocity = belief.dynamics.advance([0, 0], [0, 0], command, 0.05)
    np.testing.assert_allclose(belief.xy, xy)
    np.testing.assert_allclose(belief.velocity, velocity)
    assert belief.xy[0] > 0
    np.testing.assert_allclose(
        belief.target_velocity_center, belief.dynamics.target(command)
    )


def initialized_belief(velocity, radius, dynamics=None):
    belief = CommandBelief(dynamics or CommandDynamics(), OcclusionParameters())
    belief.predict(0, (0, 0), applied_command_intervals=[])
    belief.observe(measurement(0))
    # Unit-test initial uncertainty sets; no such input enters the controller.
    belief.velocity = np.asarray(velocity, float)
    belief.velocity_radius = radius
    return belief


def test_unsaturated_speed_mode_straddle_adds_no_cap_gap():
    d = CommandDynamics()
    command = (0.4, 0)
    target = d.target(command)
    belief = initialized_belief(target, 0.02, d)
    old_position_radius = belief.position_radius
    dt = 0.005
    belief.predict(
        dt, command, applied_command_intervals=[acknowledged(0, dt, command)]
    )
    assert belief.velocity_radius == pytest.approx(
        0.02 * np.exp(-dt / d.response_time) + d.acceleration_uncertainty * dt
    )
    assert belief.position_radius == pytest.approx(
        old_position_radius + 0.02 * dt + 0.5 * d.acceleration_uncertainty * dt**2
    )


def test_saturated_speed_mode_straddle_retains_cap_gap():
    d = CommandDynamics()
    command = (0.4, 0)
    # Opposite heading at equal speed straddles the speed-based cap switch while
    # the target error is large enough to bind the acceleration cap.
    belief = initialized_belief(-d.target(command), 0.01, d)
    old_radius = belief.position_radius
    dt = 0.005
    belief.predict(
        dt, command, applied_command_intervals=[acknowledged(0, dt, command)]
    )
    cap_gap = abs(d.max_acceleration - d.max_deceleration)
    assert belief.velocity_radius == pytest.approx(
        0.01 + (d.acceleration_uncertainty + cap_gap) * dt
    )
    assert belief.position_radius == pytest.approx(
        old_radius + 0.01 * dt + 0.5 * (d.acceleration_uncertainty + cap_gap) * dt**2
    )


def test_persistent_target_ball_brakes_after_motion_without_recentring_loss():
    d = CommandDynamics()
    belief = initialized_belief((0.14, 0.01), 0.12, d)
    dt = 0.005
    command = (0.5, 0)
    for tick in range(20):
        belief.predict(
            (tick + 1) * dt,
            command,
            applied_command_intervals=[
                acknowledged(tick * dt, (tick + 1) * dt, command)
            ],
        )
    velocity_radius_at_switch = belief.velocity_radius
    for tick in range(20, 220):
        belief.predict(
            (tick + 1) * dt,
            (0, 0),
            applied_command_intervals=[acknowledged(tick * dt, (tick + 1) * dt)],
        )
    assert np.linalg.norm(belief.velocity) < 0.004
    assert belief.velocity_radius < 0.065
    assert belief.velocity_radius < velocity_radius_at_switch
    assert belief.target_velocity_radius < 0.065
    np.testing.assert_array_equal(belief.target_velocity_center, [0, 0])


@pytest.mark.parametrize("seed", [3, 17, 61])
def test_velocity_and_position_balls_cover_adversarial_command_switches(seed):
    """Independent sampled reachable states, including outward model error."""
    rng = np.random.default_rng(seed)
    d = CommandDynamics()
    center = np.array([0.11, -0.06])
    initial_radius = 0.13
    belief = initialized_belief(center, initial_radius, d)
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    actual_velocities = center + initial_radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    actual_positions = np.zeros_like(actual_velocities)
    dt = 0.005
    commands = [(0.8, 0), (-0.4, 0.5), (0, 0), (0, -0.7), (0, 0)]
    for tick in range(300):
        command = commands[min(tick // 45, len(commands) - 1)]
        before_velocity = belief.velocity.copy()
        belief.predict(
            (tick + 1) * dt,
            command,
            applied_command_intervals=[
                acknowledged(tick * dt, (tick + 1) * dt, command)
            ],
        )
        target = d.target(command)
        for index in range(len(actual_velocities)):
            old_velocity = actual_velocities[index].copy()
            position, velocity = d.advance(
                actual_positions[index], old_velocity, command, dt
            )
            if index % 3 == 0:
                direction = old_velocity - before_velocity
            elif index % 3 == 1:
                direction = old_velocity - target
            else:
                direction = rng.normal(size=2)
            disturbance = (
                direction
                * d.acceleration_uncertainty
                / max(np.linalg.norm(direction), 1e-12)
            )
            actual_positions[index] = position + disturbance * (0.5 * dt**2)
            actual_velocities[index] = velocity + disturbance * dt
        assert np.max(np.linalg.norm(actual_velocities - belief.velocity, axis=1)) <= (
            belief.velocity_radius + 1e-10
        )
        assert np.max(np.linalg.norm(actual_velocities - target, axis=1)) <= (
            belief.target_velocity_radius + 1e-10
        )
        assert np.max(np.linalg.norm(actual_positions - belief.xy, axis=1)) <= (
            belief.position_radius + 1e-10
        )


def test_rgb_velocity_recentering_cannot_reset_persistent_target_ball():
    belief = initialized_belief((0.12, 0), 0.12)
    for tick in range(40):
        belief.predict(
            (tick + 1) * 0.005,
            (0, 0),
            applied_command_intervals=[acknowledged(tick * 0.005, (tick + 1) * 0.005)],
        )
    before = belief.target_velocity_radius
    target = belief.target_velocity_center.copy()
    assert belief.observe(measurement(0.2, belief.xy))
    assert belief.target_velocity_radius <= before
    np.testing.assert_array_equal(belief.target_velocity_center, target)
    # Moving the nominal center alone must never overwrite/recreate the target
    # radius. A deliberately looser nominal enclosure adds no new information.
    belief.velocity = np.array([0.3, 0.2])
    belief.velocity_radius = 0.8
    belief.predict(0.205, (0, 0), applied_command_intervals=[acknowledged(0.2, 0.205)])
    assert belief.target_velocity_radius < before


def test_velocity_initialization_blocks_early_motion_even_in_unbounded_free_space():
    control = controller(segment_free=lambda a, b, radius: True)
    for index in range(10):
        action = step(control, index * 0.05)
        assert control.belief.velocity_radius > 0.05
        assert not control.velocity_initialized
        assert control.status == "warming_up"
        np.testing.assert_array_equal(action, [0, 0])
        assert control.route is None
    assert control.belief.samples >= control.parameters.minimum_visual_samples


def test_visible_zero_acknowledgements_initialize_without_reset_velocity_truth():
    control = controller(segment_free=lambda a, b, radius: True)
    initialized_at = None
    for index in range(60):
        timestamp = index * 0.05
        intervals = [] if index == 0 else [acknowledged(timestamp - 0.05, timestamp)]
        action = step(control, timestamp, applied_command_intervals=intervals)
        if index == 0:
            assert control.belief.velocity_radius == 0.35
        if control.velocity_initialized:
            initialized_at = timestamp
            assert control.belief.measured
            assert control.belief.velocity_radius <= 0.05
            assert action[0] > 0
            assert control.diagnostics["velocity_initialized"]
            break
        np.testing.assert_array_equal(action, [0, 0])
    assert initialized_at is not None
    # Initialization is a one-time restriction; later uncertainty still enters
    # the ordinary stopping envelope instead of recreating the startup gate.
    control.belief.velocity_radius = 0.06
    step(
        control,
        initialized_at + 0.05,
        measurement(initialized_at + 0.05, status="missing"),
    )
    assert control.velocity_initialized
    assert control.status != "warming_up"


def test_velocity_initialization_cannot_latch_from_hidden_prediction():
    control = controller(segment_free=lambda a, b, radius: True)
    warm(control, count=5)
    assert control.belief.samples >= 3
    for index in range(5, 50):
        timestamp = index * 0.05
        action = step(control, timestamp, measurement(timestamp, status="missing"))
        assert not control.velocity_initialized
        np.testing.assert_array_equal(action, [0, 0])
    assert control.belief.velocity_radius < 0.05
    assert not control.belief.measured
    step(control, 2.5)
    assert control.velocity_initialized
    assert control.belief.measured


@pytest.mark.parametrize("invalidation", ["version", "gap", "timeline"])
def test_invalidated_velocity_initialization_never_authorizes_hidden_motion(
    invalidation,
):
    control, last = stationary_ready()
    assert control.velocity_initialized
    if invalidation == "version":
        step(control, last + 0.05, map_version="changed")
    elif invalidation == "gap":
        step(control, last + 0.5)
    else:
        step(control, last + 0.05, applied_command_intervals=[])
    assert not control.velocity_initialized
    assert not control.diagnostics["motion_authorized"]
    assert np.all(control.issued_action == 0)


def arrival_ready():
    control, last = stationary_ready()
    # This independent fixture enters the goal neighborhood after initialization.
    control.goal = np.array([0.03, 0])
    control.route = None
    return control, last


def settle_until_visible_dwell(control, timestamp):
    """Advance visible zero-ack captures to the calculated settling deadline."""
    step(control, timestamp)
    deadline = control.arrival_settle_until
    assert deadline is not None
    while control.visible_since is None:
        timestamp += 0.05
        step(control, timestamp)
        assert control.arrival_settle_until == deadline
        assert not control.arrived
    assert control.visible_since >= deadline - 1e-9
    return timestamp


def test_arrival_dwell_waits_for_pending_motion_to_expire_then_full_visible_hold():
    control, last = arrival_ready()
    control.issued_history.append((last, last + 0.15, 0.15))
    for offset in (0.05, 0.10, 0.15):
        action = step(control, last + offset)
        assert control.belief.measured
        assert np.linalg.norm(control.belief.velocity) < 0.03
        assert control.status == "settling_visible"
        assert control.visible_since is None
        assert not control.arrived
        np.testing.assert_array_equal(action, [0, 0])
    start = settle_until_visible_dwell(control, last + 0.20)
    for index in range(1, 10):
        step(control, start + index * 0.05)
        assert control.visible_since == pytest.approx(start)
        assert not control.arrived
    step(control, start + 0.50)
    assert control.arrived
    assert control.status == "arrived_visible"


def test_arrival_dwell_rejects_nonzero_ack_even_with_low_estimated_speed():
    control, last = arrival_ready()
    step(control, last + 0.05, applied=(0.06, 0))
    assert np.linalg.norm(control.belief.velocity) < 0.03
    assert control.status == "settling_visible"
    assert control.visible_since is None
    assert np.all(control.issued_action == 0)
    step(control, last + 0.10)
    assert control.arrival_settle_until > last + 0.10
    assert control.visible_since is None
    assert not control.arrived


def test_arrival_dwell_cannot_use_unknown_history_or_commands_expiring_after_capture():
    control, last = arrival_ready()
    step(control, last + 0.05, command_history_valid=False)
    assert control.status == "settling_visible"
    assert control.visible_since is None
    control.issued_history.append((last, last + 0.15, 0.15))
    step(control, last + 0.10, now=last + 0.19)
    assert control.status == "settling_visible"
    assert control.visible_since is None


def test_missing_frame_restarts_settled_visible_dwell():
    control, last = arrival_ready()
    for index in range(1, 8):
        step(control, last + index * 0.05)
        assert not control.arrived
    step(control, last + 0.40, measurement(last + 0.40, status="missing"))
    assert control.visible_since is None
    assert not control.arrived
    assert control.arrival_settle_until is None
    start = settle_until_visible_dwell(control, last + 0.45)
    for index in range(1, 10):
        step(control, start + index * 0.05)
        assert not control.arrived
    step(control, start + 0.50)
    assert control.arrived


@pytest.mark.parametrize("speed", [0.0, 0.02, 0.03, 0.06, 0.175, 0.35])
def test_arrival_settle_time_matches_saturated_then_exponential_braking(speed):
    control = controller()
    duration = control._arrival_braking_time(speed)
    d, target = control.dynamics, control.parameters.arrival_speed
    threshold = d.braking_acceleration_lower * d.response_time_upper
    saturated_time = max(0, (speed - threshold) / d.braking_acceleration_lower)
    if duration <= saturated_time:
        remaining_speed = speed - d.braking_acceleration_lower * duration
    else:
        remaining_speed = min(speed, threshold) * np.exp(
            -(duration - saturated_time) / d.response_time_upper
        )
    assert remaining_speed <= target + 1e-12
    if speed > target:
        assert remaining_speed == pytest.approx(target)
        assert duration > 0
    else:
        assert duration == 0


def test_arrival_uncertainty_adds_persistent_settle_before_unchanged_visible_hold():
    control, last = arrival_ready()
    # A low nominal speed is insufficient while its velocity-error ball is wide.
    control.belief.velocity = np.array([0.02, 0])
    control.belief.velocity_radius = 0.06
    control.belief.target_velocity_center = None
    first = last + 0.05
    action = step(control, first)
    bound = np.linalg.norm(control.belief.velocity) + control.belief.velocity_radius
    assert np.linalg.norm(control.belief.velocity) < 0.03
    assert control.arrival_settle_speed_bound == pytest.approx(bound)
    assert control.arrival_settle_until == pytest.approx(
        first + control._arrival_braking_time(bound)
    )
    deadline = control.arrival_settle_until
    assert deadline > first
    assert control.visible_since is None
    np.testing.assert_array_equal(action, [0, 0])
    timestamp = first
    while timestamp + 0.05 < deadline - 1e-9:
        timestamp += 0.05
        step(control, timestamp)
        assert control.status == "settling_visible"
        assert control.visible_since is None
        assert control.arrival_settle_until == deadline
    start = settle_until_visible_dwell(control, timestamp + 0.05)
    for index in range(1, 10):
        step(control, start + index * 0.05)
        assert not control.arrived
    step(control, start + 0.5)
    assert control.arrived


@pytest.mark.parametrize(
    "interruption",
    ["missing", "ambiguous", "ack", "pending", "history", "unsafe", "far", "stale"],
)
def test_arrival_settle_resets_on_visual_or_command_or_clearance_loss(interruption):
    control, last = arrival_ready()
    control.belief.velocity_radius = 0.10
    control.belief.target_velocity_center = None
    first = last + 0.05
    step(control, first)
    previous_deadline = control.arrival_settle_until
    assert previous_deadline > first + 0.05
    timestamp = first + 0.05
    kwargs = {}
    item = measurement(timestamp)
    if interruption in ("missing", "ambiguous"):
        item = measurement(timestamp, status=interruption)
    elif interruption == "ack":
        kwargs["applied"] = (0.06, 0)
    elif interruption == "pending":
        control.issued_history.append((first, first + 0.15, 0.10))
    elif interruption == "history":
        kwargs["command_history_valid"] = False
    elif interruption == "unsafe":
        control.segment_free = lambda a, b, radius: False
    elif interruption == "far":
        control.goal = np.array([0.6, 0])
    elif interruption == "stale":
        kwargs["now"] = timestamp + 0.2
    step(control, timestamp, item, **kwargs)
    assert control.arrival_settle_until is None
    assert control.arrival_settle_speed_bound is None
    assert control.visible_since is None
    assert not control.arrived


def disk_obstacle_certificate(start, end, radius):
    """Independent analytic segment-to-disk clearance for the planner tests."""
    start, end = np.asarray(start), np.asarray(end)
    obstacle = np.array([0.3, 0.0])
    direction = end - start
    fraction = np.clip(
        np.dot(obstacle - start, direction) / max(np.dot(direction, direction), 1e-12),
        0,
        1,
    )
    return np.linalg.norm(obstacle - start - fraction * direction) > 0.03 + radius


def test_radius_aware_planner_receives_current_error_margin_and_finds_clear_detour():
    calls = []

    def radius_planner(start, goal, radius):
        calls.append((start.copy(), goal.copy(), radius))
        return np.array([start, [0.1, 0.25], [0.5, 0.25], goal])

    control = controller(
        route_planner=radius_planner, segment_free=disk_obstacle_certificate
    )
    fallback = controller(segment_free=disk_obstacle_certificate)
    for index in range(40):
        timestamp = index * 0.05
        item = measurement(timestamp, sigma=0.053 / 3)
        step(control, timestamp, item)
        step(fallback, timestamp, item)
    assert calls and calls[0][2] == pytest.approx(0.037 + 0.040 + 0.053)
    assert control.route is not None
    assert control.status == "tracking_visible"
    assert control.diagnostics["route_planning_radius_m"] == pytest.approx(0.13)
    assert fallback.route is None
    assert fallback.status == "route_not_certified"
    np.testing.assert_array_equal(fallback.issued_action, [0, 0])


def test_radius_planner_result_still_needs_independent_capsule_certificate():
    control = controller(
        route_planner=lambda start, goal, radius: np.array([start, goal]),
        segment_free=disk_obstacle_certificate,
    )
    warm(control)
    assert control.status == "route_not_certified"
    assert control.route is None
    assert control.diagnostics["requested_planning_radius_m"] == pytest.approx(0.083)
    assert control.diagnostics["route_planning_radius_m"] is None
    np.testing.assert_array_equal(control.issued_action, [0, 0])


@pytest.mark.parametrize("problem", ["wrong_endpoint", "nonfinite", "raises"])
def test_invalid_radius_planner_never_falls_back_to_different_uncertainty(problem):
    class ForbiddenFallback:
        def route(self, start, goal):
            raise AssertionError("An explicitly supplied planner must not be bypassed")

    def planner(start, goal, radius):
        if problem == "raises":
            raise ValueError("No path at requested radius")
        return [start, [np.nan, 0] if problem == "nonfinite" else goal + 1]

    control = controller(route_planner=planner)
    control.grid = ForbiddenFallback()
    warm(control)
    assert control.status == (
        "no_proven_route_memory" if problem == "raises" else "invalid_route"
    )
    assert control.route is None
    np.testing.assert_array_equal(control.issued_action, [0, 0])


def test_lost_segment_stops_then_replans_only_on_later_visible_observation():
    calls = []
    reject_old = [False]

    def planner(start, goal, radius):
        calls.append(radius)
        return (
            np.array([start, [0, 0.25], goal])
            if reject_old[0]
            else np.array([start, goal])
        )

    def certificate(start, end, radius):
        # The old next segment loses clearance; a separate known detour exists.
        return not (
            reject_old[0] and np.allclose(start, [0, 0]) and np.allclose(end, [0.6, 0])
        )

    control = controller(route_planner=planner, segment_free=certificate)
    last = warm(control)
    assert len(calls) == 1
    reject_old[0] = True
    action = step(control, last + 0.05)
    assert control.status == "route_not_certified"
    assert control.route is None
    assert len(calls) == 1  # No immediate replan on the failure observation.
    np.testing.assert_array_equal(action, [0, 0])
    step(control, last + 0.10, measurement(last + 0.10, status="missing"))
    assert control.status == "no_proven_route_memory"
    assert len(calls) == 1
    step(control, last + 0.15)
    assert len(calls) == 2
    assert control.route is not None
    assert control.status == "tracking_visible"
