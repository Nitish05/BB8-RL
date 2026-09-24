"""Loss is a motion lifecycle event, not just formatting a large error radius."""

from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.interactive_runtime import ControlSession


class ClearMemory:
    def planning_grid(self, radius):
        return self

    def segment_free(self, start, end, radius):
        return all(np.max(np.abs(point)) < 1 for point in (start, end))

    def route(self, start, goal):
        return np.array([start, goal])


def make_session():
    stops = []
    session = ControlSession(
        policy=lambda vector: np.array([0.4, 0.0]),
        memory=ClearMemory(),
        stop=lambda: stops.append(True),
        map_version="map",
        calibration_version="rig",
        demo_goal=[0.5, 0],
        clock=lambda: 0.0,
    )
    return session, stops


def frame(
    session,
    index,
    *,
    status="visible",
    xy=(0, 0),
    intervals=None,
    covariance=None,
    applied=(0, 0),
):
    timestamp = index * 0.05
    measurement = SimpleNamespace(
        timestamp=timestamp,
        status=status,
        xy=np.asarray(xy, float) if status == "visible" else None,
        covariance=np.eye(2) * 0.002**2 if covariance is None else covariance,
    )
    if status != "visible":
        measurement.covariance = None
    if intervals is None:
        intervals = (
            []
            if index == 0
            else [
                {
                    "start": timestamp - 0.05 + tick * 0.005,
                    "end": timestamp - 0.05 + (tick + 1) * 0.005,
                    "command": list(applied),
                }
                for tick in range(10)
            ]
        )
    return session.decide(timestamp, measurement, list(applied), intervals)


def warm(session):
    for index in range(30):
        frame(session, index)


def lose(session):
    for index in range(30, 56):
        assert np.all(frame(session, index, status="missing") == 0)
    assert session.state()["localization_status"] == "lost"


def test_short_prediction_expires_and_retains_last_seen_not_current_pose():
    session, stops = make_session()
    warm(session)
    frame(session, 30, status="missing")
    state = session.state()
    assert state["localization_status"] == "predicted" and state["localization_valid"]
    assert state["last_seen_age_s"] == pytest.approx(0.05)
    for index in range(31, 70):
        frame(session, index, status="missing")
    state = session.state()
    assert state["localization_status"] == "lost" and not state["localization_valid"]
    assert state["pose"] is None and state["position_radius"] is None
    assert state["last_seen_pose"] == [0, 0]
    assert state["last_seen_age_s"] == pytest.approx(2.0)
    assert state["raw_position_radius"] > 0.06
    assert state["requires_new_goal"] and len(stops) == 1


def test_loss_cancels_goal_and_rejects_new_pending_motion():
    session, stops = make_session()
    session.receive([{"action": "goal", "generation": 1, "x": 0.5, "y": 0}])
    warm(session)
    assert session.controller is not None
    for index in range(30, 65):
        action = frame(session, index, status="missing")
        if session.localization_lost:
            assert np.all(action == 0)
    assert (
        session.goal is None
        and session.pending_goal is None
        and session.controller is None
    )
    records = session.receive([{"action": "goal", "generation": 2, "x": 0.4, "y": 0}])
    assert records[0]["outcome"] == "rejected"
    assert session.pending_goal is None
    assert len(stops) == 2


def test_recovery_needs_fresh_consistent_observations_and_explicit_new_goal():
    session, _ = make_session()
    warm(session)
    lose(session)
    assert np.all(frame(session, 56) == 0)
    assert session.state()["localization_status"] == "reacquiring"
    assert session.state()["pose"] is None
    frame(session, 57, status="missing")
    assert session.recovery_belief is None
    for index in range(58, 90):
        assert np.all(frame(session, index) == 0)
    assert session.state()["localization_valid"]
    assert session.state()["requires_new_goal"]
    assert session.controller is None and session.goal is None
    session.receive([{"action": "goal", "generation": 1, "x": 0.4, "y": 0}])
    for index in range(90, 125):
        action = frame(session, index)
    assert np.linalg.norm(action) > 0
    assert not session.state()["requires_new_goal"]


@pytest.mark.parametrize(
    "xy,covariance",
    [
        ((0.5, 0), None),
        ((np.nan, 0), None),
        ((0, 0), np.eye(2) * 1.0),
    ],
)
def test_spurious_or_uncertain_fixes_do_not_relocalize(xy, covariance):
    session, _ = make_session()
    warm(session)
    lose(session)
    for index in range(56, 90):
        assert np.all(frame(session, index, xy=xy, covariance=covariance) == 0)
        assert not session.state()["localization_valid"]
    assert session.state()["last_seen_pose"] == [0, 0]


def test_invalid_command_history_loses_localization_and_recovery_is_fresh():
    session, _ = make_session()
    warm(session)
    assert np.all(frame(session, 30, intervals=[]) == 0)
    assert session.localization_lost
    assert session.braking_prediction.record["valid"] is False
    for index in range(31, 66):
        frame(session, index)
    assert session.state()["localization_valid"]
    assert session.goal is None


def test_duplicate_capture_cannot_count_towards_recovery():
    session, _ = make_session()
    warm(session)
    lose(session)
    frame(session, 56)
    assert session.recovery_belief.samples == 1
    frame(session, 56)
    assert session.recovery_belief is None
    assert not session.state()["localization_valid"]


def test_loss_during_pending_goal_never_starts_it_on_return():
    session, _ = make_session()
    warm(session)
    frame(session, 30, status="missing")
    session.receive([{"action": "goal", "generation": 1, "x": 0.4, "y": 0}])
    for index in range(31, 65):
        frame(session, index, status="missing")
    assert session.pending_goal is None and session.goal is None
    for index in range(65, 100):
        assert np.all(frame(session, index) == 0)
    assert session.state()["localization_valid"] and session.controller is None


def test_new_controller_velocity_warmup_cannot_publish_expired_pose():
    session, _ = make_session()
    warm(session)
    session.receive([{"action": "goal", "generation": 1, "x": 0.4, "y": 0}])
    frame(session, 30)
    for index in range(31, 38):
        assert np.all(frame(session, index, status="missing") == 0)
    assert session.idle_belief.position_radius < session.parameters.max_position_radius
    assert session.state()["localization_status"] == "lost"
    assert session.state()["pose"] is None
    assert session.state()["position_radius"] is None
    assert session.goal is None and session.controller is None


def test_returning_frame_after_timeout_cannot_bypass_loss_cancellation():
    session, _ = make_session()
    session.receive([{"action": "goal", "generation": 1, "x": 0.4, "y": 0}])
    warm(session)
    for index in range(30, 50):
        frame(session, index, status="missing")
    assert not session.localization_lost  # Last missing capture is exactly 1 s.
    assert np.all(frame(session, 50) == 0)  # RGB returns at 1.05 s.
    assert session.localization_lost
    assert session.localization_status == "reacquiring"
    assert session.goal is None and session.controller is None


def commanded_motion_then_loss():
    """Pure command/measurement fixture; this does not simulate physical motion."""
    session, _ = make_session()
    session.receive([{"action": "goal", "generation": 1, "x": 0.5, "y": 0}])
    warm(session)
    moving = []
    for index in range(30, 50):
        moving.append(frame(session, index, status="missing", applied=(0.4, 0)))
        assert session.state()["localization_valid"]
    previous = session.state()
    assert any(np.linalg.norm(action) > 0 for action in moving)
    assert previous["localization_status"] == "predicted"
    assert np.all(frame(session, 50, status="missing") == 0)
    assert session.localization_lost
    assert session.recovery_anchor_time == previous["raw_prediction_time"]
    assert session.recovery_anchor_source == "predicted"
    np.testing.assert_array_equal(session.recovery_anchor_pose, previous["raw_pose"])
    return session


def test_recovery_anchor_accounts_for_previously_valid_commanded_motion():
    session = commanded_motion_then_loss()
    anchor = session.recovery_anchor_pose.copy()
    returning_xy = anchor + [0.06, 0.0]
    assert np.linalg.norm(returning_xy - session.last_seen_pose) > 0.12
    assert np.linalg.norm(returning_xy - anchor) < 0.12
    for index in range(51, 60):
        assert np.all(frame(session, index, status="missing") == 0)
    assert np.all(frame(session, 60, xy=returning_xy) == 0)
    assert session.localization_status == "reacquiring"
    assert session.state()["pose"] is None
    for index in range(61, 95):
        assert np.all(frame(session, index, xy=returning_xy) == 0)
    state = session.state()
    assert state["localization_valid"] and state["localization_status"] == "measured"
    np.testing.assert_array_equal(state["pose"], returning_xy)
    assert state["requires_new_goal"] and session.goal is None
    assert session.controller is None


def test_recovery_anchor_does_not_follow_returning_outliers_or_raw_growth():
    session = commanded_motion_then_loss()
    anchor = session.recovery_anchor_pose.copy()
    anchor_time = session.recovery_anchor_time
    for index in range(51, 251):
        assert np.all(frame(session, index, status="missing") == 0)
    assert session.idle_belief.position_radius > 0.3
    for index, offset in enumerate(np.linspace(0.06, 0.24, 10), start=251):
        assert np.all(frame(session, index, xy=anchor + [offset, 0]) == 0)
        np.testing.assert_array_equal(session.recovery_anchor_pose, anchor)
        assert session.recovery_anchor_time == anchor_time
        assert session.goal is None and session.controller is None
    assert session.localization_status == "lost"
    assert session.recovery_belief is None
    assert session.state()["pose"] is None
    # Rejected recovery candidates may enter raw diagnostics, never the gate.
    assert np.linalg.norm(session.idle_belief.xy - anchor) > 0.12


@pytest.mark.parametrize("version_field", ["map_version", "calibration_version"])
def test_context_change_clears_frozen_recovery_anchor(version_field):
    session = commanded_motion_then_loss()
    assert session.recovery_anchor_pose is not None
    setattr(session, version_field, "unregistered-change")
    assert np.all(frame(session, 51, xy=(0.1, 0)) == 0)
    state = session.state()
    assert state["recovery_anchor_pose"] is None
    assert state["recovery_anchor_time"] is None
    assert state["recovery_anchor_source"] is None
    assert session._loss_reference is None
    assert not state["localization_valid"]
    assert session.recovery_belief is None


@pytest.mark.parametrize(
    "distance,accepted", [(0.12, True), (np.nextafter(0.12, np.inf), False)]
)
def test_recovery_gate_remains_exactly_twelve_centimeters(distance, accepted):
    session, _ = make_session()
    warm(session)
    lose(session)
    np.testing.assert_array_equal(session.recovery_anchor_pose, [0, 0])
    assert np.all(frame(session, 56, xy=(distance, 0)) == 0)
    assert (session.localization_status == "reacquiring") is accepted
    assert session.state()["pose"] is None


def test_public_pose_uses_one_belief_when_controller_rejects_different_jitter():
    session, _ = make_session()
    warm(session)
    session.receive([{"action": "goal", "generation": 1, "x": 0.4, "y": 0}])
    frame(session, 30)
    frame(session, 31, xy=(0.03, 0))
    assert not session.idle_belief.measured and session.controller.belief.measured
    state = session.state()
    assert state["localization_status"] == state["pose_source"] == "predicted"
    assert state["pose"] == state["raw_pose"] == state["last_seen_pose"] == [0, 0]
    assert state["position_radius"] == session.idle_belief.position_radius
    assert state["last_seen_age_s"] == pytest.approx(0.05)
    frame(session, 32, xy=(-0.015, 0))
    assert session.idle_belief.measured and not session.controller.belief.measured
    state = session.state()
    assert state["localization_status"] == state["pose_source"] == "measured"
    assert state["pose"] == state["raw_pose"] == state["last_seen_pose"] == [-0.015, 0]
    assert state["position_radius"] == state["last_seen_radius_m"]
    assert state["raw_position_radius"] == state["position_radius"]
    np.testing.assert_array_equal(state["raw_velocity"], session.idle_belief.velocity)
    assert state["raw_velocity_radius"] == session.idle_belief.velocity_radius
    assert state["last_seen_age_s"] == 0
