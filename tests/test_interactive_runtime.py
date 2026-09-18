import importlib
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.interactive_runtime import (
    ControlSession,
    drain_commands,
    publish_latest,
    worker_main,
)


class Memory:
    def __init__(self):
        self.blocked = False
        self.route_blocked = False
        self.queries = []

    def planning_grid(self, radius):
        return self

    def segment_free(self, a, b, radius):
        self.queries.append((np.array(a), np.array(b), radius))
        return (
            not self.blocked
            and max(abs(np.asarray(a))) < 1
            and max(abs(np.asarray(b))) < 1
        )

    def route(self, start, goal):
        if self.route_blocked:
            raise ValueError("No certified path")
        return np.array([start, goal])


def observed(t, status="visible"):
    return SimpleNamespace(
        timestamp=t,
        status=status,
        xy=np.array([0.0, 0.0]) if status == "visible" else None,
        covariance=np.eye(2) * 0.002**2 if status == "visible" else None,
    )


def session():
    now, stops, memory = [10.0], [], Memory()
    control = ControlSession(
        policy=lambda vector: np.array([0.4, 0]),
        memory=memory,
        stop=lambda: stops.append(now[0]),
        map_version="map",
        calibration_version="rig",
        demo_goal=[0.5, 0],
        clock=lambda: now[0],
    )
    return control, memory, now, stops


def advance(control, index, status="visible"):
    t = index * 0.05
    intervals = [] if index == 0 else [{"start": t - 0.05, "end": t, "command": [0, 0]}]
    return control.decide(t, observed(t, status), [0, 0], intervals)


def test_stop_has_batch_priority_and_stale_goal_cannot_resume():
    control, _, _, stops = session()
    records = control.receive(
        [
            {"action": "goal", "generation": 1, "x": 0.5, "y": 0},
            {"action": "stop", "generation": 2},
            {"action": "demo", "generation": 3},
        ]
    )
    assert [r["outcome"] for r in records] == [
        "cancelled_by_stop",
        "accepted",
        "cancelled_by_stop",
    ]
    assert len(stops) == 1
    assert control.goal is None and control.controller is None
    assert control.generation == 3
    control.receive([{"action": "demo", "generation": 3}])
    assert control.goal is None
    assert np.all(advance(control, 0) == 0)
    control.receive([{"action": "demo", "generation": 4}])
    assert control.goal is not None


def test_goal_change_brakes_discards_previous_controller_and_reinitializes():
    control, _, _, stops = session()
    control.receive([{"action": "demo", "generation": 1}])
    for index in range(30):
        action = advance(control, index)
    assert action[0] > 0
    old = control.controller
    control.receive([{"action": "goal", "generation": 2, "x": 0.4, "y": 0.1}])
    assert len(stops) == 2
    assert control.controller is None
    action = advance(control, 30)
    assert control.controller is not old
    assert not control.controller.velocity_initialized
    assert np.all(action == 0)
    assert control.phase == "braking"


def test_heartbeat_expiry_stops_and_returned_heartbeat_does_not_resume():
    control, _, now, stops = session()
    control.receive([{"action": "demo", "generation": 1}])
    advance(control, 0)
    now[0] += 3.01
    assert np.all(advance(control, 1) == 0)
    assert control.phase == "heartbeat_expired"
    assert control.controller is None and control.goal is None
    stopped_count = len(stops)
    control.receive([{"action": "heartbeat", "generation": 1}])
    advance(control, 2)
    assert control.phase == "heartbeat_expired"
    assert len(stops) == stopped_count
    control.receive([{"action": "demo", "generation": 2}])
    advance(control, 3)
    assert control.controller is not None


@pytest.mark.parametrize("point", [[2, 0], [np.nan, 0], [None, 0]])
def test_goal_rejection_never_calls_policy_or_preserves_old_motion(point):
    control, _, _, stops = session()
    records = control.receive(
        [{"action": "goal", "generation": 1, "x": point[0], "y": point[1]}]
    )
    assert records[0]["outcome"] == "rejected"
    assert len(stops) == 1
    assert control.phase == "goal_rejected"
    assert control.goal is None
    assert np.all(advance(control, 0) == 0)


def test_goal_waits_for_measured_pose_and_rejects_unknown_route():
    control, memory, _, _ = session()
    control.receive([{"action": "demo", "generation": 1}])
    for index in range(4):
        advance(control, index, "missing")
        assert control.controller is None
    memory.route_blocked = True
    advance(control, 4)
    assert control.phase == "goal_rejected"
    assert control.goal is None and control.controller is None
    assert "certified" in control.message


def test_idle_localization_continues_with_zero_action_and_no_truth_input():
    control, _, _, _ = session()
    assert control.phase == "localizing"
    for index in range(10):
        assert np.all(advance(control, index) == 0)
    assert control.state()["pose"] == [0, 0]
    assert control.idle_belief.samples == 10
    assert control.phase == "idle"
    advance(control, 10, "missing")
    assert control.phase == "localizing"
    assert control.state()["pose_source"] == "predicted"


def test_latest_queue_is_bounded_and_newest_event_survives():
    events = queue.Queue(maxsize=1)
    assert publish_latest(events, {"type": "old"})
    assert publish_latest(events, {"type": "new"})
    assert events.get_nowait() == {"type": "new"}
    commands = queue.Queue()
    for index in range(5):
        commands.put(index)
    assert drain_commands(commands, limit=3) == [0, 1, 2]
    assert drain_commands(commands) == [3, 4]


def test_runtime_import_does_not_require_genesis_or_torch(monkeypatch):
    import bb8_rl.interactive_runtime as runtime

    monkeypatch.setitem(sys.modules, "genesis", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    importlib.reload(runtime)


def test_worker_configuration_failure_emits_error_without_native_import(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "genesis", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    events = queue.Queue(maxsize=8)
    worker_main(
        {"mode": 4, "generation": 7},
        queue.Queue(),
        events,
        SimpleNamespace(is_set=lambda: False),
    )
    outputs = drain_commands(events)
    assert outputs[0]["type"] == "error"
    assert outputs[1]["state"]["phase"] == "error"
    assert outputs[1]["state"]["generation"] == 7
