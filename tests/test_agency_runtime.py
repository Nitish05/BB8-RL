"""Agency cannot turn its preference scores into authority over navigation."""

from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.agency import AgencyEngine, AgencyStore
from bb8_rl.interactive import validate_command
from bb8_rl.interactive_runtime import ControlSession


class Memory:
    extent = 1.0

    def __init__(self):
        self.queries = []
        self.block_routes = False

    def planning_grid(self, radius):
        return self

    def segment_free(self, a, b, radius):
        self.queries.append((tuple(a), tuple(b), radius))
        return max(np.max(np.abs(a)), np.max(np.abs(b))) < 0.9

    def route(self, start, goal):
        if self.block_routes:
            raise ValueError("No certified route")
        return np.array([start, goal])


@pytest.fixture
def control(tmp_path):
    store = AgencyStore(tmp_path / "memory.sqlite3", map_version="map")
    result = ControlSession(
        policy=lambda _: np.array([0.4, 0.0]),
        memory=Memory(),
        stop=lambda: None,
        map_version="map",
        calibration_version="rig",
        demo_goal=[0.5, 0],
        agency_engine=AgencyEngine(store),
        clock=lambda: 10.0,
    )
    yield result
    store.close()


def frame(control, index, visible=True):
    t = index * 0.05
    measurement = SimpleNamespace(
        timestamp=t,
        status="visible" if visible else "missing",
        xy=np.zeros(2) if visible else None,
        covariance=np.eye(2) * 0.002**2 if visible else None,
    )
    intervals = [] if index == 0 else [{"start": t - 0.05, "end": t, "command": [0, 0]}]
    return control.decide(t, measurement, np.zeros(2), intervals)


def start(control):
    frame(control, 0)
    records = control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0.0
    )
    assert records[0]["outcome"] == "accepted"
    action = frame(control, 1)
    assert np.all(action == 0)  # Selection itself never emits motor commands.
    assert control.agency.intention


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", [], {}])
def test_autonomy_is_explicit_boolean(enabled):
    with pytest.raises(ValueError):
        validate_command({"action": "autonomy", "enabled": enabled})


def test_only_current_certified_candidates_and_no_learning_from_proposals(control):
    start(control)
    assert control.controller is None  # The next frame must re-certify the route.
    assert control.state()["agency"]["episodes"] == 0
    goal = tuple(control.goal)
    assert any(
        a == b == goal and radius >= control.planning_radius
        for a, b, radius in control.memory.queries
    )
    frame(control, 2)
    assert control.controller is not None
    assert control.route_planning_details["route_certified"]


def test_disconnected_routes_cannot_become_intentions(control):
    control.memory.block_routes = True
    frame(control, 0)
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )
    assert np.all(frame(control, 1) == 0)
    assert control.agency.intention is None
    assert control.goal is None


@pytest.mark.parametrize("action", ["stop", "pause"])
def test_stop_and_pause_win_batch_over_new_enable(control, action):
    start(control)
    command = (
        {"action": "stop", "generation": 2}
        if action == "stop"
        else {"action": "autonomy", "enabled": False, "generation": 2}
    )
    control.receive(
        [command, {"action": "autonomy", "enabled": True, "generation": 3}],
        sim_time=0.1,
    )
    assert not control.agency.enabled
    assert control.goal is None
    assert control.agency.intention is None
    events = control.agency.snapshot()["episodes"]
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 3}], sim_time=0.1
    )
    frame(control, 2)
    assert not control.agency.enabled
    assert control.agency.snapshot()["episodes"] == events == 1
    assert all(p["visits"] == 0 for p in control.agency.snapshot()["preferences"])


def test_manual_goal_takes_over_without_autonomous_followup(control):
    start(control)
    control.receive(
        [{"action": "goal", "x": 0.5, "y": 0, "generation": 2}], sim_time=0.1
    )
    assert not control.agency.enabled
    assert control.goal.tolist() == [0.5, 0]
    assert control.agency.recent_experience["outcome"] == "cancelled"


def test_heartbeat_expiry_retains_memory_but_never_restarts(control):
    start(control)
    assert not control.heartbeat_guard(14)
    assert not control.agency.enabled
    control.receive([{"action": "heartbeat", "generation": 2}], wall_time=14)
    frame(control, 2)
    assert not control.agency.enabled
    assert control.goal is None


def test_expired_localization_cancels_intention(control):
    start(control)
    for i in range(2, 27):
        frame(control, i, visible=False)
    assert control.localization_lost
    assert not control.agency.enabled
    assert control.agency.recent_experience["outcome"] == "localization_lost"
    for i in range(27, 40):
        frame(control, i)
    assert not control.agency.enabled
    assert control.goal is None


def test_memory_failure_stops_and_disables(control, monkeypatch):
    frame(control, 0)
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )

    def fail(*args, **kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(control.agency.engine, "select", fail)
    assert np.all(frame(control, 1) == 0)
    assert not control.agency.enabled and not control.agency.available
    assert control.goal is None


def test_map_change_cannot_reuse_old_preferences_to_authorize_motion(control):
    start(control)
    control.map_version = "different"
    assert np.all(frame(control, 2) == 0)
    assert not control.agency.enabled
    assert control.goal is None


def test_restart_preserves_outcome_but_does_not_enable(control):
    start(control)
    control.phase = "arrived"  # Simulate the terminal event from the navigation layer.
    control.agency.tick(control, 1.0)
    assert control.agency.snapshot()["episodes"] == 1
    control.agency.tick(control, 1.05)
    assert control.agency.snapshot()["episodes"] == 1
    from bb8_rl.agency_runtime import AutonomyCoordinator

    restarted = AutonomyCoordinator(control.agency.engine)
    assert not restarted.enabled and restarted.intention is None
    assert restarted.snapshot()["episodes"] == 1
    assert max(p["value"] for p in restarted.snapshot()["preferences"]) > 0
