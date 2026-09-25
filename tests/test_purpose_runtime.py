"""Outcome learning requires authorized fresh interaction evidence, not arrival."""

from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.interaction_environment import DEFAULT_STATIONS, SOURCE
from bb8_rl.purpose import PurposeEngine, PurposeStore
from bb8_rl.purpose_runtime import PurposeCoordinator


class Session:
    def __init__(self):
        self.localization_lost = False
        self.phase = "idle"
        self.pose = (0.0, 1.0)
        self.measured = True
        self.planning_radius = 0.1
        self.parameters = SimpleNamespace(robot_radius=0.04, clearance_margin=0.03)
        self.pending_goal = self.goal = None
        self.requires_new_goal = False
        self.routes, self.cancellations = [], []
        self.blocked = False

    def state(self):
        return {
            "phase": self.phase,
            "localization_valid": not self.localization_lost,
            "localization_status": "measured" if self.measured else "predicted",
            "pose": self.pose,
            "position_radius": 0.03,
        }

    def route_with_clearance(self, start, goal, radius):
        self.routes.append((tuple(start), tuple(goal), radius))
        if self.blocked:
            raise ValueError("No route")

    def cancel(self, phase, message):
        self.cancellations.append(message)
        self.phase = phase
        self.goal = self.pending_goal = None


@pytest.fixture
def fixture(tmp_path):
    store = PurposeStore(tmp_path / "purpose.sqlite3", "map")
    coordinator = PurposeCoordinator(PurposeEngine(store), DEFAULT_STATIONS)
    yield coordinator, Session()
    store.close()


def observe(coordinator, now, resource=0.35, outcome=None):
    observation = {"resource": resource, "timestamp": now, "source": SOURCE}
    if outcome is not None:
        observation["outcome"] = outcome
    coordinator.observe(observation)


def start(coordinator, session):
    observe(coordinator, 0)
    coordinator.enable(0)
    assert coordinator.tick(session, 0)
    assert coordinator.intention
    assert tuple(session.goal) in [s.xy for s in DEFAULT_STATIONS]
    assert session.routes
    assert coordinator.snapshot()["episodes"] == 0


def arrive(coordinator, session):
    session.phase = "arrived"
    session.pose = tuple(coordinator.intention["goal"])
    observe(coordinator, 1)
    coordinator.tick(session, 1)
    assert coordinator.status == "interacting"
    return coordinator.interaction_request()


def test_arrival_never_rewards_and_only_station_response_learns(fixture):
    c, s = fixture
    start(c, s)
    request = arrive(c, s)
    assert c.snapshot()["episodes"] == 0
    observe(c, 1.5)
    c.tick(s, 1.5)
    assert c.snapshot()["episodes"] == 0
    outcome = dict(**request, before=0.35, after=0.35, timestamp=2.1, source=SOURCE)
    observe(c, 2.1, outcome=outcome)
    c.tick(s, 2.1)
    assert c.snapshot()["episodes"] == 1
    assert c.recent_experience["outcome"] == "no_effect"
    assert c.interaction_request() is None
    observe(c, 2.15, outcome=outcome)
    c.tick(s, 2.15)
    assert c.snapshot()["episodes"] == 1


@pytest.mark.parametrize("mutation", ["request", "station", "source", "stale", "early"])
def test_invalid_receipts_cannot_train(fixture, mutation):
    c, s = fixture
    start(c, s)
    request = arrive(c, s)
    outcome = dict(**request, before=0.35, after=0.8, timestamp=2.1, source=SOURCE)
    now = 2.1
    if mutation == "request":
        outcome["request_id"] = "another"
    elif mutation == "station":
        outcome["station_id"] = "another"
    elif mutation == "source":
        outcome["source"] = "camera_inferred"
    elif mutation == "stale":
        outcome["timestamp"] = 1.5
    elif mutation == "early":
        outcome["timestamp"] = now = 1.5
    observe(c, now, 0.8, outcome)
    assert c.tick(s, now)
    assert not c.enabled and c.interaction_request() is None
    assert c.snapshot()["episodes"] == 0
    assert s.goal is None


@pytest.mark.parametrize("failure", ["lost", "predicted", "stale", "stop"])
def test_interrupted_interaction_does_not_learn(fixture, failure):
    c, s = fixture
    start(c, s)
    request = arrive(c, s)
    if failure == "lost":
        s.localization_lost = True
    elif failure == "predicted":
        s.measured = False
    elif failure == "stop":
        c.disable(1.5, "Stopped")
    if failure != "stale":
        observe(
            c,
            2.1,
            0.8,
            dict(**request, before=0.35, after=0.8, timestamp=2.1, source=SOURCE),
        )
    c.tick(s, 2.1)
    assert not c.enabled and c.snapshot()["episodes"] == 0
    assert c.interaction_request() is None


def test_satisfied_idle_does_not_keep_deciding_or_moving(fixture):
    c, s = fixture
    observe(c, 0, 0.9)
    c.enable(0)
    assert c.tick(s, 0)
    assert c.status == "satisfied" and s.goal is None
    decisions = c.snapshot()["decisions"]
    routes = len(s.routes)
    for now in (3, 10, 100):
        observe(c, now, 0.9)
        assert not c.tick(s, now)
    assert c.snapshot()["decisions"] == decisions
    assert len(s.routes) == routes


def test_unreachable_station_is_not_learned_as_ineffective(fixture):
    c, s = fixture
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    assert c.status == "idle" and c.intention is None
    assert c.snapshot()["episodes"] == 0


def test_memory_failure_cancels_authority(fixture, monkeypatch):
    c, s = fixture
    observe(c, 0)
    c.enable(0)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(c.engine, "select", fail)
    assert c.tick(s, 0)
    assert not c.enabled and not c.available and s.goal is None


def test_restart_retains_effects_but_not_authority(fixture):
    c, s = fixture
    start(c, s)
    request = arrive(c, s)
    observe(
        c,
        2.1,
        0.8,
        dict(**request, before=0.35, after=0.8, timestamp=2.1, source=SOURCE),
    )
    c.tick(s, 2.1)
    restarted = PurposeCoordinator(c.engine, DEFAULT_STATIONS)
    assert restarted.snapshot()["episodes"] == 1
    assert not restarted.enabled and restarted.intention is None


def test_control_session_stop_wins_and_map_change_cancels(fixture):
    from bb8_rl.interactive_runtime import ControlSession

    c, _ = fixture

    class FreeMap:
        def planning_grid(self, radius):
            return self

        def segment_free(self, *args):
            return True

        def route(self, start, goal):
            return np.array([start, goal])

    control = ControlSession(
        policy=lambda _: np.zeros(2),
        memory=FreeMap(),
        stop=lambda: None,
        map_version="map",
        calibration_version="rig",
        demo_goal=(0.5, 1.5),
        agency_coordinator=c,
        clock=lambda: 0.0,
    )
    frame = SimpleNamespace(
        timestamp=0.0, status="visible", xy=np.zeros(2), covariance=np.eye(2) * 0.002**2
    )
    observe(c, 0)
    control.decide(0, frame, np.zeros(2), [])
    control.receive([{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0)
    assert c.enabled
    control.receive(
        [
            {"action": "stop", "generation": 2},
            {"action": "autonomy", "enabled": True, "generation": 3},
        ],
        sim_time=0,
    )
    assert not c.enabled and control.goal is None
    control.receive([{"action": "autonomy", "enabled": True, "generation": 4}], sim_time=0)
    control.map_version = "changed"
    assert np.all(control.decide(0, frame, np.zeros(2), []) == 0)
    assert not c.enabled and control.goal is None
