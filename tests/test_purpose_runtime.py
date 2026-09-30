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
        self.position_radius = 0.03
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
            "position_radius": self.position_radius,
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


@pytest.mark.parametrize("delayed", [False, True])
def test_blocked_route_recovers_without_enable_or_resource_change(fixture, delayed):
    c, s = fixture
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    assert c.status == "idle"
    if delayed:
        for now in (2, 6, 14, 30, 60, 90, 120):
            observe(c, now)
            assert not c.tick(s, now)
        now = 150
    else:
        now = 3  # The original audit's first recovery check.
    s.blocked = False
    observe(c, now)
    assert c.tick(s, now)
    assert c.enabled and c.status == "travelling"
    assert c.intention and s.goal is not None
    assert c.snapshot()["episodes"] == 0


def test_static_unreachable_routes_back_off_without_motion_or_idle_event_spam(fixture):
    c, s = fixture
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    for frame in range(1, 6001):
        now = frame / 10
        observe(c, now)
        assert not c.tick(s, now)
    # Ten minutes of 10 Hz ticks: only one initial scan and bounded backoff
    # queries, not 6,000 scans, decisions, cancellation events or actions.
    assert 20 * len(c.stations) <= len(s.routes) <= 25 * len(c.stations)
    assert len(s.cancellations) == 1
    assert c.snapshot()["decisions"] == c.snapshot()["episodes"] == 0
    assert c.intention is None and c.interaction_request() is None
    assert c.status == "idle" and s.goal is None


def test_learned_useless_stations_stay_quiet_even_when_routes_are_blocked(fixture):
    original, s = fixture
    engine = original.engine
    engine.select(list(DEFAULT_STATIONS), resource=0.35, pose=s.pose, now=0)
    for station in DEFAULT_STATIONS:
        for trial in range(4):
            engine.record_outcome(
                f"{station.id}-{trial}",
                station.id,
                before=0.35,
                after=0.35,
                now=trial,
            )
    c = PurposeCoordinator(engine, DEFAULT_STATIONS)
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    decisions = c.snapshot()["decisions"]
    for now in (3, 10, 100, 1000):
        observe(c, now)
        assert not c.tick(s, now)
    assert c.status == "idle" and c.intention is None
    assert s.routes == [] and len(s.cancellations) == 1
    assert c.snapshot()["decisions"] == decisions


def test_satisfied_idle_ignores_route_context_changes_and_camera_return(fixture):
    c, s = fixture
    observe(c, 0, 0.9)
    c.enable(0)
    c.tick(s, 0)
    for now in (3, 10, 100):
        s.pose = (now, now)
        s.position_radius = 0.06
        s.measured = False
        observe(c, now, 0.9)
        assert not c.tick(s, now)
        s.measured = True
        assert not c.tick(s, now)
    assert c.status == "satisfied"
    assert "Resource need satisfied" in c.message
    assert not s.routes and len(s.cancellations) == 1
    assert c.snapshot()["decisions"] == 0


def reject_only_station(c, s):
    c.stations = DEFAULT_STATIONS[:1]
    start(c, s)
    s.phase = "goal_rejected"
    observe(c, 1)
    assert c.tick(s, 1)
    assert c.blocked == {c.stations[0].id}


def test_failed_dispatched_goal_is_not_retried_on_identical_dry_route_success(fixture):
    c, s = fixture
    reject_only_station(c, s)
    for now in range(2, 602):
        # Sub-threshold camera noise must not renew motion authority.
        s.pose = (0.001 * (now % 2), 1.0)
        s.position_radius = 0.03 + 0.001 * (now % 2)
        observe(c, now)
        c.tick(s, now)
        assert c.intention is None and s.goal is None
    assert len(s.routes) <= 26
    assert len(s.cancellations) == 3
    assert c.snapshot()["decisions"] == 1
    assert c.snapshot()["episodes"] == 0


@pytest.mark.parametrize("changed", ["pose", "clearance", "route"])
def test_failed_goal_reopens_only_with_relevant_changed_route_evidence(
    fixture, changed
):
    c, s = fixture
    if changed == "clearance":
        s.position_radius = 0.10
    reject_only_station(c, s)
    if changed == "route":
        s.blocked = True
    observe(c, 2)
    c.tick(s, 2)
    assert c.intention is None
    if changed == "pose":
        s.pose = (0.2, 1.0)
    elif changed == "clearance":
        s.position_radius = 0.03
    else:
        s.blocked = False
    observe(c, 4)
    assert c.tick(s, 4)
    assert not c.blocked
    assert c.intention and c.status == "travelling"
    assert c.snapshot()["episodes"] == 0


def test_material_route_evidence_can_wake_backoff_but_not_on_every_frame(fixture):
    c, s = fixture
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    for now in (2, 6, 14, 30):
        observe(c, now)
        c.tick(s, now)
    routes = len(s.routes)
    s.blocked = False
    s.pose = (0.2, 1.0)
    observe(c, 30.1)
    assert not c.tick(s, 30.1)
    assert len(s.routes) == routes
    observe(c, 32)
    assert c.tick(s, 32)
    assert c.intention and c.enabled


@pytest.mark.parametrize("failure", ["stale", "lost", "heartbeat", "stop"])
def test_idle_retry_cannot_override_revoked_authority(fixture, failure):
    c, s = fixture
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    routes = len(s.routes)
    s.blocked = False
    if failure != "stale":
        observe(c, 3)
    if failure == "lost":
        s.localization_lost = True
    elif failure == "heartbeat":
        s.phase = "heartbeat_expired"
    elif failure == "stop":
        c.disable(3, "Stopped")
    c.tick(s, 3)
    assert not c.enabled and c.intention is None
    assert len(s.routes) == routes and s.goal is None
    s.localization_lost = False
    s.phase = "idle"
    observe(c, 100)
    assert not c.tick(s, 100)
    assert len(s.routes) == routes


def test_predicted_pose_cannot_admit_idle_route_retry(fixture):
    c, s = fixture
    s.blocked = True
    observe(c, 0)
    c.enable(0)
    c.tick(s, 0)
    routes = len(s.routes)
    s.blocked = False
    s.measured = False
    observe(c, 3)
    assert not c.tick(s, 3)
    assert len(s.routes) == routes and c.intention is None
    s.measured = True
    observe(c, 3.1)
    assert c.tick(s, 3.1)
    assert c.intention and c.enabled


def test_explicit_new_authority_clears_session_blocked_routes(fixture):
    c, s = fixture
    reject_only_station(c, s)
    c.disable(2, "Stopped")
    assert not c.blocked
    c.enable(10)
    observe(c, 10)
    assert c.tick(s, 10)
    assert c.intention and c.enabled


@pytest.mark.parametrize("resource", [0.35, 0.9])
def test_legacy_reversal_reconsideration_requires_enabled_measured_need(
    fixture, resource
):
    original, s = fixture
    engine = original.engine
    engine.select(list(DEFAULT_STATIONS), resource=0.35, pose=s.pose, now=0)
    for index in range(4):
        engine.record_outcome(
            f"old-a-{index}", "station-a", before=0.35, after=0.35, now=index
        )
    for index, after in enumerate((0.8, 0.8, 0.35, 0.35, 0.35, 0.35)):
        engine.record_outcome(
            f"old-b-{index}", "station-b", before=0.35, after=after, now=4 + index
        )
    # Recreate the audited old metadata using only this test's isolated memory.
    with engine.store.db:
        engine.store.db.execute("UPDATE purpose_scopes SET change_epoch=0")
        engine.store.db.execute(
            "UPDATE purpose_stations SET probes_remaining=0,last_change_epoch=0"
        )
    before = engine.snapshot()
    c = PurposeCoordinator(engine, DEFAULT_STATIONS)
    observe(c, 20, resource)
    assert not c.tick(s, 20)
    assert engine.snapshot() == before
    c.enable(20)
    s.measured = False
    assert not c.tick(s, 20)
    assert engine.snapshot() == before
    s.measured = True
    c.tick(s, 20)
    if resource < 0.8:
        assert c.intention["candidate_id"] == "station-a"
        assert engine.snapshot()["change_epoch"] == 1
        assert c.status == "travelling"
    else:
        assert engine.snapshot() == before
        assert c.intention is None and c.status == "satisfied"
        assert not s.routes


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


def control_session(coordinator):
    from bb8_rl.interactive_runtime import ControlSession

    class FreeMap:
        def certified_route(self, start, goal, radius_m, *, grid=None):
            return (self if grid is None else grid).route(start, goal)

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
        agency_coordinator=coordinator,
        clock=lambda: 0.0,
    )
    frame = SimpleNamespace(
        timestamp=0.0, status="visible", xy=np.zeros(2), covariance=np.eye(2) * 0.002**2
    )
    return control, frame


@pytest.mark.parametrize("version", ["map_version", "calibration_version"])
def test_control_session_stop_wins_and_version_change_cancels(fixture, version):
    c, _ = fixture
    control, frame = control_session(c)
    observe(c, 0)
    control.decide(0, frame, np.zeros(2), [])
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )
    assert c.enabled
    control.receive(
        [
            {"action": "stop", "generation": 2},
            {"action": "autonomy", "enabled": True, "generation": 3},
        ],
        sim_time=0,
    )
    assert not c.enabled and control.goal is None
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 4}], sim_time=0
    )
    setattr(control, version, "changed")
    assert np.all(control.decide(0, frame, np.zeros(2), []) == 0)
    assert not c.enabled and control.goal is None


@pytest.mark.parametrize("action", ["goal", "demo"])
def test_manual_destination_revokes_pending_idle_retry(fixture, monkeypatch, action):
    c, _ = fixture
    control, frame = control_session(c)
    observe(c, 0)
    control.decide(0, frame, np.zeros(2), [])
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )

    def unavailable(*args):
        raise ValueError("Temporarily no route")

    with monkeypatch.context() as patch:
        patch.setattr(control, "route_with_clearance", unavailable)
        assert c.tick(control, 0)
        assert c.status == "idle"
    control.receive(
        [{"action": action, "generation": 2, "x": 0.5, "y": 0.5}], sim_time=0
    )
    assert not c.enabled and control.pending_goal is not None
    manual_goal = control.pending_goal.copy()
    for now in (3, 100):
        observe(c, now)
        assert not c.tick(control, now)
        np.testing.assert_array_equal(control.pending_goal, manual_goal)
    assert c.intention is None and c.interaction_request() is None
