"""Persistent completion, planner pagination and finite exploration regression."""

from types import SimpleNamespace

import numpy as np

from bb8_rl.agency import AgencyEngine, AgencyStore, Candidate
from bb8_rl.agency_runtime import AutonomyCoordinator


class OpenMemory:
    extent = 2.0

    def segment_free(self, a, b, radius):
        return True


class Session:
    def __init__(self, coordinator, pose=(0, 0)):
        self.agency = coordinator
        self.memory = OpenMemory()
        self.parameters = SimpleNamespace(robot_radius=0.05, clearance_margin=0.03)
        self.planning_radius = 0.10
        self.localization_lost = False
        self.phase = "idle"
        self.pose = pose
        self.goal = self.pending_goal = None
        self.queries = []
        self.allowed = None
        self.stops = 0

    def state(self):
        return {
            "pose": self.pose,
            "position_radius": 0.02,
            "phase": self.phase,
            "localization_valid": True,
            "localization_status": "measured",
        }

    def route_with_clearance(self, pose, xy, radius):
        self.queries.append(tuple(xy))
        if self.allowed is not None and tuple(xy) not in self.allowed:
            raise ValueError("Route blocked")
        return np.array([pose, xy])

    def stop(self):
        self.stops += 1

    def cancel(self, phase, message):
        self.stop()
        self.phase = phase
        self.goal = self.pending_goal = None


def seed(engine, candidate, outcome="arrived", count=1):
    engine.select([candidate], now=0, pose=(-3, -3))
    for i in range(count):
        engine.record_outcome(
            event_id=f"seed:{candidate.id}:{i}",
            candidate_id=candidate.id,
            outcome=outcome,
            now=float(i),
        )


def test_old_two_point_rewards_cannot_override_completion_after_reopen(tmp_path):
    path = tmp_path / "old-memory.sqlite"
    store = AgencyStore(path, map_version="same-map")
    engine = AgencyEngine(store)
    for c in [Candidate("place:1:3", (0.5, 1.5)), Candidate("place:0:2", (0, 1))]:
        seed(engine, c, count=50)
    store.close()
    store = AgencyStore(path, map_version="same-map")
    agency = AutonomyCoordinator(AgencyEngine(store))
    session = Session(agency, (0, 1))
    agency.enable(0)
    assert agency.tick(session, 0)
    assert agency.intention["candidate_id"] not in {"place:1:3", "place:0:2"}
    assert agency.snapshot()["completed_targets"] == 2
    assert (0.5, 1.5) not in session.queries
    store.close()


def test_farther_candidate_is_checked_after_first_twelve_blocked_routes():
    store = AgencyStore(":memory:")
    agency = AutonomyCoordinator(AgencyEngine(store))
    session = Session(agency)
    session.allowed = {(1.5, 1.5)}
    agency.enable(0)
    assert agency.tick(session, 0) is False
    assert agency.enabled and agency.search_pending
    assert len(session.queries) == 12
    for i in range(1, 8):
        before = len(session.queries)
        agency.tick(session, i * 0.1)
        assert len(session.queries) - before <= 12
        if agency.intention:
            break
    assert agency.intention["goal"] == [1.5, 1.5]
    store.close()


def test_all_completed_targets_pause_without_automatically_resetting_history():
    store = AgencyStore(":memory:")
    engine = AgencyEngine(store)
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            seed(engine, Candidate(f"place:{ix}:{iy}", (ix / 2, iy / 2)))
    agency = AutonomyCoordinator(engine)
    session = Session(agency)
    agency.enable(0)
    assert agency.tick(session, 0)
    assert not agency.enabled and agency.status == "exhausted"
    assert session.goal is None and not session.queries
    assert agency.snapshot()["completed_targets"] == 49
    agency.tick(session, 10000)
    assert session.goal is None and not agency.enabled
    assert AutonomyCoordinator(engine).snapshot()["completed_targets"] == 49
    store.close()


def test_unsuccessful_outcomes_are_not_completed_coverage():
    store = AgencyStore(":memory:")
    engine = AgencyEngine(store)
    outcomes = ["rejected", "cancelled", "localization_lost"]
    points = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    for outcome, xy in zip(outcomes, points):
        seed(engine, Candidate(f"place:{int(xy[0] * 2)}:{int(xy[1] * 2)}", xy), outcome)
    agency = AutonomyCoordinator(engine)
    session = Session(agency)
    # Constrain test memory to the three unfinished anchors.
    session.memory.segment_free = lambda a, b, r: tuple(a) in points
    offered = agency._candidates(session, session.state(), 0)
    assert {c.xy for c in offered} == set(points)
    assert agency.snapshot()["completed_targets"] == 0
    store.close()


def test_three_arrivals_choose_three_different_places():
    store = AgencyStore(":memory:")
    agency = AutonomyCoordinator(AgencyEngine(store))
    session = Session(agency, (0, 1))
    agency.enable(0)
    goals = []
    for i in range(6):
        t = i * 10.0
        agency.tick(session, t)
        assert agency.intention
        goal = tuple(agency.intention["goal"])
        assert goal not in goals
        goals.append(goal)
        session.pose, session.phase = goal, "arrived"
        agency.tick(session, t + 8)
    assert len(set(goals)) == 6
    assert agency.snapshot()["completed_targets"] == 6
    store.close()
