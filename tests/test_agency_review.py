"""Independent integration checks for optional-memory failure and goal authority."""

from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.agency import AgencyEngine, AgencyStore
from bb8_rl.agency_runtime import AutonomyCoordinator
from bb8_rl.interactive_runtime import ControlSession


class OpenMemory:
    extent = 1.0

    def certified_route(self, start, goal, radius_m, *, grid=None):
        return (self if grid is None else grid).route(start, goal)

    def planning_grid(self, _radius):
        return self

    def segment_free(self, start, goal, _radius):
        return max(np.max(np.abs(start)), np.max(np.abs(goal))) < 0.9

    def route(self, start, goal):
        return np.array([start, goal])


@pytest.fixture
def review_control(tmp_path):
    store = AgencyStore(tmp_path / "review.sqlite", map_version="review-map")
    control = ControlSession(
        policy=lambda _: np.zeros(2),
        memory=OpenMemory(),
        stop=lambda: None,
        map_version="review-map",
        calibration_version="review-camera",
        demo_goal=[0.5, 0],
        agency_engine=AgencyEngine(store),
        clock=lambda: 10.0,
    )
    yield control
    store.close()


def frame(control, tick):
    timestamp = tick * 0.05
    measurement = SimpleNamespace(
        timestamp=timestamp,
        status="visible",
        xy=np.zeros(2),
        covariance=np.eye(2) * 0.002**2,
    )
    intervals = (
        []
        if tick == 0
        else [{"start": timestamp - 0.05, "end": timestamp, "command": [0, 0]}]
    )
    return control.decide(timestamp, measurement, np.zeros(2), intervals)


def enable(control):
    frame(control, 0)
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )
    frame(control, 1)
    assert control.agency.intention is not None, control.agency.snapshot()


def unavailable(*_args, **_kwargs):
    raise OSError("Memory temporarily unavailable")


def test_stop_succeeds_when_cancellation_cannot_be_written(review_control, monkeypatch):
    control = review_control
    enable(control)
    monkeypatch.setattr(control.agency.engine, "record_outcome", unavailable)
    records = control.receive([{"action": "stop", "generation": 2}], sim_time=0.1)
    assert records[0]["outcome"] == "accepted"
    assert control.goal is None and control.controller is None
    assert not control.agency.enabled and not control.agency.available
    assert np.all(frame(control, 2) == 0)


def test_manual_goal_remains_usable_after_memory_failure(review_control, monkeypatch):
    control = review_control
    enable(control)
    monkeypatch.setattr(control.agency.engine, "record_outcome", unavailable)
    records = control.receive(
        [{"action": "goal", "x": 0.5, "y": 0.0, "generation": 2}],
        sim_time=0.1,
    )
    assert records[0]["outcome"] == "accepted_pending_visible_route"
    assert control.goal.tolist() == [0.5, 0.0]
    assert not control.agency.enabled and not control.agency.available
    frame(control, 2)
    assert control.controller is not None


def test_reenable_reports_memory_failure_without_killing_control(
    review_control, monkeypatch
):
    control = review_control
    enable(control)
    monkeypatch.setattr(control.agency.engine, "record_outcome", unavailable)
    records = control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 2}],
        sim_time=0.1,
    )
    assert records[0]["outcome"] == "rejected"
    assert control.goal is None and not control.agency.enabled
    assert not control.agency.available


def test_startup_snapshot_failure_only_disables_optional_agency():
    coordinator = AutonomyCoordinator(SimpleNamespace(snapshot=unavailable))
    assert not coordinator.available and not coordinator.enabled
    assert coordinator.snapshot()["status"] == "unavailable"


def test_proposal_coordinates_never_override_certified_candidate(
    review_control, monkeypatch
):
    control = review_control
    frame(control, 0)
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )
    original_select = control.agency.engine.select

    def corrupt_coordinates(*args, **kwargs):
        chosen = original_select(*args, **kwargs)
        return {**chosen, "goal": [1000, 1000], "label": "Untrusted proposal"}

    monkeypatch.setattr(control.agency.engine, "select", corrupt_coordinates)
    assert np.all(frame(control, 1) == 0)
    assert max(abs(x) for x in control.goal) < 0.9
    assert control.agency.intention["label"].startswith("Place (")


def test_unknown_proposal_id_cancels_instead_of_accepting_coordinates(
    review_control, monkeypatch
):
    control = review_control
    frame(control, 0)
    control.receive(
        [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0
    )
    monkeypatch.setattr(
        control.agency.engine,
        "select",
        lambda *_args, **_kwargs: {"candidate_id": "invented", "goal": [0.5, 0]},
    )
    assert np.all(frame(control, 1) == 0)
    assert control.goal is None and not control.agency.enabled
    assert control.agency.snapshot()["episodes"] == 0
