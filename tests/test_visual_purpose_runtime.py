"""Pixel fixtures exercise the real pure control path; no native evidence claim."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_visual_interaction import calibration, render

from bb8_rl.camera_rig import CameraFrame
from bb8_rl.interactive_runtime import ControlSession
from bb8_rl.purpose import PurposeEngine, PurposeStore
from bb8_rl.visual_interaction import PixelVisualObserver
from bb8_rl.visual_purpose_runtime import VisualPurposeCoordinator


class ClearMemory:
    def certified_route(self, start, goal, radius_m, *, grid=None):
        return (self if grid is None else grid).route(start, goal)

    def planning_grid(self, radius):
        return self

    def segment_free(self, start, goal, radius):
        return max(abs(np.asarray(start))) < 1 and max(abs(np.asarray(goal))) < 1

    def route(self, start, goal):
        return np.asarray([start, goal])


class VisualLoop:
    """Stationary synthetic RGB, real ControlSession authority and learner."""

    def __init__(self, path):
        self.store = PurposeStore(path, "visual-map")
        self.engine = PurposeEngine(self.store)
        self.agency = VisualPurposeCoordinator(self.engine, "cal-v1")
        self.clock = [10.0]
        self.stops = []
        self.control = ControlSession(
            policy=lambda _vector: np.zeros(2),
            memory=ClearMemory(),
            stop=lambda: self.stops.append(True),
            map_version="visual-map",
            calibration_version="nav-cal",
            demo_goal=[0, 0],
            agency_coordinator=self.agency,
            clock=lambda: self.clock[0],
        )
        self.observer = PixelVisualObserver(calibration(), "semantic-A", "cal-v1")
        self.before = render((("marker-01", (0, 0.25), 11),))
        self.after = render((("marker-01", (0, 0.25), 27),))
        self.request = self.request_started = self.completion = None
        self.index = 0

    def step(self, *, changed=False, rgb=None, timestamp=None, measurement="visible"):
        t = round(self.index * 0.05, 6)
        pixels = self.after if changed else self.before
        frame = CameraFrame(
            "semantic-A",
            "cal-v1",
            pixels if rgb is None else rgb,
            t if timestamp is None else timestamp,
        )
        scene = self.observer.observe(frame, now=t)
        correlated = self.agency.completion_for(
            self.completion, self.control.generation
        )
        self.agency.feed_visual(scene, correlated)
        intervals = (
            []
            if self.index == 0
            else [{"start": t - 0.05, "end": t, "command": [0, 0]}]
        )
        observed = SimpleNamespace(
            timestamp=t,
            status=measurement,
            xy=np.zeros(2) if measurement == "visible" else None,
            covariance=np.eye(2) * 0.002**2 if measurement == "visible" else None,
        )
        action = self.control.decide(t, observed, [0, 0], intervals)
        if self.request is None and self.agency.world_request() is not None:
            self.request = self.agency.world_request()
            self.request_started = t
        self.index += 1
        return action

    def start_request(self):
        for _ in range(6):
            self.step()
        assert self.control.state()["localization_valid"]
        assert self.engine.snapshot()["episodes"] == 0
        assert self.agency.gate.generation is None
        record = self.control.receive(
            [{"action": "autonomy", "enabled": True, "generation": 1}], sim_time=0.30
        )
        assert record[0]["outcome"] == "accepted"
        for _ in range(70):
            self.step()
            if self.request:
                break
        assert self.request and self.agency.gate.pending
        assert self.request["station_id"] == "marker-01"
        assert self.agency.interaction_request()["station_id"] != "marker-01"
        return self

    def finish(self):
        for _ in range(50):
            t = round(self.index * 0.05, 6)
            if self.completion is None and t >= self.request_started + 1.0 - 1e-9:
                self.completion = {**self.request, "timestamp": t}
            self.step(changed=self.completion is not None)
        return self.engine.snapshot()


@pytest.fixture
def loop(tmp_path):
    value = VisualLoop(tmp_path / "fresh-visual.sqlite3")
    yield value
    value.store.close()


def test_rgb_control_loop_learns_once_and_restart_requires_fresh_pixels_and_enable(
    loop,
):
    loop.start_request()
    assert loop.engine.snapshot()["episodes"] == 0
    snapshot = loop.finish()
    assert snapshot["episodes"] == 1
    assert snapshot["source"] == "native_scene_rgb"
    assert loop.agency.last_visual_outcome["classification"] == "useful"
    assert loop.agency.last_visual_outcome["gain_interval"][0] >= 0.04
    assert loop.agency.status == "satisfied"
    assert loop.agency.interaction_request() is None
    assert loop.control.goal is None
    restarted = VisualPurposeCoordinator(PurposeEngine(loop.store), "cal-v1")
    assert restarted.snapshot()["episodes"] == 1
    assert not restarted.enabled and restarted.gate.generation is None
    assert restarted.stations == () and restarted.world_request() is None


@pytest.mark.parametrize(
    "fault",
    [
        "stop",
        "manual",
        "generation",
        "heartbeat",
        "missing",
        "stale",
        "moved",
        "predicted",
    ],
)
def test_pending_visual_work_cannot_train_after_authority_or_evidence_loss(loop, fault):
    loop.start_request()
    t = round(loop.index * 0.05, 6)
    if fault == "stop":
        loop.control.receive([{"action": "stop", "generation": 2}], sim_time=t)
    elif fault == "manual":
        loop.control.receive(
            [{"action": "goal", "x": 0.2, "y": 0, "generation": 2}], sim_time=t
        )
    elif fault == "generation":
        loop.control.generation += 1
    elif fault == "heartbeat":
        loop.clock[0] += 4
    elif fault == "missing":
        loop.step(rgb=np.zeros_like(loop.before))
    elif fault == "stale":
        loop.step(timestamp=t - 0.05)
    elif fault == "moved":
        loop.step(rgb=render((("marker-01", (0.15, 0.25), 11),)))
    elif fault == "predicted":
        loop.step(measurement="missing")
    assert loop.finish()["episodes"] == 0
    assert not loop.agency.enabled
    assert loop.agency.gate.pending is None
    assert loop.agency.world_request() is None


def test_completion_correlation_rejects_numeric_telemetry_and_malformed_fields(loop):
    loop.start_request()
    valid = {**loop.request, "timestamp": loop.request_started + 1.0}
    for bad in [
        {**valid, "before": 0.3, "after": 0.9},
        {**valid, "resource": 0.9},
        {key: value for key, value in valid.items() if key != "timestamp"},
        {**valid, "timestamp": float("nan")},
        {**valid, "timestamp": True},
        {**valid, "request_id": "other"},
        {**valid, "station_id": "marker-02"},
    ]:
        assert loop.agency.completion_for(bad, 1) is None
    assert loop.agency.completion_for(valid, True) is None
    assert loop.agency.completion_for(valid, 2) is None
    assert (
        loop.agency.completion_for(valid, 1)["entity_id"]
        == loop.agency.interaction_request()["station_id"]
    )


def test_failed_rgb_evidence_retention_cancels_without_learning(loop):
    calls = []

    def fail_storage(evidence):
        calls.append(evidence["request_id"])
        assert loop.engine.snapshot()["episodes"] == 0
        raise OSError("simulated evidence storage failure")

    loop.agency.evidence_retainer = fail_storage
    loop.start_request()
    assert loop.finish()["episodes"] == 0
    assert len(calls) == 1
    assert not loop.agency.enabled
    assert loop.agency.gate.pending is None
    assert loop.agency.world_request() is None
    assert loop.agency.last_visual_outcome is None
    assert loop.control.goal is None
    assert "retained" in loop.agency.message


def test_evidence_is_retained_before_learning_transaction(loop):
    retained = []

    def retain(evidence):
        assert loop.engine.snapshot()["episodes"] == 0
        assert (
            loop.store.db.execute("SELECT count(*) FROM purpose_events").fetchone()[0]
            == 0
        )
        retained.append(copy.deepcopy(evidence))

    loop.agency.evidence_retainer = retain
    loop.start_request()
    assert loop.finish()["episodes"] == 1
    assert retained == [loop.agency.last_visual_outcome]


def rehash(receipt):
    receipt.pop("evidence_sha256")
    receipt["evidence_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "change", ["map", "entity", "marker", "calibration", "anchor", "radius", "source"]
)
def test_visual_engine_requires_exact_durable_identity_scope_and_source(loop, change):
    loop.start_request().finish()
    original = loop.agency.last_visual_outcome
    evidence = copy.deepcopy(original)
    evidence["request_id"] = evidence["completion"]["request_id"] = "different-event"
    if change == "map":
        evidence["map_version"] = "other-map"
    elif change == "entity":
        evidence["entity_id"] = evidence["completion"]["entity_id"] = "other-entity"
    elif change in ("marker", "calibration"):
        field, value = (
            ("marker_id", "marker-02")
            if change == "marker"
            else ("calibration_version", "other-calibration")
        )
        evidence[field] = value
        for key in ("before_frames", "during_frames", "after_frames"):
            for frame in evidence[key]:
                frame[field] = value
    elif change == "anchor":
        evidence["anchor_xy"][0] += 0.001
    elif change == "radius":
        evidence["anchor_radius_m"] += 0.001
    rehash(evidence)
    with pytest.raises(ValueError):
        loop.engine.record_outcome(
            evidence["request_id"],
            evidence["entity_id"],
            before=evidence["before"],
            after=evidence["after"],
            now=evidence["timestamp"],
            source="simulated_station_telemetry"
            if change == "source"
            else "native_scene_rgb",
            visual_evidence=evidence,
        )
    assert loop.engine.snapshot()["episodes"] == 1


def test_resource_uncertainty_does_not_mark_straddling_target_satisfied(loop):
    for _ in range(6):
        loop.step(rgb=render((("marker-01", (0, 0.25), 26),)))
    assert loop.agency.observation["resource"] == (26 - 0.5) / 32 < 0.8
    assert not loop.agency.enabled


@pytest.mark.parametrize("fault", ["hidden_post_action", "missing_completion"])
def test_changed_pixels_without_complete_post_action_evidence_never_train(loop, fault):
    loop.start_request()
    completion_time = loop.request_started + 1.0
    hidden_once = False
    for _ in range(130):
        t = round(loop.index * 0.05, 6)
        changed = t >= completion_time - 1e-9
        if changed and fault == "hidden_post_action" and loop.completion is None:
            loop.completion = {**loop.request, "timestamp": t}
        hide = (
            fault == "hidden_post_action"
            and t > completion_time + 0.05
            and not hidden_once
        )
        loop.step(changed=changed, rgb=np.zeros_like(loop.before) if hide else None)
        hidden_once |= hide
    assert loop.engine.snapshot()["episodes"] == 0
    assert not loop.agency.enabled and loop.agency.gate.pending is None
    assert loop.agency.world_request() is None


def test_telemetry_cannot_enter_a_scope_that_already_contains_visual_learning(loop):
    loop.start_request().finish()
    station_id = loop.agency.last_visual_outcome["entity_id"]
    with pytest.raises(ValueError, match="Different evidence sources"):
        loop.engine.record_outcome(
            "telemetry-injection",
            station_id,
            before=0.3,
            after=0.9,
            now=10,
            source="simulated_station_telemetry",
        )
    assert loop.engine.snapshot()["episodes"] == 1
    payload = loop.store.db.execute("SELECT payload FROM purpose_events").fetchone()[0]
    assert json.loads(payload)["source"] == "native_scene_rgb"


def test_failed_sql_commit_does_not_publish_uncommitted_visual_outcome(
    loop, monkeypatch
):
    import sqlite3

    retained = []
    loop.agency.evidence_retainer = lambda evidence: retained.append(
        copy.deepcopy(evidence)
    )

    def fail_commit(*args, **kwargs):
        raise sqlite3.OperationalError("simulated database write failure")

    monkeypatch.setattr(loop.engine, "record_outcome", fail_commit)
    loop.start_request()
    assert loop.finish()["episodes"] == 0
    assert len(retained) == 1
    assert (
        loop.store.db.execute("SELECT count(*) FROM purpose_events").fetchone()[0] == 0
    )
    assert loop.agency.snapshot()["visual_outcome"] is None
    assert not loop.agency.enabled
    assert loop.agency.status == "unavailable"
    assert loop.agency.gate.pending is None
    assert loop.agency.world_request() is None
    assert loop.control.goal is None


def test_exact_deduplicated_commit_still_publishes_the_single_persisted_receipt(
    loop, monkeypatch
):
    original = loop.engine.record_outcome

    def already_committed(*args, **kwargs):
        assert original(*args, **kwargs) is True
        assert original(*args, **kwargs) is False
        return False

    monkeypatch.setattr(loop.engine, "record_outcome", already_committed)
    loop.start_request()
    assert loop.finish()["episodes"] == 1
    payload = json.loads(
        loop.store.db.execute("SELECT payload FROM purpose_events").fetchone()[0]
    )
    assert loop.agency.snapshot()["visual_outcome"] == payload["visual_evidence"]


def test_archive_directory_sync_failure_cancels_before_sql_commit(
    loop, tmp_path, monkeypatch
):
    from bb8_rl import visual_evidence

    archive = visual_evidence.VisualEvidenceArchive(tmp_path)
    archive.capture(loop.before, step=0, timestamp=0)
    archive.capture(loop.after, step=1, timestamp=0.05)
    loop.agency.evidence_retainer = archive.retain_outcome

    def fail_directory_sync(_directory):
        raise OSError("directory sync failed")

    monkeypatch.setattr(visual_evidence, "_sync_directory", fail_directory_sync)
    loop.start_request()
    assert loop.finish()["episodes"] == 0
    assert (
        loop.store.db.execute("SELECT count(*) FROM purpose_events").fetchone()[0] == 0
    )
    assert loop.agency.snapshot()["visual_outcome"] is None
    assert not loop.agency.enabled and loop.control.goal is None
    assert not archive.retained
