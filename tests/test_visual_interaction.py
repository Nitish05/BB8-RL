"""Contract images are synthetic test inputs, not native rendering evidence."""

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from bb8_rl.camera import Calibration
from bb8_rl.camera_rig import CameraFrame
from bb8_rl.purpose import PurposeEngine, PurposeStore
from bb8_rl.visual_interaction import (
    ID_COLOR,
    ID_PATTERNS,
    PANEL_HEIGHT_M,
    AssociatedEntity,
    PixelVisualObserver,
    VisualEntityRegistry,
    VisualOutcomeGate,
    VisualReading,
    VisualSceneObservation,
    gauge_interval,
    gauge_rect,
    validate_visual_outcome,
)


def calibration():
    return Calibration(
        np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]]),
        np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 3], [0, 0, 0, 1]]),
        (1024, 768),
        2.0,
    )


def panel(code="marker-01", level=11):
    board = np.zeros((160, 280, 3), np.uint8)
    board[:10] = board[-10:] = [240, 0, 240]
    board[:, :10] = board[:, -10:] = [240, 0, 240]
    for y, row in enumerate(ID_PATTERNS[code]):
        for x, bit in enumerate(row):
            board[(y + 1) * 20 : (y + 2) * 20, (x + 1) * 20 : (x + 2) * 20] = (
                np.asarray(ID_COLOR[:3]) * 240 * bit
            )
    for index in range(level):
        x0, y0, x1, y1 = gauge_rect(index)
        board[round(y0 * 20) : round(y1 * 20), round(x0 * 20) : round(x1 * 20)] = [
            0,
            230,
            0,
        ]
    return board


def render(
    panels=(("marker-01", (-0.3, 0.3), 11), ("marker-02", (0.3, 0.3), 11)), angle=0
):
    rgb = np.full((768, 1024, 3), [30, 50, 60], np.uint8)
    cal = calibration()
    for code, center, level in panels:
        local = np.array(
            [[-0.175, 0.10], [0.175, 0.10], [0.175, -0.10], [-0.175, -0.10]]
        )
        theta = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        pixels = cal.to_pixel(local @ rotation.T + center, PANEL_HEIGHT_M).astype(
            np.float32
        )
        transform = cv2.getPerspectiveTransform(
            np.array([[0, 0], [280, 0], [280, 160], [0, 160]], np.float32), pixels
        )
        board = cv2.warpPerspective(panel(code, level), transform, (1024, 768))
        mask = (
            cv2.warpPerspective(
                np.full((160, 280), 255, np.uint8), transform, (1024, 768)
            )
            > 0
        )
        rgb[mask] = board[mask]
    return rgb


def decode(rgb, timestamp=0, observer=None, now=None):
    observer = observer or PixelVisualObserver(calibration(), "semantic-A", "cal-v1")
    return observer.observe(
        CameraFrame("semantic-A", "cal-v1", rgb, timestamp),
        now=timestamp if now is None else now,
    )


@pytest.mark.parametrize("level", range(33))
def test_pixel_decoder_reads_unique_codes_and_all_visible_gauge_levels(level):
    result = decode(
        render((("marker-01", (-0.3, 0.3), level), ("marker-02", (0.3, 0.3), level)))
    )
    assert result.status == "visible", result.reason
    assert [r.marker_id for r in result.readings] == ["marker-01", "marker-02"]
    assert all(
        r.gauge_level == level and r.interval == gauge_interval(level)
        for r in result.readings
    )
    assert np.linalg.norm(np.array(result.readings[0].center_xy) - [-0.3, 0.3]) < 0.01
    assert 0 < result.readings[0].radius_m < 0.04
    assert (
        result.frame_sha256
        == hashlib.sha256(
            render(
                (("marker-01", (-0.3, 0.3), level), ("marker-02", (0.3, 0.3), level))
            ).tobytes()
        ).hexdigest()
    )


@pytest.mark.parametrize("angle", [0, 25, 90, 180, 270])
def test_pixel_ids_follow_patterns_not_position_or_orientation(angle):
    result = decode(
        render((("marker-02", (-0.3, 0.3), 6), ("marker-01", (0.3, 0.3), 6)), angle)
    )
    assert result.status == "visible", result.reason
    assert result.readings[0].center_xy[0] > 0
    assert result.readings[1].center_xy[0] < 0


@pytest.mark.parametrize("color", [[230, 230, 230], [0, 230, 0], [230, 0, 230]])
def test_unapproved_glyph_color_abstains(color):
    rgb = render()
    yellow = (rgb[..., 0] > 150) & (rgb[..., 1] > 150) & (rgb[..., 2] < 80)
    rgb[yellow] = color
    assert decode(rgb).status == "ambiguous"


def test_yellow_glyph_contract_handles_bounded_lighting_without_gray_acceptance():
    patch = np.full((10, 10, 3), [160, 160, 30], np.uint8)
    assert PixelVisualObserver._classify(patch, "id") == 1
    patch[0] = [45, 45, 45]
    assert PixelVisualObserver._classify(patch, "id") == 1
    patch[1] = [45, 45, 45]
    assert PixelVisualObserver._classify(patch, "id") is None


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "different_gauges", "covered", "gauge_gap", "id_corrupt"],
)
def test_absent_ambiguous_or_incomplete_pixels_abstain(fault):
    panels = (
        ()
        if fault == "missing"
        else (
            ("marker-01", (-0.3, 0.3), 11),
            (
                "marker-01" if fault == "duplicate" else "marker-02",
                (0.3, 0.3),
                12 if fault == "different_gauges" else 11,
            ),
        )
    )
    rgb = render(panels)
    if fault in ("covered", "gauge_gap", "id_corrupt"):
        center = (-0.3, 0.3)
        cell = (
            (3, 3)
            if fault == "covered"
            else (4.0, 5.8)
            if fault == "gauge_gap"
            else (1.5, 1.5)
        )
        point = np.asarray(center) + [(cell[0] - 7) * 0.025, (4 - cell[1]) * 0.025]
        x, y = calibration().to_pixel(point, PANEL_HEIGHT_M).round().astype(int)
        rgb[y - 4 : y + 5, x - 4 : x + 5] = (
            [150, 100, 40] if fault != "id_corrupt" else [0, 0, 0]
        )
    result = decode(rgb)
    assert result.status != "visible"
    assert result.readings == ()


def test_fresh_frames_required_and_observer_does_not_mutate_pixels():
    observer = PixelVisualObserver(calibration(), "semantic-A", "cal-v1")
    rgb = render()
    original = rgb.copy()
    assert decode(rgb, 0, observer).status == "visible"
    assert decode(rgb, 0, observer).status == "stale"
    assert decode(rgb, 0.05, observer, 0.20).status == "stale"
    assert np.array_equal(rgb, original)
    wrong = CameraFrame("other", "cal-v1", rgb, 0.25)
    assert observer.observe(wrong, now=0.25).status == "invalid"


def reading(t, level=11, marker="marker-01", center=(0.0, 1.25)):
    return VisualReading(
        marker,
        center,
        0.015,
        level,
        gauge_interval(level),
        t,
        "a" * 64,
        "semantic-A",
        "cal-v1",
    )


def scene(t, *items):
    return VisualSceneObservation(
        t,
        "semantic-A",
        "cal-v1",
        "a" * 64,
        "visible",
        None,
        tuple(items or [reading(t)]),
    )


def test_registry_bootstraps_fixed_calibrated_anchor_and_reacquires_after_restart(
    tmp_path,
):
    path = tmp_path / "fresh.sqlite3"
    store = PurposeStore(path, "map-v1")
    registry = VisualEntityRegistry(store, "cal-v1")
    assert registry.stations() == ()
    for t in (0, 0.05):
        assert not registry.observe(scene(t))
    entity = registry.observe(scene(0.10))[0]
    assert entity.station.xy == (0.0, 1.0)
    assert registry.stations() == (entity.station,)
    registry.observe(scene(0.15, reading(0.15, center=(0.004, 1.25))))
    assert registry.stations()[0].xy == entity.station.xy
    store.close()
    store = PurposeStore(path, "map-v1")
    restored = VisualEntityRegistry(store, "cal-v1")
    assert restored.stations() == ()
    for t in (1, 1.05, 1.10):
        current = restored.observe(scene(t))
    assert current[0].entity_id == entity.entity_id
    assert current[0].station.xy == entity.station.xy
    other_context = VisualEntityRegistry(store, "different-calibration")
    assert not other_context.observe(scene(2))
    assert store.db.execute("SELECT COUNT(*) FROM visual_entities").fetchone()[0] == 1
    store.close()


def test_registry_missing_duplicate_stale_and_moved_entities_are_not_offered():
    store = PurposeStore(":memory:", "map-v1")
    registry = VisualEntityRegistry(store, "cal-v1")
    for t in (0, 0.05, 0.10):
        registry.observe(scene(t))
    assert registry.stations()
    assert not registry.observe(scene(0.10))
    for t in (0.15, 0.20, 0.25):
        registry.observe(scene(t))
    assert registry.stations()
    assert not registry.observe(scene(0.30, reading(0.30), reading(0.30)))
    assert not registry.observe(scene(0.35, reading(0.35, center=(0.10, 1.25))))
    for t in (0.4, 0.45, 0.50):
        assert not registry.observe(scene(t))
    assert store.db.execute("SELECT x,y FROM visual_entities").fetchone()[:] == (
        0,
        1.25,
    )
    store.close()


def entity(t, level=11):
    return AssociatedEntity("entity-one", (0.0, 1.25), 0.015, reading(t, level))


def ready_gate():
    gate = VisualOutcomeGate("map-v1", "cal-v1")
    assert gate.generation is None and gate.pending is None
    gate.enable(4)
    for t in (0, 0.05, 0.10):
        assert (
            gate.observe(
                [entity(t)], now=t, generation=4, measured=True, stationary=True
            )
            is None
        )
    assert gate.begin("request-one", "entity-one", now=0.1, generation=4)
    return gate


def complete(gate, after=25, fault=None):
    outcome = None
    for index in range(3, 27):
        t = round(index * 0.05, 6)
        options = {"now": t, "generation": 4, "measured": True, "stationary": True}
        if index == 22:
            options["completion"] = {
                "request_id": "request-one",
                "entity_id": "entity-one",
                "generation": 4,
                "timestamp": 1.1,
            }
        if fault and index == 12:
            options.update(fault)
        result = gate.observe([entity(t, after if index >= 22 else 11)], **options)
        outcome = result or outcome
    return outcome


@pytest.mark.parametrize(
    "after,classification",
    [(25, "useful"), (11, "no_meaningful_gain"), (10, "no_meaningful_gain")],
)
def test_gate_admits_only_causal_continuous_visual_outcomes(after, classification):
    gate = ready_gate()
    receipt = complete(gate, after)
    assert receipt and receipt["classification"] == classification
    assert validate_visual_outcome(receipt) == receipt
    assert (
        receipt["request_id"] == "request-one" and receipt["entity_id"] == "entity-one"
    )
    assert receipt["before"] == 11 / 32 and receipt["after"] == after / 32
    assert receipt["after_frames"][0]["timestamp"] > receipt["completed_at"]
    assert not gate.begin("request-one", "entity-one", now=1.3, generation=4)
    assert VisualOutcomeGate("map-v1", "cal-v1").generation is None


def test_threshold_crossing_visual_gain_abstains_instead_of_learning_noise():
    gate = ready_gate()
    assert complete(gate, 12) is None
    assert gate.generation is None and gate.reason == "ambiguous_visual_gain"


@pytest.mark.parametrize(
    "fault", [{"generation": 5}, {"measured": False}, {"stationary": False}]
)
def test_authority_or_stationarity_loss_cancels_without_negative_outcome(fault):
    gate = ready_gate()
    assert complete(gate, fault=fault) is None
    assert gate.generation is None and gate.pending is None


def test_stop_missing_pixels_stale_frames_and_completion_alone_never_credit():
    for fault in ("stop", "missing", "stale", "completion"):
        gate = ready_gate()
        if fault == "stop":
            gate.cancel("stop")
        elif fault == "missing":
            gate.observe([], now=0.15, generation=4, measured=True, stationary=True)
        elif fault == "stale":
            gate.observe(
                [entity(0.1)], now=0.1, generation=4, measured=True, stationary=True
            )
        else:
            gate.observe(
                [entity(0.15)],
                now=0.15,
                generation=4,
                measured=True,
                stationary=True,
                completion={
                    "request_id": "wrong",
                    "entity_id": "entity-one",
                    "generation": 4,
                    "timestamp": 0.15,
                },
            )
        assert complete(gate) is None
        assert gate.generation is None


@pytest.mark.parametrize(
    "change",
    [
        {"source": "simulated_station_telemetry"},
        {"before": 0.1},
        {"entity_id": "other"},
        {"gain_interval": [0.5, 0.6]},
        {"timestamp": 2},
        {"classification": "no_meaningful_gain"},
    ],
)
def test_outcome_validator_rejects_conflicting_or_invented_receipts(change):
    receipt = complete(ready_gate())
    receipt.update(change)
    # Recompute integrity hash to test semantic checks, not only tamper detection.
    receipt.pop("evidence_sha256")
    receipt["evidence_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_visual_outcome(receipt)


def test_returned_receipt_is_independent_and_interrupted_history_cannot_validate():
    receipt = complete(ready_gate())
    validated = validate_visual_outcome(receipt)
    validated["before_frames"][0]["gauge_level"] = 0
    assert receipt["before_frames"][0]["gauge_level"] == 11
    broken = copy.deepcopy(receipt)
    broken["during_frames"].pop(4)
    broken["during_frames"].pop(4)
    broken["during_frames"].pop(4)
    broken.pop("evidence_sha256")
    broken["evidence_sha256"] = hashlib.sha256(
        json.dumps(broken, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_visual_outcome(broken)


@pytest.mark.parametrize(
    "mutation",
    ["generation", "missing_after", "dropped_completion", "calibration", "camera"],
)
def test_late_visual_faults_and_missing_completion_cannot_train(mutation):
    gate = ready_gate()
    outputs = []
    for index in range(3, 110):
        t = round(index * 0.05, 6)
        entities = [entity(t, 25 if index >= 22 else 11)]
        correlation = None
        if index == 22 and mutation != "dropped_completion":
            correlation = {
                "request_id": "request-one",
                "entity_id": "entity-one",
                "generation": True if mutation == "generation" else 4,
                "timestamp": 1.1,
            }
        if index == 24 and mutation == "missing_after":
            entities = []
        if index == 24 and mutation in ("calibration", "camera"):
            record = entities[0].reading.evidence()
            record[
                "calibration_version" if mutation == "calibration" else "camera_id"
            ] = "changed"
            entities = [
                AssociatedEntity(
                    "entity-one", (0, 1.25), 0.015, VisualReading(**record)
                )
            ]
        receipt = gate.observe(
            entities,
            now=t,
            generation=4,
            measured=True,
            stationary=True,
            completion=correlation,
        )
        if receipt:
            outputs.append(receipt)
    assert outputs == []
    assert gate.pending is None and gate.generation is None


def test_visual_receipt_updates_persistent_purpose_memory_exactly_once(tmp_path):
    path = tmp_path / "visual-only.sqlite3"
    store = PurposeStore(path, "map-v1")
    registry = VisualEntityRegistry(store, "cal-v1")
    for t in (0, 0.05, 0.10):
        associated = registry.observe(scene(t))
    station = associated[0].station
    engine = PurposeEngine(store)
    engine.select([station], resource=0.3, pose=(0, 1), now=0.1)
    receipt = complete(ready_gate())
    receipt["entity_id"] = receipt["completion"]["entity_id"] = station.id
    receipt.pop("evidence_sha256")
    receipt["evidence_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    kwargs = {
        "before": receipt["before"],
        "after": receipt["after"],
        "now": receipt["timestamp"],
        "source": "native_scene_rgb",
        "visual_evidence": receipt,
    }
    assert engine.record_outcome("request-one", station.id, **kwargs)
    assert not engine.record_outcome("request-one", station.id, **kwargs)
    store.close()
    restored = PurposeStore(path, "map-v1")
    engine = PurposeEngine(restored)
    assert not engine.record_outcome("request-one", station.id, **kwargs)
    assert engine.snapshot()["episodes"] == 1
    assert restored.db.execute("SELECT COUNT(*) FROM purpose_events").fetchone()[0] == 1
    assert VisualEntityRegistry(restored, "cal-v1").stations() == ()
    assert VisualOutcomeGate("map-v1", "cal-v1").generation is None
    with pytest.raises(ValueError, match="does not match"):
        engine.record_outcome("request-two", station.id, **kwargs)
    with pytest.raises(ValueError, match="does not match"):
        engine.record_outcome("request-one", station.id, **{**kwargs, "after": 0.5})
    restored.close()
