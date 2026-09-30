"""Pixel-only contracts for two deliberately marked synthetic interaction panels.

This is a fixture decoder, not general object recognition. Registered camera
geometry supplies metric scale; it never supplies entity positions. Outcomes
require visible before/after gauge evidence and explicit uninterrupted authority.
Native rendering and provenance must be established independently by the caller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Calibration
from .camera_rig import CameraFrame
from .purpose import MEANINGFUL_GAIN, Station

SOURCE = "native_scene_rgb"
PANEL_COLUMNS, PANEL_ROWS = 14, 8
CELL_SIZE_M, PANEL_HEIGHT_M = 0.025, 0.010
ID_ORIGIN = (1, 1)
ID_COLOR = (1.0, 1.0, 0.0, 1.0)
GAUGE_BINS = 32
INTERACTION_OFFSET = (0.0, -0.25)
ID_PATTERNS = {
    "marker-01": ((1, 0, 0, 1), (1, 1, 0, 0), (1, 0, 1, 1), (0, 0, 1, 1)),
    "marker-02": ((1, 1, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0), (0, 1, 0, 1)),
}
MAX_FRAME_AGE = 0.10
MAX_FRAME_GAP = 0.15
WINDOW_SAMPLES = 3
WINDOW_SECONDS = 0.09


def gauge_rect(index):
    """Relative cell rectangle; no world entity position or effect is encoded."""
    if type(index) is not int or not 0 <= index < GAUGE_BINS:
        raise ValueError("Invalid gauge bin")
    row, column = divmod(index, 16)
    x, y = 1 + column * 0.75, 5.25 + row * 1.05
    return x, y, x + 0.75, y + 0.8


def _number(value, *, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError("Expected a finite number in range")
    return float(value)


def _identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Invalid visual evidence identifier")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def gauge_interval(level):
    if type(level) is not int or not 0 <= level <= GAUGE_BINS:
        raise ValueError("Invalid visual gauge level")
    return max(0.0, (level - 0.5) / GAUGE_BINS), min(1.0, (level + 0.5) / GAUGE_BINS)


@dataclass(frozen=True)
class VisualReading:
    marker_id: str
    center_xy: tuple[float, float]
    radius_m: float
    gauge_level: int
    interval: tuple[float, float]
    timestamp: float
    frame_sha256: str
    camera_id: str
    calibration_version: str

    def __post_init__(self):
        if self.marker_id not in ID_PATTERNS:
            raise ValueError("Unknown visible marker")
        if len(self.center_xy) != 2:
            raise ValueError("Invalid visual anchor")
        center = tuple(_number(x, minimum=-100) for x in self.center_xy)
        radius = _number(self.radius_m)
        if radius > 0.04 or tuple(self.interval) != gauge_interval(self.gauge_level):
            raise ValueError("Invalid visual uncertainty")
        _number(self.timestamp)
        for value in (self.camera_id, self.calibration_version):
            _identifier(value)
        if (
            not isinstance(self.frame_sha256, str)
            or len(self.frame_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.frame_sha256)
        ):
            raise ValueError("Invalid raw RGB hash")
        object.__setattr__(self, "center_xy", center)
        object.__setattr__(self, "interval", tuple(self.interval))

    def evidence(self):
        return {
            "timestamp": self.timestamp,
            "frame_sha256": self.frame_sha256,
            "camera_id": self.camera_id,
            "calibration_version": self.calibration_version,
            "marker_id": self.marker_id,
            "center_xy": list(self.center_xy),
            "radius_m": self.radius_m,
            "gauge_level": self.gauge_level,
            "interval": list(self.interval),
        }


@dataclass(frozen=True)
class VisualSceneObservation:
    timestamp: float
    camera_id: str
    calibration_version: str
    frame_sha256: str | None
    status: str
    reason: str | None
    readings: tuple[VisualReading, ...] = ()
    source: str = SOURCE


class PixelVisualObserver:
    """Decode at most two fixed panels using RGB and a registered pinhole camera.

    A whole magenta border, exact 4x4 glyph and all 32 gauge cells must be visible.
    Native preflight must establish actual pixel resolution/occlusion behavior.
    The declared two-pixel localization allowance is an engineering assumption.
    """

    def __init__(
        self,
        calibration,
        camera_id,
        calibration_version,
        *,
        panel_height_m=PANEL_HEIGHT_M,
    ):
        if not isinstance(calibration, Calibration):
            raise TypeError("Expected registered Calibration")
        self.calibration = copy.deepcopy(calibration)
        self.camera_id = _identifier(camera_id)
        self.calibration_version = _identifier(calibration_version)
        self.panel_height = _number(panel_height_m)
        self.last_capture = -math.inf

    @staticmethod
    def _classify(patch, kind):
        values = patch.astype(float)
        r, g, b = (values[..., i] for i in range(3))
        dark = np.maximum.reduce([r, g, b]) < 85
        if kind == "id":
            # Saturated yellow is distinct from the gray/white BB8 head proposal.
            light = (
                (r > 110)
                & (g > 110)
                & (np.minimum(r, g) > 1.5 * b)
                & (np.abs(r - g) < 70)
            )
        else:
            light = (g > 80) & (g > 1.5 * r) & (g > 1.35 * b)
        if float(light.mean()) >= 0.85:
            return 1
        if float(dark.mean()) >= 0.85:
            return 0
        return None

    @staticmethod
    def _patch(board, rect):
        # Rectified cells are 20px. Keep only the middle half to avoid borders.
        x0, y0, x1, y1 = rect
        dx, dy = (x1 - x0) / 4, (y1 - y0) / 4
        return board[
            round((y0 + dy) * 20) : round((y1 - dy) * 20),
            round((x0 + dx) * 20) : round((x1 - dx) * 20),
        ]

    def _decode(self, rgb, quad):
        destination = np.array([[0, 0], [280, 0], [280, 160], [0, 160]], np.float32)
        decoded = []
        for rotation in range(4):
            corners = np.roll(quad, rotation, axis=0).astype(np.float32)
            transform = cv2.getPerspectiveTransform(corners, destination)
            board = cv2.warpPerspective(rgb, transform, (280, 160))
            bits = tuple(
                tuple(
                    self._classify(
                        self._patch(board, (x + 1, y + 1, x + 2, y + 2)), "id"
                    )
                    for x in range(4)
                )
                for y in range(4)
            )
            codes = [code for code, pattern in ID_PATTERNS.items() if bits == pattern]
            if len(codes) != 1:
                continue
            # Reject undersampled boards; interpolation cannot create evidence.
            width = min(
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[3]),
            )
            height = min(
                np.linalg.norm(corners[3] - corners[0]),
                np.linalg.norm(corners[2] - corners[1]),
            )
            gx0, gy0, gx1, gy1 = gauge_rect(0)
            if min(width / 14 * (gx1 - gx0), height / 8 * (gy1 - gy0)) < 3:
                continue
            gauge = [
                self._classify(self._patch(board, gauge_rect(i)), "gauge")
                for i in range(GAUGE_BINS)
            ]
            if None in gauge or gauge != sorted(gauge, reverse=True):
                continue
            center_pixel = cv2.perspectiveTransform(
                np.array([[[140.0, 80.0]]], np.float32), np.linalg.inv(transform)
            )[0, 0]
            center = self.calibration.to_plane(center_pixel, self.panel_height)
            world_corners = self.calibration.to_plane(corners, self.panel_height)
            lengths = [
                np.linalg.norm(world_corners[(i + 1) % 4] - world_corners[i])
                for i in range(4)
            ]
            if any(
                abs(length - expected) > expected * 0.20
                for length, expected in zip(lengths, (0.35, 0.20, 0.35, 0.20))
            ):
                continue
            offsets = np.array([[2, 0], [-2, 0], [0, 2], [0, -2]])
            radius = 0.005 + max(
                float(
                    np.linalg.norm(
                        self.calibration.to_plane(
                            center_pixel + delta, self.panel_height
                        )
                        - center
                    )
                )
                for delta in offsets
            )
            if radius > 0.04 or np.max(abs(center)) >= self.calibration.extent:
                continue
            decoded.append(
                (codes[0], tuple(float(x) for x in center), radius, sum(gauge))
            )
        return decoded[0] if len(decoded) == 1 else None

    def observe(self, frame, *, now):
        timestamp = getattr(frame, "capture_time", now)
        digest = None
        status, reason, readings = "invalid", "invalid_frame", ()
        try:
            now = _number(now)
            timestamp = _number(timestamp)
            if (
                not isinstance(frame, CameraFrame)
                or frame.camera_id != self.camera_id
                or frame.calibration_version != self.calibration_version
                or frame.rgb.shape != (*reversed(self.calibration.resolution), 3)
                or frame.rgb.size > 8_000_000 * 3
            ):
                raise ValueError("Invalid visual camera frame")
            if (
                not -1e-9 <= now - timestamp <= MAX_FRAME_AGE + 1e-9
                or timestamp <= self.last_capture
            ):
                return VisualSceneObservation(
                    timestamp,
                    self.camera_id,
                    self.calibration_version,
                    None,
                    "stale",
                    "stale_or_repeated_frame",
                )
            self.last_capture = timestamp
            rgb = frame.rgb
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
            mask = ((r > 90) & (b > 75) & (g < 0.65 * np.minimum(r, b))).astype(
                np.uint8
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            candidates = [c for c in contours if cv2.contourArea(c) >= 150]
            if len(candidates) > 4:
                raise ValueError("Too many visual panel candidates")
            found, rejected = [], False
            for contour in candidates:
                quad = cv2.approxPolyDP(
                    contour, 0.025 * cv2.arcLength(contour, True), True
                )
                if len(quad) != 4 or not cv2.isContourConvex(quad):
                    rejected = True
                    continue
                quad = quad.reshape(4, 2)
                # OpenCV contours are counterclockwise in image coordinates.
                if cv2.contourArea(quad, oriented=True) < 0:
                    quad = quad[::-1]
                decoded = self._decode(rgb, quad)
                if decoded is None:
                    rejected = True
                    continue
                code, center, radius, level = decoded
                found.append(
                    VisualReading(
                        code,
                        center,
                        radius,
                        level,
                        gauge_interval(level),
                        timestamp,
                        digest,
                        self.camera_id,
                        self.calibration_version,
                    )
                )
            if (
                rejected
                or len({r.marker_id for r in found}) != len(found)
                or len({r.gauge_level for r in found}) > 1
            ):
                status, reason = (
                    "ambiguous",
                    "incomplete_duplicate_or_undecodable_panel",
                )
            elif found:
                status, reason, readings = (
                    "visible",
                    None,
                    tuple(sorted(found, key=lambda r: r.marker_id)),
                )
            else:
                status, reason = "missing", "no_complete_panel"
        except (TypeError, ValueError, AttributeError, cv2.error):
            pass
        return VisualSceneObservation(
            timestamp,
            self.camera_id,
            self.calibration_version,
            digest,
            status,
            reason,
            readings,
        )


@dataclass(frozen=True)
class AssociatedEntity:
    entity_id: str
    anchor_xy: tuple[float, float]
    anchor_radius_m: float
    reading: VisualReading

    def __post_init__(self):
        _identifier(self.entity_id)
        if not isinstance(self.reading, VisualReading) or len(self.anchor_xy) != 2:
            raise ValueError("Invalid associated visual entity")
        object.__setattr__(
            self, "anchor_xy", tuple(_number(x, minimum=-100) for x in self.anchor_xy)
        )
        if _number(self.anchor_radius_m) > 0.04:
            raise ValueError("Uncertain associated anchor")

    @property
    def station(self):
        return Station(
            self.entity_id,
            tuple(x + delta for x, delta in zip(self.anchor_xy, INTERACTION_OFFSET)),
            f"Visual {self.reading.marker_id}",
        )


class VisualEntityRegistry:
    """Stable SQLite anchors; every process restart requires fresh reacquisition."""

    def __init__(self, store, calibration_version):
        self.store = store
        self.calibration_version = _identifier(calibration_version)
        self._windows = {}
        self._moved = set()
        self.current = ()
        self.last_timestamp = -math.inf
        with store.lock, store.db:
            store.db.execute("""
                CREATE TABLE IF NOT EXISTS visual_entities (
                    agent TEXT NOT NULL, map TEXT NOT NULL, calibration TEXT NOT NULL,
                    marker TEXT NOT NULL, entity TEXT NOT NULL, x REAL NOT NULL,
                    y REAL NOT NULL, radius REAL NOT NULL, evidence TEXT NOT NULL,
                    PRIMARY KEY(agent,map,calibration,marker),
                    UNIQUE(agent,map,entity))
            """)

    def stations(self):
        return tuple(entity.station for entity in self.current)

    def observe(self, scene):
        self.current = ()
        if (
            not isinstance(scene, VisualSceneObservation)
            or scene.source != SOURCE
            or scene.calibration_version != self.calibration_version
            or not isinstance(scene.timestamp, (int, float))
            or not math.isfinite(scene.timestamp)
            or scene.timestamp <= self.last_timestamp
            or scene.status != "visible"
            or not 0 < len(scene.readings) <= 2
            or any(
                not isinstance(item, VisualReading)
                or item.timestamp != scene.timestamp
                or item.camera_id != scene.camera_id
                or item.calibration_version != scene.calibration_version
                or item.frame_sha256 != scene.frame_sha256
                for item in scene.readings
            )
        ):
            self._windows.clear()
            return self.current
        if scene.timestamp - self.last_timestamp > MAX_FRAME_GAP + 1e-9:
            self._windows.clear()
        self.last_timestamp = scene.timestamp
        visible = {reading.marker_id for reading in scene.readings}
        if len(visible) != len(scene.readings):
            self._windows.clear()
            return self.current
        self._windows = {
            key: value for key, value in self._windows.items() if key in visible
        }
        admitted = []
        with self.store.lock, self.store.db:
            for reading in scene.readings:
                marker = reading.marker_id
                if marker not in ID_PATTERNS or marker in self._moved:
                    continue
                row = self.store.db.execute(
                    "SELECT * FROM visual_entities WHERE agent=? AND map=? AND calibration=? AND marker=?",
                    (*self.store.scope, self.calibration_version, marker),
                ).fetchone()
                if (
                    row
                    and math.dist(reading.center_xy, (row["x"], row["y"]))
                    > row["radius"] + reading.radius_m + 0.01
                ):
                    self._moved.add(marker)
                    self._windows.pop(marker, None)
                    continue
                window = self._windows.setdefault(marker, deque(maxlen=WINDOW_SAMPLES))
                if (
                    window
                    and math.dist(reading.center_xy, window[-1].center_xy) > 0.015
                ):
                    window.clear()
                window.append(reading)
                if (
                    len(window) < WINDOW_SAMPLES
                    or window[-1].timestamp - window[0].timestamp < WINDOW_SECONDS
                ):
                    continue
                if row is None:
                    anchor = np.median([item.center_xy for item in window], axis=0)
                    radius = max(
                        item.radius_m + math.dist(item.center_xy, anchor)
                        for item in window
                    )
                    if radius > 0.04:
                        continue
                    entity_id = str(uuid.uuid4())
                    self.store.db.execute(
                        "INSERT INTO visual_entities VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            *self.store.scope,
                            self.calibration_version,
                            marker,
                            entity_id,
                            *map(float, anchor),
                            radius,
                            _canonical([item.evidence() for item in window]),
                        ),
                    )
                    anchor = tuple(float(x) for x in anchor)
                else:
                    entity_id, anchor, radius = (
                        row["entity"],
                        (row["x"], row["y"]),
                        row["radius"],
                    )
                admitted.append(AssociatedEntity(entity_id, anchor, radius, reading))
        self.current = tuple(admitted)
        return self.current


def _window_interval(window):
    return [
        min(item["interval"][0] for item in window),
        max(item["interval"][1] for item in window),
    ]


def validate_visual_outcome(evidence):
    """Validate a complete receipt before PurposeEngine's atomic event insertion.

    This checks the admission contract, not authenticity of caller-provided RGB.
    Independent recording must retain the images addressed by raw RGB hashes.
    """
    fields = {
        "schema",
        "source",
        "request_id",
        "entity_id",
        "marker_id",
        "generation",
        "map_version",
        "calibration_version",
        "anchor_xy",
        "anchor_radius_m",
        "started_at",
        "completed_at",
        "timestamp",
        "completion",
        "before_frames",
        "during_frames",
        "after_frames",
        "before_interval",
        "after_interval",
        "gain_interval",
        "before",
        "after",
        "classification",
        "evidence_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != fields:
        raise ValueError("Invalid visual outcome fields")
    result = copy.deepcopy(evidence)
    if result["schema"] != "bb8.visual-outcome.v1" or result["source"] != SOURCE:
        raise ValueError("Invalid visual outcome source or schema")
    for key in ("request_id", "entity_id", "map_version", "calibration_version"):
        _identifier(result[key])
    if (
        result["marker_id"] not in ID_PATTERNS
        or type(result["generation"]) is not int
        or result["generation"] < 0
    ):
        raise ValueError("Invalid visual identity or authority generation")
    anchor = result["anchor_xy"]
    if not isinstance(anchor, list) or len(anchor) != 2:
        raise ValueError("Invalid observed anchor")
    for coordinate in anchor:
        _number(coordinate, minimum=-100)
    radius = _number(result["anchor_radius_m"])
    if radius > 0.04:
        raise ValueError("Uncertain observed anchor")
    start, completed, timestamp = (
        _number(result[key]) for key in ("started_at", "completed_at", "timestamp")
    )
    if not start + 1.0 - 1e-9 <= completed < timestamp <= start + 5.0:
        raise ValueError("Invalid visual interaction timing")
    completion = {
        "request_id": result["request_id"],
        "entity_id": result["entity_id"],
        "generation": result["generation"],
        "timestamp": completed,
    }
    if (
        result["completion"] != completion
        or type(result["completion"].get("generation")) is not int
        or isinstance(result["completion"].get("timestamp"), bool)
    ):
        raise ValueError("Completion does not correlate with visual request")
    for name in ("before_frames", "during_frames", "after_frames"):
        window = result[name]
        if not isinstance(window, list) or not (
            WINDOW_SAMPLES <= len(window) <= 102
            if name == "during_frames"
            else len(window) == WINDOW_SAMPLES
        ):
            raise ValueError("Visual outcome requires complete observation windows")
        for index, frame in enumerate(window):
            if not isinstance(frame, dict) or set(frame) != {
                "timestamp",
                "frame_sha256",
                "camera_id",
                "calibration_version",
                "marker_id",
                "center_xy",
                "radius_m",
                "gauge_level",
                "interval",
            }:
                raise ValueError("Invalid visual frame evidence")
            t = _number(frame["timestamp"])
            _identifier(frame["camera_id"])
            digest = frame["frame_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid raw RGB hash")
            if (
                frame["calibration_version"] != result["calibration_version"]
                or frame["marker_id"] != result["marker_id"]
            ):
                raise ValueError("Changed visual association")
            if frame["interval"] != list(gauge_interval(frame["gauge_level"])):
                raise ValueError("Gauge interval does not follow pixel quantization")
            xy = frame["center_xy"]
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError("Invalid RGB entity position")
            for coordinate in xy:
                _number(coordinate, minimum=-100)
            observed_radius = _number(frame["radius_m"])
            if (
                observed_radius > 0.04
                or math.dist(anchor, xy) > radius + observed_radius + 0.01
            ):
                raise ValueError("Moved or uncertain visual entity")
            if (
                index
                and not 0 < t - window[index - 1]["timestamp"] <= MAX_FRAME_GAP + 1e-9
            ):
                raise ValueError("Visual window is stale or discontinuous")
        if window[-1]["timestamp"] - window[0]["timestamp"] < WINDOW_SECONDS:
            raise ValueError("Visual window lacks temporal support")
        if (
            name != "during_frames"
            and len({frame["gauge_level"] for frame in window}) != 1
        ) or len({frame["camera_id"] for frame in window}) != 1:
            raise ValueError("Visual window has inconsistent readings")
    before, after = result["before_frames"], result["after_frames"]
    during = result["during_frames"]
    if (
        not start < during[0]["timestamp"] <= start + MAX_FRAME_GAP + 1e-9
        or during[0]["timestamp"] - before[-1]["timestamp"] > MAX_FRAME_GAP + 1e-9
        or during[-WINDOW_SAMPLES:] != after
    ):
        raise ValueError("Incomplete continuous interaction evidence")
    if (
        not 0 <= start - before[-1]["timestamp"] <= MAX_FRAME_AGE + 1e-9
        or not completed < after[0]["timestamp"]
    ):
        raise ValueError("Before/after evidence is not causal and fresh")
    if (
        after[-1]["timestamp"] != timestamp
        or before[0]["camera_id"] != after[0]["camera_id"]
    ):
        raise ValueError("Visual outcome changed camera or timestamp")
    before_interval, after_interval = _window_interval(before), _window_interval(after)
    gain = [
        after_interval[0] - before_interval[1],
        after_interval[1] - before_interval[0],
    ]
    classification = (
        "useful"
        if gain[0] >= MEANINGFUL_GAIN
        else "no_meaningful_gain"
        if gain[1] < MEANINGFUL_GAIN
        else None
    )
    if classification is None or result["classification"] != classification:
        raise ValueError("Visual gain is ambiguous at the meaningful threshold")
    expected = {
        "before_interval": before_interval,
        "after_interval": after_interval,
        "gain_interval": gain,
        "before": sum(before_interval) / 2,
        "after": sum(after_interval) / 2,
    }
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError("Outcome values do not follow visual intervals")
    digest = result.pop("evidence_sha256")
    if digest != _hash(result):
        raise ValueError("Visual outcome evidence was changed")
    result["evidence_sha256"] = digest
    return result


class VisualOutcomeGate:
    """One pending authorized request; missing evidence never becomes a failure.

    The returned receipt must be inserted through PurposeEngine.record_outcome's
    transaction, which owns durable event-ID deduplication. Pending work and
    authority are deliberately not restored by this component on restart.
    """

    def __init__(self, map_version, calibration_version):
        self.map_version = _identifier(map_version)
        self.calibration_version = _identifier(calibration_version)
        self.generation = None
        self.pending = None
        self.reason = "disabled"
        self._windows = {}
        self._seen = set()
        self._last_time = -math.inf

    def enable(self, generation):
        if type(generation) is not int or generation < 0:
            raise ValueError("Invalid visual authority generation")
        self.cancel("new_authority")
        self._last_time = -math.inf
        self.generation = generation
        self.reason = "observing"

    def cancel(self, reason="cancelled"):
        self.generation = None
        self.pending = None
        self._windows.clear()
        self.reason = _identifier(reason)

    def begin(self, request_id, entity_id, *, now, generation):
        request_id, entity_id, now = (
            _identifier(request_id),
            _identifier(entity_id),
            _number(now),
        )
        window = self._windows.get(entity_id)
        if (
            self.generation is None
            or type(generation) is not int
            or generation != self.generation
            or self.pending is not None
            or request_id in self._seen
            or len(self._seen) >= 4096
            or not window
            or len(window) != WINDOW_SAMPLES
            or not 0 <= now - window[-1].reading.timestamp <= MAX_FRAME_AGE + 1e-9
            or window[-1].reading.timestamp - window[0].reading.timestamp
            < WINDOW_SECONDS
            or len({item.reading.gauge_level for item in window}) != 1
            or len(
                {
                    (item.anchor_xy, item.reading.marker_id, item.reading.camera_id)
                    for item in window
                }
            )
            != 1
        ):
            return False
        entity = window[-1]
        self._seen.add(request_id)
        self.pending = {
            "request_id": request_id,
            "entity": entity,
            "started_at": now,
            "before": [item.reading.evidence() for item in window],
            "completion": None,
            "during": [],
            "after": [],
        }
        self.reason = "awaiting_completion"
        return True

    def observe(
        self, entities, *, now, generation, measured, stationary, completion=None
    ):
        if self.generation is None:
            return None
        try:
            now = _number(now)
            entities = tuple(entities)
            valid = (
                self.generation is not None
                and type(generation) is int
                and generation == self.generation
                and measured is True
                and stationary is True
                and (
                    self._last_time == -math.inf
                    or 0 < now - self._last_time <= MAX_FRAME_GAP + 1e-9
                )
                and 0 < len(entities) <= 2
                and all(isinstance(entity, AssociatedEntity) for entity in entities)
                and len({entity.entity_id for entity in entities}) == len(entities)
                and all(
                    entity.reading.calibration_version == self.calibration_version
                    and entity.reading.timestamp == now
                    for entity in entities
                )
            )
            self._last_time = now
            if not valid:
                self.cancel("authority_or_visual_evidence_lost")
                return None
            current = {entity.entity_id: entity for entity in entities}
            self._windows = {
                key: value for key, value in self._windows.items() if key in current
            }
            for entity in entities:
                window = self._windows.setdefault(
                    entity.entity_id, deque(maxlen=WINDOW_SAMPLES)
                )
                window.append(entity)
            if self.pending is None:
                return None
            pending = self.pending
            target = current.get(pending["entity"].entity_id)
            if (
                target is None
                or target.anchor_xy != pending["entity"].anchor_xy
                or target.reading.marker_id != pending["entity"].reading.marker_id
                or now - pending["started_at"] > 5.0
            ):
                self.cancel("entity_missing_moved_or_timeout")
                return None
            pending["during"].append(target.reading.evidence())
            if len(pending["during"]) > 102:
                self.cancel("interaction_evidence_budget_exceeded")
                return None
            if completion is not None:
                if not isinstance(completion, dict) or set(completion) != {
                    "request_id",
                    "entity_id",
                    "timestamp",
                    "generation",
                }:
                    raise ValueError("Invalid completion correlation")
                completed = _number(completion["timestamp"])
                if (
                    completion["request_id"] != pending["request_id"]
                    or completion["entity_id"] != target.entity_id
                    or type(completion["generation"]) is not int
                    or completion["generation"] != self.generation
                    or not pending["started_at"] + 1.0 - 1e-9 <= completed <= now
                    or pending["completion"] is not None
                    and pending["completion"] != completion
                ):
                    raise ValueError("Unmatched or premature completion")
                pending["completion"] = dict(completion)
            if (
                pending["completion"] is None
                or now <= pending["completion"]["timestamp"]
            ):
                return None
            pending["after"].append(target.reading.evidence())
            if len(pending["after"]) < WINDOW_SAMPLES:
                self.reason = "observing_after"
                return None
            before_interval, after_interval = (
                _window_interval(pending["before"]),
                _window_interval(pending["after"]),
            )
            gain = [
                after_interval[0] - before_interval[1],
                after_interval[1] - before_interval[0],
            ]
            classification = (
                "useful"
                if gain[0] >= MEANINGFUL_GAIN
                else "no_meaningful_gain"
                if gain[1] < MEANINGFUL_GAIN
                else None
            )
            if classification is None:
                self.cancel("ambiguous_visual_gain")
                return None
            result = {
                "schema": "bb8.visual-outcome.v1",
                "source": SOURCE,
                "request_id": pending["request_id"],
                "entity_id": target.entity_id,
                "marker_id": target.reading.marker_id,
                "generation": self.generation,
                "map_version": self.map_version,
                "calibration_version": self.calibration_version,
                "anchor_xy": list(target.anchor_xy),
                "anchor_radius_m": target.anchor_radius_m,
                "started_at": pending["started_at"],
                "completed_at": pending["completion"]["timestamp"],
                "timestamp": now,
                "completion": pending["completion"],
                "before_frames": pending["before"],
                "during_frames": pending["during"],
                "after_frames": pending["after"],
                "before_interval": before_interval,
                "after_interval": after_interval,
                "gain_interval": gain,
                "before": sum(before_interval) / 2,
                "after": sum(after_interval) / 2,
                "classification": classification,
            }
            result["evidence_sha256"] = _hash(result)
            result = validate_visual_outcome(result)
            self.pending = None
            self.reason = "admitted"
            self._windows.clear()
            return result
        except (ValueError, TypeError, AttributeError):
            self.cancel("invalid_visual_evidence")
            return None
