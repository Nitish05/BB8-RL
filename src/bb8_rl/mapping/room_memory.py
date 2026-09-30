"""Conservative voxel memory from explicit volumes, not an RGB reconstructer.

Free volumes must already be certified by an upstream geometry producer. Missing
points, a hidden current view, and absent object boxes never create free evidence.
Scene/camera changes are supplied by the caller; this module does not detect them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .contracts import (
    Bounds3D,
    EvidenceSource,
    ObjectBounds,
    SpaceState,
    VolumeEvidence,
    finite_vector3,
)


class RoomMemory:
    SCHEMA_VERSION = 1
    _UNKNOWN, _FREE, _OCCUPIED = 0, 1, 2

    def __init__(
        self,
        bounds: Bounds3D,
        resolution_m: float,
        *,
        scene_version: str,
        calibration_version: str,
        world_frame: str = "room",
    ):
        if not isinstance(bounds, Bounds3D):
            raise TypeError("Room bounds must be validated")
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("Resolution must be positive and finite")
        for identifier in (scene_version, calibration_version, world_frame):
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Scene, calibration and world-frame IDs are required")
        dimensions = np.asarray(bounds.size) / resolution_m
        if np.any(dimensions > 10000):
            raise ValueError("Voxel dimensions exceed the bounded prototype budget")
        shape = np.ceil(dimensions).astype(int)
        if math.prod(shape) > 2_000_000:
            raise ValueError("Voxel count exceeds the bounded prototype budget")
        self.bounds = bounds
        self.resolution_m = float(resolution_m)
        self.scene_version = scene_version
        self.calibration_version = calibration_version
        self.world_frame = world_frame
        self.map_version = 0
        self._cells = np.zeros(tuple(shape), dtype=np.uint8)
        self._volumes: list[VolumeEvidence] = []
        self._objects: list[ObjectBounds] = []
        self._invalidations: list[dict] = []

    @property
    def valid(self):
        return not self._invalidations

    @property
    def objects(self):
        return tuple(self._objects)

    @property
    def evidence(self):
        return tuple(self._volumes)

    def _axes(self, bounds, *, fully_contained):
        axes = []
        for dimension, count in enumerate(self._cells.shape):
            # Derive shared faces once: adding resolution to a rounded lower
            # edge can disagree with the next voxel's edge (e.g. .1 + .02).
            edges = self.bounds.minimum[dimension] + self.resolution_m * np.arange(
                count + 1
            )
            lo, hi = edges[:-1], edges[1:]
            selected = (
                (lo >= bounds.minimum[dimension]) & (hi <= bounds.maximum[dimension])
                if fully_contained
                else (lo <= bounds.maximum[dimension])
                & (hi >= bounds.minimum[dimension])
            )
            # The trailing partial voxel must never certify outside-room space.
            if fully_contained:
                selected &= hi <= self.bounds.maximum[dimension]
            axes.append(np.flatnonzero(selected))
        return tuple(axes)

    def _integrate(self, evidence):
        margin = evidence.source.uncertainty_m
        if evidence.state is SpaceState.FREE:
            if any(size <= 2 * margin for size in evidence.bounds.size):
                return
            bounds = evidence.bounds.expanded(-margin)
            axes = self._axes(bounds, fully_contained=True)
            indices = np.ix_(*axes)
            old = self._cells[indices]
            self._cells[indices] = np.where(old == self._OCCUPIED, old, self._FREE)
        else:
            axes = self._axes(evidence.bounds.expanded(margin), fully_contained=False)
            self._cells[np.ix_(*axes)] = self._OCCUPIED

    def _require_valid(self):
        if not self.valid:
            raise ValueError(
                "Invalidated memory requires a new registered reconstruction"
            )

    def observe_volume(
        self, bounds: Bounds3D, state: SpaceState, source: EvidenceSource
    ):
        self._require_valid()
        evidence = VolumeEvidence(bounds, state, source)
        self._integrate(evidence)
        self._volumes.append(evidence)
        self.map_version += 1

    def remember_object(self, obj: ObjectBounds):
        self._require_valid()
        if not isinstance(obj, ObjectBounds):
            raise TypeError("Object must use the validated bounds contract")
        if any(existing.object_id == obj.object_id for existing in self._objects):
            raise ValueError(
                "Object ID already exists; scene changes require invalidation"
            )
        self._integrate(VolumeEvidence(obj.bounds, SpaceState.OCCUPIED, obj.source))
        self._objects.append(obj)
        self.map_version += 1

    def query(self, bounds: Bounds3D) -> SpaceState:
        """All touched voxels must be free; even boundary-touching occupancy blocks."""
        if not isinstance(bounds, Bounds3D):
            raise TypeError("Query requires validated 3D bounds")
        if not self.valid or not self.bounds.contains(bounds):
            return SpaceState.UNKNOWN
        values = self._cells[np.ix_(*self._axes(bounds, fully_contained=False))]
        if values.size == 0:
            return SpaceState.UNKNOWN
        if np.any(values == self._OCCUPIED):
            return SpaceState.OCCUPIED
        if np.any(values == self._UNKNOWN):
            return SpaceState.UNKNOWN
        return SpaceState.FREE

    def segment_free(self, start, end, *, radius_m, height_m, floor_z=0.0):
        """Certify the whole bounding prism of a horizontal swept body.

        This deliberately overapproximates diagonal paths. Radius is a caller-
        supplied total envelope including robot size and any error/braking margin.
        It does not derive a safe margin or sample past thin unknown regions.
        """
        if (
            not all(math.isfinite(v) for v in (radius_m, height_m, floor_z))
            or radius_m <= 0
            or height_m <= 0
        ):
            raise ValueError("Radius and height must be positive finite meters")
        start = finite_vector3((*start, floor_z))
        end = finite_vector3((*end, floor_z))
        bounds = Bounds3D(
            (
                min(start[0], end[0]) - radius_m,
                min(start[1], end[1]) - radius_m,
                floor_z,
            ),
            (
                max(start[0], end[0]) + radius_m,
                max(start[1], end[1]) + radius_m,
                floor_z + height_m,
            ),
        )
        return self.query(bounds) is SpaceState.FREE

    def invalidate_if_changed(self, *, scene_version, calibration_version, at_time):
        if not math.isfinite(at_time) or at_time < 0:
            raise ValueError("Invalidation time must be finite and nonnegative")
        for kind, original, current in (
            ("scene", self.scene_version, scene_version),
            ("camera_registration", self.calibration_version, calibration_version),
        ):
            if not isinstance(current, str) or not current:
                raise ValueError("Context versions must be nonempty strings")
            if current != original:
                change = {
                    "kind": kind,
                    "original_version": original,
                    "observed_version": current,
                    "at_time": at_time,
                }
                if change not in self._invalidations:
                    self._invalidations.append(change)
                    self.map_version += 1
        return self.valid

    def to_dict(self):
        return {
            "schema_version": self.SCHEMA_VERSION,
            "units": "meters",
            "bounds": self.bounds.to_dict(),
            "resolution_m": self.resolution_m,
            "world_frame": self.world_frame,
            "scene_version": self.scene_version,
            "calibration_version": self.calibration_version,
            "map_version": self.map_version,
            "volumes": [v.to_dict() for v in self._volumes],
            "objects": [o.to_dict() for o in self._objects],
            "invalidations": [dict(i) for i in self._invalidations],
        }

    @classmethod
    def from_dict(cls, value):
        if value["schema_version"] != cls.SCHEMA_VERSION or value["units"] != "meters":
            raise ValueError("Unsupported memory schema or non-metric units")
        room = cls(
            Bounds3D.from_dict(value["bounds"]),
            value["resolution_m"],
            scene_version=value["scene_version"],
            calibration_version=value["calibration_version"],
            world_frame=value["world_frame"],
        )
        for raw in value["volumes"]:
            evidence = VolumeEvidence.from_dict(raw)
            room.observe_volume(evidence.bounds, evidence.state, evidence.source)
        for raw in value["objects"]:
            room.remember_object(ObjectBounds.from_dict(raw))
        for change in value["invalidations"]:
            kind = change["kind"]
            if kind not in ("scene", "camera_registration"):
                raise ValueError("Unknown invalidation kind")
            original = (
                room.scene_version if kind == "scene" else room.calibration_version
            )
            if change["original_version"] != original:
                raise ValueError("Invalidation does not match stored registration")
            room.invalidate_if_changed(
                scene_version=change["observed_version"]
                if kind == "scene"
                else room.scene_version,
                calibration_version=change["observed_version"]
                if kind == "camera_registration"
                else room.calibration_version,
                at_time=change["at_time"],
            )
        if (
            type(value["map_version"]) is not int
            or value["map_version"] != room.map_version
        ):
            raise ValueError("Map version does not match retained evidence and changes")
        return room

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
