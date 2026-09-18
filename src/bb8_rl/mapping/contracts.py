"""Meter-based geometric evidence, with explicit origin and registration context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SpaceState(str, Enum):
    UNKNOWN = "unknown"
    FREE = "free"
    OCCUPIED = "occupied"


class Provenance(str, Enum):
    ANALYTIC_SYNTHETIC = "analytic_synthetic"
    RGB_RECONSTRUCTION = "rgb_reconstruction"
    MEASURED_DEPTH = "measured_depth"
    MANUAL_SURVEY = "manual_survey"


def finite_vector3(values):
    values = tuple(float(v) for v in values)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected three finite meter coordinates")
    return values


@dataclass(frozen=True)
class Bounds3D:
    """Closed, axis-aligned world bounds in meters; not a watertight object mesh."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "minimum", finite_vector3(self.minimum))
        object.__setattr__(self, "maximum", finite_vector3(self.maximum))
        if any(a >= b for a, b in zip(self.minimum, self.maximum)):
            raise ValueError("Bounds must have positive extent in every dimension")

    @property
    def size(self):
        return tuple(b - a for a, b in zip(self.minimum, self.maximum))

    def expanded(self, margin):
        if not math.isfinite(margin):
            raise ValueError("Margin must be finite")
        return Bounds3D(
            tuple(v - margin for v in self.minimum),
            tuple(v + margin for v in self.maximum),
        )

    def contains(self, other):
        return all(
            a <= c and d <= b
            for a, b, c, d in zip(
                self.minimum, self.maximum, other.minimum, other.maximum
            )
        )

    def to_dict(self):
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class EvidenceSource:
    """A stated error bound is metadata supplied by the producer, not calibrated here."""

    source_id: str
    view_ids: tuple[str, ...]
    observed_at: float
    provenance: Provenance
    uncertainty_m: float = 0.0

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("Evidence needs a source ID")
        if isinstance(self.view_ids, str):
            raise TypeError("view_ids must be a sequence of IDs")
        views = tuple(self.view_ids)
        if (
            not views
            or any(not isinstance(v, str) or not v for v in views)
            or len(set(views)) != len(views)
        ):
            raise ValueError("Evidence needs distinct, nonempty source view IDs")
        object.__setattr__(self, "view_ids", views)
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if not math.isfinite(self.observed_at) or self.observed_at < 0:
            raise ValueError("Observation time must be finite and nonnegative")
        if not math.isfinite(self.uncertainty_m) or self.uncertainty_m < 0:
            raise ValueError("Geometry uncertainty must be finite and nonnegative")

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "view_ids": list(self.view_ids),
            "observed_at": self.observed_at,
            "provenance": self.provenance.value,
            "uncertainty_m": self.uncertainty_m,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class VolumeEvidence:
    bounds: Bounds3D
    state: SpaceState
    source: EvidenceSource

    def __post_init__(self):
        object.__setattr__(self, "state", SpaceState(self.state))
        if self.state is SpaceState.UNKNOWN:
            raise ValueError("Unknown means no valid evidence; invalidate changed maps")
        if not isinstance(self.bounds, Bounds3D) or not isinstance(
            self.source, EvidenceSource
        ):
            raise TypeError("Volume evidence requires validated bounds and source")

    def to_dict(self):
        return {
            "bounds": self.bounds.to_dict(),
            "state": self.state.value,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            Bounds3D.from_dict(value["bounds"]),
            SpaceState(value["state"]),
            EvidenceSource.from_dict(value["source"]),
        )


@dataclass(frozen=True)
class ObjectBounds:
    """Conservative occupied box; its enclosed empty volume is not traversable."""

    object_id: str
    bounds: Bounds3D
    source: EvidenceSource

    def __post_init__(self):
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("Object needs an ID")
        if not isinstance(self.bounds, Bounds3D) or not isinstance(
            self.source, EvidenceSource
        ):
            raise TypeError("Object requires validated bounds and source")

    def to_dict(self):
        return {
            "object_id": self.object_id,
            "bounds": self.bounds.to_dict(),
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["object_id"],
            Bounds3D.from_dict(value["bounds"]),
            EvidenceSource.from_dict(value["source"]),
        )
