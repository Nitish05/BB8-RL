from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class DriveMode(str, Enum):
    PLANAR_DRIVE = "planar_drive"


@dataclass(frozen=True)
class PlanarDriveCapabilities:
    backend_id: str
    display_name: str
    modes: frozenset[DriveMode]
    approximate: bool
    hardware_capable: bool = False
    requires_arming: bool = False

    def supports(self, mode: DriveMode) -> bool:
        return mode in self.modes


@dataclass(frozen=True)
class PlanarDriveCommand:
    """Floor-frame unit-disk request; times use the backend's monotonic clock."""

    x: float
    y: float
    timestamp: float
    expires_at: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) for v in (self.x, self.y, self.timestamp, self.expires_at)
        ):
            raise ValueError("Planar commands and timestamps must be finite")
        if self.timestamp < 0 or self.expires_at <= self.timestamp:
            raise ValueError("Command expiry must follow a nonnegative timestamp")
        if max(abs(self.x), abs(self.y)) > 1:
            raise ValueError("Planar command components must be in [-1, 1]")

    @property
    def vector(self) -> tuple[float, float]:
        scale = max(1.0, math.hypot(self.x, self.y))
        return self.x / scale, self.y / scale


@dataclass(frozen=True)
class PlanarControlState:
    position: tuple[float, float]
    velocity: tuple[float, float]
    timestamp: float
    heading: float
    command_expired: bool


@dataclass(frozen=True)
class PlanarDriveParameters:
    """Effective response parameters; SI units, radians, and normalized dead zone."""

    max_speed: float = 0.35
    response_time: float = 0.25
    max_acceleration: float = 0.7
    max_deceleration: float = 0.9
    dead_zone: float = 0.05
    latency: float = 0.05
    max_command_age: float = 0.15
    heading_offset: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Drive parameters must be finite")
        if (
            min(
                self.max_speed,
                self.response_time,
                self.max_acceleration,
                self.max_deceleration,
                self.max_command_age,
            )
            <= 0
        ):
            raise ValueError(
                "Speed, response time, acceleration and command age must be positive"
            )
        if not 0 <= self.dead_zone < 1 or not 0 <= self.latency < self.max_command_age:
            raise ValueError(
                "Dead zone must be in [0, 1); latency must be below command age"
            )


@runtime_checkable
class PlanarDriveBackend(Protocol):
    """Optional capability, independent of named-joint control and project persistence."""

    @property
    def capabilities(self) -> PlanarDriveCapabilities: ...

    @property
    def connected(self) -> bool: ...

    def command_drive(self, command: PlanarDriveCommand, *, now: float) -> None: ...

    def read_planar_state(self, *, now: float) -> PlanarControlState: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
