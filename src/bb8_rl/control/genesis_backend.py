from __future__ import annotations

import math
from collections import deque
from typing import Any

from .contract import (
    DriveMode,
    PlanarControlState,
    PlanarDriveCapabilities,
    PlanarDriveCommand,
    PlanarDriveParameters,
)


def _as_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


class GenesisPlanarDriveBackend:
    """Approximate free-body drive, with contacts integrated by Genesis.

    Supports one unbatched rigid body. Public joint motion axes are checked before
    actuator use; the integration suite verifies that translation is world-fixed
    even after shell rotation. No pose/velocity setters are used during stepping.
    """

    capabilities = PlanarDriveCapabilities(
        backend_id="genesis-planar",
        display_name="Genesis approximate planar drive",
        modes=frozenset({DriveMode.PLANAR_DRIVE}),
        approximate=True,
    )

    def __init__(self, entity: Any, parameters: PlanarDriveParameters) -> None:
        self._entity_ref = entity
        self.parameters = parameters
        self._pending: deque[PlanarDriveCommand] = deque()
        self._active: PlanarDriveCommand | None = None
        self._last_issued = -1.0
        self._last_time = 0.0
        self.heading = 0.0
        self.last_force = (0.0, 0.0)
        if entity.n_dofs != 6 or len(entity.joints) != 1:
            raise ValueError(
                "Planar drive requires a single free rigid body with six DOFs"
            )
        joint = entity.joints[0]
        expected_vel = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        expected_ang = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        if (
            _as_python(joint.dofs_motion_vel) != expected_vel
            or _as_python(joint.dofs_motion_ang) != expected_ang
            or list(joint.dofs_idx_local) != list(range(6))
        ):
            raise RuntimeError(
                "Unverified Genesis free-body DOF mapping; run the navigation integration checks"
            )
        self.mass = float(entity.get_mass())
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Planar body must have positive finite mass")
        self.force_limit = self.mass * max(
            parameters.max_acceleration, parameters.max_deceleration
        )
        entity.set_dofs_force_range(
            [-self.force_limit] * 3 + [0.0] * 3, [self.force_limit] * 3 + [0.0] * 3
        )
        self.stop()

    @property
    def connected(self) -> bool:
        return self._entity_ref is not None

    def _entity(self) -> Any:
        if not self.connected:
            raise RuntimeError("Planar simulation backend is closed")
        return self._entity_ref

    def _check_time(self, now: float) -> None:
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError(
                "Simulation time must be finite and monotonic; reset before restarting the clock"
            )
        self._last_time = now

    def command_drive(self, command: PlanarDriveCommand, *, now: float) -> None:
        self._entity()
        self._check_time(now)
        if command.timestamp > now or command.expires_at <= now:
            raise ValueError("Rejecting future or expired planar command")
        if command.timestamp <= self._last_issued:
            raise ValueError("Planar command timestamps must increase")
        if (
            command.expires_at - command.timestamp
            > self.parameters.max_command_age + 1e-9
        ):
            raise ValueError("Command lifetime exceeds the configured maximum age")
        if len(self._pending) >= 256:
            raise RuntimeError("Planar command queue is full; reduce command rate")
        self._last_issued = command.timestamp
        self._pending.append(command)

    def read_planar_state(self, *, now: float) -> PlanarControlState:
        self._check_time(now)
        entity = self._entity()
        pos, vel = _as_python(entity.get_pos()), _as_python(entity.get_vel())
        if (
            len(pos) != 3
            or len(vel) != 3
            or not all(math.isfinite(v) for v in (*pos, *vel))
        ):
            raise RuntimeError(
                "Invalid or batched planar state; stop the run and inspect the fixture"
            )
        return PlanarControlState(
            tuple(pos[:2]),
            tuple(vel[:2]),
            now,
            self.heading,
            self._active is None or now >= self._active.expires_at,
        )

    def advance(self, *, now: float) -> PlanarControlState:
        state = self.read_planar_state(now=now)
        p = self.parameters
        while self._pending and self._pending[0].timestamp + p.latency <= now + 1e-10:
            self._active = self._pending.popleft()
        target = (0.0, 0.0)
        if self._active is not None and now < self._active.expires_at:
            x, y = self._active.vector
            magnitude = math.hypot(x, y)
            if magnitude > p.dead_zone:
                self.heading = math.atan2(y, x) + p.heading_offset
                self.heading = math.atan2(
                    math.sin(self.heading), math.cos(self.heading)
                )
                speed = p.max_speed * (magnitude - p.dead_zone) / (1 - p.dead_zone)
                target = speed * math.cos(self.heading), speed * math.sin(self.heading)
        acceleration = tuple(
            (v - old) / p.response_time for v, old in zip(target, state.velocity)
        )
        limit = (
            p.max_deceleration
            if math.hypot(*target) < math.hypot(*state.velocity)
            else p.max_acceleration
        )
        scale = max(1.0, math.hypot(*acceleration) / limit)
        self.last_force = tuple(self.mass * a / scale for a in acceleration)
        self._entity().control_dofs_force([*self.last_force, 0.0, 0.0, 0.0, 0.0])
        return self.read_planar_state(now=now)

    def stop(self) -> None:
        """Clear delayed commands immediately; subsequent ticks apply bounded braking."""
        self._pending.clear()
        self._active = None
        self.last_force = (0.0, 0.0)
        self._entity().control_dofs_force([0.0] * 6)

    def reset(self) -> None:
        self.stop()
        self.heading = 0.0
        self._last_issued = -1.0
        self._last_time = 0.0

    def initialize_position(self, position: tuple[float, float, float]) -> None:
        """Episode initialization only; never called by the stepping controller."""
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise ValueError("Initial position must contain three finite coordinates")
        self._entity().set_pos(position, zero_velocity=True)
        self._entity().set_quat([1, 0, 0, 0], zero_velocity=True)
        self._entity().set_dofs_velocity([0.0] * 6)

    @property
    def applied_request(self) -> tuple[float, float]:
        if self._active is None or self._last_time >= self._active.expires_at:
            return (0.0, 0.0)
        return self._active.vector

    def close(self) -> None:
        if self.connected:
            self.stop()
            self._entity_ref = None
