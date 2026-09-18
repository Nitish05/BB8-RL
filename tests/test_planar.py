import math
from types import SimpleNamespace
from typing import ClassVar

import pytest

from bb8_rl.control.contract import (
    DriveMode,
    PlanarDriveBackend,
    PlanarDriveCommand,
    PlanarDriveParameters,
)
from bb8_rl.control.genesis_backend import GenesisPlanarDriveBackend


class Body:
    n_dofs = 6
    joints: ClassVar[list] = [
        SimpleNamespace(
            dofs_motion_vel=[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dofs_motion_ang=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dofs_idx_local=list(range(6)),
        )
    ]

    def __init__(self):
        self.pos = [0, 0, 0.037]
        self.vel = [0, 0, 0]
        self.forces = []

    def get_mass(self):
        return 0.2

    def get_pos(self):
        return self.pos

    def get_vel(self):
        return self.vel

    def set_dofs_force_range(self, lower, upper):
        self.limits = lower, upper

    def control_dofs_force(self, force):
        self.forces.append(force)


@pytest.mark.parametrize(
    "args", [(float("nan"), 0, 0, 0.1), (2, 0, 0, 0.1), (0, 0, 1, 1), (0, 0, -1, 0.1)]
)
def test_invalid_command(args):
    with pytest.raises(ValueError):
        PlanarDriveCommand(*args)


def test_disk_latency_expiry_and_force_bound():
    body = Body()
    backend = GenesisPlanarDriveBackend(body, PlanarDriveParameters())
    assert isinstance(backend, PlanarDriveBackend)
    assert backend.capabilities.supports(DriveMode.PLANAR_DRIVE)
    assert not backend.capabilities.hardware_capable
    command = PlanarDriveCommand(1, 1, 0, 0.15)
    assert math.hypot(*command.vector) == pytest.approx(1)
    backend.command_drive(command, now=0)
    backend.advance(now=0.04)
    assert body.forces[-1] == [0] * 6
    backend.advance(now=0.05)
    assert math.hypot(*body.forces[-1][:2]) <= 0.2 * 0.7 + 1e-9
    assert body.forces[-1][0] == pytest.approx(body.forces[-1][1])
    assert body.forces[-1][2:] == [0] * 4
    body.vel = [0.1, 0, 0]
    state = backend.advance(now=0.15)
    assert state.command_expired
    assert body.forces[-1][0] < 0


def test_stop_clears_delayed_motion_and_reset_clears_time_and_heading():
    body = Body()
    backend = GenesisPlanarDriveBackend(
        body, PlanarDriveParameters(heading_offset=math.pi / 2)
    )
    backend.command_drive(PlanarDriveCommand(1, 0, 0, 0.15), now=0)
    backend.advance(now=0.05)
    assert body.forces[-1][1] > 0
    backend.command_drive(PlanarDriveCommand(-1, 0, 0.06, 0.21), now=0.06)
    backend.stop()
    backend.advance(now=0.12)
    assert body.forces[-1] == [0] * 6
    assert backend.heading == pytest.approx(math.pi / 2)
    backend.reset()
    assert backend.heading == 0
    backend.command_drive(PlanarDriveCommand(1, 0, 0, 0.15), now=0)
    backend.close()
    assert not backend.connected
    with pytest.raises(RuntimeError):
        backend.advance(now=0)


def test_invalid_stale_future_replayed_and_long_lived_commands():
    backend = GenesisPlanarDriveBackend(Body(), PlanarDriveParameters())
    for command, now in [
        (PlanarDriveCommand(1, 0, 1, 1.1), 0),
        (PlanarDriveCommand(1, 0, 0, 0.1), 0.1),
        (PlanarDriveCommand(1, 0, 0.1, 0.5), 0.1),
    ]:
        with pytest.raises(ValueError):
            backend.command_drive(command, now=now)
    backend.command_drive(PlanarDriveCommand(1, 0, 0.1, 0.2), now=0.1)
    with pytest.raises(ValueError):
        backend.command_drive(PlanarDriveCommand(0, 1, 0.1, 0.2), now=0.1)
    with pytest.raises(ValueError):
        backend.advance(now=0.05)
    with pytest.raises(ValueError):
        backend.advance(now=float("nan"))


def test_dead_zone_preserves_heading_and_nonfinite_state_fails():
    body = Body()
    backend = GenesisPlanarDriveBackend(body, PlanarDriveParameters(latency=0))
    backend.command_drive(PlanarDriveCommand(0, 1, 0, 0.15), now=0)
    backend.advance(now=0)
    backend.command_drive(PlanarDriveCommand(0.01, 0, 0.01, 0.16), now=0.01)
    backend.advance(now=0.01)
    assert body.forces[-1] == [0] * 6
    assert backend.heading == pytest.approx(math.pi / 2)
    body.pos[0] = float("nan")
    with pytest.raises(RuntimeError):
        backend.advance(now=0.02)


def test_rejects_unverified_axes_and_invalid_parameters():
    body = Body()
    body.n_dofs = 7
    with pytest.raises(ValueError):
        GenesisPlanarDriveBackend(body, PlanarDriveParameters())
    for kwargs in (
        {"max_speed": 0},
        {"latency": 0.2},
        {"response_time": float("inf")},
        {"dead_zone": 1},
    ):
        with pytest.raises(ValueError):
            PlanarDriveParameters(**kwargs)
