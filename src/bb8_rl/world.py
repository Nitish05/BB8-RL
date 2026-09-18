"""Single-environment M1 fixture. Genesis owns integration and contacts."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from genesis_studio.models import BoxObjectConfig, SphereObjectConfig
from genesis_studio.project import load_project
from genesis_studio_desktop.compatibility import require_supported_genesis
from genesis_studio_desktop.scene_builder import create_genesis_scene, make_scene_spec

from bb8_rl.control.contract import PlanarDriveCommand
from bb8_rl.control.genesis_backend import GenesisPlanarDriveBackend, _as_python

from .config import load_config


def validate_world(config_path: Path):
    config, project_path = load_config(config_path)
    project = load_project(project_path)
    if project.all_robots or any(
        o.format not in ("sphere", "box", "cylinder") for o in project.objects
    ):
        raise ValueError(
            "M1 supports a primitive-only world; remove articulated or external assets"
        )
    objects = {o.name: o for o in project.objects}
    body, head = objects.get(config.body), objects.get(config.head)
    if not isinstance(body, SphereObjectConfig) or body.fixed:
        raise ValueError("Navigation body must be a dynamic sphere in the Project JSON")
    if (
        not isinstance(head, SphereObjectConfig)
        or not head.fixed
        or head.collision_mask
        or head.collision_group
    ):
        raise ValueError(
            "Approximate head must be a fixed sphere with both collision masks zero"
        )
    if any(not o.fixed for o in project.objects if o.name != config.body):
        raise ValueError("M1 permits only the drive body to be dynamic")
    if not project.environment.ground or project.environment.gravity != (
        0.0,
        0.0,
        -9.81,
    ):
        raise ValueError("M1 requires a horizontal ground plane and Earth gravity")
    if abs(body.position[2] - body.radius) > 0.005:
        raise ValueError("Body must start at its radius above the ground")
    if max(abs(v) for v in body.position[:2]) + body.radius >= config.arena_half_extent:
        raise ValueError("Body start lies outside the footprint-inflated arena")
    # M1's square workspace has physical boundaries, not only a software flag.
    for axis in (0, 1):
        for sign in (-1, 1):
            if not any(
                isinstance(o, BoxObjectConfig)
                and o.fixed
                and o.euler == (0, 0, 0)
                and o.collision_group & body.collision_mask
                and o.collision_mask & body.collision_group
                and abs(
                    sign * o.position[axis]
                    - o.size[axis] / 2
                    - config.arena_half_extent
                )
                < 1e-6
                and o.size[1 - axis] / 2 - abs(o.position[1 - axis])
                >= config.arena_half_extent
                and o.position[2] - o.size[2] / 2 <= body.radius
                and o.position[2] + o.size[2] / 2 >= 2 * body.radius
                for o in project.objects
            ):
                raise ValueError(
                    "M1 requires four colliding walls enclosing the configured square arena"
                )
    camera = next((s for s in project.sensors if s.name == config.camera), None)
    for name in config.obstacle_names:
        item = objects.get(name)
        if (
            not isinstance(item, BoxObjectConfig)
            or not item.fixed
            or item.euler != (0, 0, 0)
        ):
            raise ValueError(
                "Procedural obstacle slots must be fixed axis-aligned boxes"
            )
        if not (
            item.collision_group & body.collision_mask
            and item.collision_mask & body.collision_group
        ):
            raise ValueError("Obstacle slots must collide with the navigation body")
    if camera is None or camera.type != "camera" or camera.parent or not camera.enabled:
        raise ValueError(
            "Choose an enabled, world-fixed RGB camera in the Project JSON"
        )
    if config.action_steps * project.physics.time_step >= config.drive.max_command_age:
        raise ValueError("Command lifetime must exceed the action period")
    return config, project, body, head, camera


@dataclass(frozen=True)
class CameraFrame:
    timestamp: float
    rgb: object


class NavigationWorld:
    def __init__(
        self,
        config_path: Path,
        *,
        backend: str = "cpu",
        viewer: bool = False,
        render: bool = False,
        camera_positions=None,
    ):
        if camera_positions is not None and (
            not render
            or not 1 <= len(camera_positions) <= 3
            or any(
                len(p) != 3 or not all(math.isfinite(v) for v in p)
                for p in camera_positions
            )
            or len({tuple(p) for p in camera_positions}) != len(camera_positions)
        ):
            raise ValueError(
                "A rendered rig requires one to three distinct finite positions"
            )
        require_supported_genesis()
        if backend not in ("cpu", "metal"):
            raise ValueError("Choose cpu or metal; backend fallback is disabled")
        self.config, self.project, self.body_spec, self.head_spec, camera = (
            validate_world(config_path)
        )
        import genesis as gs

        self.gs = gs
        self._owns_genesis = False
        self.scene = None
        self.backend = None
        self.camera = None
        self.cameras = {}
        self.frames: deque[CameraFrame] = deque(maxlen=4)
        self.camera_period = 1 / camera.rate
        self._next_frame = 0.0
        self.steps = 0
        self.contacts = 0
        self.boundary_failure = False
        self.dt = self.project.physics.time_step
        try:
            gs.init(
                backend=getattr(gs, backend),
                precision="32",
                seed=self.project.seed,
                logging_level="warning",
            )
            self._owns_genesis = True
            # A requested backend is a hard requirement, including under sandboxed execution.
            if gs.backend != getattr(gs, backend):
                raise RuntimeError(
                    f"Requested {backend}, but Genesis resolved {gs.backend}"
                )
            self.scene, _ = create_genesis_scene(
                gs, make_scene_spec(self.project, None), show_viewer=viewer
            )
            if render:
                positions = (
                    [camera.position] if camera_positions is None else camera_positions
                )
                for name, position in zip("ABC", positions):
                    self.cameras[name] = self.scene.add_camera(
                        res=camera.resolution,
                        pos=position,
                        lookat=camera.lookat,
                        fov=camera.fov,
                        GUI=False,
                    )
                self.camera = self.cameras["A"]
            self.scene.build()
            self.body = self.scene.get_entity(name=self.config.body)
            self.head = self.scene.get_entity(name=self.config.head)
            self.backend = GenesisPlanarDriveBackend(
                self.body, self.config.drive.parameters()
            )
            self._initial_state = self.scene.get_state()
            self._ground_links = {
                link.idx for link in self.scene.get_entity(name="ground").links
            }
            self.reset()
        except BaseException:
            self.close()
            raise

    @property
    def time(self) -> float:
        return self.steps * self.dt

    def reset(
        self,
        *,
        start: tuple[float, float] | None = None,
        obstacle_positions: tuple[tuple[float, float], ...] = (),
    ):
        if start is not None and (
            len(start) != 2
            or not all(math.isfinite(v) for v in start)
            or max(abs(v) for v in start) + self.body_spec.radius
            >= self.config.arena_half_extent
        ):
            raise ValueError("Reset start must be finite and inside the arena")
        if len(obstacle_positions) > len(self.config.obstacle_names) or any(
            len(p) != 2 or not all(math.isfinite(v) for v in p)
            for p in obstacle_positions
        ):
            raise ValueError("Invalid obstacle slot positions")
        # Public scene reset restores all fixture poses/velocities and clears engine time.
        # This is an episode-boundary reset, not a deterministic solver checkpoint.
        self.scene.reset(self._initial_state)
        self.backend.reset()
        if start is not None:
            self.backend.initialize_position((*start, self.body_spec.radius))
        specifications = {o.name: o for o in self.project.objects}
        for name, position in zip(self.config.obstacle_names, obstacle_positions):
            self.scene.get_entity(name=name).set_pos(
                (*position, specifications[name].position[2])
            )
        self.steps = 0
        self.contacts = 0
        self.boundary_failure = False
        self.frames.clear()
        self._next_frame = 0.0
        self._pose_head()
        return self.backend.read_planar_state(now=0.0)

    def drive(self, x: float, y: float):
        if self.boundary_failure:
            raise RuntimeError(
                "Body left the valid arena; reset the world before commanding it"
            )
        command = PlanarDriveCommand(
            x, y, self.time, self.time + self.config.drive.max_command_age
        )
        self.backend.command_drive(command, now=self.time)

    def _pose_head(self):
        position = _as_python(self.body.get_pos())
        offset = self.head_spec.position[2] - self.body_spec.position[2]
        self.head.set_pos([position[0], position[1], position[2] + offset])
        # This is a non-colliding visual, never a body-motion command.
        self.head.set_quat([1, 0, 0, 0])

    def step(self, count: int = 1):
        if count < 1:
            raise ValueError("Step count must be positive")
        for _ in range(count):
            self.backend.advance(now=self.time)
            self.scene.step()
            self.steps += 1
            state = self.backend.read_planar_state(now=self.time)
            if (
                max(abs(v) for v in state.position) + self.body_spec.radius
                > self.config.arena_half_extent + 0.005
            ):
                self.boundary_failure = True
                self.backend.stop()
            self._pose_head()
            contacts = {
                key: _as_python(value)
                for key, value in self.body.get_contacts().items()
            }
            for i, (a, b) in enumerate(
                zip(contacts.get("link_a", []), contacts.get("link_b", []))
            ):
                if not contacts.get("valid_mask", [True] * len(contacts["link_a"]))[i]:
                    continue
                if a not in self._ground_links and b not in self._ground_links:
                    self.contacts += 1
            if self.camera is not None and self.time + 1e-10 >= self._next_frame:
                rgb = self.camera.render(rgb=True)[0]
                self.frames.append(CameraFrame(self.time, rgb.copy()))
                while self._next_frame <= self.time + 1e-10:
                    self._next_frame += self.camera_period
        return self.backend.read_planar_state(now=self.time)

    def stop(self):
        self.backend.stop()

    def capture_rig(self, calibration_version="synthetic-v1"):
        """On-demand RGB frames at one paused simulation instant (at most three).

        Sequential render wall time is not hardware synchronization or a claim
        of real-time performance. Privileged labels are never requested here.
        """
        from .camera_rig import CameraFrame as RigFrame

        if not self.cameras:
            raise RuntimeError("World was not created with rendering enabled")
        return tuple(
            RigFrame(
                name,
                calibration_version,
                camera.render(rgb=True, force_render=True)[0],
                self.time,
            )
            for name, camera in self.cameras.items()
        )

    def close(self):
        try:
            if self.backend is not None:
                self.backend.close()
        finally:
            self.backend = None
            try:
                if self.scene is not None:
                    self.scene.destroy()
            finally:
                self.scene = None
                if self._owns_genesis:
                    self.gs.destroy()
                    self._owns_genesis = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
