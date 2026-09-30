"""Scalar Gymnasium goal task; oracle observations, no goal-dependent termination."""

from collections import deque
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from genesis_studio.models import BoxObjectConfig
from gymnasium import spaces

from .layouts import Layout, layout_grid, make_layout, sample_endpoints
from .task import SPLITS, load_task, require_split_seed
from .world import NavigationWorld, validate_world


class NavigationEnv(gym.Env):
    metadata: ClassVar[dict] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        task_path=Path("configs/navigation/bb8-task.yaml"),
        *,
        split="train",
        backend="cpu",
        render_mode=None,
        camera_positions=None,
        visual_fixture=None,
    ):
        if split not in SPLITS or render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Invalid task split or render mode")
        self.task, self.world_path = load_task(Path(task_path))
        if split == "heldout" and self.task.layout_mode != "authored":
            raise ValueError("The held-out seed domain requires an authored fixture")
        self.config, project, self.body_spec, _, _ = validate_world(self.world_path)
        self.sizes = [
            next(o.size for o in project.objects if o.name == name)
            for name in self.config.obstacle_names
        ]
        if self.task.layout_mode == "procedural" and (
            len(self.sizes) != 8
            or any(not np.allclose(size, (0.3, 0.3, 0.3)) for size in self.sizes)
        ):
            raise ValueError("M3 layouts require eight 0.3 m cube slots")
        self.authored_positions = tuple(
            next(o.position[:2] for o in project.objects if o.name == name)
            for name in self.config.obstacle_names
        )
        extent = self.config.arena_half_extent
        for obj in project.objects:
            if obj.name in self.config.obstacle_names:
                if (
                    obj.position[2] - obj.size[2] / 2 > 0
                    or obj.position[2] + obj.size[2] / 2 < 2 * self.body_spec.radius
                    or (
                        self.task.layout_mode == "procedural"
                        and all(
                            abs(obj.position[i]) - obj.size[i] / 2
                            <= extent + self.body_spec.radius
                            for i in (0, 1)
                        )
                    )
                ):
                    raise ValueError(
                        "Obstacle slots must reach the floor and start parked outside the arena"
                    )
            elif (
                obj.name != self.config.body
                and (
                    obj.collision_group & self.body_spec.collision_mask
                    and obj.collision_mask & self.body_spec.collision_group
                )
                and not (
                    isinstance(obj, BoxObjectConfig)
                    and obj.euler == (0, 0, 0)
                    and any(
                        abs(obj.position[i]) - obj.size[i] / 2 >= extent - 1e-6
                        for i in (0, 1)
                    )
                )
            ):
                # The procedural map represents slots and arena boundaries only.
                # Reject authored interior colliders that would be invisible to it.
                raise ValueError(
                    "Every interior collider must be a configured obstacle slot"
                )
        self.split, self.backend_name, self.render_mode = split, backend, render_mode
        self.camera_positions = camera_positions
        self.visual_fixture = visual_fixture
        self.world = None
        self._done = True
        self.grid = None
        self.goal = None
        self.history = deque(maxlen=self.task.history_length)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        state_size = 6 + 6 * self.task.history_length
        map_size = 3 * self.task.local_map_size**2
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    np.r_[np.full(state_size, -np.inf), np.zeros(map_size)].astype(
                        np.float32
                    ),
                    np.r_[np.full(state_size, np.inf), np.ones(map_size)].astype(
                        np.float32
                    ),
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"layout_seed", "start", "goal"}:
            raise ValueError("Reset accepts only layout_seed, start and goal options")
        lo, hi = SPLITS[self.split]
        layout_seed = int(options.get("layout_seed", self.np_random.integers(lo, hi)))
        require_split_seed(self.split, layout_seed)
        layout = (
            Layout(layout_seed, "authored", self.authored_positions)
            if self.task.layout_mode == "authored"
            else make_layout(layout_seed, self.split)
        )
        grid = layout_grid(
            layout,
            extent=self.config.arena_half_extent,
            resolution=self.task.grid_resolution,
            inflation=self.body_spec.radius + self.task.clearance_margin,
            sizes=self.sizes,
        )
        if ("start" in options) != ("goal" in options):
            raise ValueError("Supply both start and goal, or neither")
        if "start" in options:
            start, goal = (
                np.asarray(options[k], dtype=float) for k in ("start", "goal")
            )
            grid.route(start, goal)  # validates finiteness, footprint and connectivity
        else:
            start, goal = sample_endpoints(
                grid,
                self.np_random,
                layout,
                self.task.min_goal_distance,
                self.task.max_goal_distance,
            )
        if self.world is None:
            self.world = NavigationWorld(
                self.world_path,
                backend=self.backend_name,
                viewer=self.render_mode == "human",
                render=self.render_mode == "rgb_array",
                camera_positions=self.camera_positions,
                visual_fixture=self.visual_fixture,
            )
        self.state = self.world.reset(
            start=tuple(start), obstacle_positions=layout.positions
        )
        self.layout, self.grid = layout, grid
        self.goal = np.array([*goal, 0], dtype=np.float32)
        self.steps = 0
        self.dwell = 0.0
        self._done = False
        self.history.clear()
        self.history.extend(
            np.zeros(6, dtype=np.float32) for _ in range(self.task.history_length)
        )
        return self._observation(), self._info(False, False)

    def _observation(self):
        state = np.array(
            [*self.state.position, *self.state.velocity, 0.0, 0.0], dtype=np.float32
        )
        local_map = self.grid.local_map(self.state.position, self.task.local_map_size)
        return {
            "observation": np.concatenate(
                (state, *self.history, local_map.ravel())
            ).astype(np.float32),
            "achieved_goal": np.array(
                [*self.state.position, np.linalg.norm(self.state.velocity)],
                dtype=np.float32,
            ),
            "desired_goal": self.goal.copy(),
        }

    def goal_matches(self, achieved, desired):
        a, d = np.asarray(achieved), np.asarray(desired)
        return (
            np.linalg.norm(a[..., :2] - d[..., :2], axis=-1)
            <= self.task.position_tolerance
        ) & (abs(a[..., 2] - d[..., 2]) <= self.task.speed_tolerance)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Relabel-safe scalar/batched reward, including retained goal-independent costs."""
        success = self.goal_matches(achieved_goal, desired_goal)
        if isinstance(info, dict):
            penalty = np.asarray(info.get("event_penalty", 0.0), dtype=np.float32)
        else:
            penalty = np.array(
                [item.get("event_penalty", 0.0) for item in info], dtype=np.float32
            )
        return np.asarray(
            -np.logical_not(success).astype(np.float32) - penalty, dtype=np.float32
        )

    def _info(self, collision, boundary, penalty=0.0):
        return {
            "is_success": bool(self.dwell + 1e-9 >= self.task.dwell_seconds),
            "collision": bool(collision),
            "boundary_failure": bool(boundary),
            "event_penalty": float(penalty),
            "layout_seed": self.layout.seed,
            "layout_kind": self.layout.kind,
            "split": self.split,
            "observation_mode": "oracle",
            "sim_time": self.world.time,
            "dwell_seconds": self.dwell,
        }

    def step(self, action):
        if self._done or self.world is None:
            raise RuntimeError("Call reset before stepping a new or completed episode")
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != (2,)
            or not np.isfinite(action).all()
            or np.any(abs(action) > 1)
        ):
            raise ValueError("Action must contain two finite components in [-1, 1]")
        normalized = action / max(1.0, float(np.linalg.norm(action)))
        before = self.world.contacts
        self.world.drive(*normalized)
        collision = boundary = False
        for _ in range(self.config.action_steps):
            try:
                self.state = self.world.step()
            except BaseException:
                self._done = True
                self.world.stop()
                raise
            collision = self.world.contacts > before
            boundary = self.world.boundary_failure
            achieved = np.array(
                [*self.state.position, np.linalg.norm(self.state.velocity)]
            )
            self.dwell = (
                self.dwell + self.world.dt
                if self.goal_matches(achieved, self.goal)
                else 0.0
            )
            if collision or boundary:
                self.dwell = 0.0
                break
        self.steps += 1
        self.history.append(
            np.array(
                [
                    *self.state.velocity,
                    *normalized,
                    *self.world.backend.applied_request,
                ],
                dtype=np.float32,
            )
        )
        penalty = (
            self.task.collision_penalty * collision
            + self.task.boundary_penalty * boundary
            + self.task.control_cost * float(normalized @ normalized)
        )
        info = self._info(collision, boundary, penalty)
        observation = self._observation()
        terminated = bool(collision or boundary)
        truncated = bool(self.steps >= self.task.max_episode_steps and not terminated)
        self._done = terminated or truncated
        if self._done:
            self.world.stop()
        reward = float(
            self.compute_reward(
                observation["achieved_goal"], observation["desired_goal"], info
            )
        )
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.world is None:
            raise RuntimeError("Reset before rendering")
        if self.render_mode == "rgb_array":
            return self.world.camera.render(rgb=True)[0].copy()
        return None

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None
        self._done = True
