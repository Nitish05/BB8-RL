"""A* route plus bounded feedback, consuming the task's oracle observation/map."""

import numpy as np


class RouteController:
    def __init__(self, task, drive_parameters):
        self.task, self.drive = task, drive_parameters
        self.route = None
        self.index = 0

    def reset(self, grid, observation):
        self.route = grid.route(
            observation["achieved_goal"][:2], observation["desired_goal"][:2]
        )
        self.index = min(1, len(self.route) - 1)

    def action(self, observation):
        pos = observation["achieved_goal"][:2]
        vel = observation["observation"][2:4]
        if self.route is None:
            raise RuntimeError("Reset the controller with a route first")
        error = self.route[self.index] - pos
        if (
            self.index < len(self.route) - 1
            and np.linalg.norm(error) < self.task.waypoint_tolerance
            and np.linalg.norm(vel) < 0.06
        ):
            self.index += 1
            error = self.route[self.index] - pos
        target = (
            self.task.baseline_gain * error - self.task.baseline_velocity_damping * vel
        )
        if self.index == len(self.route) - 1 and np.linalg.norm(error) < 0.035:
            target = np.zeros(2)
        norm = float(np.linalg.norm(target))
        speed = min(norm, self.task.baseline_max_speed, self.drive.max_speed)
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float32)
        # Invert the drive adapter's normalized dead zone and heading calibration.
        magnitude = (
            self.drive.dead_zone
            + (1 - self.drive.dead_zone) * speed / self.drive.max_speed
        )
        angle = -self.drive.heading_offset
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return (rotation @ target / norm * magnitude).astype(np.float32)
