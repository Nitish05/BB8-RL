"""Camera-only navigation contracts; no simulator or hardware dependency.

The initial segmenter is explicitly specific to the synthetic room palette.
Foundation-model/student segmenters can replace it without changing control.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .planner import OccupancyGrid


@dataclass(frozen=True)
class Calibration:
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    resolution: tuple[int, int]
    extent: float
    head_height: float = 0.083
    provenance: str = "synthetic_exact"

    def __post_init__(self):
        k, t = np.asarray(self.intrinsics), np.asarray(self.world_to_camera)
        if (
            k.shape != (3, 3)
            or t.shape != (4, 4)
            or not np.isfinite(k).all()
            or not np.isfinite(t).all()
            or abs(np.linalg.det(k)) < 1e-10
            or len(self.resolution) != 2
            or min(self.resolution) < 1
            or not np.isfinite([self.extent, self.head_height]).all()
            or self.extent <= 0
            or self.head_height <= 0
        ):
            raise ValueError("Invalid pinhole calibration")
        if not np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-5):
            raise ValueError("Calibration rotation must be orthonormal")

    def homography(self, height=0.0):
        t = np.asarray(self.world_to_camera)
        return (
            np.asarray(self.intrinsics) @ np.c_[t[:3, :2], t[:3, 3] + height * t[:3, 2]]
        )

    def to_pixel(self, xy, height=0.0):
        xy = np.asarray(xy, dtype=float)
        p = np.c_[xy.reshape(-1, 2), np.ones(xy.size // 2)] @ self.homography(height).T
        if not np.isfinite(p).all() or np.any(p[:, 2] <= 1e-8):
            raise ValueError("Point is behind the camera")
        return (p[:, :2] / p[:, 2:3]).reshape(xy.shape)

    def to_plane(self, pixels, height=0.0):
        pixels = np.asarray(pixels, dtype=float)
        if pixels.shape[-1:] != (2,) or not np.isfinite(pixels).all():
            raise ValueError("Pixels must have two finite coordinates")
        p = np.c_[pixels.reshape(-1, 2), np.ones(pixels.size // 2)]
        p = p @ np.linalg.inv(self.homography(height)).T
        if np.any(abs(p[:, 2]) < 1e-8):
            raise ValueError("Ray is parallel to the requested plane")
        result = (p[:, :2] / p[:, 2:3]).reshape(pixels.shape)
        self.to_pixel(result, height)  # Reject intersections behind the camera.
        return result

    def measurement_covariance(self, pixel, sigma_pixels=1.0):
        pixel = np.asarray(pixel, dtype=float)
        origin = self.to_plane(pixel, self.head_height)
        jacobian = np.column_stack(
            [self.to_plane(pixel + d, self.head_height) - origin for d in np.eye(2)]
        )
        return sigma_pixels**2 * jacobian @ jacobian.T + np.eye(2) * 0.002**2


@dataclass(frozen=True)
class VisualMeasurement:
    timestamp: float
    xy: np.ndarray | None
    covariance: np.ndarray | None
    visible_floor: np.ndarray
    robot_pixels: np.ndarray
    status: str


class SyntheticPaletteObserver:
    """RGB baseline for the current gray head and blue checkerboard only.

    This is neither SAM inference nor a trained perception model. Unknown colors
    are not called free floor. More than one plausible head is ambiguous.
    """

    def __init__(self, calibration):
        self.calibration = calibration

    def observe(self, rgb, timestamp):
        rgb = np.asarray(rgb)
        width, height = self.calibration.resolution
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Frame must be uint8 RGB at calibrated resolution")
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid capture timestamp")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        floor = (hue >= 102) & (hue <= 109) & (saturation >= 112)
        floor &= (saturation <= 190) & (value >= 15) & (value <= 120)
        heads = ((saturation < 45) & (value > 85)).astype(np.uint8)
        count, _labels, stats, centers = cv2.connectedComponentsWithStats(heads)
        candidates = []
        for i in range(1, count):
            area = stats[i, cv2.CC_STAT_AREA]
            if not 3 <= area <= 100:
                continue
            xy = self.calibration.to_plane(centers[i], self.calibration.head_height)
            if np.max(abs(xy)) < self.calibration.extent - 0.06:
                candidates.append((i, xy))
        robot = np.zeros(floor.shape, bool)
        if len(candidates) != 1:
            return VisualMeasurement(
                timestamp,
                None,
                None,
                floor,
                robot,
                "missing" if not candidates else "ambiguous",
            )
        label, xy = candidates[0]
        yy, xx = np.indices(floor.shape)
        # Group neutral body/head pixels around the detected head; never use an
        # entity ID, simulator segmentation, pose, velocity or depth buffer.
        nearby = (xx - centers[label, 0]) ** 2 + (yy - centers[label, 1]) ** 2 < 14**2
        robot = nearby & (saturation < 65) & (value > 15)
        return VisualMeasurement(
            timestamp,
            xy,
            self.calibration.measurement_covariance(centers[label]),
            floor,
            robot,
            "visible",
        )


class PositionBelief:
    """Causal constant-velocity Kalman filter with acceleration process noise."""

    def __init__(self):
        self.state = None
        self.covariance = None
        self.timestamp = None
        self.last_seen = None
        self.samples = 0
        self.status = "uninitialized"

    def update(self, measurement):
        now = measurement.timestamp
        if self.timestamp is not None and now <= self.timestamp:
            raise ValueError("Capture timestamps must increase strictly")
        if self.state is not None:
            dt = now - self.timestamp
            f = np.eye(4)
            f[:2, 2:] = np.eye(2) * dt
            g = np.vstack((np.eye(2) * dt**2 / 2, np.eye(2) * dt))
            self.state = f @ self.state
            self.covariance = f @ self.covariance @ f.T + 0.8**2 * g @ g.T
        self.timestamp, self.status = now, measurement.status
        if measurement.xy is None:
            return
        if self.state is None:
            self.state = np.r_[measurement.xy, 0.0, 0.0]
            self.covariance = np.zeros((4, 4))
            self.covariance[:2, :2] = measurement.covariance
            self.covariance[2:, 2:] = np.eye(2) * 0.1**2
        else:
            residual = measurement.xy - self.state[:2]
            innovation = self.covariance[:2, :2] + measurement.covariance
            if residual @ np.linalg.solve(innovation, residual) > 16:
                self.status = "outlier"
                return
            gain = np.linalg.solve(innovation, self.covariance[:2, :]).T
            self.state += gain @ residual
            h = np.c_[np.eye(2), np.zeros((2, 2))]
            correction = np.eye(4) - gain @ h
            self.covariance = (
                correction @ self.covariance @ correction.T
                + gain @ measurement.covariance @ gain.T
            )
        self.samples += 1
        self.last_seen = now

    @property
    def sigma(self):
        if self.covariance is None:
            return float("inf")
        return float(np.sqrt(np.linalg.eigvalsh(self.covariance[:2, :2]).max()))

    def ready(self, now):
        return (
            self.state is not None
            and self.samples >= 3
            and self.status == "visible"
            and self.last_seen is not None
            and -1e-9 <= now - self.last_seen <= 0.10 + 1e-9
            and self.sigma <= 0.025
        )


def camera_grid(
    calibration, measurement, *, resolution=0.05, inflation=0.10, method="legacy"
):
    """Visible-free-space map; silhouettes and occlusions stay blocked.

    The robot's neutral-pixel mask is treated as self occupancy, not an obstacle.
    This is a synthetic baseline assumption, not inferred hidden scene geometry.
    """
    grid = OccupancyGrid(calibration.extent, resolution, inflation, [])
    cells = np.indices((grid.n, grid.n)).reshape(2, -1).T
    points = -grid.extent + (cells[:, ::-1] + 0.5) * grid.resolution
    visible = measurement.visible_floor | measurement.robot_pixels
    if method == "metric":
        # Enclose the projected cell quadrilateral by native pixel bounds. Taking
        # every pixel in this superset prevents holes between sparse probes.
        corners = (
            points[:, None]
            + np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * resolution / 2
        )
        uv = calibration.to_pixel(corners)
        low = np.floor(uv.min(1)).astype(int)
        high = np.ceil(uv.max(1)).astype(int)
        width, height = calibration.resolution
        inside = (
            (low[:, 0] >= 0)
            & (low[:, 1] >= 0)
            & (high[:, 0] < width)
            & (high[:, 1] < height)
        )
        x0, y0 = np.clip(low[:, 0], 0, width - 1), np.clip(low[:, 1], 0, height - 1)
        x1, y1 = (
            np.clip(high[:, 0], 0, width - 1) + 1,
            np.clip(high[:, 1], 0, height - 1) + 1,
        )
        integral = cv2.integral((~visible).astype(np.uint8), sdepth=cv2.CV_32S)
        unknown = (
            integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        )
        blocked = (~(inside & (unknown == 0))).reshape(grid.n, grid.n).astype(np.uint8)
        # Exact minimum distance between two axis-aligned square cells. Inflate
        # entire cells, retaining the requested metric clearance without turning
        # a small uncertainty change into an extra full ceil-radius disc.
        radius = int(np.ceil(inflation / resolution)) + 1
        offsets = np.arange(-radius, radius + 1)
        dx, dy = np.meshgrid(offsets, offsets)
        distance = resolution * np.hypot(
            np.maximum(abs(dx) - 1, 0), np.maximum(abs(dy) - 1, 0)
        )
        kernel = (distance <= inflation + 1e-12).astype(np.uint8)
        grid.blocked |= cv2.dilate(blocked, kernel).astype(bool)
        return grid
    if method != "legacy":
        raise ValueError("Unknown camera map method")
    # Historical five-probe / rounded ellipse method, retained for comparison.
    free = np.ones(len(points), bool)
    width, height = calibration.resolution
    for offset in (
        np.array([[0, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]) * resolution / 2
    ):
        uv = np.rint(calibration.to_pixel(points + offset)).astype(int)
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        )
        free &= (
            inside
            & visible[np.clip(uv[:, 1], 0, height - 1), np.clip(uv[:, 0], 0, width - 1)]
        )
    blocked = (~free).reshape(grid.n, grid.n).astype(np.uint8)
    radius = int(np.ceil(inflation / resolution))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    grid.blocked |= cv2.dilate(blocked, kernel).astype(bool)
    return grid


class CameraController:
    """Input boundary: RGB + timestamp + calibrated goal pixel, never environment."""

    def __init__(
        self,
        calibration,
        policy,
        goal_pixel,
        observer=None,
        *,
        map_resolution=0.05,
        map_method="legacy",
    ):
        self.calibration, self.policy = calibration, policy
        # Validate with the same map geometry constraints used during planning.
        OccupancyGrid(calibration.extent, map_resolution, 0.1, [])
        self.map_resolution = map_resolution
        if map_method not in ("legacy", "metric", "continuous"):
            raise ValueError("Unknown camera map method")
        self.map_method = map_method
        self.observer = observer or SyntheticPaletteObserver(calibration)
        self.goal = calibration.to_plane(goal_pixel)
        if np.max(abs(self.goal)) >= calibration.extent:
            raise ValueError("Clicked goal is outside the calibrated room")
        self.belief = PositionBelief()
        self.grid = self.route = None
        self.clearance = None
        self.index = 0
        self.status = "warming_up"
        self.last_measurement = None
        self.map_sigma = 0.0
        self.last_route_error = None

    def action(self, rgb, timestamp, *, now=None):
        now = timestamp if now is None else now
        if (
            not np.isfinite(now)
            or timestamp > now + 1e-9
            or now - timestamp > 0.10 + 1e-9
        ):
            self.status = "stale_frame"
            return np.zeros(2, np.float32)
        if self.belief.timestamp is not None and timestamp <= self.belief.timestamp:
            self.status = "repeated_frame"
            return np.zeros(2, np.float32)
        measurement = self.observer.observe(rgb, timestamp)
        self.last_measurement = measurement
        self.belief.update(measurement)
        if not self.belief.ready(now):
            self.status = self.belief.status
            if self.status == "visible":
                self.status = "warming_up" if self.belief.samples < 3 else "uncertain"
            return np.zeros(2, np.float32)
        position, velocity = self.belief.state[:2], self.belief.state[2:]
        if self.belief.sigma > max(self.map_sigma * 1.25, self.map_sigma + 0.002):
            self.route = None
        if self.route is None:
            self.map_sigma = self.belief.sigma
            self.grid = camera_grid(
                self.calibration,
                measurement,
                resolution=self.map_resolution,
                method="metric" if self.map_method == "continuous" else self.map_method,
                inflation=0.037 + 0.04 + 3 * self.belief.sigma,
            )
            if self.map_method == "continuous":
                from .camera_clearance import CameraClearance

                self.clearance = CameraClearance(
                    self.calibration,
                    measurement.visible_floor | measurement.robot_pixels,
                    self.grid.inflation,
                )
            try:
                self.route = (
                    self.clearance.route(self.grid, position, self.goal)
                    if self.clearance is not None
                    else self.grid.route(position, self.goal)
                )
                self.index = min(1, len(self.route) - 1)
                self.last_route_error = None
            except ValueError as error:
                self.last_route_error = str(error)
                self.status = "no_visible_route"
                return np.zeros(2, np.float32)
        segment_free = (
            self.clearance.segment_free
            if self.clearance is not None
            else self.grid.segment_free
        )
        delta = self.route[self.index] - position
        if (
            self.index < len(self.route) - 1
            and np.linalg.norm(delta) < 0.045
            and np.linalg.norm(velocity) < 0.06
            and segment_free(position, self.route[self.index + 1])
        ):
            self.index += 1
            delta = self.route[self.index] - position
        if not segment_free(position, self.route[self.index]):
            self.route = None
            self.status = "route_invalid"
            return np.zeros(2, np.float32)
        vector = np.asarray(
            [*delta, *(velocity / 0.35), self.index == len(self.route) - 1], np.float32
        )
        action = np.asarray(self.policy(vector), np.float32)
        if (
            action.shape != (2,)
            or not np.isfinite(action).all()
            or np.max(abs(action)) > 1
        ):
            raise ValueError("Policy returned invalid drive action")
        self.status = "tracking"
        return action
