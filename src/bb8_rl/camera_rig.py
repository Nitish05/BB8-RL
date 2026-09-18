"""Up to three calibrated RGB views with conservative observed-position fusion.

This module neither fuses masks nor predicts a hidden robot position. A fresh
single-camera detection can take over when another view is missing or invalid.
Fresh ambiguity or incompatible detections stop fusion instead of choosing an
identity. Covariance intersection handles unknown inter-camera correlation,
conditional on each supplied covariance being a valid error bound; the current
visual observer's heuristic covariance does not establish that calibration.
"""

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from .camera import Calibration, VisualMeasurement


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _same_calibration(first, second):
    return (
        isinstance(first, Calibration)
        and tuple(first.resolution) == tuple(second.resolution)
        and first.extent == second.extent
        and first.head_height == second.head_height
        and np.allclose(first.intrinsics, second.intrinsics, rtol=0, atol=1e-10)
        and np.allclose(
            first.world_to_camera, second.world_to_camera, rtol=0, atol=1e-10
        )
    )


def calibration_from_live_camera(
    camera, extent, *, head_height=0.083, provenance="synthetic_exact_live_transform"
):
    """Read a Genesis camera's current OpenGL camera-to-world transform.

    Genesis's cached ``extrinsics`` property can retain an earlier camera pose.
    Convert the live transform to OpenCV axes and invert it instead. This helper
    reads geometry only, renders nothing and makes no simulator/hardware import.
    A caller must issue a new calibration version whenever a fixed rig changes.
    """
    camera_to_world = np.array(camera.transform, dtype=float, copy=True)
    if (
        camera_to_world.shape != (4, 4)
        or not np.isfinite(camera_to_world).all()
        or not np.allclose(camera_to_world[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
    ):
        raise ValueError("Invalid live camera transform")
    camera_to_world[:3, 1:3] *= -1
    return Calibration(
        np.array(camera.intrinsics, dtype=float, copy=True),
        np.linalg.inv(camera_to_world),
        tuple(camera.res),
        float(extent),
        head_height=head_height,
        provenance=provenance,
    )


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    calibration_version: str
    rgb: np.ndarray
    capture_time: float

    def __post_init__(self):
        _identifier(self.camera_id, "Camera ID")
        _identifier(self.calibration_version, "Calibration version")
        rgb = np.asarray(self.rgb)
        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 1
            or rgb.dtype != np.uint8
        ):
            raise ValueError("Camera frame must contain uint8 RGB pixels")
        if not np.isfinite(self.capture_time) or self.capture_time < 0:
            raise ValueError("Capture time must be finite and nonnegative")
        # A retained frame must not change when a renderer reuses its buffer.
        frozen_rgb = np.array(rgb, copy=True, order="C")
        frozen_rgb.setflags(write=False)
        object.__setattr__(self, "rgb", frozen_rgb)


@dataclass(frozen=True)
class ViewObservation:
    camera_id: str
    calibration_version: str
    capture_time: float | None
    measurement: VisualMeasurement | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class RigObservation:
    timestamp: float
    xy: np.ndarray | None
    covariance: np.ndarray | None
    status: str
    source_ids: tuple[str, ...]
    views: dict[str, ViewObservation]
    handover: bool = False


class CameraView:
    """Bind one observer to one immutable calibration-version contract."""

    def __init__(self, camera_id, calibration_version, calibration, observer):
        _identifier(camera_id, "Camera ID")
        _identifier(calibration_version, "Calibration version")
        if not isinstance(calibration, Calibration) or not callable(
            getattr(observer, "observe", None)
        ):
            raise TypeError("A camera view needs calibration and an RGB observer")
        bound = getattr(observer, "calibration", calibration)
        if not _same_calibration(bound, calibration):
            raise ValueError("Observer calibration differs from registered camera")
        self.camera_id = camera_id
        self.calibration_version = calibration_version
        intrinsics = np.array(calibration.intrinsics, dtype=float, copy=True)
        transform = np.array(calibration.world_to_camera, dtype=float, copy=True)
        intrinsics.setflags(write=False)
        transform.setflags(write=False)
        self.calibration = replace(
            calibration, intrinsics=intrinsics, world_to_camera=transform
        )
        self.observer = observer
        self.last_capture_time = None

    def observe(self, frame, *, now, max_age=0.10):
        if not np.isfinite(now) or now < 0 or not np.isfinite(max_age) or max_age < 0:
            raise ValueError("Invalid observation time or maximum age")

        def rejected(status, reason=None):
            return ViewObservation(
                self.camera_id,
                self.calibration_version,
                frame.capture_time,
                None,
                status,
                reason,
            )

        if frame.camera_id != self.camera_id:
            return rejected("wrong_camera")
        if frame.calibration_version != self.calibration_version:
            return rejected("calibration_mismatch")
        if not _same_calibration(
            getattr(self.observer, "calibration", self.calibration), self.calibration
        ):
            return rejected("calibration_mismatch", "Observer calibration changed")
        width, height = self.calibration.resolution
        if frame.rgb.shape != (height, width, 3):
            return rejected(
                "invalid_frame", "Frame resolution differs from calibration"
            )
        if frame.capture_time > now + 1e-9:
            return rejected("future_frame")
        if now - frame.capture_time > max_age + 1e-9:
            return rejected("stale_frame")
        if (
            self.last_capture_time is not None
            and frame.capture_time <= self.last_capture_time
        ):
            return rejected("repeated_frame")
        self.last_capture_time = frame.capture_time
        try:
            measurement = self.observer.observe(frame.rgb, frame.capture_time)
        except (ValueError, np.linalg.LinAlgError) as error:
            return rejected("invalid_measurement", str(error))
        if not isinstance(measurement, VisualMeasurement):
            return rejected(
                "invalid_measurement", "Observer must return VisualMeasurement"
            )
        try:
            valid_metadata = (
                np.isfinite(measurement.timestamp)
                and abs(measurement.timestamp - frame.capture_time) <= 1e-9
                and np.asarray(measurement.visible_floor).shape == (height, width)
                and np.asarray(measurement.robot_pixels).shape == (height, width)
                and np.asarray(measurement.visible_floor).dtype == np.bool_
                and np.asarray(measurement.robot_pixels).dtype == np.bool_
            )
        except (TypeError, ValueError):
            valid_metadata = False
        if not valid_metadata:
            return rejected(
                "invalid_measurement", "Invalid measurement timestamp or masks"
            )
        if measurement.status not in ("visible", "missing", "ambiguous"):
            return rejected(
                "unobserved", "Predicted positions are not RGB observations"
            )
        if measurement.status != "visible":
            if measurement.xy is not None or measurement.covariance is not None:
                return rejected(
                    "invalid_measurement", "Unobserved measurement contains a pose"
                )
        else:
            try:
                xy = np.asarray(measurement.xy, dtype=float)
                covariance = np.asarray(measurement.covariance, dtype=float)
            except (TypeError, ValueError):
                return rejected(
                    "invalid_measurement", "Non-numeric position or covariance"
                )
            if (
                xy.shape != (2,)
                or covariance.shape != (2, 2)
                or not np.isfinite(xy).all()
                or not np.isfinite(covariance).all()
                or np.max(abs(xy)) >= self.calibration.extent
                or not np.allclose(covariance, covariance.T, rtol=1e-6, atol=1e-12)
                or np.linalg.eigvalsh(covariance).min() <= 0
            ):
                return rejected(
                    "invalid_measurement", "Invalid measured position or covariance"
                )
        return ViewObservation(
            self.camera_id,
            self.calibration_version,
            frame.capture_time,
            measurement,
            measurement.status,
        )


def _inflate_for_skew(covariance, displacement_bound):
    """Bound error plus unobserved motion without assuming independence.

    Young's inequality gives (1+eta) P + (1+1/eta) d² I for bounded displacement
    d and unknown correlation. Isotropic input becomes (sigma+d)² I.
    """
    covariance = np.asarray(covariance, dtype=float)
    if displacement_bound == 0:
        return covariance.copy()
    eta = displacement_bound / np.sqrt(np.linalg.eigvalsh(covariance).max())
    return (1 + eta) * covariance + (1 + 1 / eta) * displacement_bound**2 * np.eye(2)


class CameraRig:
    """Fresh RGB-only fusion for one to three registered views.

    Accepted measurements refer to the latest capture time. Older measurements
    inside max_skew retain their measured mean and acquire a bounded-motion
    covariance inflation. Views outside that window are excluded, permitting a
    fresh remaining camera to take over. Missing output has no position and does
    not preserve a prior measurement as a new observation. A capture at or before
    the last returned timestamp is never fused as a new temporal update.

    Equal-weight covariance intersection does not divide identical uncertainty
    by the number of cameras. Pairwise spatial/innovation gates are engineering
    identity-consistency checks, not calibrated identity probabilities. The last
    accepted position is retained only for the same consistency gate across
    time and handover; it is never returned as a fresh observed position.
    """

    def __init__(
        self,
        views,
        *,
        max_age=0.10,
        max_skew=0.05,
        velocity_bound=0.35,
        identity_gate=16.0,
        max_disagreement=0.10,
    ):
        views = list(views)
        if not 1 <= len(views) <= 3 or len({view.camera_id for view in views}) != len(
            views
        ):
            raise ValueError("A camera rig requires one to three unique views")
        if (
            not np.isfinite(
                [max_age, max_skew, velocity_bound, identity_gate, max_disagreement]
            ).all()
            or not 0 <= max_skew <= max_age
            or max_age <= 0
            or velocity_bound < 0
            or identity_gate <= 0
            or max_disagreement <= 0
        ):
            raise ValueError("Invalid camera rig fusion limits")
        extents = {view.calibration.extent for view in views}
        heights = {view.calibration.head_height for view in views}
        if len(extents) != 1 or len(heights) != 1:
            raise ValueError("Rig views must share room extent and robot head height")
        self.views = {view.camera_id: view for view in views}
        self.max_age, self.max_skew = float(max_age), float(max_skew)
        self.velocity_bound = float(velocity_bound)
        self.identity_gate, self.max_disagreement = (
            float(identity_gate),
            float(max_disagreement),
        )
        self.previous_source_ids = ()
        self.last_timestamp = None
        self.last_measurement = None
        self.last_call_time = None

    def observe(self, frames, *, now):
        frames = list(frames)
        if not np.isfinite(now) or now < 0:
            raise ValueError("Rig time must be finite and nonnegative")
        if self.last_call_time is not None and now < self.last_call_time:
            raise ValueError("Rig time must not move backwards")
        ids = [frame.camera_id for frame in frames]
        if len(frames) > 3 or len(set(ids)) != len(ids):
            raise ValueError("A capture must contain at most three unique views")
        if any(camera_id not in self.views for camera_id in ids):
            raise ValueError("Capture contains an unregistered camera")
        self.last_call_time = float(now)
        by_id = {frame.camera_id: frame for frame in frames}
        observations = {
            camera_id: view.observe(by_id[camera_id], now=now, max_age=self.max_age)
            if camera_id in by_id
            else ViewObservation(
                camera_id, view.calibration_version, None, None, "missing_frame"
            )
            for camera_id, view in self.views.items()
        }

        def unavailable(status):
            self.last_timestamp = max(now, self.last_timestamp or 0.0)
            return RigObservation(float(now), None, None, status, (), observations)

        fresh = [
            o
            for o in observations.values()
            if o.status in ("visible", "missing", "ambiguous")
        ]
        if not fresh:
            return unavailable("no_fresh_measurement")
        latest = max(o.capture_time for o in fresh)
        for observation in fresh:
            if latest - observation.capture_time > self.max_skew + 1e-9:
                observations[observation.camera_id] = replace(
                    observation,
                    status="skewed_frame",
                    reason="Outside capture skew window",
                )
        if any(o.status == "ambiguous" for o in observations.values()):
            return unavailable("ambiguous")
        accepted = sorted(
            (o for o in observations.values() if o.status == "visible"),
            key=lambda observation: observation.camera_id,
        )
        if not accepted:
            return unavailable("missing")
        timestamp = max(o.capture_time for o in accepted)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return unavailable("out_of_order")
        means = [np.array(o.measurement.xy, dtype=float) for o in accepted]
        covariances = [
            _inflate_for_skew(
                o.measurement.covariance,
                self.velocity_bound * (timestamp - o.capture_time),
            )
            for o in accepted
        ]
        for a, b in combinations(range(len(accepted)), 2):
            residual = means[a] - means[b]
            if (
                np.linalg.norm(residual) > self.max_disagreement
                or residual @ np.linalg.solve(covariances[a] + covariances[b], residual)
                > self.identity_gate
            ):
                return unavailable("inconsistent_views")
        try:
            information = [np.linalg.inv(covariance) for covariance in covariances]
            covariance = np.linalg.inv(sum(information) / len(information))
            xy = (
                covariance
                @ sum(p @ mean for p, mean in zip(information, means))
                / len(means)
            )
        except np.linalg.LinAlgError:
            return unavailable("invalid_fusion")
        if not np.isfinite(xy).all() or not np.isfinite(covariance).all():
            return unavailable("invalid_fusion")
        if self.last_measurement is not None:
            previous_time, previous_xy, previous_covariance = self.last_measurement
            displacement_bound = self.velocity_bound * (timestamp - previous_time)
            previous_covariance = _inflate_for_skew(
                previous_covariance, displacement_bound
            )
            residual = xy - previous_xy
            if (
                np.linalg.norm(residual) > self.max_disagreement + displacement_bound
                or residual
                @ np.linalg.solve(previous_covariance + covariance, residual)
                > self.identity_gate
            ):
                return unavailable("inconsistent_history")
        source_ids = tuple(o.camera_id for o in accepted)
        handover = bool(
            self.previous_source_ids and source_ids != self.previous_source_ids
        )
        self.previous_source_ids = source_ids
        self.last_timestamp = timestamp
        self.last_measurement = (timestamp, xy.copy(), covariance.copy())
        return RigObservation(
            float(timestamp),
            xy,
            covariance,
            "visible",
            source_ids,
            observations,
            handover,
        )
