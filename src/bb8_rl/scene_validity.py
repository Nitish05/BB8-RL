"""Bounded RGB change detection for an already registered, fixed camera scene.

This guard compares the current session with its stationary startup images. It
does not certify that those images match a saved map, repair maps, recalibrate a
camera, infer hidden obstacles, or establish a physical collision guarantee.
Only RGB frames, declared calibrations and fresh RGB-derived robot measurements
enter the detector. A suspected change denies navigation immediately; repeated
evidence latches invalidation until an explicit, stopped original-reference recheck.
"""

import copy
import hashlib
import math
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .camera import Calibration
from .camera_rig import CameraFrame


@dataclass(frozen=True)
class SceneValidityParameters:
    """Declared engineering thresholds, not learned or held-out calibrated bounds."""

    sample_seconds: float = 0.5
    max_frame_age: float = 0.10
    warmup_samples: int = 3
    confirmation_samples: int = 2
    recheck_samples: int = 3
    recheck_min_seconds: float = 1.0
    recheck_timeout_seconds: float = 10.0
    analysis_width: int = 320
    analysis_height: int = 240
    max_features: int = 128
    minimum_features: int = 12
    camera_motion_pixels: float = 2.5
    camera_inlier_fraction: float = 0.65
    pixel_difference: float = 24.0
    component_fraction: float = 0.0015
    minimum_component_pixels: int = 24
    minimum_contrast: float = 8.0
    maximum_mask_fraction: float = 0.15
    robot_radius_m: float = 0.10
    robot_height_m: float = 0.16
    robot_pixel_padding: int = 4

    def __post_init__(self):
        integers = {
            "warmup_samples": (2, 10),
            "confirmation_samples": (2, 10),
            "recheck_samples": (3, 10),
            "analysis_width": (64, 640),
            "analysis_height": (48, 480),
            "max_features": (16, 256),
            "minimum_features": (8, 64),
            "minimum_component_pixels": (4, 1024),
            "robot_pixel_padding": (0, 32),
        }
        for name, (low, high) in integers.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid scene validity parameter: {name}")
        for name in set(asdict(self)) - integers.keys():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid scene validity parameter: {name}")
        if (
            self.minimum_features > self.max_features
            or self.camera_inlier_fraction > 1
            or self.component_fraction > 1
            or self.maximum_mask_fraction >= 0.5
            or self.recheck_timeout_seconds <= self.recheck_min_seconds
        ):
            raise ValueError("Invalid scene validity fractions or feature bounds")


class SceneValidityGuard:
    """One to three fixed views; no automatic reset or reference replacement.

    ``observe(frames, views, now=...)`` accepts the same CameraFrame objects and
    per-view observations used by CameraRig. ``navigation_allowed`` is false
    while establishing a reference, on the first suspicion, and after any latch.
    The caller owns Stop, braking and revocation: this module issues no commands.
    """

    def __init__(self, calibrations, calibration_versions, *, parameters=None):
        self.parameters = (
            SceneValidityParameters() if parameters is None else parameters
        )
        if not isinstance(self.parameters, SceneValidityParameters):
            raise TypeError("Expected SceneValidityParameters")
        if (
            not isinstance(calibrations, dict)
            or not isinstance(calibration_versions, dict)
            or not 1 <= len(calibrations) <= 3
            or set(calibrations) != set(calibration_versions)
        ):
            raise ValueError("Need one to three calibrated, versioned scene views")
        self.calibrations, self.versions = {}, dict(calibration_versions)
        for name, calibration in calibrations.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(self.versions[name], str)
                or not self.versions[name]
                or not isinstance(calibration, Calibration)
            ):
                raise ValueError("Invalid scene camera identity or calibration")
            self.calibrations[name] = replace(
                calibration,
                intrinsics=np.array(calibration.intrinsics, copy=True),
                world_to_camera=np.array(calibration.world_to_camera, copy=True),
            )
        self._reference = {}
        self._last_capture = {}
        self._reports = {}
        self._strikes = {name: 0 for name in calibrations}
        self._next_check = 0.0
        self._timestamp = self._checked_at = self._invalidated_at = None
        self._reason = self._invalid_camera = None
        self._checks = 0
        self._ready = False
        self._invalidated = False
        self._suspected = False
        self._fault_epoch = 0
        self._recovery = None
        self._last_recheck_generation = -1
        self._recovery_first_stable = None

    def snapshot(self):
        return {
            "schema": "bb8.scene-validity.v1",
            "timestamp": self._timestamp,
            "observed_at": self._timestamp,
            "checked_at": self._checked_at,
            "capture_times": dict(self._last_capture),
            "ready": self._ready,
            "invalidated": self._invalidated,
            "invalidated_at": self._invalidated_at,
            "fault_epoch": self._fault_epoch,
            "recovery": copy.deepcopy(self._recovery),
            "navigation_allowed": self._ready
            and not self._invalidated
            and not self._suspected,
            "status": "invalidated"
            if self._invalidated
            else "suspected_change"
            if self._suspected
            else "stable"
            if self._ready
            else "warming",
            "reason": self._reason,
            "camera_id": self._invalid_camera,
            "checks": self._checks,
            "parameters": asdict(self.parameters),
            "cameras": copy.deepcopy(self._reports),
            "scope": "Same-session RGB change detector; startup is not saved-map validation",
            "limits": (
                "Only visible changes above declared image thresholds are detectable. "
                "Robot/reference masks exclude their pixels; hidden or visually "
                "indistinguishable changes can remain undetected. Strong local lighting "
                "changes may conservatively invalidate. No map repair or relocalization."
            ),
        }

    def _invalidate(self, reason, camera=None):
        if not self._invalidated:
            self._fault_epoch += 1
        self._invalidated = True
        self._invalidated_at = self._timestamp
        self._reason, self._invalid_camera = reason, camera
        if self._recovery and self._recovery["status"] == "checking":
            self._recovery.update(
                status="rejected", completed_at=self._timestamp, reason=reason
            )
        return self.snapshot()

    def request_recheck(self, generation, *, now):
        """Recheck the immutable original reference, never adopt a changed scene.

        Authority and stopped acknowledgements belong to the caller. A newer
        request may repeat a successful attempt whose acknowledgement was lost
        or superseded by Stop before the supervisor accepted it.
        """
        if (
            type(generation) is not int
            or generation < 0
            or generation <= self._last_recheck_generation
            or isinstance(now, bool)
            or not isinstance(now, (float, int))
            or not math.isfinite(now)
            or now < 0
            or (self._timestamp is not None and now < self._timestamp)
            or not self._ready
            or set(self._reference) != set(self.calibrations)
            or not self._fault_epoch
            or not (
                self._invalidated
                or (self._recovery and self._recovery["status"] == "succeeded")
            )
        ):
            raise ValueError(
                "Recheck requires a fresh request and original ready reference"
            )
        self._last_recheck_generation = generation
        self._recovery = {
            "generation": generation,
            "fault_epoch": self._fault_epoch,
            "status": "checking",
            "requested_at": float(now),
            "completed_at": None,
            "stable_checks": 0,
            "reference_sha256": {
                name: hashlib.sha256(reference["gray"].tobytes()).hexdigest()
                for name, reference in self._reference.items()
            },
        }
        self._timestamp = float(now)
        self._invalidated = True  # A retry is the same fault, not a new epoch.
        self._suspected = False
        self._reason = "explicit_original_reference_recheck"
        self._recovery_first_stable = None
        self._next_check = float(now)
        return self.snapshot()

    def cancel_recheck(self, reason):
        if self._recovery and self._recovery["status"] == "checking":
            self._recovery.update(
                status="cancelled", completed_at=self._timestamp, reason=str(reason)
            )
            self._invalidated = True
            self._reason = str(reason)
        return self.snapshot()

    def _frames(self, frames, views, now):
        if not isinstance(frames, (tuple, list)) or not isinstance(views, dict):
            raise TypeError("Missing scene frames or RGB observations")
        by_id = {
            frame.camera_id: frame for frame in frames if isinstance(frame, CameraFrame)
        }
        if (
            len(by_id) != len(frames)
            or set(by_id) != set(self.calibrations)
            or set(views) != set(self.calibrations)
        ):
            raise ValueError("Missing, duplicate or unexpected scene camera")
        poses = []
        for name, frame in by_id.items():
            calibration = self.calibrations[name]
            width, height = calibration.resolution
            view = views[name]
            if (
                frame.rgb.shape != (height, width, 3)
                or frame.rgb.dtype != np.uint8
                or frame.calibration_version != self.versions[name]
                or getattr(view, "camera_id", None) != name
                or getattr(view, "calibration_version", None) != self.versions[name]
                or getattr(view, "capture_time", None) != frame.capture_time
                or not -1e-9
                <= now - frame.capture_time
                <= self.parameters.max_frame_age + 1e-9
                or frame.capture_time <= self._last_capture.get(name, -math.inf)
            ):
                raise ValueError(f"Invalid, stale or repeated scene evidence: {name}")
            measurement = getattr(view, "measurement", None)
            if getattr(view, "status", None) not in ("visible", "missing", "ambiguous"):
                raise ValueError(f"Rejected scene observer evidence: {name}")
            if (
                measurement is None
                or measurement.timestamp != frame.capture_time
                or measurement.status != view.status
            ):
                raise ValueError(f"Missing current scene observation: {name}")
            if measurement.status == "visible":
                xy = np.asarray(measurement.xy, dtype=float)
                covariance = np.asarray(measurement.covariance, dtype=float)
                if (
                    xy.shape != (2,)
                    or covariance.shape != (2, 2)
                    or not np.isfinite(xy).all()
                    or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T)
                    or np.linalg.eigvalsh(covariance).min() < 0
                ):
                    raise ValueError(f"Invalid RGB robot estimate: {name}")
                poses.append((xy, 3 * math.sqrt(np.linalg.eigvalsh(covariance).max())))
            elif (
                measurement.status not in ("missing", "ambiguous")
                or measurement.xy is not None
                or measurement.covariance is not None
            ):
                raise ValueError(f"Robot estimate is not fresh RGB evidence: {name}")
        return by_id, poses

    def _image(self, name, frame, poses):
        calibration, p = self.calibrations[name], self.parameters
        width, height = calibration.resolution
        scale = min(1.0, p.analysis_width / width, p.analysis_height / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.cvtColor(
            cv2.resize(frame.rgb, size, interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY,
        )
        mask = np.zeros(gray.shape, np.uint8)
        angles = np.arange(16) * (2 * math.pi / 16)
        ring = np.c_[np.cos(angles), np.sin(angles)]
        for xy, uncertainty in poses:
            boundary = xy + ring * (p.robot_radius_m + uncertainty)
            pixels = np.concatenate(
                [
                    calibration.to_pixel(boundary, height=z)
                    for z in (0.0, p.robot_height_m)
                ]
            )
            if not np.isfinite(pixels).all() or np.max(abs(pixels)) > 1e7:
                raise ValueError("Unusable projected RGB robot exclusion")
            hull = cv2.convexHull(np.rint(pixels * scale).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1)
        padding = max(1, round(p.robot_pixel_padding * scale))
        if poses:
            mask = cv2.dilate(mask, np.ones((2 * padding + 1,) * 2, np.uint8))
        return gray, mask.astype(bool)

    def _normalized(self, reference, current, static):
        ref = reference[static].astype(float)
        cur = current[static].astype(float)
        ref_low, ref_high = np.percentile(ref, [10, 90])
        cur_low, cur_high = np.percentile(cur, [10, 90])
        if (
            min(ref_high - ref_low, cur_high - cur_low)
            < self.parameters.minimum_contrast
        ):
            raise ValueError("Insufficient static image contrast")
        gain = (ref_high - ref_low) / (cur_high - cur_low)
        corrected = (current.astype(float) - np.median(cur)) * gain + np.median(ref)
        return np.clip(corrected, 0, 255).astype(np.uint8), float(gain)

    def _compare(self, reference, current, static):
        p = self.parameters
        corrected, gain = self._normalized(reference, current, static)
        mask = static.astype(np.uint8) * 255
        points = cv2.goodFeaturesToTrack(reference, p.max_features, 0.02, 5, mask=mask)
        count = 0 if points is None else len(points)
        if count < p.minimum_features:
            raise ValueError("Insufficient static reference features")
        tracked, valid, _ = cv2.calcOpticalFlowPyrLK(
            reference, corrected, points, None, winSize=(21, 21), maxLevel=2
        )
        back, reverse_valid, _ = cv2.calcOpticalFlowPyrLK(
            corrected, reference, tracked, None, winSize=(21, 21), maxLevel=2
        )
        starts, ends = points[:, 0], tracked[:, 0]
        height, width = reference.shape
        inside = (
            (ends[:, 0] >= 0)
            & (ends[:, 0] < width)
            & (ends[:, 1] >= 0)
            & (ends[:, 1] < height)
        )
        safe_ends = np.clip(np.rint(ends).astype(int), [0, 0], [width - 1, height - 1])
        usable = valid[:, 0].astype(bool) & reverse_valid[:, 0].astype(bool) & inside
        usable &= np.linalg.norm(back[:, 0] - starts, axis=1) <= 1.5
        usable &= static[safe_ends[:, 1], safe_ends[:, 0]]
        starts, ends = starts[usable], ends[usable]
        motion, inlier_fraction, regions = 0.0, 0.0, 0
        if len(starts) >= p.minimum_features:
            transform, inliers = cv2.estimateAffinePartial2D(
                starts,
                ends,
                method=cv2.RANSAC,
                ransacReprojThreshold=1.5,
                maxIters=500,
                confidence=0.99,
                refineIters=5,
            )
            if transform is not None and inliers is not None:
                supported = starts[inliers[:, 0].astype(bool)]
                inlier_fraction = float(np.mean(inliers))
                regions = len(
                    {(int(x >= width / 2), int(y >= height / 2)) for x, y in supported}
                )
                corners = np.array(
                    [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
                )
                moved = np.c_[corners, np.ones(4)] @ transform.T
                motion = float(np.median(np.linalg.norm(moved - corners, axis=1)))
        residual = cv2.absdiff(
            cv2.GaussianBlur(reference, (3, 3), 0),
            cv2.GaussianBlur(corrected, (3, 3), 0),
        )
        changed = ((residual > p.pixel_difference) & static).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(changed)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
        threshold = max(
            p.minimum_component_pixels,
            math.ceil(p.component_fraction * int(static.sum())),
        )
        camera_change = (
            regions >= 3
            and inlier_fraction >= p.camera_inlier_fraction
            and motion >= p.camera_motion_pixels
        )
        reason = (
            "camera_or_global_scene_change"
            if camera_change
            else "scene_content_changed"
            if largest >= threshold
            else None
        )
        return {
            "reason": reason,
            "camera_motion_pixels": motion,
            "feature_count": count,
            "tracked_features": len(starts),
            "camera_inlier_fraction": inlier_fraction,
            "camera_support_regions": regions,
            "largest_changed_component": largest,
            "component_threshold": threshold,
            "changed_pixels": int(changed.sum()),
            "photometric_gain": gain,
        }

    def observe(self, frames, views, *, now):
        """Validate each capture; compare structure at a bounded simulation cadence."""
        if (
            isinstance(now, bool)
            or not isinstance(now, (float, int))
            or not math.isfinite(now)
            or now < 0
        ):
            self._timestamp = None
            return self._invalidate("Invalid scene observation timestamp")
        self._timestamp = float(now)
        rechecking = self._recovery and self._recovery["status"] == "checking"
        if self._invalidated and not rechecking:
            return self.snapshot()
        try:
            if (
                rechecking
                and now - self._recovery["requested_at"]
                > self.parameters.recheck_timeout_seconds
            ):
                raise ValueError("Original-reference recheck timed out")
            by_id, poses = self._frames(frames, views, now)
            self._last_capture = {
                name: frame.capture_time for name, frame in by_id.items()
            }
            if rechecking and any(
                frame.capture_time <= self._recovery["requested_at"]
                for frame in by_id.values()
            ):
                return self.snapshot()  # The request's own capture is not new evidence.
            if now + 1e-9 < self._next_check:
                return self.snapshot()
            self._next_check = now + self.parameters.sample_seconds
            self._checked_at = float(now)
            self._checks += 1
            self._suspected = False
            self._reason = self._invalid_camera = None
            for name, frame in by_id.items():
                gray, robot = self._image(name, frame, poses)
                if name not in self._reference:
                    self._reference[name] = {
                        "gray": gray.copy(),
                        "mask": robot.copy(),
                        "samples": 0,
                        "timestamp": frame.capture_time,
                    }
                reference = self._reference[name]
                static = ~(reference["mask"] | robot)
                masked_fraction = float(1 - static.mean())
                if masked_fraction > self.parameters.maximum_mask_fraction:
                    raise ValueError(f"Insufficient unmasked static coverage: {name}")
                report = self._compare(reference["gray"], gray, static)
                report.update(
                    masked_fraction=masked_fraction,
                    reference_timestamp=reference["timestamp"],
                    reference_sha256=hashlib.sha256(
                        reference["gray"].tobytes()
                    ).hexdigest(),
                    analysis_resolution=[gray.shape[1], gray.shape[0]],
                    robot_estimates=len(poses),
                )
                self._reports[name] = report
                if report["reason"]:
                    if rechecking:
                        return self._invalidate(report["reason"], name)
                    self._strikes[name] += 1
                    self._suspected = True
                    self._reason, self._invalid_camera = report["reason"], name
                else:
                    self._strikes[name] = 0
                    if not self._ready:
                        reference["mask"] |= robot
                        reference["samples"] += 1
                report["consecutive_changes"] = self._strikes[name]
                report["reference_samples"] = reference["samples"]
                if self._strikes[name] >= self.parameters.confirmation_samples:
                    return self._invalidate(report["reason"], name)
            self._ready = self._ready or all(
                reference["samples"] >= self.parameters.warmup_samples
                for reference in self._reference.values()
            )
            if rechecking:
                if self._recovery_first_stable is None:
                    self._recovery_first_stable = float(now)
                self._recovery["stable_checks"] += 1
                if (
                    self._recovery["stable_checks"] >= self.parameters.recheck_samples
                    and now - self._recovery_first_stable + 1e-9
                    >= self.parameters.recheck_min_seconds
                ):
                    self._recovery.update(status="succeeded", completed_at=float(now))
                    self._invalidated = False
                    self._reason = self._invalid_camera = None
        except (
            ValueError,
            TypeError,
            AttributeError,
            cv2.error,
            np.linalg.LinAlgError,
        ) as error:
            return self._invalidate(f"Scene evidence unavailable: {error}")
        return self.snapshot()
