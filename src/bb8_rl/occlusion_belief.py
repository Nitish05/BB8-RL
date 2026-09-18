"""Causal motion-estimation baselines; predictions never authorize movement.

CV wraps the unchanged PositionBelief. CA uses a six-state linear Kalman model
with an assumed, discrete white-jerk prior. Neither estimates dynamics from
hidden truth or treats a prediction as a visual measurement.
"""

import numpy as np

from .camera import PositionBelief, VisualMeasurement


def observation_input(record, stream):
    """Select only capture time and recorded causal RGB observation fields."""
    if stream not in ("A", "B", "fused"):
        raise ValueError("Use A, B or fused observation stream")
    observed = record["rig"] if stream == "fused" else record["views"][stream]
    return {
        "timestamp": record["time"],
        "xy": observed["position"],
        "covariance": observed["covariance"],
        "status": observed["status"],
    }


class OcclusionBelief:
    """Return measured/predicted modes and transparent assumed uncertainty.

    State order is x/y/vx/vy[/ax/ay]. The reported position radius is three times
    the largest position standard deviation, with a nondecreasing floor during
    missing/outlier intervals. This is an assumed envelope, not calibrated
    coverage or an obstacle-clearance certificate. Covariance itself remains
    the Kalman covariance. Last-seen time changes only for accepted RGB data.
    """

    def __init__(self, model="cv", *, jerk_sigma=0.8):
        if model not in ("cv", "ca") or not np.isfinite(jerk_sigma) or jerk_sigma <= 0:
            raise ValueError("Need CV/CA and a positive finite jerk sigma")
        self.model, self.jerk_sigma = model, float(jerk_sigma)
        self._cv = PositionBelief() if model == "cv" else None
        self._state = self._covariance = self._timestamp = self._last_seen = None
        self._last_mode = "uninitialized"
        self._last_radius = 0.0

    @property
    def assumptions(self):
        return {
            "model": self.model,
            "process": "unchanged PositionBelief acceleration sigma0.8m/s² per interval"
            if self.model == "cv"
            else f"constant acceleration; discrete jerk sigma{self.jerk_sigma}m/s³ per interval",
            "initial_velocity_sigma_m_s": 0.1,
            "initial_acceleration_sigma_m_s2": 0.7 if self.model == "ca" else None,
            "innovation_mahalanobis_squared_gate": 16.0,
            "position_radius": "3*sqrt(max_eigenvalue(Pxy)); monotone floor while not measured",
            "noise_calibrated": False,
            "command_conditioned": False,
            "motion_authorized": False,
        }

    def _predict_ca(self, dt):
        transition = np.eye(6)
        transition[:2, 2:4] = np.eye(2) * dt
        transition[:2, 4:] = np.eye(2) * dt**2 / 2
        transition[2:4, 4:] = np.eye(2) * dt
        jerk = np.vstack((np.eye(2) * dt**3 / 6, np.eye(2) * dt**2 / 2, np.eye(2) * dt))
        self._state = transition @ self._state
        self._covariance = (
            transition @ self._covariance @ transition.T
            + self.jerk_sigma**2 * jerk @ jerk.T
        )

    def update(self, timestamp, *, xy=None, covariance=None, status="missing"):
        if (
            not np.isfinite(timestamp)
            or timestamp < 0
            or (self._timestamp is not None and timestamp <= self._timestamp)
        ):
            raise ValueError("Capture timestamps must increase strictly")
        if xy is None:
            if covariance is not None or status == "visible":
                raise ValueError(
                    "Unobserved input cannot contain covariance or claim visible"
                )
        else:
            xy, covariance = np.asarray(xy, float), np.asarray(covariance, float)
            if (
                status != "visible"
                or xy.shape != (2,)
                or covariance.shape != (2, 2)
                or not np.isfinite(xy).all()
                or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, rtol=1e-6, atol=1e-12)
                or np.linalg.eigvalsh(covariance).min() <= 0
            ):
                raise ValueError(
                    "Need a finite visible RGB position and positive covariance"
                )
        prior = None
        if self._state is not None:
            dt = timestamp - self._timestamp
            if self.model == "cv":
                prior = self._state[:2] + dt * self._state[2:4]
            else:
                self._predict_ca(dt)
                prior = self._state[:2].copy()
        residual = xy - prior if xy is not None and prior is not None else None
        returning = xy is not None and self._last_mode == "predicted"
        filter_status = status
        if self.model == "cv":
            empty = np.zeros((0, 0), dtype=bool)
            self._cv.update(
                VisualMeasurement(timestamp, xy, covariance, empty, empty, status)
            )
            self._state = self._cv.state
            self._covariance = self._cv.covariance
            self._last_seen = self._cv.last_seen
            filter_status = self._cv.status
        elif xy is not None:
            if self._state is None:
                self._state = np.r_[xy, 0.0, 0.0, 0.0, 0.0]
                self._covariance = np.diag([0.0, 0.0, 0.1**2, 0.1**2, 0.7**2, 0.7**2])
                self._covariance[:2, :2] = covariance
                self._last_seen = timestamp
            else:
                innovation = self._covariance[:2, :2] + covariance
                if residual @ np.linalg.solve(innovation, residual) > 16:
                    filter_status = "outlier"
                else:
                    gain = np.linalg.solve(innovation, self._covariance[:2, :]).T
                    self._state += gain @ residual
                    h = np.c_[np.eye(2), np.zeros((2, 4))]
                    correction = np.eye(6) - gain @ h
                    self._covariance = (
                        correction @ self._covariance @ correction.T
                        + gain @ covariance @ gain.T
                    )
                    self._last_seen = timestamp
        self._timestamp = float(timestamp)
        accepted = self._last_seen == timestamp
        mode = (
            "uninitialized"
            if self._state is None
            else "measured"
            if accepted
            else "predicted"
        )
        radius = None
        if self._state is not None:
            radius = 3 * float(
                np.sqrt(np.linalg.eigvalsh(self._covariance[:2, :2]).max())
            )
            if not accepted:
                radius = max(radius, self._last_radius)
            self._last_radius = radius
        result = {
            "time": float(timestamp),
            "model": self.model,
            "mode": mode,
            "input_status": status,
            "filter_status": filter_status,
            "measurement_accepted": accepted,
            "measured_xy": xy.tolist() if accepted else None,
            "position": self._state[:2].tolist() if self._state is not None else None,
            "velocity": self._state[2:4].tolist() if self._state is not None else None,
            "speed_m_s": float(np.linalg.norm(self._state[2:4]))
            if self._state is not None
            else None,
            "acceleration": self._state[4:].tolist()
            if self.model == "ca" and self._state is not None
            else None,
            "covariance": self._covariance.tolist()
            if self._covariance is not None
            else None,
            "assumed_position_radius_m": radius,
            "last_visual_time": self._last_seen,
            "time_since_visual_seconds": timestamp - self._last_seen
            if self._last_seen is not None
            else None,
            "pre_update_position": prior.tolist() if prior is not None else None,
            "reacquisition_residual": residual.tolist() if returning else None,
            "reacquisition_residual_m": float(np.linalg.norm(residual))
            if returning
            else None,
            "free_space_query": "unavailable",
            "motion_authorized": False,
        }
        self._last_mode = mode
        return result
