"""Conditional, non-authoritative position region during acknowledged braking.

This is not a covariance calibration, stationarity measurement, or control bound.
The authoritative predictor still permits sustained acceleration mismatch. This
separate diagnostic instead assumes an unforced dissipative zero-command tail.
"""

import math
from collections.abc import Mapping

import numpy as np


class ConditionalBrakingRegion:
    """Read-only snapshots plus complete command timing produce an advisory disk.

    ``update`` must run BEFORE advancing the source belief. ``previous_snapshot``
    is a mapping with timestamp, xy, velocity, position_radius, velocity_radius,
    map_version and calibration_version. Its state refers to the START of the
    supplied acknowledged interval. Each interval is a dict with exactly start,
    end and command, using the same half-open command-only timing as the runtime.

    A valid update requires a complete contiguous partition, every tick at most
    ``max_tick_seconds``, exact zero commands, zero endpoint acknowledgement, no
    pending nonzero command, valid inputs, and matching versions. Missing RGB is
    not itself invalid: the snapshot may be a valid prediction. Invalidations
    discard the anchor; the next update needs a new valid pre-interval snapshot.
    Repeated/backwards calls are rejected. Timeline comparisons allow only the
    existing command contract's 1 ns floating-point timestamp tolerance.

    For speed upper bound u, assume du/dt <= -min(a, u/tau), with a the lower
    braking acceleration and tau the upper response time. No sustained q-force,
    external push or nonzero future target is allowed by this conditional model.
    The fixed-anchor disk integrates this tail. A left-endpoint sum on ticks of
    maximum length delta exceeds its continuous integral by at most u0*delta;
    the returned radius adds that allowance (u0*min(t,delta) before one tick).
    The speed diagnostic likewise allows one tick of delay. Neither diagnostic
    may authorize motion, certify free space, reset a belief, or declare arrival.
    """

    _TIME_TOLERANCE = 1e-9

    def __init__(
        self,
        *,
        braking_acceleration_lower=0.5,
        response_time_upper=0.35,
        max_tick_seconds=0.005,
        max_interval_seconds=0.15,
        enabled=True,
    ):
        values = (
            braking_acceleration_lower,
            response_time_upper,
            max_tick_seconds,
            max_interval_seconds,
        )
        if (
            not all(math.isfinite(v) and v > 0 for v in values)
            or max_tick_seconds > max_interval_seconds
            or type(enabled) is not bool
        ):
            raise ValueError("Need positive finite braking/timing parameters")
        self.acceleration = float(braking_acceleration_lower)
        self.response_time = float(response_time_upper)
        self.max_tick = float(max_tick_seconds)
        self.max_interval = float(max_interval_seconds)
        self.enabled = enabled
        self._anchor = None
        self._last_update = None
        self._last_valid_end = None
        self._elapsed = 0.0
        self._reason = "not_started" if enabled else "disabled"

    def _invalidate(self, reason):
        self._anchor = None
        self._last_valid_end = None
        self._elapsed = 0.0
        self._reason = reason
        return self.record

    def invalidate(self, reason="invalidated"):
        """Discard the advisory after a new nonzero request/context change.

        The monotonic-call watermark remains: invalidation does not make a
        previously consumed command interval fresh again.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("Need a nonempty invalidation reason")
        return self._invalidate(reason)

    @staticmethod
    def _zero(command):
        command = np.asarray(command, dtype=float)
        return bool(
            command.shape == (2,)
            and np.isfinite(command).all()
            and not np.any(command != 0)
        )

    def _tail(self, speed, elapsed):
        threshold = self.acceleration * self.response_time
        exponential_start = min(speed, threshold)
        saturated_time = max(0.0, (speed - threshold) / self.acceleration)
        if elapsed <= saturated_time:
            return (
                speed - self.acceleration * elapsed,
                speed * elapsed - 0.5 * self.acceleration * elapsed**2,
            )
        remaining = elapsed - saturated_time
        saturated_distance = (
            (speed - exponential_start)
            * (speed + exponential_start)
            / (2 * self.acceleration)
        )
        return (
            exponential_start * math.exp(-remaining / self.response_time),
            saturated_distance
            - exponential_start
            * self.response_time
            * math.expm1(-remaining / self.response_time),
        )

    def update(
        self,
        *,
        previous_snapshot,
        timestamp,
        acknowledged_intervals,
        endpoint_command,
        map_version,
        calibration_version,
        pending_nonzero=False,
        inputs_valid=True,
    ):
        """Return a JSON-safe advisory record; never mutate the supplied snapshot.

        ``map_version`` and ``calibration_version`` describe current context.
        ``inputs_valid`` must be false for invalid belief/timing/context, and
        ``pending_nonzero`` true while any nonzero target can still take effect.
        Callers must not substitute an issued zero request for an acknowledged
        timeline. Returned ``motion_authority`` and ``observed`` are always false.
        """
        if not self.enabled:
            return self._invalidate("disabled")
        try:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp < 0:
                return self._invalidate("invalid_time")
            if self._last_update is not None and timestamp <= self._last_update:
                return self._invalidate("repeated_or_backwards_time")
            self._last_update = timestamp
            if type(inputs_valid) is not bool or not inputs_valid:
                return self._invalidate("invalid_inputs")
            if type(pending_nonzero) is not bool or pending_nonzero:
                return self._invalidate("pending_nonzero_command")
            if not self._zero(endpoint_command):
                return self._invalidate("nonzero_or_invalid_endpoint_command")
            if not isinstance(previous_snapshot, Mapping):
                return self._invalidate("invalid_snapshot")
            source = previous_snapshot
            start = float(source["timestamp"])
            xy = np.array(source["xy"], dtype=float, copy=True)
            velocity = np.array(source["velocity"], dtype=float, copy=True)
            radius = float(source["position_radius"])
            velocity_radius = float(source["velocity_radius"])
            if (
                xy.shape != (2,)
                or velocity.shape != (2,)
                or not np.isfinite(xy).all()
                or not np.isfinite(velocity).all()
                or not all(
                    math.isfinite(v) and v >= 0
                    for v in (start, radius, velocity_radius)
                )
                or timestamp <= start
                or timestamp - start > self.max_interval + self._TIME_TOLERANCE
            ):
                return self._invalidate("invalid_snapshot")
            versions = (map_version, calibration_version)
            if (
                not all(isinstance(v, str) and v for v in versions)
                or versions != (source["map_version"], source["calibration_version"])
                or (self._anchor is not None and versions != self._anchor["versions"])
            ):
                return self._invalidate("version_mismatch")
            if (
                self._last_valid_end is not None
                and abs(start - self._last_valid_end) > self._TIME_TOLERANCE
            ):
                return self._invalidate("timeline_gap")
            if (
                not isinstance(acknowledged_intervals, (list, tuple))
                or not acknowledged_intervals
            ):
                return self._invalidate("missing_timeline")
            cursor = start
            for interval in acknowledged_intervals:
                if not isinstance(interval, Mapping) or set(interval) != {
                    "start",
                    "end",
                    "command",
                }:
                    return self._invalidate("invalid_timeline")
                left, right = float(interval["start"]), float(interval["end"])
                if (
                    not math.isfinite(left)
                    or not math.isfinite(right)
                    or abs(left - cursor) > self._TIME_TOLERANCE
                    or right <= left
                    or right <= cursor
                    or right - left > self.max_tick + self._TIME_TOLERANCE
                    or right > timestamp + self._TIME_TOLERANCE
                ):
                    return self._invalidate("invalid_timeline")
                if not self._zero(interval["command"]):
                    return self._invalidate("nonzero_or_invalid_interval_command")
                cursor = right
            if abs(cursor - timestamp) > self._TIME_TOLERANCE:
                return self._invalidate("incomplete_timeline")
            if self._anchor is None:
                speed = math.hypot(*velocity) + velocity_radius
                _, distance_limit = self._tail(speed, float("inf"))
                limit = radius + distance_limit + speed * self.max_tick
                if not math.isfinite(speed) or not math.isfinite(limit):
                    return self._invalidate("invalid_snapshot")
                self._anchor = {
                    "xy": xy,
                    "time": start,
                    "radius": radius,
                    "speed": speed,
                    "limit": limit,
                    "versions": versions,
                }
            self._elapsed = timestamp - self._anchor["time"]
            self._last_valid_end = timestamp
            self._reason = "complete_zero_command_history"
            return self.record
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._invalidate("invalid_input_structure")

    @property
    def record(self):
        """Fresh JSON-safe value; no references to the source belief or anchor."""
        valid = self._anchor is not None
        result = {
            "enabled": self.enabled,
            "valid": valid,
            "kind": "conditional_braking_region",
            "motion_authority": False,
            "observed": False,
            "status": "settling_conditional" if valid else "unavailable",
            "reason": self._reason,
            "anchor_xy": None,
            "anchor_time": None,
            "radius_m": None,
            "radius_limit_m": None,
            "speed_upper_m_s": None,
            "assumptions": [
                "Dissipative zero-target braking; no sustained acceleration mismatch or external push.",
                "Zero acknowledged and pending targets; unchanged map/calibration context.",
                "Initial position and speed enclosures are engineering assumptions, not calibrated confidence.",
                "Advisory only: never a position observation, stationarity proof, or motion/arrival authority.",
            ],
            "braking_acceleration_lower_m_s2": self.acceleration,
            "response_time_upper_s": self.response_time,
            "max_tick_seconds": self.max_tick,
            "elapsed_seconds": None,
            "anchor_radius_m": None,
            "anchor_speed_upper_m_s": None,
            "discrete_padding_m": None,
        }
        if valid:
            anchor = self._anchor
            _, distance = self._tail(anchor["speed"], self._elapsed)
            speed, _ = self._tail(
                anchor["speed"], max(0.0, self._elapsed - self.max_tick)
            )
            padding = anchor["speed"] * min(self._elapsed, self.max_tick)
            result.update(
                anchor_xy=anchor["xy"].tolist(),
                anchor_time=anchor["time"],
                radius_m=anchor["radius"] + distance + padding,
                radius_limit_m=anchor["limit"],
                speed_upper_m_s=speed,
                elapsed_seconds=self._elapsed,
                anchor_radius_m=anchor["radius"],
                anchor_speed_upper_m_s=anchor["speed"],
                discrete_padding_m=padding,
            )
        return result
