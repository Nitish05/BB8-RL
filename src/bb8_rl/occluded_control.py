"""Guarded motion during brief RGB occlusion, using commands and scan memory.

This experimental dynamics model is conditional on assumed response/error bounds;
its radii are not calibrated confidence guarantees. It never reads a simulator
pose, velocity, collision, renderer label, or depth. A zero returned request asks
the existing actuator adapter for bounded braking; it does not teleport/stop the
body. Memory queries must certify the ENTIRE requested capsule and block unknown.
"""

from collections import deque
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class CommandDynamics:
    """Nominal adapter defaults plus explicit, unvalidated mismatch assumptions.

    Applied commands refer to the preceding sensor interval. A sampled command
    can miss its transition within that interval; latency_mismatch adds a bound
    when it changes. This assumes at most one transition per sensor interval.
    Braking response_time_upper and braking_acceleration_lower bound the assumed
    zero-command response, not arbitrary external pushes, slopes, or collisions.
    """

    max_speed: float = 0.35
    response_time: float = 0.25
    max_acceleration: float = 0.7
    max_deceleration: float = 0.9
    dead_zone: float = 0.05
    heading_offset: float = 0.0
    acceleration_uncertainty: float = 0.12
    latency_mismatch: float = 0.05
    response_time_upper: float = 0.35
    braking_acceleration_lower: float = 0.5
    max_command_age: float = 0.15

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Dynamics parameters must be finite")
        if (
            min(
                self.max_speed,
                self.response_time,
                self.max_acceleration,
                self.max_deceleration,
                self.response_time_upper,
                self.braking_acceleration_lower,
                self.max_command_age,
            )
            <= 0
            or self.acceleration_uncertainty < 0
            or self.latency_mismatch < 0
            or self.latency_mismatch >= self.max_command_age
            or not 0 <= self.dead_zone < 1
            or self.response_time_upper < self.response_time
            or self.braking_acceleration_lower > self.max_deceleration
        ):
            raise ValueError("Invalid response or uncertainty bounds")

    def target(self, command):
        command = np.asarray(command, float)
        if (
            command.shape != (2,)
            or not np.isfinite(command).all()
            or np.max(abs(command)) > 1
        ):
            raise ValueError(
                "Applied normalized command must have two finite components"
            )
        magnitude = min(float(np.linalg.norm(command)), 1.0)
        if magnitude <= self.dead_zone:
            return np.zeros(2)
        angle = np.arctan2(command[1], command[0]) + self.heading_offset
        speed = self.max_speed * (magnitude - self.dead_zone) / (1 - self.dead_zone)
        return speed * np.array([np.cos(angle), np.sin(angle)])

    def advance(self, position, velocity, command, duration):
        """Small-step nominal response only; never a measured state."""
        target = self.target(command)
        position, velocity = np.array(position, float), np.array(velocity, float)
        steps = max(1, int(np.ceil(duration / 0.005)))
        dt = duration / steps
        for _ in range(steps):
            difference = target - velocity
            distance = np.linalg.norm(difference)
            limit = (
                self.max_deceleration
                if np.linalg.norm(target) < np.linalg.norm(velocity)
                else self.max_acceleration
            )
            fraction = min(
                1 - np.exp(-dt / self.response_time), limit * dt / max(distance, 1e-12)
            )
            updated = velocity + difference * fraction
            position += (velocity + updated) * (dt / 2)
            velocity = updated
        return position, velocity


@dataclass(frozen=True)
class OcclusionParameters:
    max_occlusion_seconds: float = 1.0
    max_frame_age: float = 0.10
    max_interval: float = 0.15
    speed_cap: float = 0.15
    max_position_radius: float = 0.08
    measurement_sigma_multiplier: float = 3.0
    measurement_radius_floor: float = 0.005
    innovation_slack: float = 0.015
    max_reacquisition_distance: float = 0.12
    robot_radius: float = 0.037
    clearance_margin: float = 0.04
    command_horizon: float = 0.05
    minimum_visual_samples: int = 3
    max_initial_velocity_radius: float = 0.05
    arrival_radius: float = 0.10
    arrival_speed: float = 0.03
    visible_dwell: float = 0.5

    def __post_init__(self):
        if not all(np.isfinite(value) and value > 0 for value in asdict(self).values()):
            raise ValueError("Controller parameters must be finite and positive")
        if self.minimum_visual_samples < 3:
            raise ValueError("At least three accepted RGB fixes are required")


class CommandBelief:
    """Position/velocity estimate and conservative engineering error radii.

    RGB covariance is converted to a radius by a declared sigma multiplier;
    this is an assumption, not a proof that the visual errors are bounded.
    Position is replaced only by accepted measured XY. Velocity uses applied
    commands and a bounded finite-difference estimate from previous RGB fixes.
    """

    def __init__(self, dynamics, parameters):
        self.dynamics, self.parameters = dynamics, parameters
        self.xy = self.velocity = None
        self.timestamp = self.last_visual_time = None
        self.position_radius = self.velocity_radius = float("inf")
        self.measured = False
        self.samples = 0
        self.history = deque()
        self.previous_applied = np.zeros(2)
        self.status = "uninitialized"
        self.last_prediction_mode = "uninitialized"
        self.last_prediction_segments = 0
        self.target_velocity_center = None
        self.target_velocity_radius = float("inf")

    def _validated_intervals(self, timestamp, applied_command, intervals):
        """Validate command-only acknowledged timing before mutating state."""
        if not isinstance(intervals, (list, tuple)):
            raise TypeError("Acknowledged intervals must be an ordered list")
        self.dynamics.target(applied_command)
        if self.timestamp is None:
            if intervals:
                raise ValueError("Initial belief has no preceding interval")
            return []
        if (
            not intervals
            or not np.isfinite(timestamp)
            or timestamp <= self.timestamp
            or timestamp - self.timestamp > self.parameters.max_interval + 1e-9
        ):
            raise ValueError("Need a complete bounded preceding interval")
        cursor = self.timestamp
        validated = []
        for record in intervals:
            if not isinstance(record, dict) or set(record) != {
                "start",
                "end",
                "command",
            }:
                raise ValueError(
                    "Only start/end/command are allowed in actuator history"
                )
            try:
                start, end = float(record["start"]), float(record["end"])
                command = np.asarray(record["command"], float)
                self.dynamics.target(command)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Invalid acknowledged command interval") from error
            if (
                not np.isfinite([start, end]).all()
                or abs(start - cursor) > 1e-9
                or end <= start
                or end <= cursor
                or end > timestamp + 1e-9
            ):
                raise ValueError(
                    "Acknowledged intervals must be positive and contiguous"
                )
            if abs(end - timestamp) <= 1e-9:
                end = float(timestamp)
            normalized = command / max(1.0, float(np.linalg.norm(command)))
            validated.append((end, normalized.copy()))
            cursor = end
        if abs(cursor - timestamp) > 1e-9:
            raise ValueError("Acknowledged timeline does not reach the capture time")
        return validated

    def predict(self, timestamp, applied_command, *, applied_command_intervals=None):
        """Propagate from the last state using only actuator acknowledgements.

        Optional intervals are dictionaries with exactly start/end/command,
        describing the command active during each half-open [start,end). They
        must partition the whole preceding capture interval, including zeros.
        The endpoint acknowledgement may change exactly at capture (expiry or
        activation); it is a separate future-guard input, not the last half-open
        interval's command. Past propagation uses only the supplied intervals.
        A validated timeline removes only uncertain transition-time inflation;
        acceleration mismatch and all other model bounds remain unchanged.
        Missing timelines retain the original scalar-endpoint approximation.
        """
        if applied_command_intervals is None:
            self._advance_interval(timestamp, applied_command)
            self.last_prediction_mode = "scalar_endpoint_with_timing_uncertainty"
            self.last_prediction_segments = 1
            return
        intervals = self._validated_intervals(
            timestamp, applied_command, applied_command_intervals
        )
        if not intervals:
            self._advance_interval(
                timestamp, applied_command, transition_time_known=True
            )
        else:
            for end, command in intervals:
                self._advance_interval(end, command, transition_time_known=True)
        self.last_prediction_mode = "acknowledged_command_timeline"
        self.last_prediction_segments = len(intervals)

    def _advance_interval(
        self, timestamp, applied_command, *, transition_time_known=False
    ):
        d = self.dynamics
        target = d.target(applied_command)
        self.measured = False
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("Belief timestamps must increase")
        if self.xy is not None:
            dt = timestamp - self.timestamp
            old_velocity = self.velocity.copy()
            # Both enclosures independently contain the true velocity. Translate
            # the persistent target-centered ball only when its center changes;
            # then take the smaller of two valid radii about the new target.
            nominal_to_target = (
                np.linalg.norm(target - old_velocity) + self.velocity_radius
            )
            if self.target_velocity_center is None:
                target_radius = nominal_to_target
            else:
                translated_radius = self.target_velocity_radius + np.linalg.norm(
                    target - self.target_velocity_center
                )
                target_radius = min(translated_radius, nominal_to_target)
            unsaturated = (
                nominal_to_target + d.acceleration_uncertainty * dt
                <= min(d.max_acceleration, d.max_deceleration) * d.response_time
            )
            # The adapter switches acceleration caps at |v|=|target|. If the
            # velocity set might straddle this switching surface during this
            # interval, its nominal-centered comparison needs the cap gap too.
            # This is deterministic mode uncertainty, not a smaller model q.
            possible_cap_switch = (
                abs(np.linalg.norm(old_velocity) - np.linalg.norm(target))
                <= self.velocity_radius
                + (
                    max(d.max_acceleration, d.max_deceleration)
                    + d.acceleration_uncertainty
                )
                * dt
            )
            # When the whole ball is unsaturated, neither cap can bind even if
            # velocities lie on both sides of the speed-based mode boundary.
            cap_gap = (
                abs(d.max_acceleration - d.max_deceleration)
                if possible_cap_switch and not unsaturated
                else 0.0
            )
            self.xy, self.velocity = d.advance(
                self.xy, self.velocity, applied_command, dt
            )
            mismatch = min(dt, d.latency_mismatch) * min(
                d.max_acceleration + d.max_deceleration,
                np.linalg.norm(target - d.target(self.previous_applied))
                / d.response_time,
            )
            if transition_time_known:
                mismatch = 0.0
            self.position_radius += (
                self.velocity_radius * dt
                + 0.5 * (d.acceleration_uncertainty + cap_gap) * dt**2
                + mismatch * dt
            )
            # Contract only when the full velocity-error ball is unsaturated.
            decay = np.exp(-dt / d.response_time) if unsaturated else 1.0
            radius = self.velocity_radius * decay
            # Also propagate a ball centered on the fixed target. This reduces
            # initial unknown velocity during observed zero-command warm-up,
            # including the acceleration-saturated part of the response.
            limit = min(d.max_acceleration, d.braking_acceleration_lower)
            response_upper = d.response_time_upper
            saturated_time = max(0, (target_radius - limit * response_upper) / limit)
            remaining = max(0, dt - saturated_time)
            target_radius = max(
                0, target_radius - limit * min(dt, saturated_time)
            ) * np.exp(-remaining / response_upper)
            self.target_velocity_center = target.copy()
            self.target_velocity_radius = float(
                target_radius + d.acceleration_uncertainty * dt + mismatch
            )
            self.velocity_radius = float(
                min(
                    radius + (d.acceleration_uncertainty + cap_gap) * dt + mismatch,
                    np.linalg.norm(self.velocity - target)
                    + self.target_velocity_radius,
                )
            )
        self.previous_applied = np.asarray(applied_command, float).copy()
        self.timestamp = float(timestamp)
        self.status = "predicted" if self.xy is not None else "uninitialized"

    def observe(self, measurement):
        p, d = self.parameters, self.dynamics
        if getattr(measurement, "status", None) != "visible":
            self.status = getattr(measurement, "status", "invalid_measurement")
            return False
        try:
            xy, covariance = (
                np.asarray(measurement.xy, float),
                np.asarray(measurement.covariance, float),
            )
            valid = (
                xy.shape == (2,)
                and covariance.shape == (2, 2)
                and np.isfinite(xy).all()
                and np.isfinite(covariance).all()
                and np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-6)
                and np.linalg.eigvalsh(covariance).min() > 0
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            valid = False
        if not valid:
            self.status = "invalid_measurement"
            return False
        radius = max(
            p.measurement_radius_floor,
            p.measurement_sigma_multiplier
            * np.sqrt(np.linalg.eigvalsh(covariance).max()),
        )
        if radius > p.max_position_radius:
            self.status = "uncertain_measurement"
            return False
        if self.xy is not None:
            innovation = np.linalg.norm(xy - self.xy)
            gate = min(
                p.max_reacquisition_distance,
                self.position_radius + radius + p.innovation_slack,
            )
            if innovation > gate:
                self.status = "reacquisition_rejected"
                return False
        else:
            self.velocity = np.zeros(2)
            self.velocity_radius = d.max_speed
        self.xy = xy.copy()
        self.position_radius = float(radius)
        # Endpoint uncertainty plus bounded acceleration bounds final velocity
        # relative to the interval-average finite difference. No truth velocity.
        for old_time, old_xy, old_radius in self.history:
            dt = self.timestamp - old_time
            if 0.15 - 1e-9 <= dt <= 0.4 + 1e-9:
                bound = (old_radius + radius) / dt + 0.5 * (
                    max(d.max_acceleration, d.max_deceleration)
                    + d.acceleration_uncertainty
                ) * dt
                if bound < self.velocity_radius:
                    self.velocity = (xy - old_xy) / dt
                    self.velocity_radius = float(bound)
        self.history.append((self.timestamp, xy.copy(), radius))
        if self.target_velocity_center is not None:
            # An accepted RGB finite-difference ball may add information; it
            # never discards the persistent target-centered reachable set.
            self.target_velocity_radius = float(
                min(
                    self.target_velocity_radius,
                    np.linalg.norm(self.velocity - self.target_velocity_center)
                    + self.velocity_radius,
                )
            )
        while self.history and self.timestamp - self.history[0][0] > 0.4:
            self.history.popleft()
        self.last_visual_time = self.timestamp
        self.samples += 1
        self.measured = True
        self.status = "visible"
        return True


class OccludedController:
    """SAC waypoint actions gated by a remembered map and stopping envelope.

    ``segment_free(a,b,radius_m)`` must certify ALL of the capsule including its
    radius, against raw map evidence; do not pass a centerline-only predicate.
    The grid may be footprint-inflated for A*, but the independent certificate
    receives the full physical/error radius. Routes are created only while RGB
    is measured, then retained during bounded occlusion. Frozen versions must
    match each call; replacing memory/calibration requires a new controller.

    ``action`` timestamp and measurement.timestamp refer to the sensor/sim
    clock, with now used to reject stale frames. applied_command is the actual
    preceding actuator request, never a proposed but unexecuted policy action.
    No success is declared until accepted visible fixes satisfy a continuous
    dwell; missing, rejected, stale, and repeated frames break that dwell.

    V2 assumes this wrapper is the sole command issuer and returned commands
    expire within dynamics.max_command_age (the adapter default is 0.15 s).
    Every returned target is retained for that entire lifetime, regardless of
    acknowledgement value: equal values cannot identify a queued command.
    Until one complete lifetime of history exists, or if the caller declares
    history unknown, the universal maximum target speed supplies the bound.
    The candidate speed is reduced only along the SAC-requested heading.

    V3 optionally accepts a complete command-only acknowledgement timeline for
    the preceding interval. These are actuator inputs with observed transition
    times, not physics state. Missing/overlapping/contradictory timelines brake
    without advancing the belief. Future queued-command latency and all map,
    model, uncertainty, occlusion and arrival gates are unchanged.

    V4 preserves the target-centered velocity reachable ball between intervals;
    recentering on the nominal estimate never overwrites that persistent ball.

    V5 requests zero until an accepted visible fix and command-model contraction
    establish the initial velocity-error radius. No reset velocity is assumed.
    Initialization latches; subsequent uncertainty uses the existing guards.

    V6 starts arrival dwell only after a zero endpoint acknowledgement and a
    complete command history with no unexpired nonzero targets. A pending or
    still-acknowledged motion request requires visible settling with zero output.

    An optional ``route_planner(start, goal, radius)`` can construct a route for
    the current physical margin plus position uncertainty. Its result still
    requires independent whole-capsule certification. New routes require a
    current visible fix; losing a retained segment's clearance first stops.

    Arrival additionally waits for the assumed zero-command braking response
    to reduce the initial speed enclosure to arrival_speed, before the full
    visible dwell begins. This uses the same lower acceleration / upper response
    time assumption as the stopping-envelope tail, not an adversarial sustained
    q-disturbance guarantee or calibrated physical-speed measurement.
    """

    def __init__(
        self,
        policy,
        goal_xy,
        grid,
        segment_free,
        *,
        map_version,
        calibration_version,
        parameters=None,
        dynamics=None,
        route_planner=None,
    ):
        self.parameters = parameters or OcclusionParameters()
        self.dynamics = dynamics or CommandDynamics()
        if self.parameters.speed_cap > self.dynamics.max_speed:
            raise ValueError("Speed cap exceeds configured drive maximum")
        self.goal = np.asarray(goal_xy, float)
        if self.goal.shape != (2,) or not np.isfinite(self.goal).all():
            raise ValueError("Goal must be two finite calibrated coordinates")
        if (
            not callable(policy)
            or not callable(segment_free)
            or not callable(getattr(grid, "route", None))
            or (route_planner is not None and not callable(route_planner))
        ):
            raise TypeError("Need policy, route grid and whole-capsule certificate")
        if (
            not isinstance(map_version, str)
            or not map_version
            or not isinstance(calibration_version, str)
            or not calibration_version
        ):
            raise ValueError("Frozen map/calibration versions are required")
        self.policy, self.grid, self.segment_free = policy, grid, segment_free
        self.route_planner = route_planner
        self.requested_planning_radius = self.route_planning_radius = None
        self.map_version, self.calibration_version = map_version, calibration_version
        self.belief = CommandBelief(self.dynamics, self.parameters)
        self.route = None
        self.index = 0
        self.status = "uninitialized"
        self.arrived = False
        self.visible_since = None
        self.arrival_settle_until = self.arrival_settle_speed_bound = None
        self.envelope_radius = None
        self.segment_certificate = False
        self.issued_action = np.zeros(2, np.float32)
        self.consulted_versions = {}
        self.request_timestamp = None
        self.input_accepted = False
        self.issued_history = deque()
        self.history_started_at = None
        self.request_now = None
        self.command_history_valid = True
        self.command_history_complete = False
        self.speed_scale = None
        self.envelope_trials = []
        self.velocity_initialized = False

    def _finish(self, status, command=None):
        self.status = status
        self.issued_action = (
            np.zeros(2, np.float32)
            if command is None
            else np.asarray(command, np.float32)
        )
        if status not in ("tracking_visible", "arrived_visible"):
            self.visible_since = None
            if status != "settling_visible":
                self._reset_arrival()
        if self.request_now is not None:
            if self.history_started_at is None:
                self.history_started_at = self.request_now
            self.issued_history.append(
                (
                    self.request_now,
                    self.request_now + self.dynamics.max_command_age,
                    float(np.linalg.norm(self.dynamics.target(self.issued_action))),
                )
            )
        return self.issued_action.copy()

    def _reset_arrival(self):
        self.visible_since = None
        self.arrival_settle_until = self.arrival_settle_speed_bound = None

    def _arrival_braking_time(self, speed_bound):
        """Time to the speed gate under the existing conditional braking tail.

        For ds/dt=-min(a_lower,s/tau_upper), first reach a_lower*tau_upper
        with saturated braking, then decay exponentially to arrival_speed.
        The generic model-error q is not a sustained adversarial force during
        this tail: q*tau_upper can exceed the speed gate. This extra wait thus
        preserves the declared braking assumption, not a physical guarantee.
        """
        d, target = self.dynamics, self.parameters.arrival_speed
        threshold = d.braking_acceleration_lower * d.response_time_upper
        saturated = (
            max(0.0, speed_bound - max(target, threshold))
            / d.braking_acceleration_lower
        )
        exponential = d.response_time_upper * np.log(
            max(target, min(speed_bound, threshold)) / target
        )
        return float(saturated + exponential)

    def _refresh_command_history(self, timestamp, now):
        """Retain every issued target alive at capture, even if expired by now."""
        self.issued_history = deque(
            entry for entry in self.issued_history if entry[1] >= timestamp - 1e-12
        )
        self.command_history_complete = bool(
            self.command_history_valid
            and self.history_started_at is not None
            and timestamp - self.history_started_at
            >= self.dynamics.max_command_age - 1e-12
            and all(entry[0] <= now + 1e-12 for entry in self.issued_history)
        )

    def _reaction_envelope(self, command, applied_command, timestamp, now):
        """Same whole-stop proof with a command-aware reachable-speed bound.

        A saturated first-order response is always directed toward its target.
        Without mismatch it cannot leave the speed ball containing the initial
        velocity and every possible target. A bounded additive acceleration q
        expands that speed ball by q*t and its path length by q*t²/2.
        Retain targets alive at CAPTURE time, including ones that expire before
        `now`: they can still have affected motion during the observation age.
        """
        p, d, b = self.parameters, self.dynamics, self.belief
        self._refresh_command_history(timestamp, now)
        candidate_speed = float(np.linalg.norm(d.target(command)))
        acknowledged_speed = float(np.linalg.norm(d.target(applied_command)))
        retained_speed = max((entry[2] for entry in self.issued_history), default=0.0)
        target_bound = max(candidate_speed, acknowledged_speed, retained_speed)
        if not self.command_history_complete:
            target_bound = max(target_bound, d.max_speed)
        initial_speed_bound = float(np.linalg.norm(b.velocity) + b.velocity_radius)
        nominal_speed_bound = max(initial_speed_bound, target_bound)
        reaction = p.command_horizon + d.latency_mismatch + max(0, now - timestamp)
        speed_bound = nominal_speed_bound + d.acceleration_uncertainty * reaction
        reaction_distance = (
            nominal_speed_bound * reaction
            + 0.5 * d.acceleration_uncertainty * reaction**2
        )
        threshold = d.braking_acceleration_lower * d.response_time_upper
        stopping_distance = (
            speed_bound * d.response_time_upper
            if speed_bound <= threshold
            else (speed_bound**2 - threshold**2) / (2 * d.braking_acceleration_lower)
            + threshold * d.response_time_upper
        )
        base_radius = p.robot_radius + p.clearance_margin + b.position_radius
        radius = float(base_radius + reaction_distance + stopping_distance)
        endpoint, _ = d.advance(b.xy, b.velocity, command, reaction)
        return (
            endpoint,
            radius,
            {
                "candidate_target_speed_m_s": candidate_speed,
                "acknowledged_target_speed_m_s": acknowledged_speed,
                "retained_target_speed_m_s": retained_speed,
                "initial_speed_bound_m_s": initial_speed_bound,
                "target_speed_bound_m_s": target_bound,
                "reaction_speed_bound_m_s": speed_bound,
                "reaction_seconds": reaction,
                "reaction_path_length_m": reaction_distance,
                "stopping_distance_m": stopping_distance,
                "envelope_radius_m": radius,
                "command_history_complete": self.command_history_complete,
            },
        )

    def _scaled_command(self, command, scale):
        """Scale target speed, preserving heading and the calibrated dead zone."""
        d = self.dynamics
        speed = np.linalg.norm(d.target(command))
        if speed == 0 or scale == 0:
            return np.zeros(2)
        magnitude = d.dead_zone + (1 - d.dead_zone) * speed * scale / d.max_speed
        return np.asarray(command) / np.linalg.norm(command) * magnitude

    def _certify(self, start, end, radius):
        try:
            return bool(
                self.segment_free(np.asarray(start), np.asarray(end), float(radius))
            )
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            return False

    def action(
        self,
        timestamp,
        measurement,
        *,
        applied_command,
        map_version,
        calibration_version,
        now=None,
        memory_valid=True,
        command_history_valid=True,
        applied_command_intervals=None,
    ):
        now = timestamp if now is None else now
        self.belief.measured = False
        self.request_timestamp = float(timestamp) if np.isfinite(timestamp) else None
        self.input_accepted = False
        self.request_now = float(now) if np.isfinite(now) and now >= 0 else None
        self.command_history_valid = bool(command_history_valid)
        self.command_history_complete = False
        self.speed_scale = None
        self.envelope_trials = []
        self.segment_certificate = False
        self.envelope_radius = None
        self.requested_planning_radius = None
        self.arrived = False
        self.consulted_versions = {
            "map": map_version,
            "calibration": calibration_version,
        }
        if (
            not memory_valid
            or map_version != self.map_version
            or calibration_version != self.calibration_version
        ):
            self.route = None
            self.route_planning_radius = None
            self.velocity_initialized = False
            return self._finish("memory_or_calibration_invalid")
        p, d, b = self.parameters, self.dynamics, self.belief
        if (
            not np.isfinite(timestamp)
            or not np.isfinite(now)
            or timestamp < 0
            or timestamp > now + 1e-9
            or now - timestamp > p.max_frame_age + 1e-9
        ):
            return self._finish("stale_or_invalid_frame")
        if b.timestamp is not None and timestamp <= b.timestamp:
            return self._finish("repeated_frame")
        if (
            measurement is None
            or not np.isfinite(getattr(measurement, "timestamp", np.nan))
            or abs(measurement.timestamp - timestamp) > 1e-9
        ):
            return self._finish("invalid_measurement_time")
        if getattr(measurement, "status", None) == "missing" and (
            getattr(measurement, "xy", None) is not None
            or getattr(measurement, "covariance", None) is not None
        ):
            return self._finish("invalid_missing_measurement")
        if b.timestamp is not None and timestamp - b.timestamp > p.max_interval + 1e-9:
            self.route = None
            self.route_planning_radius = None
            self.velocity_initialized = False
            # State is stale; require a new controller / fresh initialization.
            return self._finish("control_interval_exceeded")
        try:
            b.predict(
                timestamp,
                applied_command,
                applied_command_intervals=applied_command_intervals,
            )
        except (ValueError, TypeError):
            self.velocity_initialized = False
            return self._finish(
                "invalid_applied_command"
                if applied_command_intervals is None
                else "invalid_applied_command_intervals"
            )
        self.input_accepted = True
        visible = b.observe(measurement)
        if not visible and getattr(measurement, "status", None) != "missing":
            return self._finish(b.status)
        if (
            b.xy is None
            or b.last_visual_time is None
            or b.samples < p.minimum_visual_samples
        ):
            return self._finish("warming_up")
        if not self.velocity_initialized:
            if not visible or b.velocity_radius > p.max_initial_velocity_radius:
                return self._finish("warming_up")
            self.velocity_initialized = True
        if timestamp - b.last_visual_time > p.max_occlusion_seconds + 1e-9:
            return self._finish("occlusion_timeout")
        if b.position_radius > p.max_position_radius:
            return self._finish("uncertain_prediction")
        base_radius = p.robot_radius + p.clearance_margin + b.position_radius
        if self.route is None:
            if not visible:
                return self._finish("no_proven_route_memory")
            self.requested_planning_radius = float(base_radius)
            try:
                planned = (
                    self.grid.route(b.xy, self.goal)
                    if self.route_planner is None
                    else self.route_planner(b.xy.copy(), self.goal.copy(), base_radius)
                )
                route = np.asarray(planned, float)
            except (TypeError, ValueError, RuntimeError):
                return self._finish("no_proven_route_memory")
            if (
                route.ndim != 2
                or route.shape[1] != 2
                or len(route) < 1
                or not np.isfinite(route).all()
                or not np.allclose(route[0], b.xy)
                or not np.allclose(route[-1], self.goal)
            ):
                return self._finish("invalid_route")
            if not all(
                self._certify(a, z, base_radius) for a, z in pairwise(route)
            ) or not self._certify(b.xy, b.xy, base_radius):
                return self._finish("route_not_certified")
            self.route, self.index = route.copy(), min(1, len(route) - 1)
            self.route_planning_radius = float(base_radius)
        if (
            self.index < len(self.route) - 1
            and np.linalg.norm(self.route[self.index] - b.xy) < 0.045
            and np.linalg.norm(b.velocity) < 0.06
            and self._certify(b.xy, self.route[self.index + 1], base_radius)
        ):
            self.index += 1
        if not self._certify(b.xy, self.route[self.index], base_radius):
            self.route = None
            self.route_planning_radius = None
            return self._finish("route_not_certified")
        near = (
            np.linalg.norm(b.xy - self.goal) + b.position_radius <= p.arrival_radius
            and np.linalg.norm(b.velocity) <= p.arrival_speed
        )
        if visible and near:
            self._refresh_command_history(timestamp, now)
            if (
                not self.command_history_complete
                or np.linalg.norm(d.target(applied_command)) > 0
                or any(speed > 0 for _, _, speed in self.issued_history)
            ):
                self._reset_arrival()
                return self._finish("settling_visible")
            if self.arrival_settle_until is None:
                self.arrival_settle_speed_bound = float(
                    np.linalg.norm(b.velocity) + b.velocity_radius
                )
                self.arrival_settle_until = timestamp + self._arrival_braking_time(
                    self.arrival_settle_speed_bound
                )
            if timestamp < self.arrival_settle_until - 1e-9:
                return self._finish("settling_visible")
            if self.visible_since is None:
                self.visible_since = timestamp
            if timestamp - self.visible_since >= p.visible_dwell - 1e-9:
                self.arrived = True
                return self._finish("arrived_visible")
            return self._finish("tracking_visible")
        else:
            self._reset_arrival()
        vector = np.asarray(
            [
                *(self.route[self.index] - b.xy),
                *(b.velocity / d.max_speed),
                self.index == len(self.route) - 1,
            ],
            np.float32,
        )
        try:
            command = np.asarray(self.policy(vector), float)
            d.target(command)
        except (ValueError, TypeError, RuntimeError):
            return self._finish("invalid_policy_action")
        magnitude = np.linalg.norm(command)
        maximum = d.dead_zone + (1 - d.dead_zone) * p.speed_cap / d.max_speed
        command *= min(1, maximum / max(magnitude, 1e-12))
        # Every candidate preserves SAC heading. Rejected candidates are never
        # issued/added to history. Current momentum and all possible queued
        # targets remain in every candidate's bound, including reduced speeds.
        scales = (
            (1.0,) if np.linalg.norm(d.target(command)) == 0 else (1.0, 0.75, 0.5, 0.25)
        )
        for scale in scales:
            candidate = self._scaled_command(command, scale)
            endpoint, radius, trial = self._reaction_envelope(
                candidate, applied_command, timestamp, now
            )
            self.envelope_radius = radius
            self.segment_certificate = self._certify(b.xy, endpoint, radius)
            trial.update(speed_scale=scale, certified=self.segment_certificate)
            self.envelope_trials.append(trial)
            if self.segment_certificate:
                self.speed_scale = scale
                return self._finish(
                    "tracking_visible" if visible else "tracking_hidden", candidate
                )
        return self._finish("stopping_envelope_not_certified")

    @property
    def diagnostics(self):
        b = self.belief
        current = self.input_accepted and b.timestamp == self.request_timestamp
        return {
            "status": self.status,
            "pose_source": "measured"
            if b.measured
            else (
                ("predicted" if current else "stale")
                if b.xy is not None
                else "uninitialized"
            ),
            "xy": None if b.xy is None else b.xy.tolist(),
            "velocity": None if b.velocity is None else b.velocity.tolist(),
            "speed_m_s": None
            if b.velocity is None
            else float(np.linalg.norm(b.velocity)),
            "position_radius_m": None
            if not np.isfinite(b.position_radius)
            else b.position_radius,
            "velocity_radius_m_s": None
            if not np.isfinite(b.velocity_radius)
            else b.velocity_radius,
            "last_visual_time": b.last_visual_time,
            "timestamp": b.timestamp,
            "last_state_time": b.timestamp,
            "request_timestamp": self.request_timestamp,
            "state_is_current": current,
            "input_accepted": self.input_accepted,
            "occlusion_seconds": None
            if b.last_visual_time is None
            else b.timestamp - b.last_visual_time,
            "envelope_radius_m": self.envelope_radius,
            "motion_authorized": self.status in ("tracking_visible", "tracking_hidden")
            and bool(np.linalg.norm(self.issued_action) > 0),
            "issued_action": self.issued_action.tolist(),
            "versions": self.consulted_versions,
            "route_available": self.route is not None,
            "route_index": self.index,
            "segment_certificate": self.segment_certificate,
            "arrived": self.arrived,
            "dynamics_bounds_calibrated": False,
            "controller_revision": "v8-model-settling-before-visible-dwell",
            "arrival_settle_until": self.arrival_settle_until,
            "arrival_settle_speed_bound_m_s": self.arrival_settle_speed_bound,
            "arrival_visible_since": self.visible_since,
            "requested_planning_radius_m": self.requested_planning_radius,
            "route_planning_radius_m": self.route_planning_radius,
            "velocity_initialized": self.velocity_initialized,
            "max_initial_velocity_radius_m_s": self.parameters.max_initial_velocity_radius,
            "prediction_mode": b.last_prediction_mode if current else "stale",
            "acknowledged_prediction_segments": b.last_prediction_segments
            if current
            else 0,
            "target_velocity_center_m_s": None
            if b.target_velocity_center is None
            else b.target_velocity_center.tolist(),
            "target_velocity_radius_m_s": None
            if not np.isfinite(b.target_velocity_radius)
            else b.target_velocity_radius,
            "speed_scale": self.speed_scale,
            "command_history_complete": self.command_history_complete,
            "retained_issued_targets": [
                {"issued_at": start, "expires_at": end, "target_speed_m_s": speed}
                for start, end, speed in self.issued_history
            ],
            "envelope_trials": self.envelope_trials,
        }
