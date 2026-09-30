"""Spawned, lockstep RGB navigation worker; importing this module starts nothing.

Genesis and learned-model imports occur only inside ``worker_main``. Simulation
time pauses during rendering/inference: processing_ms is a measurement, not a
real-time guarantee. Truth/renderer labels are consumed only after each decision
for a separate audit log. ControlSession accepts measurements and command inputs.
"""

import hashlib
import json
import math
import queue
import time
from itertools import pairwise
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def set_render_camera_pose(camera, record):
    """Reproduce the declared fixture independently of previous camera roll.

    Render setup only. Estimated registration remains the controller calibration.
    Existing bundles use world-Z up; an explicitly declared up is honored.
    """
    camera.set_pose(
        pos=record["position"],
        lookat=record.get("lookat", [0, 0, 0.1]),
        up=record.get("up", [0, 0, 1]),
    )


def camera_pixel_coordinates(record):
    """Untagged installed bundles retain their original renderer convention."""
    value = record.get("pixel_coordinates", "genesis_viewport")
    if value not in ("genesis_viewport", "opencv_integer_center"):
        raise ValueError("Unknown camera pixel coordinate convention")
    return value


def validate_render_camera_intrinsics(camera, record, extent, *, scale=1):
    """Check renderer setup in the declared convention, never install true W2C."""
    from .camera_rig import calibration_from_live_camera, scale_intrinsics

    coordinates = camera_pixel_coordinates(record)
    live = calibration_from_live_camera(camera, extent, pixel_coordinates=coordinates)
    expected = scale_intrinsics(
        record["intrinsics"], scale, pixel_coordinates=coordinates
    )
    if not np.allclose(live.intrinsics, expected, atol=1e-5) or tuple(
        live.resolution
    ) != tuple(scale * value for value in record["resolution"]):
        raise ValueError("Live camera intrinsics/resolution differ from bundle")


def publish_latest(events, event):
    """Never block actuation on a slow consumer or an already-full queue."""
    for _ in range(4):
        try:
            events.put_nowait(event)
            return True
        except queue.Full:
            try:
                events.get_nowait()
            except queue.Empty:
                pass  # multiprocessing feeder may not yet expose the old item
    return False


def drain_commands(commands, limit=256):
    result = []
    for _ in range(limit):
        try:
            result.append(commands.get_nowait())
        except queue.Empty:
            break
    return result


def _point(value):
    point = np.asarray(value, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Goal must have two finite metric coordinates")
    return point


def evaluation_camera_changes(config, camera_ids):
    """Validate explicit render-only test disturbances; never control inputs."""
    changes = config.get("evaluation_camera_changes", [])
    if not isinstance(changes, list) or len(changes) > 8:
        raise ValueError("At most eight evaluation camera changes are supported")
    if changes and config.get("recording_mode") != "audit":
        raise ValueError("Evaluation camera changes require full audit recording")
    checked = []
    for change in changes:
        if not isinstance(change, dict) or set(change) != {
            "at_time",
            "camera_id",
            "position",
            "lookat",
        }:
            raise ValueError("Invalid evaluation camera change fields")
        at = change["at_time"]
        if type(at) not in (int, float) or not math.isfinite(at) or at < 0:
            raise ValueError("Invalid evaluation camera change time")
        if change["camera_id"] not in camera_ids:
            raise ValueError("Evaluation camera change must use an active view")
        for key in ("position", "lookat"):
            value = np.asarray(change[key], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Camera pose must be a finite three-vector")
        if np.linalg.norm(np.asarray(change["position"]) - change["lookat"]) < 1e-6:
            raise ValueError("Camera position and lookat must differ")
        checked.append(dict(change))
    return sorted(checked, key=lambda change: change["at_time"])


class ControlSession:
    """Pure control coordinator; no simulator state or renderer labels accepted.

    A target replacement first calls the existing backend stop and discards the
    old controller. A new V6 controller must repeat its conservative visible
    velocity initialization. Idle localization continues with zero commands.
    STOP wins a drained batch; commands at/below its generation cannot resume it.
    Heartbeat expiry cancels the target, rather than automatically resuming when
    a heartbeat returns. New target commands are required to resume.
    """

    def __init__(
        self,
        *,
        policy,
        memory,
        stop,
        map_version,
        calibration_version,
        demo_goal,
        generation=0,
        planning_radius=0.10,
        parameters=None,
        planning_reserve=0.0,
        heartbeat_seconds=3.0,
        clock=time.monotonic,
        agency_engine=None,
        agency_coordinator=None,
        visibility_planner=None,
        scene_validity_required=False,
    ):
        from .agency_runtime import AutonomyCoordinator
        from .occluded_control import (
            CommandBelief,
            CommandDynamics,
            OcclusionParameters,
        )

        self.agency = agency_coordinator or AutonomyCoordinator(agency_engine)

        if heartbeat_seconds <= 0 or not np.isfinite(heartbeat_seconds):
            raise ValueError("Heartbeat timeout must be finite and positive")
        self.policy, self.memory, self.stop = policy, memory, stop
        self.map_version, self.calibration_version = map_version, calibration_version
        self.registered_versions = (map_version, calibration_version)
        self.demo_goal = _point(demo_goal)
        self.generation = int(generation)
        self.last_motion_generation = self.generation - 1
        self.planning_radius = float(planning_radius)
        if not math.isfinite(planning_reserve) or planning_reserve < 0:
            raise ValueError("Planning reserve must be finite and nonnegative")
        self.planning_reserve = float(planning_reserve)
        self.grid = memory.planning_grid(self.planning_radius)
        self.route_grids = {}
        self.route_planning_details = None
        self.visibility_planner = visibility_planner
        self.scene_validity_required = bool(scene_validity_required)
        self.scene_validity = {
            "enabled": self.scene_validity_required,
            "navigation_allowed": not self.scene_validity_required,
            "ready": not self.scene_validity_required,
            "invalidated": False,
            "fault_epoch": 0,
            "recovery": None,
            "timestamp": None,
            "reason": "awaiting_reference" if scene_validity_required else "disabled",
        }
        self.scene_recheck = None
        self.scene_recheck_cancel_reason = None
        self.scene_recheck_cancellation = None
        self.clock, self.heartbeat_seconds = clock, float(heartbeat_seconds)
        self.last_heartbeat = clock()
        self.parameters = parameters or OcclusionParameters()
        self.dynamics = CommandDynamics()
        self.idle_belief = CommandBelief(self.dynamics, self.parameters)
        from .braking_prediction import ConditionalBrakingRegion

        self.braking_prediction = ConditionalBrakingRegion(
            braking_acceleration_lower=self.dynamics.braking_acceleration_lower,
            response_time_upper=self.dynamics.response_time_upper,
            max_interval_seconds=self.parameters.max_interval,
        )
        self.capture_time = None
        self.localization_lost = False
        self.localization_status = "uninitialized"
        self.recovery_belief = None
        self.recovery_started = None
        self.last_seen_pose = self.last_seen_time = self.last_seen_radius = None
        self.recovery_anchor_pose = self.recovery_anchor_time = None
        self.recovery_anchor_source = None
        self._loss_reference = None
        self.requires_new_goal = False
        self.controller = None
        self.goal = self.pending_goal = None
        self.phase, self.message, self.status = (
            "localizing",
            "Waiting for a fresh RGB position.",
            "localizing",
        )

    def route_with_clearance(self, start, goal, radius):
        """Prefer cached coarse grids; retry the exact required reserve if needed."""
        if not self.scene_validity["navigation_allowed"]:
            raise ValueError("Scene/camera validity does not permit navigation")
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("Route clearance must be finite and nonnegative")
        required = max(self.planning_radius, radius + self.planning_reserve)
        if not math.isfinite(required):
            raise ValueError("Route clearance including reserve must be finite")
        # Division may put an exact boundary just above its integer index:
        # .14/.02 is 7.000000000000001. Compare the actual product, without
        # epsilon, so a nextafter value above the boundary still rounds up.
        steps = math.floor(required / 0.02)
        rounded = steps * 0.02
        if rounded < required:
            rounded = (steps + 1) * 0.02
        self.route_planning_details = {
            "required_radius_m": required,
            "rounded_radius_m": rounded,
            "selected_radius_m": rounded,
            "exact_fallback_used": False,
            "certificate_radius_m": required,
            "route_certified": False,
        }

        def grid_for(clearance):
            # Exact retries share the same bounded cache; no unbounded set of
            # covariance-dependent radii accumulates.
            if clearance not in self.route_grids:
                if len(self.route_grids) >= 16:
                    self.route_grids.pop(next(iter(self.route_grids)))
                self.route_grids[clearance] = self.memory.planning_grid(clearance)
            return self.route_grids[clearance]

        rounded_grid = grid_for(rounded)

        def plan(grid):
            if self.visibility_planner is None:
                return self.memory.certified_route(start, goal, required, grid=grid)
            try:
                return self.visibility_planner.route(
                    start, goal, grid=grid, radius=required
                )
            finally:
                self.route_planning_details["visibility"] = (
                    self.visibility_planner.diagnostics
                )

        try:
            route = plan(rounded_grid)
        except ValueError:
            if rounded == required:
                raise
            self.route_planning_details.update(
                selected_radius_m=required, exact_fallback_used=True
            )
            # Retain every millimeter of requested clearance and reserve.
            route = plan(grid_for(required))
        route = np.asarray(route, float)
        if (
            route.ndim != 2
            or route.shape[1] != 2
            or len(route) < 1
            or not np.isfinite(route).all()
            or not np.allclose(route[0], start)
            or not np.allclose(route[-1], goal)
            or not self.memory.segment_free(start, start, required)
            or not self.memory.segment_free(goal, goal, required)
            or not all(
                self.memory.segment_free(a, b, required) for a, b in pairwise(route)
            )
        ):
            raise ValueError(
                "Planned route lacks the full requested clearance and reserve"
            )
        self.route_planning_details["route_certified"] = True
        return route

    def update_scene_validity(self, record, timestamp):
        """Readmit a latch only for this session's explicit recovery ticket."""
        if not self.scene_validity_required:
            return False
        previously_allowed = self.scene_validity["navigation_allowed"]
        valid_record = (
            isinstance(record, dict)
            and type(record.get("navigation_allowed")) is bool
            and type(record.get("invalidated")) is bool
            and type(record.get("ready")) is bool
            and record.get("timestamp") == timestamp
            and math.isfinite(timestamp)
            and timestamp >= 0
            and (self.capture_time is None or timestamp > self.capture_time)
            and not (
                record["navigation_allowed"]
                and (record["invalidated"] or not record["ready"])
            )
        )
        recovery = record.get("recovery") if isinstance(record, dict) else None
        ticket = self.scene_recheck
        recovered = bool(
            valid_record
            and ticket
            and record["navigation_allowed"]
            and isinstance(recovery, dict)
            and recovery.get("status") == "succeeded"
            and type(recovery.get("generation")) is int
            and type(recovery.get("fault_epoch")) is int
            and type(recovery.get("stable_checks")) is int
            and recovery["stable_checks"] >= 3
            and recovery.get("generation") == ticket["generation"] == self.generation
            and type(record.get("fault_epoch")) is int
            and record["fault_epoch"]
            == recovery.get("fault_epoch")
            == ticket["fault_epoch"]
            and recovery.get("requested_at") == ticket["requested_at"]
            and recovery.get("completed_at") == timestamp
            and timestamp - ticket["requested_at"] >= 1.0
            and recovery.get("reference_sha256") == ticket.get("reference_sha256")
            and ticket.get("reference_sha256")
            and (self.map_version, self.calibration_version) == self.registered_versions
        )
        # Ordinary healthy or old-generation records cannot clear a latch.
        if (
            self.scene_validity["invalidated"]
            and not recovered
            and (not valid_record or not record["invalidated"])
        ):
            return False
        if recovered:
            self.scene_recheck = None
            self.cancel(
                "stopped",
                "Original scene reference restored; waiting for fresh local RGB localization and a new goal.",
            )
            self.agency.disable(timestamp, self.message)
            self.requires_new_goal = True
        self.scene_validity = (
            dict(record, enabled=True)
            if valid_record
            else {
                "enabled": True,
                "ready": False,
                "navigation_allowed": False,
                "invalidated": True,
                "fault_epoch": self.scene_validity.get("fault_epoch", 0),
                "recovery": None,
                "timestamp": timestamp,
                "reason": "invalid_or_stale_scene_evidence",
            }
        )
        if not self.scene_validity["navigation_allowed"]:
            # A suspect image cannot update the pose. Freeze the preceding
            # accepted local anchor and require the existing reacquisition gate
            # if the scene check later clears; never replay a cancelled route.
            belief = self.idle_belief
            if not self.localization_lost and belief.xy is not None:
                if (
                    belief.timestamp == self.capture_time
                    and belief.last_visual_time is not None
                    and belief.timestamp - belief.last_visual_time
                    <= self.parameters.max_occlusion_seconds
                    and belief.position_radius <= self.parameters.max_position_radius
                ):
                    self._loss_reference = (
                        belief.xy.copy(),
                        belief.timestamp,
                        "measured" if belief.measured else "predicted",
                    )
                self._lose_localization("scene evidence is not currently usable.")
            self.cancel(
                "scene_invalid"
                if self.scene_validity["invalidated"]
                else "scene_checking",
                "Scene or camera evidence changed; restore the original scene and explicitly recheck it."
                if self.scene_validity["invalidated"]
                else "Checking a stable camera reference; navigation is paused.",
            )
            self.agency.disable(timestamp, self.message)
            self.requires_new_goal = True
            if self.scene_validity["invalidated"]:
                self.localization_status = "lost"
                self.braking_prediction.invalidate("scene_invalid")
        elif not previously_allowed and not self.localization_lost:
            self.phase = self.status = "localizing"
            self.message = (
                "Scene reference established; waiting for a fresh RGB position."
            )
        return recovered

    def _cancel_scene_recheck(self, reason):
        if self.scene_recheck is not None:
            recovery = self.scene_validity.get("recovery") or {}
            self.scene_recheck_cancellation = {
                "requested_at": None,
                "stable_checks": 0,
                "reference_sha256": {},
                **recovery,
                **self.scene_recheck,
                "status": "cancelled",
                "reason": reason,
                "completed_at": None,
            }
            self.scene_validity["recovery"] = dict(self.scene_recheck_cancellation)
            self.scene_recheck = None
            self.scene_recheck_cancel_reason = reason

    def observe_scene_guard(
        self,
        guard,
        frames,
        views,
        *,
        timestamp,
        measurement,
        applied,
        intervals,
        invalidation_event=None,
    ):
        """Worker handoff, called after the post-render command drain.

        Compare pixels only after checking current authority and the complete
        stopped actuator timeline. Clearing the shared fault follows session
        admission, never merely a guard result or a lossy presentation event.
        """
        heartbeat_ok = self.heartbeat_guard()
        ticket = self.scene_recheck
        if ticket:
            try:
                stopped = (
                    heartbeat_ok
                    and ticket["generation"] == self.generation
                    and self._capture_valid(timestamp, measurement, applied, intervals)
                    and np.array_equal(applied, [0, 0])
                    and all(
                        np.array_equal(item["command"], [0, 0]) for item in intervals
                    )
                    and self.controller is None
                    and self.goal is None
                    and self.pending_goal is None
                    and not self.agency.enabled
                )
            except (KeyError, TypeError, ValueError):
                stopped = False
            if not stopped:
                self._cancel_scene_recheck(
                    "Recheck cancelled: authority, versions or stopped acknowledgements changed."
                )
            else:
                recovery = guard.snapshot().get("recovery")
                if not recovery or recovery["generation"] != ticket["generation"]:
                    try:
                        snapshot = guard.request_recheck(
                            ticket["generation"], now=timestamp
                        )
                        recovery = snapshot["recovery"]
                        ticket.update(
                            requested_at=recovery["requested_at"],
                            reference_sha256=recovery["reference_sha256"],
                        )
                    except ValueError as error:
                        self._cancel_scene_recheck(str(error))
        if self.scene_recheck is None:
            guard.cancel_recheck(
                self.scene_recheck_cancel_reason or "Recheck request no longer current."
            )
        record = guard.observe(frames, views, now=timestamp)
        # Comparison itself can consume wall time. An expired heartbeat must
        # also cancel a success produced during that comparison.
        self.heartbeat_guard()
        cancellation = self.scene_recheck_cancellation
        if cancellation and cancellation["generation"] >= (
            record.get("recovery") or {}
        ).get("generation", -1):
            if cancellation["completed_at"] is None:
                cancellation["completed_at"] = timestamp
            record["recovery"] = dict(cancellation)
        committed = self.update_scene_validity(record, timestamp)
        recovery = record.get("recovery")
        if (
            self.scene_recheck
            and recovery
            and recovery["status"] in ("rejected", "cancelled")
        ):
            self.scene_recheck = None
        if invalidation_event is not None:
            if self.scene_validity["invalidated"]:
                invalidation_event.set()
            elif committed:
                invalidation_event.clear()
        return committed

    def cancel(self, phase="stopped", message="Stopped; choose a goal to resume."):
        self.stop()
        self.controller = None
        self.goal = self.pending_goal = None
        self.phase, self.message, self.status = phase, message, phase

    def receive(self, messages, *, wall_time=None, sim_time=None):
        now = self.clock() if wall_time is None else wall_time
        records = []
        valid = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or isinstance(message.get("generation"), bool)
                or not isinstance(message.get("generation"), int)
                or message["generation"] < 0
            ):
                records.append(
                    {
                        "outcome": "invalid_command",
                        "sim_time": sim_time,
                        "wall_time": now,
                    }
                )
                continue
            record = dict(message, sim_time=sim_time, wall_time=now)
            records.append(record)
            if message.get("action") == "heartbeat":
                self.last_heartbeat = now
                record["outcome"] = "accepted"
            else:
                valid.append((message, record))
        stops = [
            (message, record)
            for message, record in valid
            if message.get("action") == "stop"
            or (message.get("action") == "autonomy" and message.get("enabled") is False)
        ]
        if stops:
            # A stop also invalidates every goal already in the drained batch.
            self.generation = max(
                self.generation, *(message["generation"] for message, _ in valid)
            )
            self.last_motion_generation = self.generation
            self._cancel_scene_recheck("Recheck cancelled by Stop.")
            self.cancel()
            self.agency.disable(sim_time or 0.0, "Autonomy paused by the user.")
            for message, record in valid:
                record["outcome"] = (
                    "accepted"
                    if message.get("action") == "stop"
                    or (
                        message.get("action") == "autonomy"
                        and message.get("enabled") is False
                    )
                    else "cancelled_by_stop"
                )
            return records
        for message, record in valid:
            action, generation = message.get("action"), message["generation"]
            if action not in ("goal", "demo", "autonomy", "recheck_scene"):
                record["outcome"] = "unsupported_worker_command"
                continue
            if action == "autonomy" and type(message.get("enabled")) is not bool:
                record["outcome"] = "invalid_command"
                continue
            if (
                generation < self.generation
                or generation <= self.last_motion_generation
            ):
                record["outcome"] = "stale_generation"
                continue
            self.generation = self.last_motion_generation = generation
            self._cancel_scene_recheck("Recheck cancelled by a newer command.")
            if action == "recheck_scene":
                epoch = message.get("fault_epoch")
                prior = self.scene_validity.get("recovery")
                can_retry = (
                    isinstance(prior, dict) and prior.get("status") == "succeeded"
                )
                if (
                    not self.scene_validity_required
                    or not self.scene_validity["ready"]
                    or not (self.scene_validity["invalidated"] or can_retry)
                    or type(epoch) is not int
                    or epoch <= 0
                    or epoch != self.scene_validity.get("fault_epoch")
                    or (self.map_version, self.calibration_version)
                    != self.registered_versions
                ):
                    record.update(
                        outcome="rejected",
                        reason="No matching original-reference fault is available for recheck.",
                    )
                    continue
                self.cancel(
                    "scene_invalid",
                    "Rechecking the original scene reference while stopped.",
                )
                self.agency.disable(sim_time or 0.0, self.message)
                self.requires_new_goal = True
                self.scene_validity.update(invalidated=True, navigation_allowed=False)
                self.scene_recheck = {"generation": generation, "fault_epoch": epoch}
                self.scene_recheck_cancel_reason = None
                self.scene_recheck_cancellation = None
                record["outcome"] = "accepted_scene_recheck"
                continue
            if not self.scene_validity["navigation_allowed"]:
                record.update(outcome="rejected", reason=self.message)
                continue
            if action == "autonomy":
                if not self.agency.available or not self.state()["localization_valid"]:
                    record.update(
                        outcome="rejected",
                        reason="Wait for available memory and localization.",
                    )
                    continue
                self.cancel("idle", "Autonomy enabled.")
                self.agency.disable(sim_time or 0.0)
                try:
                    self.agency.enable(sim_time or 0.0)
                    record["outcome"] = "accepted"
                except ValueError as error:
                    record.update(outcome="rejected", reason=str(error))
                continue
            if self.localization_lost:
                record.update(
                    outcome="rejected",
                    reason="Localization lost; wait for visual reacquisition or reset.",
                )
                continue
            self.cancel("braking", "Braking before accepting a new target.")
            self.agency.disable(sim_time or 0.0, "Manual destination selected.")
            try:
                goal = (
                    self.demo_goal.copy()
                    if action == "demo"
                    else _point([message.get("x"), message.get("y")])
                )
                if not self.memory.segment_free(goal, goal, self.planning_radius):
                    raise ValueError("Goal is occupied, unknown, or lacks clearance.")
            except (TypeError, ValueError, RuntimeError) as error:
                self.phase, self.message, self.status = (
                    "goal_rejected",
                    str(error),
                    "goal_rejected",
                )
                record["outcome"] = "rejected"
                record["reason"] = self.message
                continue
            self.pending_goal = self.goal = goal
            self.requires_new_goal = False
            record["outcome"] = "accepted_pending_visible_route"
        return records

    def heartbeat_guard(self, now=None):
        now = self.clock() if now is None else now
        if now - self.last_heartbeat > self.heartbeat_seconds:
            self._cancel_scene_recheck("Recheck cancelled by heartbeat expiry.")
            if self.phase != "heartbeat_expired":
                self.last_motion_generation = max(
                    self.last_motion_generation, self.generation
                )
                self.cancel(
                    "heartbeat_expired",
                    "Connection heartbeat expired; motion cancelled.",
                )
                self.agency.disable(
                    self.capture_time or 0.0, "Connection lost; exploration paused."
                )
            return False
        return True

    def _lose_localization(self, reason):
        if not self.localization_lost:
            # Freeze the last usable observer prediction before cancellation.
            # The robot may have moved during the permitted short blind period;
            # a gate around its older visual fix would reject that known motion.
            # This reference never follows the expanding expired prediction.
            reference = self._loss_reference
            if reference is not None:
                anchor, when, source = reference
                self.recovery_anchor_pose = anchor.copy()
                self.recovery_anchor_time, self.recovery_anchor_source = when, source
            self.cancel(
                "localization_lost",
                f"Localization lost: {reason} Goal cancelled; waiting for fresh visual observations.",
            )
            self.agency.disable(
                self.capture_time or 0.0,
                "Localization lost; exploration paused.",
                "localization_lost",
            )
            self.localization_lost = True
            self.requires_new_goal = True
            self.recovery_belief = None
            self.recovery_started = None
        self.localization_status = "lost"

    def _remember_visual(self, belief):
        self.last_seen_pose = belief.xy.copy()
        self.last_seen_time = belief.last_visual_time
        self.last_seen_radius = belief.position_radius

    def _capture_valid(self, timestamp, measurement, applied, intervals):
        from .occluded_control import CommandBelief

        try:
            if (
                not np.isfinite(timestamp)
                or timestamp < 0
                or (self.map_version, self.calibration_version)
                != self.registered_versions
                or not np.isfinite(measurement.timestamp)
                or abs(measurement.timestamp - timestamp) > 1e-9
                or (
                    measurement.status == "missing"
                    and (
                        measurement.xy is not None or measurement.covariance is not None
                    )
                )
            ):
                return False
            checker = CommandBelief(self.dynamics, self.parameters)
            checker.timestamp = self.capture_time
            checker._validated_intervals(timestamp, applied, intervals)
            return True
        except (AttributeError, TypeError, ValueError):
            return False

    def _try_reacquire(self, timestamp, measurement, applied, intervals, valid):
        """Fresh bounded RGB initialization; never resume the cancelled target."""
        from .occluded_control import CommandBelief

        p = self.parameters
        try:
            eligible = (
                valid
                and measurement.status == "visible"
                and np.array_equal(applied, [0, 0])
                and all(np.array_equal(item["command"], [0, 0]) for item in intervals)
                and np.asarray(measurement.xy).shape == (2,)
                and np.isfinite(measurement.xy).all()
                and self.recovery_anchor_pose is not None
                and np.linalg.norm(measurement.xy - self.recovery_anchor_pose)
                <= p.max_reacquisition_distance
                and self.memory.segment_free(
                    measurement.xy, measurement.xy, p.robot_radius
                )
            )
        except (TypeError, ValueError, KeyError, RuntimeError):
            eligible = False
        if not eligible:
            self.recovery_belief = self.recovery_started = None
            self.localization_status = "lost"
            if self.phase == "reacquiring":
                self.phase = self.status = "localization_lost"
                self.message = "Localization lost; waiting for consistent fresh visual observations."
            return False
        if self.recovery_belief is None:
            self.recovery_belief = CommandBelief(self.dynamics, p)
            self.recovery_started = timestamp
        candidate = self.recovery_belief
        try:
            candidate.predict(
                timestamp,
                applied,
                applied_command_intervals=[]
                if candidate.timestamp is None
                else intervals,
            )
            accepted = candidate.observe(measurement)
        except (TypeError, ValueError):
            accepted = False
        if not accepted:
            self.recovery_belief = self.recovery_started = None
            self.localization_status = "lost"
            self.phase = self.status = "localization_lost"
            self.message = (
                "Localization lost; returning observation did not pass validation."
            )
            return False
        self.localization_status = "reacquiring"
        self.phase = self.status = "reacquiring"
        self.message = "Reacquiring position from fresh RGB observations. Motion remains cancelled."
        if (
            candidate.samples >= p.minimum_visual_samples
            and timestamp - self.recovery_started >= p.max_frame_age
            and candidate.velocity_radius <= p.max_initial_velocity_radius
        ):
            self.idle_belief = candidate
            self._remember_visual(candidate)
            self.localization_lost = False
            self.localization_status = "measured"
            self.recovery_belief = self.recovery_started = None
            self.phase = self.status = "stopped"
            self.message = "Position reacquired; choose a new destination to resume."
            return True
        return False

    def decide(self, timestamp, measurement, applied, intervals, *, wall_time=None):
        if self.scene_validity_required:
            if self.scene_validity.get("timestamp") != timestamp:
                self.update_scene_validity(None, timestamp)
            if not self.scene_validity["navigation_allowed"]:
                self.heartbeat_guard(wall_time)
                self.capture_time = timestamp
                self.cancel(
                    "scene_invalid"
                    if self.scene_validity["invalidated"]
                    else "scene_checking",
                    "Scene or camera evidence changed; restore the original scene and explicitly recheck it."
                    if self.scene_validity["invalidated"]
                    else "Checking a stable camera reference; navigation is paused.",
                )
                self.agency.disable(timestamp, self.message)
                return np.zeros(2, np.float32)
        action = self._decide(
            timestamp, measurement, applied, intervals, wall_time=wall_time
        )
        if (self.map_version, self.calibration_version) != self.registered_versions:
            self.agency.disable(
                timestamp, "Map or calibration changed; exploration paused."
            )
        if self.agency.tick(self, timestamp):
            return np.zeros(2)
        return action

    def _decide(self, timestamp, measurement, applied, intervals, *, wall_time=None):
        from .occluded_control import OccludedController

        # Capture only the PREVIOUS accepted state, before this frame can change
        # the observer. A returning detection must not nominate its own gate.
        b, p = self.idle_belief, self.parameters
        if (
            b.xy is not None
            and b.timestamp == self.capture_time
            and b.last_visual_time is not None
            and b.timestamp - b.last_visual_time <= p.max_occlusion_seconds + 1e-9
            and np.isfinite(b.position_radius)
            and b.position_radius <= p.max_position_radius
        ):
            self._loss_reference = (
                b.xy.copy(),
                b.timestamp,
                "measured" if b.measured else "predicted",
            )
        if (self.map_version, self.calibration_version) != self.registered_versions:
            self._loss_reference = None
            self.recovery_anchor_pose = self.recovery_anchor_time = None
            self.recovery_anchor_source = None
        self.heartbeat_guard(wall_time)
        inputs_valid = self._capture_valid(timestamp, measurement, applied, intervals)
        prior_visual_times = [self.last_seen_time]
        if self.controller is not None:
            prior_visual_times.append(self.controller.belief.last_visual_time)
        if np.isfinite(timestamp) and any(
            seen is not None
            and timestamp - seen > self.parameters.max_occlusion_seconds + 1e-9
            for seen in prior_visual_times
        ):
            # A returning frame cannot erase the loss interval before the old
            # goal has been cancelled and a fresh recovery track is initialized.
            self._lose_localization(
                "the interval since the last visual fix exceeded the prediction limit."
            )
        previous = self.idle_belief
        snapshot = {
            "timestamp": previous.timestamp,
            "xy": previous.xy,
            "velocity": previous.velocity,
            "position_radius": previous.position_radius,
            "velocity_radius": previous.velocity_radius,
            "map_version": self.map_version,
            "calibration_version": self.calibration_version,
        }
        pending_nonzero = self.controller is not None and any(
            entry[1] > timestamp and entry[2] > 0
            for entry in self.controller.issued_history
        )
        self.braking_prediction.update(
            previous_snapshot=snapshot,
            timestamp=timestamp,
            acknowledged_intervals=intervals,
            endpoint_command=applied,
            map_version=self.map_version,
            calibration_version=self.calibration_version,
            pending_nonzero=pending_nonzero,
            inputs_valid=inputs_valid,
        )
        if np.isfinite(timestamp) and (
            self.capture_time is None or timestamp > self.capture_time
        ):
            self.capture_time = float(timestamp)
        try:
            if not inputs_valid:
                raise ValueError("Invalid capture or command timeline")
            history = [] if self.idle_belief.timestamp is None else intervals
            self.idle_belief.predict(
                timestamp, applied, applied_command_intervals=history
            )
            self.idle_belief.observe(measurement)
        except (TypeError, ValueError):
            self.idle_belief.measured = False
            if self.last_seen_time is not None:
                self._lose_localization("invalid prediction input.")
        belief = self.idle_belief
        if self.last_seen_time is not None and (
            not inputs_valid
            or timestamp - belief.last_visual_time
            > self.parameters.max_occlusion_seconds + 1e-9
            or belief.position_radius > self.parameters.max_position_radius
        ):
            self._lose_localization("the visual prediction has expired.")
        if self.localization_lost:
            self._try_reacquire(
                timestamp, measurement, applied, intervals, inputs_valid
            )
            return np.zeros(2, np.float32)
        if not inputs_valid:
            self.braking_prediction.invalidate("invalid_capture")
            return np.zeros(2, np.float32)
        self.localization_status = (
            "measured"
            if belief.measured
            else "predicted"
            if belief.xy is not None
            else "uninitialized"
        )
        if belief.measured:
            self._remember_visual(belief)
        fresh_visible = (
            measurement.status == "visible"
            and abs(measurement.timestamp - timestamp) <= 1e-9
            and self.idle_belief.measured
        )
        if self.phase in ("idle", "localizing"):
            self.phase = self.status = "idle" if fresh_visible else "localizing"
            self.message = (
                "Choose a clear goal or Demo."
                if fresh_visible
                else "Waiting for a fresh RGB position."
            )
        if self.pending_goal is not None and fresh_visible:
            try:
                radius = max(
                    self.planning_radius,
                    self.parameters.robot_radius
                    + self.parameters.clearance_margin
                    + self.idle_belief.position_radius,
                )
                route = np.asarray(
                    self.route_with_clearance(
                        measurement.xy, self.pending_goal, radius
                    ),
                    float,
                )
                if (
                    route.ndim != 2
                    or route.shape[1] != 2
                    or len(route) < 1
                    or not np.isfinite(route).all()
                    or not np.allclose(route[0], measurement.xy)
                    or not np.allclose(route[-1], self.pending_goal)
                    or not self.memory.segment_free(
                        measurement.xy, measurement.xy, radius
                    )
                    or not all(
                        self.memory.segment_free(a, b, radius)
                        for a, b in pairwise(route)
                    )
                ):
                    raise ValueError(
                        "No certified route from the current RGB position."
                    )
                self.controller = OccludedController(
                    self.policy,
                    self.pending_goal,
                    self.grid,
                    self.memory.segment_free,
                    map_version=self.map_version,
                    calibration_version=self.calibration_version,
                    route_planner=self.route_with_clearance,
                    parameters=self.parameters,
                )
                self.pending_goal = None
                self.message = (
                    "RGB navigation; physics pauses during capture and inference."
                )
            except (TypeError, ValueError, RuntimeError) as error:
                self.cancel("goal_rejected", str(error))
        if self.controller is None:
            return np.zeros(2, np.float32)
        history = [] if self.controller.belief.timestamp is None else intervals
        action = self.controller.action(
            timestamp,
            measurement,
            applied_command=applied,
            applied_command_intervals=history,
            map_version=self.map_version,
            calibration_version=self.calibration_version,
            now=timestamp,
        )
        active_belief = self.controller.belief
        if active_belief.last_visual_time is not None and (
            timestamp - active_belief.last_visual_time
            > self.parameters.max_occlusion_seconds + 1e-9
            or active_belief.position_radius > self.parameters.max_position_radius
        ):
            self._lose_localization(
                "the active controller's position envelope expired."
            )
            return np.zeros(2, np.float32)
        if np.any(action != 0):
            self.braking_prediction.invalidate("new_nonzero_request")
        self.status = self.controller.status
        self.message = {
            "route_not_certified": (
                "The current route lacks clearance for BB-8 and its position "
                "uncertainty. Stopped pending a new route from a visible estimate."
            ),
            "no_proven_route_memory": (
                "No clear route is available from the current camera estimate. "
                "Try a more open destination or another camera mode."
            ),
            "stopping_envelope_not_certified": (
                "Stopped: the next motion lacks stopping clearance. "
                "Try a more open destination or another camera mode."
            ),
            "proactive_envelope_not_certified": (
                "Stopped early to retain stopping clearance along the route."
            ),
        }.get(
            self.status,
            "RGB navigation; physics pauses during capture and inference.",
        )
        self.phase = (
            "arrived"
            if self.controller.arrived
            else "braking"
            if self.status == "warming_up"
            else "running"
            if self.status
            in ("tracking_visible", "tracking_hidden", "settling_visible")
            else "guarded_stop"
        )
        return action

    def state(self):
        # The persistent observer owns published pose, source and last-seen
        # metadata. A newly created controller can accept different innovations
        # during its separate velocity warm-up; its belief stays in diagnostics
        # and its stricter expiry is still enforced in decide().
        belief = self.idle_belief
        valid = (
            self.scene_validity["navigation_allowed"]
            and self.localization_status in ("measured", "predicted")
            and not self.localization_lost
            and belief.xy is not None
            and belief.timestamp == self.capture_time
            and belief.last_visual_time is not None
            and self.capture_time - belief.last_visual_time
            <= self.parameters.max_occlusion_seconds + 1e-9
            and np.isfinite(belief.position_radius)
            and belief.position_radius <= self.parameters.max_position_radius
        )
        return {
            "scene_validity": dict(self.scene_validity),
            "visibility_planning": self.visibility_planner is not None,
            "agency": self.agency.snapshot(),
            "route_planning": self.route_planning_details,
            "phase": self.phase,
            "localization_status": self.localization_status,
            "localization_valid": valid,
            "requires_new_goal": self.requires_new_goal,
            "last_seen_pose": None
            if self.last_seen_pose is None
            else self.last_seen_pose.tolist(),
            "last_seen_age_s": None
            if self.last_seen_time is None
            else max(0.0, self.capture_time - self.last_seen_time),
            "last_seen_radius_m": self.last_seen_radius,
            "recovery_anchor_pose": None
            if self.recovery_anchor_pose is None
            else self.recovery_anchor_pose.tolist(),
            "recovery_anchor_time": self.recovery_anchor_time,
            "recovery_anchor_source": self.recovery_anchor_source,
            "reacquisition_samples": 0
            if self.recovery_belief is None
            else self.recovery_belief.samples,
            "raw_position_radius": float(belief.position_radius)
            if np.isfinite(belief.position_radius)
            else None,
            "raw_prediction_time": belief.timestamp,
            "raw_pose": None if belief.xy is None else belief.xy.tolist(),
            "raw_velocity": None
            if belief.velocity is None
            else belief.velocity.tolist(),
            "raw_velocity_radius": float(belief.velocity_radius)
            if np.isfinite(belief.velocity_radius)
            else None,
            "braking_prediction": self.braking_prediction.record,
            "pose": None if not valid or belief.xy is None else belief.xy.tolist(),
            "position_radius": float(belief.position_radius)
            if valid and np.isfinite(belief.position_radius)
            else None,
            "pose_source": "unavailable"
            if not valid
            else "measured"
            if belief.measured
            else "predicted"
            if belief.xy is not None
            else "uninitialized",
            "goal": None if self.goal is None else self.goal.tolist(),
            "commanded_goal": None if self.goal is None else self.goal.tolist(),
            "route": None
            if self.controller is None or self.controller.route is None
            else self.controller.route.tolist(),
            "controller_status": self.status,
            "message": self.message,
            "generation": self.generation,
        }


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_main(
    config, command_queue, event_queue, stop_event, invalidation_event=None
):
    """Multiprocessing spawn entry; all native/model imports and lifetime live here.

    Required config: asset_dir, mode (1/2/3), run_dir. Optional generation,
    max_steps, save_frames, save_every, scoring_labels, jpeg_width, jpeg_quality,
    camera_dropouts ({ID:[[sim_start,sim_end],...]}), invalidate_memory_at,
    recording_mode (telemetry default, audit for complete physics evidence).
    The last two are explicit test interventions, never geometric oracle input.
    """
    # Presentation events are deliberately lossy; durable audit evidence is
    # finalized on disk. Joining a feeder after the supervisor detaches its
    # reader can otherwise keep an already-finished native process alive.
    cancel_join = getattr(event_queue, "cancel_join_thread", None)
    if cancel_join is not None:
        cancel_join()
    env = session = agency_store = None
    manifest = None
    recorder = None
    run_dir = None
    try:
        mode = config.get("mode", 1)
        if isinstance(mode, bool) or mode not in (1, 2, 3):
            raise ValueError("Camera mode must be 1, 2 or 3")
        save_every = config.get("save_every", 1)
        if type(save_every) is not int or save_every < 1:
            raise ValueError("save_every must be a positive integer")
        asset_dir, run_dir = (
            Path(config["asset_dir"]).resolve(),
            Path(config["run_dir"]).resolve(),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        if any(
            (run_dir / name).exists()
            for name in ("rows.jsonl", "manifest.json", "recording.json")
        ):
            raise ValueError("Use a fresh run directory for each worker generation")
        from .session_recording import SessionRecorder

        recorder = SessionRecorder(
            run_dir,
            mode=config.get("recording_mode", "telemetry"),
            segment_bytes=config.get("log_segment_bytes", 1_048_576),
            segments=config.get("log_segments", 4),
        )
        if config.get("save_frames") and recorder.mode != "audit":
            raise ValueError("Saving every frame requires explicit audit recording")
        from .demo_assets import validate_assets

        demo = validate_assets(asset_dir)
        bundle = json.loads((asset_dir / "bundle.json").read_text())
        from .control_profiles import get_profile

        profile = get_profile(config.get("control_profile"))

        def asset(name):
            path = (asset_dir / demo[name]).resolve()
            if not path.is_relative_to(asset_dir):
                raise ValueError("Demo assets must stay inside the asset bundle")
            return path

        # Task overrides must not bypass the checked dependency graph. To use a
        # different world, supply a fresh bundle with that task in demo.json.
        if config.get("task_path") and Path(config["task_path"]).resolve() != asset(
            "task"
        ):
            raise ValueError("Task override must match the checksummed demo task")
        if (
            len(
                {
                    camera_pixel_coordinates(record)
                    for record in demo["cameras"].values()
                }
            )
            != 1
        ):
            raise ValueError("Camera bundle mixes pixel coordinate conventions")

        # Heavy/native imports are intentionally below config/asset validation.
        import cv2
        import torch

        from .camera import Calibration, VisualMeasurement
        from .camera_rig import (
            CameraFrame,
            CameraRig,
            CameraView,
            scale_intrinsics,
        )
        from .env import NavigationEnv
        from .guided import GuidedSAC
        from .mapping.scan_free_space import ScanFreeMemory
        from .vision import LearnedObserver, PositionOnlyObserver

        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        memory = ScanFreeMemory.load(asset("memory"))
        map_version = _digest(asset("memory") / "manifest.json")
        import sqlite3

        from .agency import AgencyEngine, AgencyStore
        from .interaction_environment import DEFAULT_STATIONS, InteractionEnvironment
        from .purpose import PurposeEngine, PurposeStore
        from .purpose_runtime import PurposeCoordinator

        agency_mode = config.get("agency_mode", "purpose")
        if agency_mode not in ("purpose", "coverage", "visual"):
            raise ValueError("Agency mode must be purpose, coverage or visual")
        visual_fixture = visual_observer = visual_calibration = None
        visual_version = None
        if agency_mode == "visual":
            from .purpose import Station
            from .visual_fixture import DEFAULT_PANELS, VisualCamera, VisualFixture
            from .visual_interaction import INTERACTION_OFFSET, PixelVisualObserver
            from .visual_purpose_runtime import VisualPurposeCoordinator

            if not config.get("agency_memory"):
                raise ValueError(
                    "Visual learning requires a separate persistent memory"
                )
            if (
                any(
                    key in config
                    for key in (
                        "visual_effects",
                        "visual_initial_resource",
                        "visual_dropouts",
                        "visual_hide_after_completion",
                    )
                )
                and recorder.mode != "audit"
            ):
                raise ValueError("Visual evaluation overrides require audit recording")
            record = demo["cameras"]["B"]
            coordinates = camera_pixel_coordinates(record)
            intrinsics = scale_intrinsics(
                record["intrinsics"], 2, pixel_coordinates=coordinates
            )
            resolution = tuple(2 * value for value in record["resolution"])
            visual_version = _digest(asset_dir / "demo.json") + ":visual-B-scaled2-v2"
            if coordinates != "genesis_viewport":
                visual_version += ":" + coordinates
            visual_calibration = Calibration(
                intrinsics,
                np.asarray(record["world_to_camera"]),
                resolution,
                float(memory.extent),
                provenance="RGB-estimated registered pose; scaled synthetic intrinsics",
            )
            visual_fixture = VisualFixture(
                initial_resource=config.get("visual_initial_resource", 0.25),
                camera_specs=(
                    VisualCamera(
                        "V",
                        tuple(record["position"]),
                        tuple(record["lookat"]),
                        resolution,
                    ),
                ),
            )
            visual_observer = PixelVisualObserver(
                visual_calibration, "V", visual_version
            )
            world_stations = tuple(
                Station(
                    panel.visual_id,
                    tuple(
                        a + b
                        for a, b in zip(
                            panel.center_xy, INTERACTION_OFFSET, strict=True
                        )
                    ),
                    panel.visual_id,
                )
                for panel in DEFAULT_PANELS
            )
        interaction_environment = (
            InteractionEnvironment() if agency_mode == "purpose" else None
        )
        if agency_mode == "visual":
            interaction_environment = InteractionEnvironment(
                world_stations,
                initial_resource=config.get("visual_initial_resource", 0.25),
                effects=config.get(
                    "visual_effects", {"marker-01": 0.0, "marker-02": 0.45}
                ),
            )
        agency_engine = None
        agency_error = None
        if config.get("agency_memory"):
            try:
                # Scope experience to all checked map bytes, including raster
                # changes that do not alter descriptive manifest fields.
                agency_map_version = hashlib.sha256(
                    json.dumps(
                        {
                            name: digest
                            for name, digest in bundle["sha256"].items()
                            if name.startswith(demo["memory"].rstrip("/") + "/")
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                if agency_mode == "visual":
                    # Scope to observable map/calibration/protocol, never hidden effects.
                    agency_map_version += (
                        ":visual-v2:"
                        + hashlib.sha256(visual_version.encode()).hexdigest()
                    )
                    agency_store = PurposeStore(
                        config["agency_memory"], map_version=agency_map_version
                    )
                    agency_engine = PurposeEngine(agency_store)
                elif agency_mode == "purpose":
                    # Fixture identities/coordinates also scope outcome memory.
                    agency_map_version += (
                        ":stations-v1:"
                        + hashlib.sha256(
                            repr([(s.id, s.xy) for s in DEFAULT_STATIONS]).encode()
                        ).hexdigest()
                    )
                    agency_store = PurposeStore(
                        config["agency_memory"], map_version=agency_map_version
                    )
                    agency_engine = PurposeEngine(agency_store)
                else:
                    agency_store = AgencyStore(
                        config["agency_memory"], map_version=agency_map_version
                    )
                    agency_engine = AgencyEngine(agency_store)
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                agency_error = str(error)
        if agency_mode == "visual" and agency_engine is None:
            raise ValueError(f"Visual memory unavailable: {agency_error}")
        camera_ids = tuple("ABC"[:mode])
        camera_changes = evaluation_camera_changes(config, camera_ids)
        cameras = demo["cameras"]
        calibration_version = (
            _digest(asset_dir / "demo.json") + ":" + "".join(camera_ids)
        )
        calibrations, views = {}, []
        for name in camera_ids:
            record = cameras[name]
            calibration = Calibration(
                np.asarray(record["intrinsics"]),
                np.asarray(record["world_to_camera"]),
                tuple(record["resolution"]),
                float(memory.extent),
                provenance="RGB-estimated registered pose; synthetic intrinsics and metric scan frame",
            )
            calibrations[name] = calibration
            observer = LearnedObserver(
                calibration, asset("vision"), floor_refinement="guided"
            )
            views.append(
                CameraView(
                    name,
                    record["calibration_version"],
                    calibration,
                    PositionOnlyObserver(observer),
                )
            )
        rig = CameraRig(views)
        from .scene_validity import SceneValidityGuard
        from .visibility_planning import VisibilityPlanner

        for option in ("visibility_planning", "scene_validity"):
            if type(config.get(option, False)) is not bool:
                raise ValueError(f"{option} must be an explicit boolean")
        visibility_planner = (
            VisibilityPlanner(
                memory,
                calibrations,
                max_blind_distance_m=0.10,
                reference_speed_m_s=0.10,
            )
            if config.get("visibility_planning", False)
            else None
        )
        validity_guard = (
            SceneValidityGuard(
                calibrations,
                {name: cameras[name]["calibration_version"] for name in camera_ids},
            )
            if config.get("scene_validity", False)
            else None
        )
        policy = GuidedSAC.load(asset("policy"), device="cpu")
        task_path = asset("task")
        sources = [
            Path(__file__),
            Path(__file__).with_name("agency.py"),
            Path(__file__).with_name("agency_runtime.py"),
            Path(__file__).with_name("purpose.py"),
            Path(__file__).with_name("purpose_runtime.py"),
            Path(__file__).with_name("visual_interaction.py"),
            Path(__file__).with_name("visual_purpose_runtime.py"),
            Path(__file__).with_name("visual_fixture.py"),
            Path(__file__).with_name("visual_evidence.py"),
            Path(__file__).with_name("interaction_environment.py"),
            Path(__file__).with_name("braking_prediction.py"),
            Path(__file__).with_name("occluded_control.py"),
            Path(__file__).with_name("control_profiles.py"),
            Path(__file__).with_name("camera.py"),
            Path(__file__).with_name("camera_rig.py"),
            Path(__file__).with_name("vision.py"),
            Path(__file__).with_name("visibility_planning.py"),
            Path(__file__).with_name("scene_validity.py"),
            Path(__file__).parent / "control/genesis_backend.py",
            Path(__file__).parent / "mapping/scan_free_space.py",
        ]
        (run_dir / "source").mkdir(exist_ok=True)
        for source in sources:
            (run_dir / "source" / source.name).write_bytes(source.read_bytes())
        manifest = {
            "schema": "bb8.interactive-run.v2",
            "status": "running",
            "mode": mode,
            "camera_ids": camera_ids,
            "render_camera_up": {
                name: cameras[name].get("up", [0, 0, 1]) for name in camera_ids
            },
            "scope": "Interactive lockstep synthetic development session",
            "truth_to_controller": False,
            "segmentation_to_controller": False,
            "control_inputs": [
                "accepted RGB rig position/covariance/status",
                "RGB-estimated fixed calibrations",
                "frozen scan evidence",
                "user goal",
                "optional learned station-outcome chooser (coverage diagnostic available)",
                "native fixture RGB identity/anchor/gauge evidence and completion-only correlation in visual mode"
                if agency_mode == "visual"
                else "explicit simulated resource and station-response telemetry in purpose mode",
                "command acknowledgements and times",
                "heartbeat/generation/stop",
            ],
            "wall_time_note": "Physics pauses during capture, inference and audit; not real-time performance.",
            "asset_dir": str(asset_dir),
            "asset_sha256": {
                "demo.json": _digest(asset_dir / "demo.json"),
                demo["policy"]: _digest(asset("policy")),
                demo["vision"]: _digest(asset("vision")),
                "memory/manifest.json": map_version,
            },
            "assets": {
                name: {"path": str(asset_dir / name), "sha256": digest}
                for name, digest in bundle["sha256"].items()
            },
            "source_sha256": {source.name: _digest(source) for source in sources},
            "config": jsonable(config),
            "control_profile": profile.record(),
            "map_version": map_version,
            "calibration_version": calibration_version,
            "navigation_safeguards": {
                "visibility_planning": visibility_planner is not None,
                "scene_validity": validity_guard is not None,
                "scope": "Experimental geometric visibility heuristic and same-session RGB change guard; neither certifies the frozen map at startup",
            },
            "generation": int(config.get("generation", 0)),
            "goal": demo["goal"],
            "completed_steps": 0,
            "recording": recorder.summary(),
            "agency_mode": agency_mode,
            "interaction_fixture": {
                "scope": "Virtual task zones; not RGB-recognized objects or rendered station responses",
                "stations": [{"id": s.id, "xy": s.xy} for s in DEFAULT_STATIONS],
                "radius_m": 0.14,
                "dwell_s": 1.0,
                "speed_limit_m_s": 0.03,
                "initial_resource": 0.35,
                "target_resource": 0.8,
                "source": "simulated_station_telemetry",
            }
            if interaction_environment is not None
            else None,
        }
        if agency_mode == "visual":
            manifest["interaction_fixture"] = {
                "scope": "Fixture-specific native RGB panels; world-side synthetic effects, not generic object understanding",
                "rendering": visual_fixture.rendering_contract(),
                "source": "native_scene_rgb",
                "visual_camera": {
                    "id": "V",
                    "intrinsics": visual_calibration.intrinsics,
                    "world_to_camera": visual_calibration.world_to_camera,
                    "resolution": visual_calibration.resolution,
                    "calibration_version": visual_version,
                },
                "world_effects_scoring_only": dict(interaction_environment._effects),
                "initial_resource_scoring_only": interaction_environment.resource,
            }
        (run_dir / "manifest.json").write_text(
            json.dumps(jsonable(manifest), indent=2) + "\n"
        )
        evaluation_split = config.get("evaluation_split", "validation")
        if evaluation_split not in {"validation", "heldout"} or (
            evaluation_split == "heldout" and recorder.mode != "audit"
        ):
            raise ValueError("Held-out fixtures require explicit audit recording")
        env = NavigationEnv(
            task_path,
            split=evaluation_split,
            render_mode="rgb_array",
            camera_positions=[cameras[name]["position"] for name in "ABC"],
            visual_fixture=visual_fixture,
        )
        env.task.max_episode_steps = int(config.get("max_steps", 1_000_000))
        env.reset(
            seed=int(demo["seed"]),
            options={
                "layout_seed": int(demo["layout_seed"]),
                "start": demo["start"],
                "goal": demo["goal"],
            },
        )
        from .run_identity import capture_run_identity

        manifest["run_identity"] = capture_run_identity(
            run_dir / "identity",
            asset_dir=asset_dir,
            demo=demo,
            task_path=task_path,
        )
        # Record effective setup after reset as well as the on-disk dependency
        # graph. These are provenance/scoring data, never controller inputs.
        loaded_setup = {
            "task": env.task.model_dump(mode="json"),
            "world": env.config.model_dump(mode="json"),
            "project": env.world.project.model_dump(mode="json"),
            "layout": {
                "seed": env.layout.seed,
                "kind": env.layout.kind,
                "positions": env.layout.positions,
            },
            "reset_start": demo["start"],
            "reset_goal": demo["goal"],
            "camera_fixture": cameras,
            "scope": "Effective synthetic setup; not observations supplied to control",
        }
        loaded_path = run_dir / "loaded-setup.json"
        loaded_path.write_text(json.dumps(jsonable(loaded_setup), indent=2) + "\n")
        manifest["loaded_setup"] = {
            "path": loaded_path.name,
            "sha256": _digest(loaded_path),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(jsonable(manifest), indent=2) + "\n"
        )
        if not math.isclose(env.config.action_steps * env.world.dt, 0.05, abs_tol=1e-9):
            raise ValueError(
                "Interactive controller requires the existing 50 ms action period"
            )
        env.world.camera_period = env.world._next_frame = 1e9
        for name in "ABC":
            record, camera = cameras[name], env.world.cameras[name]
            set_render_camera_pose(camera, record)
            # Render geometry is fixture setup only; never replace estimated W2C.
            validate_render_camera_intrinsics(camera, record, memory.extent)
        if visual_fixture is not None:
            validate_render_camera_intrinsics(
                visual_fixture.cameras["V"], cameras["B"], memory.extent, scale=2
            )
        session = ControlSession(
            policy=lambda vector: policy.predict(vector, deterministic=True)[0],
            memory=memory,
            stop=env.world.backend.stop,
            map_version=map_version,
            calibration_version=calibration_version,
            demo_goal=demo["goal"],
            generation=config.get("generation", 0),
            planning_radius=demo.get("planning_radius", 0.10),
            parameters=profile.parameters(),
            planning_reserve=profile.planning_reserve_m,
            agency_engine=agency_engine,
            agency_coordinator=(
                VisualPurposeCoordinator(agency_engine, visual_version)
                if agency_mode == "visual"
                else PurposeCoordinator(agency_engine, DEFAULT_STATIONS)
                if agency_mode == "purpose"
                else None
            ),
            visibility_planner=visibility_planner,
            scene_validity_required=validity_guard is not None,
        )
        if agency_error:
            session.agency.message = f"Memory unavailable: {agency_error}"
        # World-side mechanics produce a resource sensor observation. Kinematics
        # are never passed to the chooser; its navigation pose comes from RGB.
        interaction_observation = None
        if interaction_environment is not None:
            interaction_observation = interaction_environment.advance(
                now=env.world.time,
                position=env.state.position[:2],
                velocity=env.state.velocity[:2],
                request=None,
            )
        command_trace, physics_trace, pending = [], [], []
        original_advance, original_step = env.world.backend.advance, env.world.step

        def acknowledged_advance(*, now):
            result = original_advance(now=now)
            pending.append(
                {
                    "start": now,
                    "end": now + env.world.dt,
                    "command": list(env.world.backend.applied_request),
                }
            )
            return result

        interaction_request = None

        def mechanics_and_scoring_step(*args, **kwargs):
            nonlocal interaction_observation
            before = list(env.world.backend.applied_request)
            pending.clear()
            state = original_step(*args, **kwargs)
            if interaction_environment is not None:
                interaction_observation = interaction_environment.advance(
                    now=env.world.time,
                    position=state.position[:2],
                    velocity=state.velocity[:2],
                    request=interaction_request,
                )
            command_trace.extend(pending)
            physics_trace.append(
                {
                    "time": env.world.time,
                    "position": list(state.position),
                    "velocity": list(state.velocity),
                    "command_active_before_tick": before,
                    "command_active_after_tick": list(
                        env.world.backend.applied_request
                    ),
                }
            )
            return state

        env.world.backend.advance, env.world.step = (
            acknowledged_advance,
            mechanics_and_scoring_step,
        )
        labels = env.world.scene.visualizer.segmentation_idx_dict
        head_ids = [
            key
            for key, value in labels.items()
            if isinstance(value, tuple) and value[0] == env.world.head.idx
        ]
        intervals = []
        previous_scene_recovery = None
        save_frames, score_labels = (
            bool(config.get("save_frames", False)),
            bool(config.get("scoring_labels", recorder.mode == "audit")),
        )
        visual_evidence = None
        if visual_fixture is not None:
            from .visual_evidence import VisualEvidenceArchive

            visual_evidence = VisualEvidenceArchive(
                run_dir, audit=recorder.mode == "audit"
            )
            session.agency.evidence_retainer = visual_evidence.retain_outcome
        for step in range(env.task.max_episode_steps):
            if stop_event.is_set():
                session.cancel("stopped", "Worker stopped by supervisor.")
                break
            started = time.monotonic()
            timestamp = env.world.time

            def receive_pending(capture_time=timestamp):
                for record in session.receive(
                    drain_commands(command_queue), sim_time=capture_time
                ):
                    recorder.command(jsonable(record))

            receive_pending()
            session.heartbeat_guard()
            while camera_changes and timestamp + 1e-9 >= camera_changes[0]["at_time"]:
                change = camera_changes.pop(0)
                # This changes the evaluation renderer, not a body pose or the
                # registered calibration. The RGB guard receives only images.
                set_render_camera_pose(env.world.cameras[change["camera_id"]], change)
                recorder.event("evaluation_camera_change", time=timestamp, **change)
            frames, rgbs, segmentations = [], {}, {}
            for name in camera_ids:
                rendered = env.world.cameras[name].render(
                    rgb=True, segmentation=score_labels, force_render=True
                )
                rgbs[name] = np.ascontiguousarray(rendered[0])
                if score_labels:
                    segmentations[name] = rendered[2]
                dropped = any(
                    a <= timestamp < b
                    for a, b in config.get("camera_dropouts", {}).get(name, [])
                )
                pixels = np.zeros_like(rgbs[name]) if dropped else rgbs[name]
                frames.append(
                    CameraFrame(
                        name, cameras[name]["calibration_version"], pixels, timestamp
                    )
                )
            visual_scene = visual_artifact = None
            if visual_fixture is not None:
                # Numeric mechanics reach the renderer only; the learner sees RGB.
                visual_fixture.set_resource(interaction_environment.resource)
                if (
                    config.get("visual_hide_after_completion", False)
                    and (interaction_observation or {}).get("outcome") is not None
                ):
                    visual_fixture.set_visible_panels(())
                visual_rgb = np.ascontiguousarray(
                    visual_fixture.cameras["V"].render(rgb=True, force_render=True)[0]
                )
                if any(
                    a <= timestamp < b for a, b in config.get("visual_dropouts", [])
                ):
                    visual_rgb = np.zeros_like(visual_rgb)
                visual_scene = visual_observer.observe(
                    CameraFrame("V", visual_version, visual_rgb, timestamp),
                    now=timestamp,
                )
                visual_artifact = visual_evidence.capture(
                    visual_rgb, step=step, timestamp=timestamp
                )
            observation = rig.observe(frames, now=timestamp)
            empty = np.zeros((0, 0), bool)
            measurement = VisualMeasurement(
                observation.timestamp,
                observation.xy,
                observation.covariance,
                empty,
                empty,
                observation.status,
            )
            # Process STOP/new generations received during rendering/inference
            # before selecting any drive request. Heartbeat is checked again.
            receive_pending()
            if stop_event.is_set():
                session.cancel("stopped", "Worker stopped by supervisor.")
                break
            changed = config.get("invalidate_memory_at")
            if changed is not None and timestamp >= changed:
                session.map_version = map_version + ":invalidated"
                session.cancel("guarded_stop", "Map validity changed; reset required.")
            applied = np.asarray(env.world.backend.applied_request)
            if validity_guard is not None:
                session.observe_scene_guard(
                    validity_guard,
                    frames,
                    observation.views,
                    timestamp=timestamp,
                    measurement=measurement,
                    applied=applied,
                    intervals=intervals,
                    invalidation_event=invalidation_event,
                )
                recovery = session.scene_validity.get("recovery")
                recovery_key = (
                    None
                    if recovery is None
                    else (
                        recovery.get("generation"),
                        recovery.get("fault_epoch"),
                        recovery.get("status"),
                        recovery.get("stable_checks"),
                    )
                )
                if recovery_key is not None and recovery_key != previous_scene_recovery:
                    recorder.event(
                        "scene_reference_recheck",
                        time=timestamp,
                        **jsonable(recovery),
                        navigation_allowed=session.scene_validity["navigation_allowed"],
                        requires_new_goal=session.requires_new_goal,
                        shared_fault_set=None
                        if invalidation_event is None
                        else invalidation_event.is_set(),
                    )
                previous_scene_recovery = recovery_key
            if agency_mode == "visual":
                receipt = (interaction_observation or {}).get("outcome")
                completion = (
                    None
                    if receipt is None
                    else session.agency.completion_for(
                        {
                            key: receipt[key]
                            for key in ("request_id", "station_id", "timestamp")
                        },
                        session.generation,
                    )
                )
                session.agency.feed_visual(visual_scene, completion)
            elif interaction_observation is not None:
                session.agency.observe(interaction_observation)
            action = session.decide(timestamp, measurement, applied, intervals)
            decision_finished = time.monotonic()
            diagnostics = (
                None if session.controller is None else session.controller.diagnostics
            )
            state = dict(
                session.state(),
                mode=mode,
                sim_time=timestamp,
                source_ids=list(observation.source_ids),
                views={
                    name: {"status": view.status}
                    for name, view in observation.views.items()
                },
                frame_seq=step,
                processing_ms=(decision_finished - started) * 1000,
            )
            # Action is frozen here. All following truth/label reads are scoring.
            head_counts, artifacts = {}, {}
            for name in camera_ids:
                mask = (
                    np.isin(segmentations[name], head_ids)
                    if name in segmentations
                    else None
                )
                head_counts[name] = None if mask is None else int(mask.sum())
                if save_frames and step % save_every == 0:
                    rgb_path = run_dir / f"frame-{step:06d}-{name}-rgb.png"
                    cv2.imwrite(
                        str(rgb_path), cv2.cvtColor(rgbs[name], cv2.COLOR_RGB2BGR)
                    )
                    artifacts[name] = {
                        "rgb_path": rgb_path.name,
                        "rgb_sha256": _digest(rgb_path),
                    }
                    if mask is not None:
                        mask_path = run_dir / f"frame-{step:06d}-{name}-head-truth.png"
                        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                        artifacts[name].update(
                            head_mask_path=mask_path.name,
                            head_mask_sha256=_digest(mask_path),
                        )
            if visual_artifact is not None:
                artifacts["V"] = visual_artifact
            row = {
                "step": step,
                "time": timestamp,
                "capture_time": timestamp,
                "action_time": timestamp,
                "mode": mode,
                "generation": session.generation,
                "commanded_goal": session.goal,
                "source_ids": observation.source_ids,
                "views": state["views"],
                "measurement": {
                    "timestamp": measurement.timestamp,
                    "status": measurement.status,
                    "xy": measurement.xy,
                    "covariance": measurement.covariance,
                },
                "prior_applied_command": applied,
                "prior_acknowledged_command_intervals": intervals,
                "action": action,
                "controller_status": session.status,
                "route_planning": session.route_planning_details,
                "scene_validity": state["scene_validity"],
                "agency": state["agency"],
                "interaction_telemetry": interaction_observation
                if agency_mode != "visual"
                else None,
                "interaction_world_scoring_only": interaction_observation
                if agency_mode == "visual"
                else None,
                "localization": {
                    key: state[key]
                    for key in (
                        "localization_status",
                        "localization_valid",
                        "pose",
                        "position_radius",
                        "last_seen_pose",
                        "last_seen_age_s",
                        "last_seen_radius_m",
                        "recovery_anchor_pose",
                        "recovery_anchor_time",
                        "recovery_anchor_source",
                        "raw_position_radius",
                        "raw_prediction_time",
                        "raw_pose",
                        "raw_velocity",
                        "raw_velocity_radius",
                        "reacquisition_samples",
                        "requires_new_goal",
                        "braking_prediction",
                    )
                },
                "controller_arrived": bool(
                    session.controller is not None and session.controller.arrived
                ),
                "controller": diagnostics,
                "truth_before_scoring_only": {
                    "position": env.state.position,
                    "velocity": env.state.velocity,
                },
                "head_pixels_scoring_only": head_counts,
                "images": artifacts,
                "wall_seconds": {
                    "capture_inference_control": decision_finished - started
                },
            }
            command_trace.clear()
            physics_trace.clear()
            interaction_request = (
                (
                    session.agency.world_request()
                    if agency_mode == "visual"
                    else session.agency.interaction_request()
                )
                if interaction_environment is not None
                else None
            )
            _, _, terminal, truncated, info = env.step(action)
            intervals = list(command_trace)
            row["after_step"] = {
                "time": env.world.time,
                "position_scoring_only": env.state.position,
                "velocity_scoring_only": env.state.velocity,
                "applied_command": list(env.world.backend.applied_request),
                "acknowledged_command_intervals": intervals,
                "collision_scoring_only": info["collision"],
                "boundary_scoring_only": info["boundary_failure"],
                "physics_trace_scoring_only": list(physics_trace),
                "interaction_telemetry": interaction_observation
                if agency_mode != "visual"
                else None,
                "interaction_world_scoring_only": interaction_observation
                if agency_mode == "visual"
                else None,
            }
            recorder.row(jsonable(row))
            manifest["completed_steps"] = step + 1
            encoded = {}
            for name, rgb in rgbs.items():
                width = min(int(config.get("jpeg_width", 640)), rgb.shape[1])
                preview = cv2.resize(
                    rgb, (width, round(rgb.shape[0] * width / rgb.shape[1]))
                )
                ok, payload = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, int(config.get("jpeg_quality", 80))],
                )
                if ok:
                    encoded[name] = payload.tobytes()
            publish_latest(
                event_queue,
                {"type": "frames", "frames": encoded, "generation": session.generation},
            )
            publish_latest(event_queue, {"type": "state", "state": state})
            if terminal or truncated:
                session.cancel(
                    "error" if terminal else "finished",
                    "Native episode ended; Reset to start another episode.",
                )
                break
        manifest["status"] = "complete"
    except Exception as error:  # noqa: BLE001 - worker boundary must brake/report native failures
        if manifest is not None:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish_latest(
            event_queue,
            {"type": "error", "message": f"{type(error).__name__}: {error}"},
        )
        if session is not None:
            session.cancel("error", f"{type(error).__name__}: {error}")
        if recorder is not None:
            try:
                recorder.event("fault", error=f"{type(error).__name__}: {error}")
            except OSError:
                pass  # Braking/error publication must survive a failed log device.
        publish_latest(
            event_queue,
            {
                "type": "state",
                "state": {
                    "phase": "error",
                    "message": str(error),
                    "generation": int(config.get("generation", 0)),
                    "mode": config.get("mode", 1),
                },
            },
        )
    finally:
        if env is not None and env.world is not None:
            env.world.backend.stop()
            if session is not None:
                if session.phase not in ("error", "finished"):
                    session.cancel("stopped", "Worker stopped.")
                session.agency.disable(
                    env.world.time, "Session ended; exploration paused."
                )
                publish_latest(
                    event_queue,
                    {
                        "type": "state",
                        "state": dict(
                            session.state(),
                            mode=config.get("mode", 1),
                            sim_time=env.world.time,
                        ),
                    },
                )
            env.close()
        if agency_store is not None:
            agency_store.close()
        if manifest is not None and run_dir is not None:
            from .session_recording import finish_recording

            finish_recording(recorder, manifest)
            if "run_identity" in manifest:
                from .run_identity import verify_run_identity

                manifest["run_identity_verification"] = verify_run_identity(
                    manifest["run_identity"]
                )
            manifest["source_files_unchanged"] = all(
                _digest(
                    Path(__file__).parent
                    / (
                        "control/"
                        if name == "genesis_backend.py"
                        else "mapping/"
                        if name == "scan_free_space.py"
                        else ""
                    )
                    / name
                )
                == digest
                for name, digest in manifest["source_sha256"].items()
            ) and manifest.get("run_identity_verification", {}).get("unchanged", False)
            (run_dir / "manifest.json").write_text(
                json.dumps(jsonable(manifest), indent=2) + "\n"
            )
