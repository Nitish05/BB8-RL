"""Learned station outcomes above unchanged camera navigation authority."""

import math
import uuid

import numpy as np

from .purpose import MEANINGFUL_GAIN


class PurposeCoordinator:
    def __init__(self, engine=None, stations=()):
        self.engine, self.stations = engine, tuple(stations)
        self.available, self.enabled = engine is not None, False
        self.status = "paused" if self.available else "unavailable"
        self.message = "Start learning to address the current simulated resource need."
        self.intention = self.recent_experience = self.observation = None
        self.next_choice = self.active_since = 0.0
        self.interaction_since = self.guarded_since = None
        self.blocked = set()
        self._idle_resource = None
        self._memory = {}
        if engine is not None:
            try:
                self._memory = engine.snapshot()
            except Exception as error:  # noqa: BLE001 — optional persistent service
                self.available = False
                self.status, self.message = (
                    "unavailable",
                    f"Memory unavailable: {error}",
                )

    def snapshot(self):
        return {
            **self._memory,
            "available": self.available,
            "enabled": self.enabled,
            "status": self.status,
            "message": self.message,
            "intention": self.intention,
            "recent_experience": self.recent_experience,
            "selection_policy": "learned_station_outcomes",
            "resource": None
            if self.observation is None
            else self.observation["resource"],
            "target": 0.8,
            "resource_source": "simulated_station_telemetry",
            "stations": [
                {"id": s.id, "goal": list(s.xy), "label": s.label, "radius": 0.14}
                for s in self.stations
            ],
            "interaction_request": self.interaction_request(),
        }

    def observe(self, observation):
        """Only resource sensor telemetry enters here, never world poses/effects."""
        if not isinstance(observation, dict):
            raise TypeError("Invalid station telemetry")
        resource, timestamp = observation.get("resource"), observation.get("timestamp")
        if (
            observation.get("source") != "simulated_station_telemetry"
            or not isinstance(resource, (int, float))
            or isinstance(resource, bool)
            or not math.isfinite(resource)
            or not 0 <= resource <= 1
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
            or timestamp < 0
            or self.observation is not None
            and timestamp < self.observation["timestamp"]
        ):
            raise ValueError("Invalid or out-of-order station telemetry")
        outcome = observation.get("outcome")
        if outcome is not None:
            fields = {
                "request_id",
                "station_id",
                "before",
                "after",
                "timestamp",
                "source",
            }
            if not isinstance(outcome, dict) or set(outcome) != fields:
                raise ValueError("Invalid station receipt fields")
            outcome = dict(outcome)
        # Whitelist and copy the observation interface: no hidden simulator fields.
        self.observation = {
            "resource": float(resource),
            "timestamp": float(timestamp),
            "source": observation["source"],
            "outcome": outcome,
        }

    def interaction_request(self):
        if not self.enabled or self.status != "interacting" or self.intention is None:
            return None
        return {
            "request_id": self.intention["event_id"],
            "station_id": self.intention["candidate_id"],
        }

    def disable(self, now=0.0, reason="Learning paused.", outcome="cancelled"):
        self.enabled = False
        if self.intention:
            self.recent_experience = {
                "label": self.intention["label"],
                "outcome": outcome,
                "summary": f"{self.intention['label']}: {outcome.replace('_', ' ')}; no response learned.",
            }
        self.intention = None
        self.interaction_since = None
        self.status, self.message = "paused", reason

    def enable(self, now):
        if not self.available:
            raise ValueError("Persistent outcome memory is unavailable")
        self.enabled = True
        self.status, self.message = (
            "choosing",
            "Comparing learned effects with the current resource need.",
        )
        self.next_choice = float(now)
        self.blocked.clear()
        self._idle_resource = None

    def _cancel(self, session, now, reason, outcome="cancelled"):
        session.cancel("idle", reason)
        self.disable(now, reason, outcome)
        return True

    def tick(self, session, now):
        if not self.enabled:
            return False
        try:
            state = session.state()
            if session.localization_lost or state["phase"] in (
                "heartbeat_expired",
                "error",
                "finished",
                "localization_lost",
            ):
                return self._cancel(
                    session,
                    now,
                    "Learning paused; explicit restart required.",
                    "localization_lost",
                )
            if (
                self.observation is None
                or not -1e-9 <= now - self.observation["timestamp"] <= 0.10
            ):
                return self._cancel(
                    session,
                    now,
                    "Resource telemetry is stale; no interaction outcome learned.",
                )
            measured = (
                state["localization_valid"]
                and state["localization_status"] == "measured"
            )
            if self.intention:
                if self.status == "interacting":
                    if not measured or state["phase"] != "arrived":
                        return self._cancel(
                            session,
                            now,
                            "Interaction interrupted; fresh stationary camera confirmation is required.",
                        )
                    outcome = self.observation.get("outcome")
                    if outcome is not None:
                        expected = self.interaction_request()
                        if (
                            outcome.get("request_id") != expected["request_id"]
                            or outcome.get("station_id") != expected["station_id"]
                            or outcome.get("source") != "simulated_station_telemetry"
                            or outcome.get("timestamp", -1) - self.interaction_since
                            < 1.0 - 1e-9
                            or not self.interaction_since
                            <= outcome.get("timestamp", -1)
                            <= now
                            or now - outcome["timestamp"] > 0.10
                        ):
                            return self._cancel(
                                session,
                                now,
                                "Unmatched interaction evidence; no outcome learned.",
                            )
                        self.engine.record_outcome(
                            expected["request_id"],
                            expected["station_id"],
                            before=outcome["before"],
                            after=outcome["after"],
                            now=outcome["timestamp"],
                            source=outcome["source"],
                        )
                        self._memory = self.engine.snapshot()
                        gain = outcome["after"] - outcome["before"]
                        self.recent_experience = {
                            **outcome,
                            "label": self.intention["label"],
                            "outcome": "restored"
                            if gain >= MEANINGFUL_GAIN - 1e-9
                            else "no_effect"
                            if gain == 0
                            else "no_meaningful_gain",
                            "summary": f"{self.intention['label']}: {outcome['before']:.0%} → {outcome['after']:.0%} simulated resource. Response remembered.",
                        }
                        self.intention = None
                        self.interaction_since = None
                        self.next_choice = now + 1.0
                        self.status, self.message = (
                            "remembering",
                            self.recent_experience["summary"],
                        )
                        return (
                            False  # Preserve stationary arrival for independent audit.
                        )
                    if now - self.interaction_since > 5.0:
                        return self._cancel(
                            session,
                            now,
                            "No confirmed station response; paused without learning a failure.",
                        )
                    return False
                if state["phase"] == "arrived" and measured:
                    self.status, self.message = (
                        "interacting",
                        "Requesting a stationary station interaction; arrival alone earns nothing.",
                    )
                    self.interaction_since = now
                    return False
                if state["phase"] == "guarded_stop":
                    self.guarded_since = (
                        now if self.guarded_since is None else self.guarded_since
                    )
                else:
                    self.guarded_since = None
                if (
                    state["phase"] == "goal_rejected"
                    or now - self.active_since >= 60
                    or (
                        self.guarded_since is not None and now - self.guarded_since >= 4
                    )
                ):
                    self.blocked.add(self.intention["candidate_id"])
                    self.recent_experience = {
                        "outcome": "route_unavailable",
                        "summary": f"{self.intention['label']}: route unavailable; station effect remains untested.",
                    }
                    self.intention = None
                    session.cancel("idle", "Checking another reachable interaction.")
                    self.next_choice = now + 1.0
                    self.status = "choosing"
                    return True
                return False
            if now < self.next_choice:
                return False
            if not measured:
                self.status, self.message = (
                    "waiting",
                    "Waiting for a fresh camera position before choosing.",
                )
                return False
            resource = self.observation["resource"]
            if self.status in ("idle", "satisfied") and resource == self._idle_resource:
                return False
            candidates = []
            radius = max(
                session.planning_radius,
                session.parameters.robot_radius
                + session.parameters.clearance_margin
                + state["position_radius"],
            )
            for station in self.stations:
                if station.id in self.blocked:
                    continue
                try:
                    session.route_with_clearance(state["pose"], station.xy, radius)
                except ValueError:
                    continue
                candidates.append(station)
            choice = self.engine.select(
                candidates, resource=resource, pose=tuple(state["pose"]), now=now
            )
            self._memory = self.engine.snapshot()
            if choice is None:
                self.status = "satisfied" if resource >= 0.8 else "idle"
                self.message = (
                    "Resource need satisfied. Waiting; idle does not drain the simulated resource."
                    if self.status == "satisfied"
                    else "No reachable interaction has enough expected benefit. Waiting; no map-point wandering."
                )
                session.cancel("idle", self.message)
                self._idle_resource = resource
                self.next_choice = now + 2.0
                return True
            station = next(s for s in candidates if s.id == choice["candidate_id"])
            # Reuse only an already-confirmed arrival at this same station.
            at_station = (
                state["phase"] == "arrived"
                and math.dist(state["pose"], station.xy) <= 0.10
            )
            self.intention = {
                **choice,
                "goal": list(station.xy),
                "label": station.label,
                "event_id": uuid.uuid4().hex,
            }
            self.active_since, self.guarded_since = now, None
            if at_station:
                self.status, self.message = "interacting", self.intention["explanation"]
                self.interaction_since = now
                return False
            session.cancel(
                "braking", "Checking the route to the selected interaction zone."
            )
            session.pending_goal = session.goal = np.asarray(station.xy, float)
            session.requires_new_goal = False
            self.status, self.message = "travelling", self.intention["explanation"]
            return True
        except Exception as error:  # noqa: BLE001 — optional learner cannot bypass control
            self._cancel(session, now, f"Outcome learning unavailable: {error}")
            self.available, self.status = False, "unavailable"
            return True
