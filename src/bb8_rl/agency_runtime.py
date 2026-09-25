"""Explicitly enabled intention selection above the existing navigation guards."""

import math
import uuid

from .agency import Candidate


class AutonomyCoordinator:
    def __init__(self, engine=None):
        self.engine = engine
        self.enabled = False
        self.available = engine is not None
        self.intention = None
        self.recent_experience = None
        self.message = "Start exploring to learn from visits."
        self.status = "paused" if self.available else "unavailable"
        self.next_choice = 0.0
        self.active_since = 0.0
        self.guarded_since = None
        self.cooldowns = {}
        self.failures = 0
        self._memory = {}
        if engine:
            try:
                self._memory = engine.snapshot()
            except Exception as error:  # noqa: BLE001 — optional memory startup
                self.available = False
                self.status, self.message = (
                    "unavailable",
                    f"Memory unavailable: {error}",
                )

    def snapshot(self):
        return {
            **self._memory,
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "message": self.message,
            "intention": self.intention,
            "recent_experience": self.recent_experience,
        }

    def _finish(self, now, outcome):
        intention, self.intention = self.intention, None
        if intention is None:
            return
        self.engine.record_outcome(
            event_id=intention["event_id"],
            candidate_id=intention["candidate_id"],
            outcome=outcome,
            now=max(0.0, float(now)),
            duration_s=max(0.0, float(now) - self.active_since),
            progress=1.0 if outcome == "arrived" else 0.0,
        )
        self._memory = self.engine.snapshot()
        self.recent_experience = {
            "label": intention["label"],
            "outcome": outcome,
            "summary": f"{intention['label']}: {outcome.replace('_', ' ')}",
        }
        self.cooldowns[intention["candidate_id"]] = float(now) + (
            20.0 if outcome == "rejected" else 3.0
        )

    def disable(self, now=0.0, reason="Exploration paused.", outcome="cancelled"):
        # Disable before persistence: an unavailable database cannot leave an
        # intention authorized to move. The caller brakes before this operation.
        self.enabled = False
        self.status, self.message = "paused", reason
        try:
            self._finish(now, outcome)
        except Exception as error:  # noqa: BLE001 — persistence must fail closed
            self.available = False
            self.status, self.message = "unavailable", f"Memory unavailable: {error}"

    def enable(self, now):
        if not self.available:
            raise ValueError("Persistent exploration memory is unavailable")
        self.enabled = True
        self.status, self.message = "choosing", "Choosing among certified places."
        self.next_choice, self.failures = float(now), 0

    def _candidates(self, session, state, now):
        # Stable half-metre anchors: IDs remain meaningful across restarts of
        # this map. No scene objects, truth poses or privileged visibility.
        extent = float(session.memory.extent)
        pose = tuple(state["pose"])
        n = math.ceil(2 * extent)
        radius = max(
            session.planning_radius,
            session.parameters.robot_radius
            + session.parameters.clearance_margin
            + state["position_radius"],
        )
        anchors = []
        for ix in range(-n + 1, n):
            for iy in range(-n + 1, n):
                xy = (ix / 2, iy / 2)
                distance = math.dist(xy, pose)
                identifier = f"place:{ix}:{iy}"
                if (
                    max(abs(xy[0]), abs(xy[1])) >= extent - radius
                    or distance < 0.55
                    or self.cooldowns.get(identifier, -1) > now
                    or not session.memory.segment_free(xy, xy, radius)
                ):
                    continue
                anchors.append((distance, identifier, xy))
        accepted = []
        # Bounded planner work per decision. Each accepted route is independently
        # checked again by ControlSession when it starts moving on a later frame.
        for _, identifier, xy in sorted(anchors)[:12]:
            try:
                session.route_with_clearance(pose, xy, radius)
            except ValueError:
                continue
            accepted.append(Candidate(identifier, xy, f"Place ({xy[0]:g}, {xy[1]:g})"))
            if len(accepted) >= 5:
                break
        return accepted

    def tick(self, session, now):
        """Return True if this tick cancelled/replaced motion (output must be zero)."""
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
                session.stop()
                self.disable(
                    now,
                    "Exploration paused; explicit restart required.",
                    "localization_lost" if session.localization_lost else "cancelled",
                )
                return True
            if self.intention:
                if state["phase"] == "arrived":
                    self._finish(now, "arrived")
                    self.failures = 0
                    self.next_choice = now + 1.0
                    self.status, self.message = (
                        "remembering",
                        "Arrival remembered. Choosing again shortly.",
                    )
                    return (
                        False  # Preserve the arrived controller for independent audit.
                    )
                if state["phase"] == "guarded_stop":
                    self.guarded_since = (
                        now if self.guarded_since is None else self.guarded_since
                    )
                else:
                    self.guarded_since = None
                stalled = (
                    self.guarded_since is not None and now - self.guarded_since >= 4.0
                )
                if (
                    state["phase"] == "goal_rejected"
                    or stalled
                    or now - self.active_since >= 60
                ):
                    session.cancel(
                        "idle", "Exploration route ended; selecting another place."
                    )
                    self._finish(now, "rejected")
                    self.failures += 1
                    self.next_choice = now + 1.0
                    if self.failures >= 3:
                        self.disable(
                            now,
                            "Three routes could not complete. Choose a manual goal or reset.",
                        )
                    return True
                return False
            if now < self.next_choice:
                return False
            if (
                not state["localization_valid"]
                or state["localization_status"] != "measured"
            ):
                self.status, self.message = (
                    "waiting",
                    "Waiting for a fresh camera position before choosing.",
                )
                return False
            candidates = self._candidates(session, state, now)
            choice = self.engine.select(candidates, now=now, pose=tuple(state["pose"]))
            self._memory = self.engine.snapshot()
            self.next_choice = now + 2.0
            if choice is None:
                self.status, self.message = (
                    "waiting",
                    "No reachable exploration place at current clearance.",
                )
                return False
            candidate = next(c for c in candidates if c.id == choice["candidate_id"])
            session.cancel("braking", "Checking the selected exploration route.")
            # The selected coordinate comes from the checked candidate, never
            # from a language-model-generated coordinate or free-form explanation.
            import numpy as np

            session.pending_goal = session.goal = np.asarray(candidate.xy, float)
            session.requires_new_goal = False
            self.intention = {
                **choice,
                "goal": list(candidate.xy),
                "label": candidate.label,
                "event_id": uuid.uuid4().hex,
            }
            self.active_since, self.guarded_since = now, None
            self.status, self.message = "exploring", choice["explanation"]
            return True
        except Exception as error:  # noqa: BLE001 — optional agency cannot bypass control
            session.cancel(
                "guarded_stop",
                "Exploration memory or planning failed; motion cancelled.",
            )
            self.disable(now, f"Exploration unavailable: {error}")
            self.available = False
            self.status = "unavailable"
            return True
