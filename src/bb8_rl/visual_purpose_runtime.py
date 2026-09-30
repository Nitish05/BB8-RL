"""Connect pixel-derived entities/outcomes to the existing bounded purpose loop.

The coordinator never receives numeric world resource/effect telemetry. A world
completion is correlation only; fresh before/after RGB is required to learn.
"""

import math

from .purpose_runtime import PurposeCoordinator
from .visual_interaction import SOURCE, VisualEntityRegistry, VisualOutcomeGate


class VisualPurposeCoordinator(PurposeCoordinator):
    def __init__(self, engine, calibration_version, *, evidence_retainer=None):
        super().__init__(engine, (), observation_source=SOURCE)
        self.registry = VisualEntityRegistry(engine.store, calibration_version)
        self.gate = VisualOutcomeGate(engine.store.map_version, calibration_version)
        self.entities = ()
        self.scene = self.completion = None
        self._active_request = None
        self.last_visual_outcome = None
        self.evidence_retainer = evidence_retainer
        self.visual_status = "awaiting_pixels"

    def feed_visual(self, scene, completion=None):
        """Capture-scoped pixel observation and completion-only acknowledgement."""
        self.scene, self.completion = scene, completion
        self.entities = self.registry.observe(scene)
        self.stations = self.registry.stations()

    def disable(self, now=0.0, reason="Visual learning paused.", outcome="cancelled"):
        self.gate.cancel(reason)
        self._active_request = None
        super().disable(now, reason, outcome)

    def interaction_request(self):
        requested = super().interaction_request()
        if (
            requested is None
            or self._active_request != requested
            or self.gate.generation is None
        ):
            return None
        return requested

    def world_request(self):
        """Translate a currently observed UUID to its visible fixture glyph.

        The renderer/mechanics resolve that glyph. Hidden efficacy never enters
        this translation, candidate coordinates, or the learner's memory scope.
        """
        request = self.interaction_request()
        if request is None:
            return None
        entity = next(
            (e for e in self.entities if e.entity_id == request["station_id"]), None
        )
        if entity is None:
            return None
        return {
            "request_id": request["request_id"],
            "station_id": entity.reading.marker_id,
        }

    def completion_for(self, world_completion, generation):
        """Allow-list a world acknowledgement, explicitly excluding its values."""
        request = self.world_request()
        if (
            request is None
            or not isinstance(world_completion, dict)
            or set(world_completion) != {"request_id", "station_id", "timestamp"}
            or type(generation) is not int
            or generation != self.gate.generation
        ):
            return None
        timestamp = world_completion["timestamp"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
            or any(world_completion[key] != request[key] for key in request)
        ):
            return None
        return {
            "request_id": request["request_id"],
            "entity_id": self._active_request["station_id"],
            "timestamp": world_completion["timestamp"],
            "generation": generation,
        }

    def tick(self, session, now):
        state = session.state()
        fresh = (
            self.scene is not None
            and self.scene.timestamp == now
            and self.scene.status == "visible"
            and bool(self.entities)
        )
        if not fresh:
            self.visual_status = "visual_evidence_unavailable"
            if self.enabled:
                return self._cancel(
                    session,
                    now,
                    "Fresh entity and gauge pixels are unavailable; no outcome learned.",
                )
            self.observation = None
            return False
        lower = max(e.reading.interval[0] for e in self.entities)
        upper = min(e.reading.interval[1] for e in self.entities)
        if lower > upper:
            self.visual_status = "gauges_disagree"
            if self.enabled:
                return self._cancel(
                    session,
                    now,
                    "Visible resource indicators disagree; no outcome learned.",
                )
            self.observation = None
            return False
        self.visual_status = "fresh_rgb"
        observation = {"resource": lower, "timestamp": now, "source": SOURCE}
        evidence = None
        if self.enabled:
            if self.intention and self.intention["candidate_id"] not in {
                e.entity_id for e in self.entities
            }:
                return self._cancel(
                    session,
                    now,
                    "Selected visual entity is unavailable; explicit restart required.",
                )
            measured = (
                state["localization_valid"]
                and state["localization_status"] == "measured"
            )
            stationary = measured and state["phase"] == "arrived"
            if (
                stationary
                and self.gate.generation is None
                and self._active_request is None
            ):
                self.gate.enable(session.generation)
            had_pending = self.gate.pending is not None
            evidence = self.gate.observe(
                self.entities,
                now=now,
                generation=session.generation,
                measured=measured,
                stationary=stationary,
                completion=self.completion,
            )
            if had_pending and evidence is None and self.gate.pending is None:
                return self._cancel(
                    session,
                    now,
                    "Visual interaction evidence was interrupted; no outcome learned.",
                )
            if evidence is not None:
                if self.evidence_retainer is not None:
                    try:
                        self.evidence_retainer(evidence)
                    except (OSError, ValueError):
                        return self._cancel(
                            session,
                            now,
                            "Visual evidence could not be retained; no outcome learned.",
                        )
                observation["outcome"] = {
                    "request_id": evidence["request_id"],
                    "station_id": evidence["entity_id"],
                    "before": evidence["before"],
                    "after": evidence["after"],
                    "timestamp": evidence["timestamp"],
                    "source": SOURCE,
                    "visual_evidence": evidence,
                }
        super().observe(observation)
        changed = super().tick(session, now)
        # Parent tick publishes this receipt only after successful persistence
        # (including an exact event replay). A failed write must not advertise
        # an uncommitted outcome through the public snapshot.
        committed = (self.recent_experience or {}).get("visual_evidence")
        if evidence is not None and committed == evidence:
            self.last_visual_outcome = committed
        if not self.enabled or self.intention is None:
            self._active_request = None
        else:
            request = super().interaction_request()
            if (
                request is not None
                and self._active_request is None
                and self.gate.begin(
                    request["request_id"],
                    request["station_id"],
                    now=now,
                    generation=session.generation,
                )
            ):
                self._active_request = request
        return changed

    def snapshot(self):
        return {
            **super().snapshot(),
            "resource_source": SOURCE,
            "visual_status": self.visual_status,
            "visual_entities": [
                {
                    "entity_id": e.entity_id,
                    "marker_id": e.reading.marker_id,
                    "anchor_xy": list(e.anchor_xy),
                    "anchor_radius_m": e.anchor_radius_m,
                    "frame_sha256": e.reading.frame_sha256,
                }
                for e in self.entities
            ],
            "visual_outcome": self.last_visual_outcome,
            "scope": "Fixture-specific native RGB panels and simulated resource; not general object understanding",
        }
