"""Wall-clock deadline gate for a calibrated RGB rig; no actuation or mapping."""

import copy
import math
import time
from dataclasses import dataclass

from .camera_rig import CameraRig, RigObservation, ViewObservation


@dataclass(frozen=True)
class DeadlineRigResult:
    observation: RigObservation
    status: str
    accepted: bool
    elapsed_seconds: float
    processing_seconds: float
    budget_seconds: float


class DeadlineCameraRig:
    """Own a rig and accept results only within a capture-to-result wall budget.

    ``capture_started`` is a monotonic timestamp taken before the first camera
    capture, not after rendering. ``now`` and each CameraFrame.capture_time keep
    their sensor/simulation clock domain; wall time never rewrites them.

    Fusion runs against copied rig/per-view temporal state. Only on-time results
    commit that state. A late result contains neither fused nor per-view poses
    or masks and cannot refresh last_accepted_capture_time. No late historical
    detection is retained by the owned rig. Observer instances are shared and
    must be stateless per frame, as LearnedObserver is. This gate does not
    preempt inference or issue a braking command; its caller must honor accepted.
    """

    def __init__(self, rig, *, budget_seconds=0.05, clock=time.monotonic):
        if not isinstance(rig, CameraRig):
            raise TypeError("Deadline processing requires CameraRig")
        if (
            not math.isfinite(budget_seconds)
            or budget_seconds <= 0
            or not callable(clock)
        ):
            raise ValueError("Invalid deadline budget or monotonic clock")
        self.rig = rig
        self.budget_seconds = float(budget_seconds)
        self.clock = clock
        self.last_accepted_capture_time = None

    def process(self, frames, *, now, capture_started):
        frames = list(frames)
        ids = [frame.camera_id for frame in frames]
        if (
            len(ids) > 3
            or len(set(ids)) != len(ids)
            or any(i not in self.rig.views for i in ids)
        ):
            raise ValueError(
                "Deadline capture requires at most three registered unique views"
            )
        started = self.clock()
        if (
            not math.isfinite(now)
            or now < 0
            or not math.isfinite(capture_started)
            or capture_started < 0
            or not math.isfinite(started)
            or started < capture_started
        ):
            raise ValueError("Invalid sensor time or monotonic capture start")

        def overdue(finished, processing):
            by_id = {frame.camera_id: frame for frame in frames}
            views = {
                name: ViewObservation(
                    name,
                    view.calibration_version,
                    by_id[name].capture_time if name in by_id else None,
                    None,
                    "deadline_exceeded",
                    "Capture-to-result wall budget exceeded",
                )
                for name, view in self.rig.views.items()
            }
            return DeadlineRigResult(
                RigObservation(float(now), None, None, "deadline_exceeded", (), views),
                "deadline_exceeded",
                False,
                finished - capture_started,
                processing,
                self.budget_seconds,
            )

        if started - capture_started > self.budget_seconds:
            return overdue(started, 0.0)
        candidate = copy.copy(self.rig)
        candidate.views = {
            name: copy.copy(view) for name, view in self.rig.views.items()
        }
        observation = candidate.observe(frames, now=now)
        finished = self.clock()
        if not math.isfinite(finished) or finished < started:
            raise ValueError("Monotonic clock moved backwards during processing")
        if finished - capture_started > self.budget_seconds:
            return overdue(finished, finished - started)
        self.rig = candidate
        accepted = observation.status == "visible" and observation.xy is not None
        if accepted:
            self.last_accepted_capture_time = observation.timestamp
        return DeadlineRigResult(
            observation,
            "accepted" if accepted else "no_accepted_pose",
            accepted,
            finished - capture_started,
            finished - started,
            self.budget_seconds,
        )
