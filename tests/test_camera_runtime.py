import numpy as np
import pytest

from bb8_rl.camera import Calibration, VisualMeasurement
from bb8_rl.camera_rig import CameraFrame, CameraRig, CameraView
from bb8_rl.camera_runtime import DeadlineCameraRig


class Observer:
    def __init__(self, calibration):
        self.calibration = calibration
        self.calls = 0
        self.status = "visible"

    def observe(self, rgb, timestamp):
        self.calls += 1
        return VisualMeasurement(
            timestamp,
            np.array([0.1, 0.1]) if self.status == "visible" else None,
            np.eye(2) * 0.0001 if self.status == "visible" else None,
            np.ones(rgb.shape[:2], bool),
            np.zeros(rgb.shape[:2], bool),
            self.status,
        )


@pytest.fixture
def rig():
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 3
    calibration = Calibration(
        np.array([[80.0, 0, 4], [0, 80.0, 3], [0, 0, 1]]), transform, (8, 6), 2
    )
    return CameraRig(
        [CameraView(name, "v1", calibration, Observer(calibration)) for name in "AB"]
    )


def frames(a=0.05, b=0.05):
    return [
        CameraFrame(name, "v1", np.zeros((6, 8, 3), np.uint8), timestamp)
        for name, timestamp in (("A", a), ("B", b))
    ]


def clock(*values):
    return iter(values).__next__


def test_capture_cost_counts_and_expired_batch_skips_processing(rig):
    runtime = DeadlineCameraRig(rig, clock=clock(10.051))
    result = runtime.process(frames(), now=0.05, capture_started=10)
    assert result.status == "deadline_exceeded" and not result.accepted
    assert result.processing_seconds == 0 and result.elapsed_seconds > 0.05
    assert result.observation.xy is None and not result.observation.source_ids
    assert all(view.measurement is None for view in result.observation.views.values())
    assert all(view.observer.calls == 0 for view in rig.views.values())
    assert runtime.last_accepted_capture_time is None


def test_late_result_never_commits_pose_masks_or_temporal_state(rig):
    runtime = DeadlineCameraRig(rig, clock=clock(10.002, 10.020, 10.102, 10.160))
    first = runtime.process(frames(), now=0.05, capture_started=10)
    assert first.accepted and runtime.last_accepted_capture_time == 0.05
    late = runtime.process(frames(0.1, 0.1), now=0.1, capture_started=10.1)
    assert late.status == "deadline_exceeded" and not late.accepted
    assert late.observation.xy is None and late.observation.covariance is None
    assert not late.observation.source_ids
    assert all(view.measurement is None for view in late.observation.views.values())
    assert runtime.last_accepted_capture_time == 0.05
    assert runtime.rig.last_timestamp == 0.05
    assert runtime.rig.last_measurement[0] == 0.05
    assert all(view.last_capture_time == 0.05 for view in runtime.rig.views.values())
    assert [view.capture_time for view in late.observation.views.values()] == [0.1, 0.1]


def test_accepted_result_preserves_sensor_timestamps_and_skew_inflation(rig):
    runtime = DeadlineCameraRig(rig, clock=clock(40.010, 40.025))
    result = runtime.process(frames(0.03, 0.05), now=0.05, capture_started=40)
    assert result.accepted and result.status == "accepted"
    assert result.observation.timestamp == 0.05
    assert [view.capture_time for view in result.observation.views.values()] == [
        0.03,
        0.05,
    ]
    assert result.observation.covariance[0, 0] > 0.0001
    assert result.elapsed_seconds == pytest.approx(0.025)
    assert result.processing_seconds == pytest.approx(0.015)


def test_timely_missing_result_has_explicit_no_accepted_pose_gate(rig):
    for view in rig.views.values():
        view.observer.status = "missing"
    runtime = DeadlineCameraRig(rig, clock=clock(0.005, 0.01))
    result = runtime.process(frames(), now=0.05, capture_started=0)
    assert result.status == "no_accepted_pose" and not result.accepted
    assert result.observation.status == "missing" and result.observation.xy is None
    assert runtime.last_accepted_capture_time is None


def test_timely_stale_frame_is_still_rejected_by_sensor_clock(rig):
    runtime = DeadlineCameraRig(rig, clock=clock(0.005, 0.01))
    result = runtime.process(frames(), now=0.2, capture_started=0)
    assert not result.accepted and result.status == "no_accepted_pose"
    assert all(
        view.status == "stale_frame" for view in result.observation.views.values()
    )


@pytest.mark.parametrize("values,start", [((10.0,), 10.1), ((10.0, 9.9), 10.0)])
def test_invalid_wall_clock_rejected(rig, values, start):
    runtime = DeadlineCameraRig(rig, clock=clock(*values))
    with pytest.raises(ValueError, match="[Mm]onotonic"):
        runtime.process(frames(), now=0.05, capture_started=start)
