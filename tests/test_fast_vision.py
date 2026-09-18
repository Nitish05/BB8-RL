import cv2
import numpy as np
import pytest

from bb8_rl.camera import Calibration
from bb8_rl.camera_rig import CameraFrame, CameraRig, CameraView
from bb8_rl.vision import LearnedObserver, PositionOnlyObserver


@pytest.fixture
def observer():
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 3
    calibration = Calibration(
        np.array([[80.0, 0, 64], [0, 80.0, 48], [0, 0, 1]]),
        transform,
        (128, 96),
        2,
    )
    observer = object.__new__(LearnedObserver)
    observer.calibration = calibration
    observer.floor_threshold = 0.5
    observer.head_threshold = 0.5
    observer.floor_refinement = "none"
    observer.probabilities = lambda patches, head=False: np.stack(
        [
            (patch[..., 0] > 100).astype(float) if head else np.ones(patch.shape[:2])
            for patch in patches
        ]
    )
    return observer


def image(heads):
    rgb = np.zeros((96, 128, 3), np.uint8)
    for center in heads:
        cv2.circle(rgb, center, 2, (180, 180, 180), -1)
    return rgb


@pytest.mark.parametrize(
    "heads,status",
    [([], "missing"), ([(64, 48)], "visible"), ([(64, 48), (75, 48)], "ambiguous")],
)
def test_position_matches_full_detection_with_no_free_space_evidence(
    observer, heads, status
):
    rgb = image(heads)
    full = observer.observe(rgb, 0.05)
    fast = observer.observe_position(rgb, 0.05)
    assert fast.status == full.status == status and fast.timestamp == full.timestamp
    if full.xy is None:
        assert fast.xy is None and fast.covariance is None
    else:
        np.testing.assert_array_equal(fast.xy, full.xy)
        np.testing.assert_array_equal(fast.covariance, full.covariance)
    assert full.visible_floor.any()
    for mask in (fast.visible_floor, fast.robot_pixels):
        assert mask.shape == (96, 128) and mask.dtype == bool
        assert not mask.any() and not mask.flags.writeable
    assert observer.observe(rgb, 0.1).visible_floor.any()


def test_position_never_runs_floor_network_refinement_or_robot_grid(
    observer, monkeypatch
):
    original = observer.probabilities
    calls = []

    def only_head(patches, head=False):
        assert head, "Localization must not infer floor"
        calls.append(len(patches))
        return original(patches, head=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Localization must not allocate/refine a full-frame mask")

    observer.probabilities = only_head
    observer.floor_refinement = "guided"
    monkeypatch.setattr("bb8_rl.vision.guided_floor_probability", forbidden)
    monkeypatch.setattr("bb8_rl.vision.np.indices", forbidden)
    assert observer.observe_position(image([(64, 48)]), 0).status == "visible"
    assert calls == [1]


def test_excessive_proposals_remain_ambiguous_without_head_inference(observer):
    def forbidden(*args, **kwargs):
        raise AssertionError("Too many proposals must stop inference")

    observer.probabilities = forbidden
    heads = [(x, y) for x in range(8, 121, 8) for y in range(8, 89, 8)]
    result = observer.observe_position(image(heads), 0)
    assert result.status == "ambiguous" and result.xy is None


def test_position_adapter_is_accepted_by_rig_without_masks(observer):
    adapted = PositionOnlyObserver(observer)
    rig = CameraRig([CameraView("A", "v1", observer.calibration, adapted)])
    result = rig.observe([CameraFrame("A", "v1", image([(64, 48)]), 0.05)], now=0.05)
    assert result.status == "visible"
    assert not result.views["A"].measurement.visible_floor.any()
    assert not result.views["A"].measurement.robot_pixels.any()


@pytest.mark.parametrize("timestamp", [-1, float("nan"), float("inf")])
def test_fast_path_retains_timestamp_validation(observer, timestamp):
    with pytest.raises(ValueError, match="Invalid RGB"):
        observer.observe_position(image([]), timestamp)


def test_fast_path_retains_rgb_validation(observer):
    with pytest.raises(ValueError, match="Invalid RGB"):
        observer.observe_position(np.zeros((96, 128, 3), float), 0)
    with pytest.raises(ValueError, match="Invalid RGB"):
        observer.observe_position(np.zeros((48, 64, 3), np.uint8), 0)
