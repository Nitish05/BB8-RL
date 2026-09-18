import cv2
import numpy as np
import pytest
import torch

from bb8_rl.camera import Calibration
from bb8_rl.vision import LearnedObserver, VisionStudent, guided_floor_probability


def test_guided_filter_preserves_constant_probability_under_textured_rgb():
    rng = np.random.default_rng(12)
    rgb = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    probability = np.full(rgb.shape[:2], 0.37, np.float32)
    result = guided_floor_probability(rgb, probability)
    np.testing.assert_allclose(result, probability, atol=1e-6)
    np.testing.assert_array_equal(probability, np.full((48, 64), 0.37, np.float32))


def test_guided_filter_sharpens_rgb_boundary_without_unbounded_probabilities():
    truth = np.zeros((48, 64), np.float32)
    truth[:, 32:] = 1
    rgb = np.repeat((truth * 180 + 30).astype(np.uint8)[..., None], 3, axis=2)
    blurred = cv2.GaussianBlur(truth, (0, 0), 3)
    result = guided_floor_probability(rgb, blurred)
    assert np.square(result - truth).mean() < np.square(blurred - truth).mean()
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    np.testing.assert_array_equal(result >= 0.5, truth.astype(bool))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_guided_filter_rejects_invalid_probability_instead_of_free_space(value):
    rgb = np.zeros((16, 16, 3), np.uint8)
    probability = np.full((16, 16), 0.5, np.float32)
    probability[8, 8] = value
    with pytest.raises(ValueError, match="Invalid RGB, probability"):
        guided_floor_probability(rgb, probability)


def test_observer_refinement_is_explicit_and_preserves_checkpoint_threshold(tmp_path):
    torch.set_num_threads(2)
    path = tmp_path / "vision.pt"
    torch.save(
        {
            "format": "bb8-vision-v1",
            "model": VisionStudent().state_dict(),
            "floor_threshold": 0.4,
            "head_threshold": 0.5,
            "floor_target": "traversable",
        },
        path,
    )
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 3
    calibration = Calibration(
        np.array([[60.0, 0, 64], [0, 60.0, 48], [0, 0, 1]]),
        transform,
        (128, 96),
        2,
    )
    default = LearnedObserver(calibration, path)
    guided = LearnedObserver(calibration, path, floor_refinement="guided")
    assert default.floor_refinement == "none"
    assert guided.floor_threshold == default.floor_threshold == 0.4
    with pytest.raises(ValueError, match="Unknown floor refinement"):
        LearnedObserver(calibration, path, floor_refinement="oracle")
    rgb = np.full((96, 128, 3), 30, np.uint8)
    rgb[:, 64:] = 210
    low_probability = np.zeros((240, 320), np.float32)
    low_probability[:, 160:] = 1
    low_probability = cv2.GaussianBlur(low_probability, (0, 0), 7.5)
    # Fixed floor probabilities isolate the inference option from model accuracy.
    for observer in (default, guided):
        observer.probabilities = lambda images, head=False: low_probability[None]
    baseline = default.observe(rgb, 0)
    refined = guided.observe(rgb, 0)
    # No head proposals exist in these large gray regions, so neither observer
    # hallucinates localization, and only the floor boundary changes.
    assert baseline.status == refined.status == "missing"
    truth = np.indices(rgb.shape[:2])[1] >= 64
    assert (refined.visible_floor != truth).sum() < (
        baseline.visible_floor != truth
    ).sum()
    state = torch.load(path, weights_only=True)
    state["calibration_refinement"] = "guided-radius4-epsilon1e-4"
    torch.save(state, path)
    with pytest.raises(ValueError, match="calibrated guided refinement"):
        LearnedObserver(calibration, path)
    assert (
        LearnedObserver(calibration, path, floor_refinement="guided").floor_threshold
        == 0.4
    )
