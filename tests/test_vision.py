import cv2
import numpy as np
import pytest
import torch

from bb8_rl.camera import Calibration, CameraController
from bb8_rl.vision import LearnedObserver, VisionStudent, candidates, crop


def test_native_crop_keeps_small_head_and_label_alignment():
    rgb = np.zeros((960, 1280, 3), np.uint8)
    label = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(rgb, (740, 515), 2, (180, 180, 180), -1)
    cv2.circle(label, (740, 515), 2, 1, -1)
    (proposal,) = candidates(rgb)
    patch, origin = crop(rgb, proposal)
    target, label_origin = crop(label, proposal)
    assert target.sum() == 13 and patch.shape == (64, 64, 3)
    np.testing.assert_array_equal(origin, label_origin)
    np.testing.assert_array_equal(patch[..., 0] > 0, target.astype(bool))


def test_checkpoint_reload_blank_frame_stops_controller(tmp_path):
    torch.set_num_threads(2)
    model = VisionStudent()
    path = tmp_path / "model.pt"
    torch.save(
        {
            "format": "bb8-vision-v1",
            "model": model.state_dict(),
            "floor_threshold": 0.95,
            "head_threshold": 0.5,
        },
        path,
    )
    k = np.array([[600.0, 0, 640], [0, 600.0, 480], [0, 0, 1]])
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 3
    calibration = Calibration(k, transform, (1280, 960), 2)
    observer = LearnedObserver(calibration, path)
    blank = np.zeros((960, 1280, 3), np.uint8)

    def forbidden(vector):
        raise AssertionError("Blank camera must not call learned driving policy")

    controller = CameraController(
        calibration, forbidden, calibration.to_pixel([1, 0]), observer=observer
    )
    for i in range(4):
        assert not controller.action(blank, i * 0.05).any()
        assert controller.status == "missing"
    with pytest.raises(ValueError, match="Invalid RGB"):
        observer.observe(blank, float("nan"))
    # Exercise mask-to-measurement geometry independently of learned accuracy.
    # Both overlapping crop proposals see both heads: never average them into
    # a single fictitious midpoint position.
    observer.probabilities = lambda patches, head=False: np.stack(
        [
            (patch[..., 0] > 100).astype(float) if head else np.ones(patch.shape[:2])
            for patch in patches
        ]
    )
    rgb = blank.copy()
    cv2.circle(rgb, (640, 480), 2, (180, 180, 180), -1)
    assert observer.observe(rgb, 1).status == "visible"
    cv2.circle(rgb, (651, 480), 2, (180, 180, 180), -1)
    assert observer.observe(rgb, 2).status == "ambiguous"


def test_both_networks_train_with_finite_gradients():
    torch.set_num_threads(2)
    model = VisionStudent()
    for network, size in ((model.floor, (60, 80)), (model.head, (64, 64))):
        logits = network(torch.rand(2, 3, *size))
        target = torch.zeros(2, *size, dtype=torch.long)
        target[:, 20:23, 30:33] = 1
        torch.nn.functional.cross_entropy(logits, target).backward()
        assert logits.shape == (2, 2, *size)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in network.parameters()
        )
