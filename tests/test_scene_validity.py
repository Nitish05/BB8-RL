"""Synthetic image contracts, separate from frozen held-out/native evaluation."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from bb8_rl.camera import Calibration, VisualMeasurement
from bb8_rl.camera_rig import CameraFrame, ViewObservation
from bb8_rl.scene_validity import SceneValidityGuard, SceneValidityParameters


def calibration():
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 3.0
    return Calibration(
        np.array([[150.0, 0, 96], [0, 150, 72], [0, 0, 1]]),
        transform,
        (192, 144),
        2.0,
    )


def background():
    # Fixed procedural texture for contract testing, not a held-out room image.
    rng = np.random.default_rng(1729)
    gray = np.repeat(np.repeat(rng.integers(45, 185, (18, 24)), 8, 0), 8, 1)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)


def scene(*, object_x=30, robot=None):
    rgb = background()
    rgb[35:64, object_x : object_x + 23] = (205, 30, 40)
    if robot is not None:
        pixel = calibration().to_pixel(robot, height=calibration().head_height)
        cv2.circle(rgb, tuple(np.rint(pixel).astype(int)), 4, (235, 235, 235), -1)
    return rgb


def guard(camera_ids=("A",), **parameters):
    return SceneValidityGuard(
        {name: calibration() for name in camera_ids},
        {name: "registered-v1" for name in camera_ids},
        parameters=SceneValidityParameters(**parameters),
    )


def evidence(rgb, now, *, robot=None, camera_ids=("A",), visible_in=None):
    frames, views = [], {}
    for name in camera_ids:
        pixels = rgb[name] if isinstance(rgb, dict) else rgb
        frames.append(CameraFrame(name, "registered-v1", pixels, now))
        visible = robot is not None and (visible_in is None or name in visible_in)
        empty = np.zeros(pixels.shape[:2], dtype=bool)
        measurement = VisualMeasurement(
            now,
            np.array(robot, float) if visible else None,
            np.eye(2) * 0.002**2 if visible else None,
            empty,
            empty,  # PositionOnlyObserver intentionally supplies no self mask.
            "visible" if visible else "missing",
        )
        views[name] = ViewObservation(
            name, "registered-v1", now, measurement, measurement.status
        )
    return frames, views


def observe(detector, rgb, now, **kwargs):
    frames, views = evidence(
        rgb, now, camera_ids=tuple(detector.calibrations), **kwargs
    )
    return detector.observe(frames, views, now=now)


def warm(detector, rgb=None, **kwargs):
    rgb = scene() if rgb is None else rgb
    first = observe(detector, rgb, 0, **kwargs)
    assert first["status"] == "warming" and not first["navigation_allowed"]
    assert not observe(detector, rgb, 0.5, **kwargs)["navigation_allowed"]
    result = observe(detector, rgb, 1, **kwargs)
    assert result["ready"] and result["navigation_allowed"], result
    return result


def test_stationary_startup_requires_distinct_fresh_samples_and_is_bounded():
    detector = guard()
    for step in range(501):
        result = observe(detector, scene(), step / 100)
        assert result["timestamp"] == result["observed_at"] == step / 100
    assert result["checks"] == 11
    assert result["checked_at"] == 5
    assert result["navigation_allowed"]
    assert result["parameters"]["sample_seconds"] == 0.5
    assert result["cameras"]["A"]["analysis_resolution"] == [192, 144]


def test_moved_obstacle_denies_first_suspicion_then_latches_without_reference_update():
    detector = guard()
    original = warm(detector)
    changed = observe(detector, scene(object_x=65), 1.5)
    assert not changed["navigation_allowed"] and not changed["invalidated"]
    assert changed["reason"] == "scene_content_changed"
    assert changed["cameras"]["A"]["largest_changed_component"] > 24
    unchanged_timestamp = observe(detector, scene(), 1.6)
    assert not unchanged_timestamp["navigation_allowed"]  # No unchecked readmission.
    latched = observe(detector, scene(object_x=65), 2)
    assert latched["invalidated"] and latched["invalidated_at"] == 2
    restored = observe(detector, scene(), 3)
    assert restored["invalidated"] and not restored["navigation_allowed"]
    assert (
        restored["cameras"]["A"]["reference_sha256"]
        == original["cameras"]["A"]["reference_sha256"]
    )


def test_transient_suspicion_clears_only_after_another_structural_check():
    detector = guard()
    warm(detector)
    assert not observe(detector, scene(object_x=65), 1.5)["navigation_allowed"]
    assert not observe(detector, scene(), 1.8)["navigation_allowed"]
    recovered = observe(detector, scene(), 2)
    assert recovered["navigation_allowed"] and not recovered["invalidated"]
    # The caller has already cancelled its goal/authority; the guard issues none.
    assert "goal" not in recovered and "action" not in recovered


@pytest.mark.parametrize("motion", ["translation", "rotation"])
def test_distributed_camera_motion_is_distinguished_from_local_object_change(motion):
    detector = guard()
    warm(detector)
    transform = (
        np.array([[1.0, 0, 6], [0, 1, 4]])
        if motion == "translation"
        else cv2.getRotationMatrix2D((96, 72), 4, 1)
    )
    moved = cv2.warpAffine(
        scene(), transform, (192, 144), borderMode=cv2.BORDER_REFLECT
    )
    first = observe(detector, moved, 1.5)
    assert first["reason"] == "camera_or_global_scene_change", first
    assert not first["navigation_allowed"]
    assert first["cameras"]["A"]["camera_support_regions"] >= 3
    assert observe(detector, moved, 2)["invalidated"]


@pytest.mark.parametrize(
    "lighting", ["brighter", "dimmer", "gain_bias", "smooth_gradient", "noise"]
)
def test_moderate_photometric_changes_do_not_change_geometry(lighting):
    detector = guard()
    warm(detector)
    image = scene().astype(float)
    if lighting == "brighter":
        image += 30
    elif lighting == "dimmer":
        image -= 20
    elif lighting == "gain_bias":
        image = image * 0.7 + 25
    elif lighting == "smooth_gradient":
        image += np.linspace(-12, 12, 192)[None, :, None]
    else:
        image += np.random.default_rng(19).integers(-5, 6, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    for now in (1.5, 2, 3):
        result = observe(detector, image, now)
        assert result["navigation_allowed"] and not result["invalidated"], result


def test_rgb_robot_projection_masks_motion_without_floor_or_native_masks():
    detector = guard()
    warm(detector, scene(robot=(0, 0)), robot=(0, 0))
    for index, x in enumerate((0.2, 0.4, 0.7, 1.0)):
        result = observe(detector, scene(robot=(x, 0)), 1.5 + index * 0.5, robot=(x, 0))
        assert result["navigation_allowed"], result
        assert 0 < result["cameras"]["A"]["masked_fraction"] < 0.15
    moved_obstacle = observe(
        detector, scene(object_x=65, robot=(1, 0)), 4, robot=(1, 0)
    )
    assert not moved_obstacle["navigation_allowed"]
    assert moved_obstacle["reason"] == "scene_content_changed"


def test_another_rgb_view_can_mask_robot_in_a_temporarily_occluded_view():
    detector = guard(("A", "B", "C"))
    warm(detector, scene(robot=(0, 0)), robot=(0, 0), visible_in={"B"})
    result = observe(
        detector, scene(robot=(0.6, 0)), 1.5, robot=(0.6, 0), visible_in={"C"}
    )
    assert result["navigation_allowed"], result
    assert all(report["robot_estimates"] == 1 for report in result["cameras"].values())
    moved = {name: scene(robot=(0.6, 0)) for name in "ABC"}
    moved["A"] = scene(object_x=65, robot=(0.6, 0))
    result = observe(detector, moved, 2, robot=(0.6, 0), visible_in={"C"})
    assert not result["navigation_allowed"] and result["camera_id"] == "A"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_frame",
        "duplicate",
        "wrong_version",
        "stale",
        "repeated",
        "out_of_order",
        "missing_view",
        "missing_measurement",
        "predicted_measurement",
        "bad_covariance",
    ],
)
def test_malformed_or_stale_evidence_fails_closed_even_between_structural_checks(
    failure,
):
    detector = guard()
    warm(detector)
    now = 1.1
    capture = (
        1
        if failure == "repeated"
        else 0.9
        if failure in ("stale", "out_of_order")
        else now
    )
    frames, views = evidence(scene(), capture)
    if failure == "missing_frame":
        frames = []
    elif failure == "duplicate":
        frames *= 2
    elif failure == "wrong_version":
        frames = [replace(frames[0], calibration_version="moved")]
    elif failure == "missing_view":
        views = {}
    elif failure == "missing_measurement":
        views["A"] = replace(views["A"], measurement=None)
    elif failure == "predicted_measurement":
        views["A"] = replace(
            views["A"], measurement=replace(views["A"].measurement, status="predicted")
        )
    elif failure == "bad_covariance":
        frames, views = evidence(scene(robot=(0, 0)), capture, robot=(0, 0))
        views["A"] = replace(
            views["A"],
            measurement=replace(views["A"].measurement, covariance=np.eye(2) * -0.1),
        )
    result = detector.observe(frames, views, now=now)
    assert result["invalidated"] and not result["navigation_allowed"]
    assert result["timestamp"] == now
    assert result["checks"] == 3  # Rejection did not conduct or credit a new check.


def test_missing_textured_reference_and_excessive_dynamic_exclusion_fail_closed():
    detector = guard()
    result = observe(detector, np.zeros((144, 192, 3), np.uint8), 0)
    assert result["invalidated"] and not result["ready"]
    assert "contrast" in result["reason"]
    detector = guard(robot_radius_m=2)
    result = observe(detector, scene(robot=(0, 0)), 0, robot=(0, 0))
    assert result["invalidated"] and "coverage" in result["reason"]


def test_snapshots_and_renderer_buffer_reuse_cannot_change_reference():
    detector = guard()
    image = scene()
    original = warm(detector, image)
    original["cameras"]["A"]["reference_sha256"] = "tampered"
    image[:] = 0
    assert detector.snapshot()["cameras"]["A"]["reference_sha256"] != "tampered"
    result = observe(detector, scene(), 1.5)
    assert result["navigation_allowed"]


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True])
def test_invalid_now_is_an_explicit_latched_rejection(value):
    detector = guard()
    frames, views = evidence(scene(), 0)
    result = detector.observe(frames, views, now=value)
    assert result["invalidated"] and not result["navigation_allowed"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"sample_seconds": 0},
        {"warmup_samples": True},
        {"confirmation_samples": 1},
        {"analysis_width": 10000},
        {"maximum_mask_fraction": 0.9},
        {"minimum_features": 40, "max_features": 20},
    ],
)
def test_parameters_have_finite_work_and_evidence_bounds(parameters):
    with pytest.raises(ValueError):
        SceneValidityParameters(**parameters)


def test_missing_calibration_reference_is_rejected():
    with pytest.raises(ValueError):
        SceneValidityGuard({}, {})
    with pytest.raises(ValueError):
        SceneValidityGuard({"A": calibration()}, {})


def faulted_guard(camera_ids=("A",)):
    detector = guard(camera_ids)
    warm(detector)
    observe(detector, scene(object_x=65), 1.5)
    fault = observe(detector, scene(object_x=65), 2)
    assert fault["fault_epoch"] == 1 and fault["invalidated"]
    return detector


def test_explicit_restoration_requires_three_spaced_fresh_checks_and_keeps_reference():
    detector = faulted_guard()
    reference = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in detector._reference["A"].items()
    }
    assert observe(detector, scene(), 2.1)["invalidated"]
    armed = detector.request_recheck(4, now=2.1)
    assert armed["recovery"]["stable_checks"] == 0
    assert observe(detector, scene(), 2.2)["recovery"]["stable_checks"] == 1
    assert observe(detector, scene(), 2.3)["recovery"]["stable_checks"] == 1
    assert observe(detector, scene(), 2.7)["invalidated"]
    result = observe(detector, scene(), 3.2)
    assert result["navigation_allowed"] and result["fault_epoch"] == 1
    assert result["recovery"] == dict(
        armed["recovery"], stable_checks=3, status="succeeded", completed_at=3.2
    )
    for key, value in reference.items():
        assert np.array_equal(detector._reference["A"][key], value)
    assert detector._last_recheck_generation == 4


@pytest.mark.parametrize(
    "bad", ["changed", "repeated", "missing", "wrong_version", "timeout"]
)
def test_recheck_failure_never_adopts_bad_reference_or_retries_itself(bad):
    detector = faulted_guard()
    original = detector._reference["A"]["gray"].copy()
    detector.request_recheck(4, now=2.1)
    observe(detector, scene(), 2.2)
    when = 2.7 if bad != "timeout" else 12.2
    frames, views = evidence(scene(object_x=65) if bad == "changed" else scene(), when)
    if bad == "repeated":
        frames, views = evidence(scene(), 2.2)
    elif bad == "missing":
        frames = []
    elif bad == "wrong_version":
        frames = [replace(frames[0], calibration_version="other")]
    result = detector.observe(frames, views, now=when)
    assert result["invalidated"] and result["recovery"]["status"] == "rejected"
    assert observe(detector, scene(), when + 1)["invalidated"]
    assert np.array_equal(detector._reference["A"]["gray"], original)
    detector.request_recheck(5, now=when + 1.1)
    for offset in (0.2, 0.7, 1.2):
        result = observe(detector, scene(), when + 1 + offset)
    assert result["navigation_allowed"] and result["fault_epoch"] == 1


def test_recheck_all_views_must_match_and_reference_cannot_be_blessed_during_warmup():
    incomplete = guard()
    observe(incomplete, scene(), 0)
    observe(incomplete, scene(object_x=65), 0.5)
    assert observe(incomplete, scene(object_x=65), 1)["invalidated"]
    with pytest.raises(ValueError, match="original ready reference"):
        incomplete.request_recheck(1, now=1.1)
    detector = faulted_guard(("A", "B"))
    detector.request_recheck(1, now=2.1)
    result = observe(detector, {"A": scene(), "B": scene(object_x=65)}, 2.2)
    assert result["invalidated"] and result["recovery"]["status"] == "rejected"
    assert result["camera_id"] == "B"


def test_late_stop_acknowledgement_loss_can_retry_same_fault_then_new_change_increments_epoch():
    detector = faulted_guard()
    detector.request_recheck(1, now=2.1)
    for timestamp in (2.2, 2.7, 3.2):
        result = observe(detector, scene(), timestamp)
    assert result["navigation_allowed"]
    # Supervisor may reject generation 1 after Stop generation 2. A retry at 3
    # reasserts the same latch and checks the original reference again.
    retried = detector.request_recheck(3, now=3.3)
    assert retried["invalidated"] and retried["fault_epoch"] == 1
    detector.cancel_recheck("Stop")
    assert observe(detector, scene(), 4)["recovery"]["status"] == "cancelled"
    with pytest.raises(ValueError):
        detector.request_recheck(3, now=4.1)
    detector.request_recheck(5, now=4.1)
    for timestamp in (4.2, 4.7, 5.2):
        result = observe(detector, scene(), timestamp)
    assert result["navigation_allowed"] and result["fault_epoch"] == 1
    observe(detector, scene(object_x=65), 5.7)
    result = observe(detector, scene(object_x=65), 6.2)
    assert result["invalidated"] and result["fault_epoch"] == 2
