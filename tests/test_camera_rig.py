"""Causal RGB rig contracts; no renderer or privileged robot state."""

from dataclasses import replace

import numpy as np
import pytest

from bb8_rl.camera import Calibration, VisualMeasurement
from bb8_rl.camera_rig import (
    CameraFrame,
    CameraRig,
    CameraView,
    calibration_from_live_camera,
)


@pytest.fixture
def calibration():
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 3.0
    return Calibration(
        np.array([[80.0, 0, 32], [0, 80.0, 24], [0, 0, 1]]),
        transform,
        (64, 48),
        1.0,
    )


class Observer:
    def __init__(
        self, calibration, *, xy=(0.1, 0.1), covariance=None, status="visible"
    ):
        self.calibration = calibration
        self.xy = xy
        self.covariance = np.eye(2) * 0.01**2 if covariance is None else covariance
        self.status = status
        self.calls = 0
        self.override = {}

    def observe(self, rgb, timestamp):
        self.calls += 1
        measurement = VisualMeasurement(
            timestamp,
            np.asarray(self.xy) if self.status == "visible" else None,
            self.covariance if self.status == "visible" else None,
            np.ones(rgb.shape[:2], bool),
            np.zeros(rgb.shape[:2], bool),
            self.status,
        )
        return replace(measurement, **self.override)


def view(calibration, camera_id="A", **kwargs):
    return CameraView(camera_id, "v1", calibration, Observer(calibration, **kwargs))


def frame(camera_id="A", time=0.0, version="v1"):
    return CameraFrame(camera_id, version, np.zeros((48, 64, 3), np.uint8), time)


def test_frame_owns_rgb_and_rejects_invalid_contract():
    pixels = np.zeros((48, 64, 3), np.uint8)
    observation = CameraFrame("A", "v1", pixels, 0)
    pixels[:] = 255
    assert not observation.rgb.any()
    assert not observation.rgb.flags.writeable
    with pytest.raises(ValueError):
        observation.rgb[:] = 255
    for camera_id, version, rgb, timestamp in [
        ("", "v1", pixels, 0),
        ("A", "", pixels, 0),
        ("A", "v1", pixels.astype(float), 0),
        ("A", "v1", pixels[:, :, 0], 0),
        ("A", "v1", pixels, float("nan")),
        ("A", "v1", pixels, -1),
    ]:
        with pytest.raises(ValueError):
            CameraFrame(camera_id, version, rgb, timestamp)


def test_rig_has_strict_three_camera_cap_and_unique_ids(calibration):
    cameras = [view(calibration, camera_id) for camera_id in "ABCD"]
    for invalid in [[], cameras, [cameras[0], cameras[0]]]:
        with pytest.raises(ValueError, match="one to three unique"):
            CameraRig(invalid)
    rig = CameraRig(cameras[:3])
    for invalid in [[frame(i) for i in "ABCD"], [frame(), frame()], [frame("D")]]:
        with pytest.raises(ValueError):
            rig.observe(invalid, now=0)
    result = rig.observe([frame(i) for i in "ABC"], now=0)
    assert result.status == "visible" and result.source_ids == ("A", "B", "C")


@pytest.mark.parametrize(
    "observation,now,expected",
    [
        (frame("B"), 0, "wrong_camera"),
        (frame(version="v2"), 0, "calibration_mismatch"),
        (frame(time=0.2), 0.1, "future_frame"),
        (frame(time=0), 0.11, "stale_frame"),
        (
            CameraFrame("A", "v1", np.zeros((10, 10, 3), np.uint8), 0),
            0,
            "invalid_frame",
        ),
    ],
)
def test_invalid_frame_never_invokes_observer(calibration, observation, now, expected):
    camera = view(calibration)
    result = camera.observe(observation, now=now)
    assert result.status == expected and result.measurement is None
    assert camera.observer.calls == 0


def test_repeated_frame_is_not_an_independent_observation(calibration):
    camera = view(calibration)
    rig = CameraRig([camera])
    assert rig.observe([frame()], now=0).status == "visible"
    result = rig.observe([frame()], now=0.01)
    assert result.status == "no_fresh_measurement" and result.xy is None
    assert result.views["A"].status == "repeated_frame"
    assert camera.observer.calls == 1


def test_identical_correlated_views_do_not_divide_covariance(calibration):
    covariance = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
    for count in (1, 2, 3):
        ids = "ABC"[:count]
        rig = CameraRig([view(calibration, i, covariance=covariance) for i in ids])
        result = rig.observe([frame(i) for i in ids], now=0)
        np.testing.assert_allclose(result.covariance, covariance, rtol=1e-12)
        np.testing.assert_allclose(result.xy, [0.1, 0.1])


def test_ci_handles_complementary_axes_and_is_order_invariant(calibration):
    first = np.diag([0.0001, 0.0009])
    second = np.diag([0.0009, 0.0001])
    expected = 2 * np.linalg.inv(np.linalg.inv(first) + np.linalg.inv(second))
    results = []
    for order in ("AB", "BA"):
        cameras = {
            "A": view(calibration, "A", covariance=first),
            "B": view(calibration, "B", covariance=second),
        }
        rig = CameraRig([cameras[i] for i in order])
        results.append(rig.observe([frame(i) for i in order], now=0))
    for result in results:
        assert result.status == "visible"
        np.testing.assert_allclose(result.covariance, expected)
        assert np.linalg.eigvalsh(result.covariance).min() > 0
    np.testing.assert_allclose(results[0].xy, results[1].xy)


def test_skew_inflation_includes_unknown_motion_error_correlation(calibration):
    # A's error and unobserved displacement could have the same direction:
    # sigma² + d² alone misses the 2 sigma d cross term.
    rig = CameraRig([view(calibration, i) for i in "AB"], velocity_bound=0.35)
    result = rig.observe([frame("A", 0), frame("B", 0.04)], now=0.04)
    older_variance = (0.01 + 0.35 * 0.04) ** 2
    expected = 2 / (1 / older_variance + 1 / 0.01**2)
    assert result.status == "visible" and result.timestamp == 0.04
    np.testing.assert_allclose(result.covariance, expected * np.eye(2), rtol=1e-12)
    assert result.covariance[0, 0] > 0.01**2


def test_excessive_skew_excludes_old_view_without_shrinking_uncertainty(calibration):
    rig = CameraRig([view(calibration, i) for i in "AB"])
    result = rig.observe([frame("A", 0), frame("B", 0.06)], now=0.06)
    assert result.status == "visible" and result.source_ids == ("B",)
    assert result.views["A"].status == "skewed_frame"
    np.testing.assert_allclose(result.covariance, np.eye(2) * 0.01**2)


@pytest.mark.parametrize("invalid", ["missing", "stale", "wrong_version"])
def test_fresh_camera_handover_survives_other_view_unavailability(calibration, invalid):
    first, second = [view(calibration, i) for i in "AB"]
    rig = CameraRig([first, second])
    initial = rig.observe([frame()], now=0)
    assert initial.status == "visible" and not initial.handover
    captures = [frame("B", 0.2)]
    if invalid == "stale":
        captures.append(frame("A", 0.05))
    elif invalid == "wrong_version":
        captures.append(frame("A", 0.2, version="v2"))
    result = rig.observe(captures, now=0.2)
    assert result.status == "visible" and result.source_ids == ("B",)
    assert result.handover
    np.testing.assert_allclose(result.xy, initial.xy)
    np.testing.assert_allclose(result.covariance, initial.covariance)


def test_missing_is_not_a_predicted_or_replayed_position(calibration):
    camera = view(calibration)
    rig = CameraRig([camera])
    assert rig.observe([frame()], now=0).status == "visible"
    camera.observer.status = "missing"
    result = rig.observe([frame(time=0.05)], now=0.05)
    assert result.status == "missing"
    assert result.xy is None and result.covariance is None and not result.source_ids
    camera.observer.status = "predicted"
    result = rig.observe([frame(time=0.1)], now=0.1)
    assert result.xy is None and result.covariance is None
    assert result.views["A"].status == "unobserved"


@pytest.mark.parametrize(
    "second_xy,covariance",
    [((0.6, 0.1), np.eye(2) * 0.2**2), ((0.15, 0.1), np.eye(2) * 0.001**2)],
)
def test_inconsistent_identity_rejects_every_view(calibration, second_xy, covariance):
    rig = CameraRig(
        [
            view(calibration, "A", covariance=covariance),
            view(calibration, "B", xy=second_xy, covariance=covariance),
        ]
    )
    result = rig.observe([frame(i) for i in "AB"], now=0)
    assert result.status == "inconsistent_views"
    assert result.xy is None and result.covariance is None and not result.source_ids


def test_ambiguous_view_does_not_get_voted_out(calibration):
    rig = CameraRig([view(calibration), view(calibration, "B", status="ambiguous")])
    result = rig.observe([frame(i) for i in "AB"], now=0)
    assert result.status == "ambiguous" and result.xy is None


def test_handover_rejects_impossible_identity_jump(calibration):
    rig = CameraRig([view(calibration), view(calibration, "B", xy=(0.7, 0.1))])
    assert rig.observe([frame()], now=0).status == "visible"
    result = rig.observe([frame("B", 0.05)], now=0.05)
    assert result.status == "inconsistent_history" and result.xy is None


def test_handover_accepts_motion_within_elapsed_velocity_bound(calibration):
    rig = CameraRig([view(calibration), view(calibration, "B", xy=(0.1, 0.14))])
    rig.observe([frame()], now=0)
    # An intervening dropout advances time but supplies no measured position.
    assert rig.observe([], now=0.1).xy is None
    result = rig.observe([frame("B", 0.2)], now=0.2)
    assert result.status == "visible" and result.handover
    np.testing.assert_allclose(result.xy, [0.1, 0.14])


def test_out_of_order_handover_is_not_a_new_measurement(calibration):
    rig = CameraRig([view(calibration), view(calibration, "B")])
    rig.observe([frame(time=0.05)], now=0.05)
    result = rig.observe([frame("B", 0.04)], now=0.06)
    assert result.status == "out_of_order" and result.xy is None
    with pytest.raises(ValueError, match="backwards"):
        rig.observe([], now=0.02)


@pytest.mark.parametrize(
    "override",
    [
        {"xy": np.array([float("nan"), 0])},
        {"xy": ["invalid", 0]},
        {"xy": np.array([2, 0])},
        {"covariance": np.diag([-0.01, 0.01])},
        {"covariance": np.zeros((2, 2))},
        {"covariance": np.array([[0.001, 0.002], [0, 0.001]])},
        {"timestamp": 0.01},
        {"timestamp": "invalid"},
        {"visible_floor": np.zeros((48, 64), np.uint8)},
        {"robot_pixels": np.zeros((1, 1), bool)},
        {"status": "missing"},
    ],
)
def test_invalid_observer_output_is_never_fused(calibration, override):
    camera = view(calibration)
    camera.observer.override = override
    result = CameraRig([camera]).observe([frame()], now=0)
    assert result.xy is None and result.views["A"].status == "invalid_measurement"


def test_view_registration_rejects_wrong_observer_calibration(calibration):
    moved = replace(calibration, world_to_camera=calibration.world_to_camera.copy())
    moved.world_to_camera[0, 3] = 0.2
    with pytest.raises(ValueError, match="differs"):
        CameraView("A", "v1", calibration, Observer(moved))


def test_calibration_array_mutation_requires_reregistration(calibration):
    camera = view(calibration)
    original_transform = camera.calibration.world_to_camera.copy()
    calibration.world_to_camera[0, 3] = 0.3
    result = camera.observe(frame(), now=0)
    assert result.status == "calibration_mismatch" and result.measurement is None
    assert camera.observer.calls == 0
    np.testing.assert_array_equal(
        camera.calibration.world_to_camera, original_transform
    )


def test_live_calibration_reads_current_transform_never_cached_extrinsics(calibration):
    class Camera:
        intrinsics = calibration.intrinsics.copy()
        res = calibration.resolution
        transform = np.linalg.inv(calibration.world_to_camera)
        transform[:3, 1:3] *= -1

        @property
        def extrinsics(self):
            raise AssertionError("Cached camera extrinsics must never be read")

    camera = Camera()
    first = calibration_from_live_camera(camera, calibration.extent)
    np.testing.assert_allclose(first.world_to_camera, calibration.world_to_camera)
    # Change translation and rotation, as Genesis set_pose does for a new view.
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    camera.transform[:3, :3] = rotation @ camera.transform[:3, :3]
    camera.transform[:3, 3] += [0.4, -0.2, 0.0]
    live_transform = camera.transform.copy()
    second = calibration_from_live_camera(camera, calibration.extent)
    expected_camera_to_world = live_transform.copy()
    expected_camera_to_world[:3, 1:3] *= -1
    np.testing.assert_allclose(
        second.world_to_camera, np.linalg.inv(expected_camera_to_world)
    )
    np.testing.assert_array_equal(camera.transform, live_transform)
    np.testing.assert_allclose(first.world_to_camera, calibration.world_to_camera)
    assert not np.allclose(first.to_pixel([0, 0]), second.to_pixel([0, 0]))
    point = np.array([0.3, -0.4])
    np.testing.assert_allclose(
        second.to_plane(second.to_pixel(point)), point, atol=1e-12
    )
    assert second.provenance == "synthetic_exact_live_transform"


@pytest.mark.parametrize(
    "positions",
    [[], [(0, 0, 2)] * 2, [(i, 0, 2) for i in range(4)], [(0, float("nan"), 2)]],
)
def test_world_rejects_invalid_rig_before_loading_scene(positions):
    from pathlib import Path

    from bb8_rl.world import NavigationWorld

    with pytest.raises(ValueError, match="one to three distinct finite"):
        NavigationWorld(
            Path("not-a-scene.yaml"), render=True, camera_positions=positions
        )
