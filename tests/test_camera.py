import cv2
import numpy as np
import pytest

from bb8_rl.camera import (
    Calibration,
    CameraController,
    PositionBelief,
    SyntheticPaletteObserver,
    VisualMeasurement,
    camera_grid,
)


@pytest.fixture
def calibration():
    k = np.array([[600.0, 0, 640], [0, 600.0, 480], [0, 0, 1]])
    t = np.diag([1.0, -1.0, -1.0, 1.0])
    t[2, 3] = 3
    return Calibration(k, t, (1280, 960), 2)


def image(calibration, position=(0, 0), *, second=None):
    rgb = np.full((960, 1280, 3), [27, 47, 67], np.uint8)
    for xy in [position, second] if second is not None else [position]:
        center = tuple(
            np.rint(calibration.to_pixel(xy, calibration.head_height)).astype(int)
        )
        cv2.circle(rgb, center, 3, (150, 150, 150), -1)
    return rgb


def test_plane_roundtrip_and_height_correction(calibration):
    points = np.array([[1.3, -1.3], [-1, 0.5], [0, 0]])
    pixels = calibration.to_pixel(points, calibration.head_height)
    np.testing.assert_allclose(
        calibration.to_plane(pixels, calibration.head_height), points, atol=1e-12
    )
    assert np.linalg.norm(calibration.to_plane(pixels)[0] - points[0]) > 0.04
    covariance = calibration.measurement_covariance(pixels[0])
    assert np.linalg.eigvalsh(covariance).min() > 0
    with pytest.raises(ValueError, match="behind"):
        calibration.to_pixel(points, 4)


def test_rgb_detection_and_ambiguity(calibration):
    observer = SyntheticPaletteObserver(calibration)
    measured = observer.observe(image(calibration, (0.4, -0.3)), 0)
    assert measured.status == "visible"
    np.testing.assert_allclose(measured.xy, [0.4, -0.3], atol=0.004)
    assert (
        observer.observe(image(calibration, second=(0.6, 0.6)), 0.1).status
        == "ambiguous"
    )
    assert observer.observe(np.zeros((960, 1280, 3), np.uint8), 0.2).status == "missing"
    with pytest.raises(ValueError, match="resolution"):
        observer.observe(np.zeros((20, 20, 3), np.uint8), 0.3)


def measurement(t, xy=None):
    return VisualMeasurement(
        t,
        np.array(xy) if xy is not None else None,
        np.eye(2) * 0.005**2 if xy is not None else None,
        np.zeros((1, 1), bool),
        np.zeros((1, 1), bool),
        "visible" if xy is not None else "missing",
    )


def test_temporal_velocity_dropout_and_outlier():
    belief = PositionBelief()
    for i in range(30):
        t = i * 0.05
        belief.update(measurement(t, [0.1 * t, -0.05 * t]))
    np.testing.assert_allclose(belief.state[2:], [0.1, -0.05], atol=0.004)
    assert belief.ready(1.45) and not belief.ready(1.6)
    belief.update(measurement(1.5, [1.5, 1.5]))
    assert belief.status == "outlier" and not belief.ready(1.5)
    belief.update(measurement(1.55))
    assert not belief.ready(1.55)
    with pytest.raises(ValueError, match="increase"):
        belief.update(measurement(1.55))


def test_unknown_floor_blocks_routes(calibration):
    observer = SyntheticPaletteObserver(calibration)
    rgb = image(calibration, (-0.7, -0.7))
    polygon = np.rint(
        calibration.to_pixel([[-0.2, -0.4], [0.2, -0.4], [0.2, 0.4], [-0.2, 0.4]])
    ).astype(np.int32)
    cv2.fillPoly(rgb, [polygon], (200, 30, 30))
    grid = camera_grid(calibration, observer.observe(rgb, 0))
    assert not grid.free(grid.cell([0, 0]))
    assert not grid.segment_free([-0.7, 0], [0.7, 0])
    route = grid.route([-0.7, 0], [0.7, 0])
    assert len(route) > 2


def test_controller_uses_visual_estimate_and_stops_on_stale_or_missing(calibration):
    seen = []

    def policy(vector):
        seen.append(vector)
        return np.array([0.2, 0], np.float32)

    controller = CameraController(calibration, policy, calibration.to_pixel([0.8, 0]))
    rgb = image(calibration)
    for i in range(2):
        assert not controller.action(rgb, i * 0.05).any()
    np.testing.assert_allclose(controller.action(rgb, 0.1), [0.2, 0])
    assert len(seen) == 1 and seen[0].shape == (5,)
    np.testing.assert_allclose(seen[0][:2], [0.8, 0], atol=0.005)
    assert not controller.action(rgb, 0.1).any()
    assert controller.status == "repeated_frame" and len(seen) == 1
    assert not controller.action(rgb, 0.15, now=0.4).any()
    assert controller.status == "stale_frame" and len(seen) == 1
    assert not controller.action(np.zeros_like(rgb), 0.2).any()
    assert controller.status == "missing" and len(seen) == 1


def test_blocked_goal_never_calls_policy(calibration):
    def forbidden(vector):
        raise AssertionError("Policy must not drive toward unknown floor")

    controller = CameraController(
        calibration, forbidden, calibration.to_pixel([0.8, 0])
    )
    rgb = image(calibration)
    goal = tuple(np.rint(calibration.to_pixel([0.8, 0])).astype(int))
    cv2.circle(rgb, goal, 50, (200, 20, 20), -1)
    for i in range(3):
        assert not controller.action(rgb, i * 0.05).any()
    assert controller.status == "no_visible_route"


def test_near_corner_does_not_skip_waypoint_across_blocked_segment(calibration):
    from bb8_rl.planner import OccupancyGrid

    position = np.array([-0.01, 0.03])

    class FixedObserver:
        def observe(self, rgb, timestamp):
            floor = np.ones((960, 1280), bool)
            return VisualMeasurement(
                timestamp,
                position.copy(),
                np.eye(2) * 0.005**2,
                floor,
                np.zeros_like(floor),
                "visible",
            )

    seen = []

    def policy(vector):
        seen.append(vector)
        return np.zeros(2)

    controller = CameraController(
        calibration,
        policy,
        calibration.to_pixel([0.01, -0.3]),
        observer=FixedObserver(),
    )
    for index in range(3):
        controller.action(None, index * 0.05)
    grid = OccupancyGrid(2, 0.02, 0.1, [])
    axis = -2 + (np.arange(grid.n) + 0.5) * grid.resolution
    grid.blocked |= (axis[:, None] < 0) & (axis[None, :] < 0)
    waypoint, goal = np.array([0.01, 0.03]), np.array([0.01, -0.3])
    assert grid.segment_free(position, waypoint)
    assert grid.segment_free(waypoint, goal)
    assert not grid.segment_free(position, goal)
    controller.grid, controller.route = grid, np.array([position, waypoint, goal])
    controller.index, controller.map_sigma = 1, controller.belief.sigma
    controller.action(None, 0.15)
    assert controller.index == 1 and controller.status == "tracking"
    np.testing.assert_allclose(seen[-1][:2], waypoint - position, atol=1e-7)
