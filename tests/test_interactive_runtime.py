import importlib
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.interactive_runtime import (
    ControlSession,
    drain_commands,
    publish_latest,
    worker_main,
)


class Memory:
    def __init__(self):
        self.blocked = False
        self.route_blocked = False
        self.queries = []
        self.planning_radii = []

    def planning_grid(self, radius):
        self.planning_radii.append(radius)
        return self

    def segment_free(self, a, b, radius):
        self.queries.append((np.array(a), np.array(b), radius))
        return (
            not self.blocked
            and max(abs(np.asarray(a))) < 1
            and max(abs(np.asarray(b))) < 1
        )

    def route(self, start, goal):
        if self.route_blocked:
            raise ValueError("No certified path")
        return np.array([start, goal])

    def certified_route(self, start, goal, radius_m, *, grid=None):
        return (self if grid is None else grid).route(start, goal)


def observed(t, status="visible"):
    return SimpleNamespace(
        timestamp=t,
        status=status,
        xy=np.array([0.0, 0.0]) if status == "visible" else None,
        covariance=np.eye(2) * 0.002**2 if status == "visible" else None,
    )


def session(**options):
    now, stops, memory = [10.0], [], Memory()
    control = ControlSession(
        policy=lambda vector: np.array([0.4, 0]),
        memory=memory,
        stop=lambda: stops.append(now[0]),
        map_version="map",
        calibration_version="rig",
        demo_goal=[0.5, 0],
        clock=lambda: now[0],
        **options,
    )
    return control, memory, now, stops


def scene_evidence(timestamp, *, ready=True, allowed=True, invalidated=False):
    return {
        "timestamp": timestamp,
        "ready": ready,
        "navigation_allowed": allowed,
        "invalidated": invalidated,
        "reason": "test_rgb_evidence",
    }


def test_scene_guard_warmup_and_suspicion_cancel_authority_without_auto_resume():
    control, _, _, _ = session(scene_validity_required=True)
    assert (
        control.receive([{"action": "demo", "generation": 1}])[0]["outcome"]
        == "rejected"
    )
    control.update_scene_validity(scene_evidence(0), 0)
    advance(control, 0)
    assert control.phase == "idle"
    control.receive([{"action": "demo", "generation": 2}])
    assert control.goal is not None
    control.update_scene_validity(scene_evidence(0.05, allowed=False), 0.05)
    assert np.array_equal(advance(control, 1), [0, 0])
    assert control.goal is None and control.state()["pose"] is None
    control.update_scene_validity(scene_evidence(0.10), 0.10)
    assert np.array_equal(advance(control, 2), [0, 0])
    assert control.goal is None and control.requires_new_goal
    for index in range(3, 30):
        control.update_scene_validity(scene_evidence(index * 0.05), index * 0.05)
        assert np.array_equal(advance(control, index), [0, 0])
    assert control.state()["localization_valid"]
    control.receive([{"action": "demo", "generation": 3}])
    assert control.goal is not None


def test_scene_invalidation_latches_against_fresh_pixels_and_new_command_generations():
    control, _, _, stops = session(scene_validity_required=True)
    control.update_scene_validity(scene_evidence(0), 0)
    advance(control, 0)
    control.receive([{"action": "demo", "generation": 1}])
    control.update_scene_validity(
        scene_evidence(0.05, allowed=False, invalidated=True), 0.05
    )
    assert np.array_equal(advance(control, 1), [0, 0])
    control.update_scene_validity(scene_evidence(0.10), 0.10)
    assert (
        control.receive([{"action": "demo", "generation": 2}])[0]["outcome"]
        == "rejected"
    )
    assert np.array_equal(advance(control, 2), [0, 0])
    assert control.state()["scene_validity"]["invalidated"]
    assert control.goal is None and control.state()["pose"] is None and stops
    assert (
        control.receive([{"action": "stop", "generation": 3}])[0]["outcome"]
        == "accepted"
    )
    assert np.array_equal(advance(control, 3), [0, 0])
    assert control.phase == "scene_invalid"


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        scene_evidence(0),
        {**scene_evidence(0.05), "ready": False},
        {**scene_evidence(0.05), "navigation_allowed": "true"},
        {**scene_evidence(0.05), "invalidated": True},
    ],
)
def test_scene_guard_rejects_malformed_stale_or_inconsistent_evidence(record):
    control, _, _, _ = session(scene_validity_required=True)
    control.update_scene_validity(scene_evidence(0), 0)
    advance(control, 0)
    control.update_scene_validity(record, 0.05)
    assert np.array_equal(advance(control, 1), [0, 0])
    assert control.scene_validity["invalidated"]


def test_missing_guard_update_cannot_reuse_last_frame_permission():
    control, _, _, _ = session(scene_validity_required=True)
    control.update_scene_validity(scene_evidence(0), 0)
    advance(control, 0)
    assert np.array_equal(advance(control, 1), [0, 0])
    assert control.scene_validity["invalidated"]


def test_visibility_veto_cannot_fall_back_to_clearance_only_route():
    class Visibility:
        def __init__(self):
            self.diagnostics = {"rejected": "blind_destination"}

        def route(self, *args, **kwargs):
            raise ValueError("blind_destination")

    control, _, _, _ = session(visibility_planner=Visibility())
    with pytest.raises(ValueError, match="blind_destination"):
        control.route_with_clearance([0, 0], [0.5, 0], 0.105)
    assert not control.route_planning_details["route_certified"]
    assert (
        control.route_planning_details["visibility"]["rejected"] == "blind_destination"
    )


def test_visibility_candidate_still_requires_full_footprint_clearance():
    class Visibility:
        def __init__(self):
            self.diagnostics = {"predicted_visible": True}

        def route(self, start, goal, **kwargs):
            return np.array([start, [2.0, 0.0], goal])

    control, _, _, _ = session(visibility_planner=Visibility())
    with pytest.raises(ValueError, match="full requested clearance"):
        control.route_with_clearance([0, 0], [0.5, 0], 0.1)


def advance(control, index, status="visible"):
    t = index * 0.05
    intervals = [] if index == 0 else [{"start": t - 0.05, "end": t, "command": [0, 0]}]
    return control.decide(t, observed(t, status), [0, 0], intervals)


def test_stop_has_batch_priority_and_stale_goal_cannot_resume():
    control, _, _, stops = session()
    records = control.receive(
        [
            {"action": "goal", "generation": 1, "x": 0.5, "y": 0},
            {"action": "stop", "generation": 2},
            {"action": "demo", "generation": 3},
        ]
    )
    assert [r["outcome"] for r in records] == [
        "cancelled_by_stop",
        "accepted",
        "cancelled_by_stop",
    ]
    assert len(stops) == 1
    assert control.goal is None and control.controller is None
    assert control.generation == 3
    control.receive([{"action": "demo", "generation": 3}])
    assert control.goal is None
    assert np.all(advance(control, 0) == 0)
    control.receive([{"action": "demo", "generation": 4}])
    assert control.goal is not None


def test_goal_change_brakes_discards_previous_controller_and_reinitializes():
    control, _, _, stops = session()
    control.receive([{"action": "demo", "generation": 1}])
    for index in range(30):
        action = advance(control, index)
    assert action[0] > 0
    old = control.controller
    control.receive([{"action": "goal", "generation": 2, "x": 0.4, "y": 0.1}])
    assert len(stops) == 2
    assert control.controller is None
    action = advance(control, 30)
    assert control.controller is not old
    assert not control.controller.velocity_initialized
    assert np.all(action == 0)
    assert control.phase == "braking"


def test_heartbeat_expiry_stops_and_returned_heartbeat_does_not_resume():
    control, _, now, stops = session()
    control.receive([{"action": "demo", "generation": 1}])
    advance(control, 0)
    now[0] += 3.01
    assert np.all(advance(control, 1) == 0)
    assert control.phase == "heartbeat_expired"
    assert control.controller is None and control.goal is None
    stopped_count = len(stops)
    control.receive([{"action": "heartbeat", "generation": 1}])
    advance(control, 2)
    assert control.phase == "heartbeat_expired"
    assert len(stops) == stopped_count
    control.receive([{"action": "demo", "generation": 2}])
    advance(control, 3)
    assert control.controller is not None


@pytest.mark.parametrize("point", [[2, 0], [np.nan, 0], [None, 0]])
def test_goal_rejection_never_calls_policy_or_preserves_old_motion(point):
    control, _, _, stops = session()
    records = control.receive(
        [{"action": "goal", "generation": 1, "x": point[0], "y": point[1]}]
    )
    assert records[0]["outcome"] == "rejected"
    assert len(stops) == 1
    assert control.phase == "goal_rejected"
    assert control.goal is None
    assert np.all(advance(control, 0) == 0)


def test_goal_waits_for_measured_pose_and_rejects_unknown_route():
    control, memory, _, _ = session()
    control.receive([{"action": "demo", "generation": 1}])
    for index in range(4):
        advance(control, index, "missing")
        assert control.controller is None
    memory.route_blocked = True
    advance(control, 4)
    assert control.phase == "goal_rejected"
    assert control.goal is None and control.controller is None
    assert "certified" in control.message


def test_idle_localization_continues_with_zero_action_and_no_truth_input():
    control, _, _, _ = session()
    assert control.phase == "localizing"
    for index in range(10):
        assert np.all(advance(control, index) == 0)
    assert control.state()["pose"] == [0, 0]
    assert control.idle_belief.samples == 10
    assert control.phase == "idle"
    advance(control, 10, "missing")
    assert control.phase == "localizing"
    assert control.state()["pose_source"] == "predicted"


def test_current_rgb_uncertainty_reaches_preflight_and_controller_planner():
    control, memory, _, _ = session()
    control.receive([{"action": "demo", "generation": 1}])
    for index in range(30):
        timestamp = index * 0.05
        measurement = observed(timestamp)
        measurement.covariance = np.eye(2) * 0.02**2
        intervals = (
            []
            if index == 0
            else [{"start": timestamp - 0.05, "end": timestamp, "command": [0, 0]}]
        )
        action = control.decide(timestamp, measurement, [0, 0], intervals)
    required = 0.037 + 0.04 + 3 * 0.02
    assert memory.planning_radii == pytest.approx([0.1, 0.14])
    assert control.controller.route_planning_radius == pytest.approx(required)
    assert action[0] > 0
    assert control.status == "tracking_visible"
    assert any(radius == pytest.approx(required) for _, _, radius in memory.queries)


def test_planning_grid_cache_rounds_up_and_is_bounded():
    control, memory, _, _ = session()
    control.route_with_clearance([0, 0], [0.5, 0], 0.13001)
    control.route_with_clearance([0, 0], [0.5, 0], 0.13999)
    assert memory.planning_radii == pytest.approx([0.1, 0.14])
    for radius in np.linspace(0.15, 0.8, 35):
        control.route_with_clearance([0, 0], [0.5, 0], radius)
        assert memory.planning_radii[-1] >= radius
    assert len(control.route_grids) == 16
    for radius in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="clearance"):
            control.route_with_clearance([0, 0], [0.5, 0], radius)


def test_selected_parameters_reach_idle_and_active_control_with_planning_reserve():
    from bb8_rl.control_profiles import get_profile

    profile, memory = get_profile("reserve-3cm"), Memory()
    parameters = profile.parameters()
    control = ControlSession(
        policy=lambda vector: np.array([0.4, 0]),
        memory=memory,
        stop=lambda: None,
        map_version="map",
        calibration_version="rig",
        demo_goal=[0.5, 0],
        parameters=parameters,
        planning_reserve=profile.planning_reserve_m,
        clock=lambda: 10.0,
    )
    assert control.idle_belief.parameters is parameters
    control.receive([{"action": "demo", "generation": 1}])
    for index in range(30):
        action = advance(control, index)
    assert control.controller.parameters is parameters
    assert 0 < np.linalg.norm(control.dynamics.target(action)) <= parameters.speed_cap
    assert memory.planning_radii[-1] == pytest.approx(0.12)
    assert all(radius >= 0.037 + 0.03 + 0.006 for _, _, radius in memory.queries)


def test_latest_queue_is_bounded_and_newest_event_survives():
    events = queue.Queue(maxsize=1)
    assert publish_latest(events, {"type": "old"})
    assert publish_latest(events, {"type": "new"})
    assert events.get_nowait() == {"type": "new"}
    commands = queue.Queue()
    for index in range(5):
        commands.put(index)
    assert drain_commands(commands, limit=3) == [0, 1, 2]
    assert drain_commands(commands) == [3, 4]


def test_runtime_import_does_not_require_genesis_or_torch(monkeypatch):
    import bb8_rl.interactive_runtime as runtime

    monkeypatch.setitem(sys.modules, "genesis", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    importlib.reload(runtime)


def test_worker_configuration_failure_emits_error_without_native_import(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "genesis", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    events = queue.Queue(maxsize=8)
    worker_main(
        {"mode": 4, "generation": 7},
        queue.Queue(),
        events,
        SimpleNamespace(is_set=lambda: False),
    )
    outputs = drain_commands(events)
    assert outputs[0]["type"] == "error"
    assert outputs[1]["state"]["phase"] == "error"
    assert outputs[1]["state"]["generation"] == 7


def test_worker_does_not_wait_on_detached_presentation_reader():
    import multiprocessing

    import bb8_rl.interactive_runtime as runtime

    context = multiprocessing.get_context("spawn")
    commands, events, stop = context.Queue(2), context.Queue(8), context.Event()
    # Fill the OS pipe, leaving no consumer. The child can enqueue a fault but
    # must not wait for its feeder at interpreter shutdown.
    events.put(b"x" * 1_000_000)
    process = context.Process(
        target=runtime.worker_main, args=({"mode": 4}, commands, events, stop)
    )
    try:
        process.start()
        process.join(timeout=8)
        assert not process.is_alive(), "Finished worker waited on presentation pipe"
        assert process.exitcode == 0
    finally:
        if process.pid is not None and process.is_alive():
            process.kill()
            process.join(timeout=2)
        for channel in (commands, events):
            channel.cancel_join_thread()
            channel.close()


def test_worker_rejects_task_override_before_loading_models(tmp_path, monkeypatch):
    import bb8_rl.demo_assets as assets

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(assets, "validate_assets", lambda _path: {"task": "task.yaml"})
    (tmp_path / "bundle.json").write_text("{}")
    events = queue.Queue(maxsize=8)
    worker_main(
        {
            "mode": 1,
            "asset_dir": str(tmp_path),
            "run_dir": str(tmp_path / "run"),
            "task_path": str(tmp_path / "unchecked-task.yaml"),
        },
        queue.Queue(),
        events,
        SimpleNamespace(is_set=lambda: False),
    )
    outputs = drain_commands(events)
    assert outputs[0]["type"] == "error"
    assert "checksummed demo task" in outputs[0]["message"]


@pytest.mark.parametrize(
    "required,expected",
    [
        (np.nextafter(0.14, -np.inf), 0.14),
        (0.14, 0.14),
        (np.nextafter(0.14, np.inf), 0.16),
    ],
)
def test_planning_quantization_preserves_exact_and_adjacent_float_boundaries(
    required, expected
):
    control, memory, _, _ = session()
    control.route_with_clearance([0, 0], [0.5, 0], float(required))
    # Exact comparisons are intentional: approx would hide an unsafe epsilon fix.
    assert memory.planning_radii[-1] == expected
    assert memory.planning_radii[-1] >= required
    assert control.route_planning_details["required_radius_m"] == required
    assert control.route_planning_details["route_certified"]


def test_quantization_never_rounds_below_request_at_all_room_scale_boundaries():
    for step in range(5, 101):
        boundary = step * 0.02
        for requested, expected in (
            (np.nextafter(boundary, -np.inf), boundary),
            (boundary, boundary),
            (np.nextafter(boundary, np.inf), (step + 1) * 0.02),
        ):
            control, memory, _, _ = session()
            control.route_with_clearance([0, 0], [0.5, 0], float(requested))
            assert memory.planning_radii[-1] == expected
            assert memory.planning_radii[-1] >= max(0.1, requested)


class RadiusLimitedMemory(Memory):
    def __init__(self, limit):
        super().__init__()
        self.limit, self.route_attempts = limit, []

    def planning_grid(self, radius):
        self.planning_radii.append(radius)

        def route(start, goal):
            self.route_attempts.append(radius)
            if radius > self.limit:
                raise ValueError("Disconnected at this conservative planning radius")
            return np.asarray([start, goal])

        return SimpleNamespace(route=route)


def clearance_session(memory, reserve=0.04):
    return ControlSession(
        policy=lambda vector: np.zeros(2),
        memory=memory,
        stop=lambda: None,
        map_version="map",
        calibration_version="rig",
        demo_goal=[0.5, 0],
        planning_reserve=reserve,
        clock=lambda: 10.0,
    )


def test_exact_boundary_with_reserve_uses_14cm_grid_without_fallback():
    memory = RadiusLimitedMemory(0.15)
    control = clearance_session(memory)
    control.route_with_clearance([0, 0], [0.5, 0], 0.1)
    assert memory.route_attempts == [0.14]
    assert memory.planning_radii == [0.1, 0.14]
    assert not control.route_planning_details["exact_fallback_used"]
    assert all(radius == 0.14 for _, _, radius in memory.queries)


def test_exact_fallback_keeps_every_requested_bit_and_full_reserve():
    memory = RadiusLimitedMemory(0.15)
    control = clearance_session(memory)
    radius = 0.100003107
    required = radius + 0.04
    route = control.route_with_clearance([0, 0], [0.5, 0], radius)
    np.testing.assert_array_equal(route, [[0, 0], [0.5, 0]])
    assert memory.route_attempts == [0.16, required]
    assert memory.planning_radii == [0.1, 0.16, required]
    assert control.route_planning_details == {
        "required_radius_m": required,
        "rounded_radius_m": 0.16,
        "selected_radius_m": required,
        "exact_fallback_used": True,
        "certificate_radius_m": required,
        "route_certified": True,
    }
    assert all(r == required for _, _, r in memory.queries)
    # Repeated requests reuse both failed rounded and successful exact grids.
    control.route_with_clearance([0, 0], [0.5, 0], radius)
    assert memory.planning_radii == [0.1, 0.16, required]


def test_rounded_route_remains_preferred_and_all_cache_entries_are_bounded():
    open_memory = RadiusLimitedMemory(1.0)
    control = clearance_session(open_memory)
    control.route_with_clearance([0, 0], [0.5, 0], 0.100003107)
    assert open_memory.route_attempts == [0.16]
    assert not control.route_planning_details["exact_fallback_used"]
    memory = RadiusLimitedMemory(0.15)
    control = clearance_session(memory)
    for index in range(40):
        control.route_with_clearance([0, 0], [0.5, 0], 0.100001 + index * 0.000001)
        assert len(control.route_grids) <= 16
    assert len(control.route_grids) == 16


def test_exact_fallback_and_final_capsule_validation_still_fail_closed():
    memory = RadiusLimitedMemory(0.139)
    control = clearance_session(memory)
    radius = 0.100003107
    with pytest.raises(ValueError, match="Disconnected"):
        control.route_with_clearance([0, 0], [0.5, 0], radius)
    assert memory.route_attempts == [0.16, radius + 0.04]
    assert not control.route_planning_details["route_certified"]
    memory = RadiusLimitedMemory(1.0)
    memory.blocked = True
    control = clearance_session(memory)
    with pytest.raises(ValueError, match="full requested clearance"):
        control.route_with_clearance([0, 0], [0.5, 0], radius)
    assert memory.route_attempts == [0.16]  # No fallback after certificate failure.
    assert not control.route_planning_details["route_certified"]


def test_render_camera_pose_resets_history_dependent_up_without_truth_calibration():
    from bb8_rl.interactive_runtime import set_render_camera_pose

    calls = []
    camera = SimpleNamespace(set_pose=lambda **values: calls.append(values))
    record = {
        "position": [4, -2, 3],
        "lookat": [0, 0, 0.1],
        "world_to_camera": object(),
    }
    set_render_camera_pose(camera, record)
    assert calls[-1] == {"pos": [4, -2, 3], "lookat": [0, 0, 0.1], "up": [0, 0, 1]}
    set_render_camera_pose(camera, {**record, "up": [0, 1, 0]})
    assert calls[-1]["up"] == [0, 1, 0]
    assert "world_to_camera" not in calls[-1]


@pytest.mark.parametrize(
    "coordinates", [None, "genesis_viewport", "opencv_integer_center"]
)
@pytest.mark.parametrize("scale", [1, 2])
def test_render_intrinsics_admit_declared_coordinates_without_using_true_pose(
    coordinates, scale
):
    from bb8_rl.interactive_runtime import validate_render_camera_intrinsics

    # Native renderer pixel centers are u+.5/v+.5. The corrected exported K
    # places the optical axis halfway between the two central array pixels.
    center = 639.5 if coordinates == "opencv_integer_center" else 640.0
    vertical = 479.5 if coordinates == "opencv_integer_center" else 480.0
    record = {
        "intrinsics": [[800, 0, center], [0, 800, vertical], [0, 0, 1]],
        "resolution": [1280, 960],
        "world_to_camera": object(),  # Must not inspect or replace estimated pose.
    }
    if coordinates is not None:
        record["pixel_coordinates"] = coordinates
    native_k = np.array(
        [[800 * scale, 0, 640 * scale], [0, 800 * scale, 480 * scale], [0, 0, 1]],
        dtype=float,
    )
    camera = SimpleNamespace(
        intrinsics=native_k,
        res=(1280 * scale, 960 * scale),
        transform=np.eye(4),
    )
    before = native_k.copy()
    assert validate_render_camera_intrinsics(camera, record, 2, scale=scale) is None
    np.testing.assert_array_equal(native_k, before)
    assert not isinstance(record["world_to_camera"], np.ndarray)


def test_render_intrinsics_reject_mislabeled_or_double_shifted_array_calibration():
    from bb8_rl.interactive_runtime import validate_render_camera_intrinsics

    camera = SimpleNamespace(
        intrinsics=np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]], dtype=float),
        res=(1280, 960),
        transform=np.eye(4),
    )
    for center, vertical in [(640, 480), (639, 479)]:
        record = {
            "intrinsics": [[800, 0, center], [0, 800, vertical], [0, 0, 1]],
            "resolution": [1280, 960],
            "pixel_coordinates": "opencv_integer_center",
        }
        with pytest.raises(ValueError, match="intrinsics/resolution"):
            validate_render_camera_intrinsics(camera, record, 2)


@pytest.mark.parametrize(
    "coordinates", [None, "", "auto", "opencv_integer_centers", []]
)
def test_unknown_explicit_camera_pixel_convention_fails_closed(coordinates):
    from bb8_rl.interactive_runtime import camera_pixel_coordinates

    with pytest.raises(ValueError, match="pixel coordinate convention"):
        camera_pixel_coordinates({"pixel_coordinates": coordinates})


@pytest.mark.parametrize("coordinates", ["opencv_integer_center", "unrecognized"])
def test_worker_rejects_mixed_or_unknown_coordinate_inventory_before_models(
    tmp_path, monkeypatch, coordinates
):
    import bb8_rl.demo_assets as assets

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        assets,
        "validate_assets",
        lambda _path: {
            "task": "task.yaml",
            "cameras": {"A": {}, "B": {"pixel_coordinates": coordinates}, "C": {}},
        },
    )
    (tmp_path / "bundle.json").write_text("{}")
    events = queue.Queue(maxsize=8)
    worker_main(
        {"mode": 1, "asset_dir": str(tmp_path), "run_dir": str(tmp_path / "run")},
        queue.Queue(),
        events,
        SimpleNamespace(is_set=lambda: False),
    )
    result = drain_commands(events)[0]
    assert result["type"] == "error"
    assert "pixel coordinate convention" in result["message"]


class SceneRecoveryHarness:
    """The worker's actual guard/session handoff with synthetic RGB and no native state."""

    def __init__(self):
        import threading

        from bb8_rl.camera import Calibration
        from bb8_rl.scene_validity import SceneValidityGuard

        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[2, 3] = 3
        calibration = Calibration(
            np.array([[150.0, 0, 96], [0, 150, 72], [0, 0, 1]]),
            transform,
            (192, 144),
            2.0,
        )
        self.guard = SceneValidityGuard({"A": calibration}, {"A": "rig"})
        self.control, self.memory, self.clock, _ = session(scene_validity_required=True)
        self.event = threading.Event()
        self.index = -1
        rng = np.random.default_rng(1729)
        gray = np.repeat(np.repeat(rng.integers(45, 185, (18, 24)), 8, 0), 8, 1)
        self.pixels = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
        for _ in range(31):
            self.step()
        self.control.receive([{"action": "demo", "generation": 1}], sim_time=1.5)
        for _ in range(20):
            self.step(changed=True)
        assert self.control.scene_validity["invalidated"] and self.event.is_set()
        assert self.control.scene_validity["fault_epoch"] == 1
        assert self.control.goal is None and self.control.controller is None

    def step(self, *, command=None, changed=False, applied=(0, 0), intervals=None):
        from bb8_rl.camera_rig import CameraFrame, ViewObservation

        self.index += 1
        timestamp = self.index * 0.05
        if command:
            self.records = self.control.receive([command], sim_time=timestamp)
        pixels = self.pixels.copy()
        x = 65 if changed else 30
        pixels[35:64, x : x + 23] = (205, 30, 40)
        frame = CameraFrame("A", "rig", pixels, timestamp)
        measurement = observed(timestamp)
        view = ViewObservation("A", "rig", timestamp, measurement, "visible")
        if intervals is None:
            intervals = (
                []
                if self.index == 0
                else [
                    {
                        "start": (self.index - 1) * 0.05,
                        "end": timestamp,
                        "command": [0, 0],
                    }
                ]
            )
        committed = self.control.observe_scene_guard(
            self.guard,
            [frame],
            {"A": view},
            timestamp=timestamp,
            measurement=measurement,
            applied=applied,
            intervals=intervals,
            invalidation_event=self.event,
        )
        action = self.control.decide(timestamp, measurement, applied, intervals)
        assert np.array_equal(action, [0, 0])
        return committed

    def request(self, generation=2, *, changed=False):
        self.step(
            command={
                "action": "recheck_scene",
                "generation": generation,
                "fault_epoch": 1,
            },
            changed=changed,
        )
        assert self.records[0]["outcome"] == "accepted_scene_recheck"

    def complete(self):
        commits = [self.step() for _ in range(24)]
        assert sum(commits) == 1
        assert not self.event.is_set()
        assert self.control.scene_validity["recovery"]["status"] == "succeeded"
        assert self.control.goal is None and self.control.pending_goal is None
        assert self.control.controller is None and not self.control.agency.enabled
        assert self.control.requires_new_goal


def test_worker_original_reference_recheck_then_local_reacquisition_needs_new_goal():
    h = SceneRecoveryHarness()
    original = h.guard.snapshot()["cameras"]["A"]["reference_sha256"]
    h.request(changed=True)
    h.step(changed=True)
    assert h.control.scene_validity["recovery"]["status"] == "rejected"
    assert h.event.is_set()
    for _ in range(12):
        h.step()
    assert h.control.scene_validity["invalidated"]
    h.request(3)
    h.complete()
    assert h.guard.snapshot()["cameras"]["A"]["reference_sha256"] == original
    # Guard success does not itself establish a measured local pose.
    assert not h.control.state()["localization_valid"]
    for _ in range(20):
        h.step()
    assert h.control.state()["localization_valid"]
    assert h.control.requires_new_goal and h.control.goal is None
    result = h.control.receive([{"action": "demo", "generation": 4}])
    assert result[0]["outcome"] == "accepted_pending_visible_route"


@pytest.mark.parametrize(
    "interruption", ["stop", "manual", "heartbeat", "version", "moving", "timeline"]
)
def test_scene_recheck_cannot_commit_after_authority_or_evidence_interruption(
    interruption,
):
    h = SceneRecoveryHarness()
    h.request()
    for _ in range(12):
        h.step()
    assert h.control.scene_validity["recovery"]["stable_checks"] == 2
    if interruption == "stop":
        h.step(command={"action": "stop", "generation": 3})
    elif interruption == "manual":
        h.step(command={"action": "demo", "generation": 3})
    elif interruption == "heartbeat":
        h.clock[0] += 4
        h.step()
    elif interruption == "version":
        h.control.map_version = "other"
        h.step()
    elif interruption == "moving":
        h.step(applied=[0.2, 0])
    else:
        h.step(intervals=[])
    assert h.control.scene_recheck is None
    assert h.control.scene_validity["recovery"]["status"] == "cancelled"
    for _ in range(20):
        h.step()
    assert h.event.is_set() and h.control.scene_validity["invalidated"]
    assert h.control.goal is None and not h.control.agency.enabled
    if interruption != "version":
        h.control.receive([{"action": "heartbeat", "generation": 4}])
        h.request(4)
        h.complete()


def test_late_stop_after_worker_success_allows_same_epoch_recheck_without_restart():
    h = SceneRecoveryHarness()
    h.request()
    h.complete()
    previous = h.control.scene_validity["recovery"].copy()
    h.step(command={"action": "stop", "generation": 3})
    assert previous["generation"] < h.control.generation
    # Supervisor rejects the generation-2 success and remains faulted; retry
    # does not require replacing the reference or worker.
    h.request(4)
    assert h.event.is_set() and h.guard.snapshot()["fault_epoch"] == 1
    h.complete()
    assert h.control.scene_validity["recovery"]["generation"] == 4
    assert (
        h.control.scene_validity["recovery"]["reference_sha256"]
        == previous["reference_sha256"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", 1),
        ("fault_epoch", 2),
        ("requested_at", 0),
        ("completed_at", 0),
        ("reference_sha256", {}),
    ],
)
def test_session_rejects_uncorrelated_guard_success(field, value):
    import copy

    h = SceneRecoveryHarness()
    h.request()
    record = copy.deepcopy(h.guard.snapshot())
    timestamp = (h.index + 22) * 0.05
    record.update(timestamp=timestamp, invalidated=False, navigation_allowed=True)
    record["recovery"].update(
        status="succeeded", completed_at=timestamp, stable_checks=3
    )
    record["recovery"][field] = value
    assert not h.control.update_scene_validity(record, timestamp)
    assert h.control.scene_validity["invalidated"] and h.event.is_set()
    assert h.control.goal is None


def test_recheck_rejects_wrong_epoch_and_healthy_unfaulted_scene():
    control, _, _, _ = session(scene_validity_required=True)
    control.update_scene_validity(dict(scene_evidence(0), fault_epoch=0), 0)
    result = control.receive(
        [{"action": "recheck_scene", "generation": 1, "fault_epoch": 0}]
    )
    assert result[0]["outcome"] == "rejected"
    h = SceneRecoveryHarness()
    h.step(command={"action": "recheck_scene", "generation": 2, "fault_epoch": 2})
    assert h.records[0]["outcome"] == "rejected"
    assert h.event.is_set() and h.control.scene_recheck is None


def test_heartbeat_expiring_inside_last_comparison_cannot_clear_shared_fault(
    monkeypatch,
):
    h = SceneRecoveryHarness()
    h.request()
    original_observe = h.guard.observe

    def observe_then_expire(*args, **kwargs):
        result = original_observe(*args, **kwargs)
        if (result.get("recovery") or {}).get("status") == "succeeded":
            h.clock[0] += 4
        return result

    monkeypatch.setattr(h.guard, "observe", observe_then_expire)
    for _ in range(24):
        assert not h.step()
    assert h.event.is_set() and h.control.scene_validity["invalidated"]
    assert h.control.scene_validity["recovery"]["status"] == "cancelled"
    monkeypatch.setattr(h.guard, "observe", original_observe)
    h.control.receive([{"action": "heartbeat", "generation": 3}])
    h.request(3)
    h.complete()
