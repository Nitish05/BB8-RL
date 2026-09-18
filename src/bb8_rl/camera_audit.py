"""Scoring-only synchronized snapshots; never imported by the RGB controller."""

from pathlib import Path

import cv2
import numpy as np

from .diagnostics import write_report


def capture_camera_audit(world, controller, rgb, timestamp, *, step, action):
    """Read privileged renderer labels only after a control action was selected.

    The caller must not advance physics between the controller observation and
    this call. Returned buffers are independent copies for offline scoring.
    """
    if not np.isclose(timestamp, world.time, rtol=0, atol=1e-8):
        raise ValueError("Cannot audit a camera frame from a different physics time")
    render_rgb, _, labels, _ = world.camera.render(rgb=True, segmentation=True)
    label_map = world.scene.visualizer.segmentation_idx_dict
    ground_entity = world.scene.get_entity(name="ground").idx
    ground_labels = [
        label
        for label, key in label_map.items()
        if isinstance(key, tuple) and key[0] == ground_entity
    ]
    self_labels = [
        label
        for label, key in label_map.items()
        if isinstance(key, tuple) and key[0] in (world.body.idx, world.head.idx)
    ]
    if not ground_labels or not self_labels:
        raise ValueError("Audit requires explicit renderer ground/self label mapping")
    measurement = controller.last_measurement
    grid = controller.grid
    calibration = controller.calibration
    arrays = {
        "rgb": np.array(rgb, copy=True),
        "render_rgb": np.array(render_rgb, copy=True),
        "labels": np.array(labels, copy=True),
        "truth_floor": np.isin(labels, ground_labels),
        "truth_self": np.isin(labels, self_labels),
        "predicted_floor": np.array(measurement.visible_floor, copy=True)
        if measurement is not None
        else np.empty((0, 0), bool),
        "predicted_self": np.array(measurement.robot_pixels, copy=True)
        if measurement is not None
        else np.empty((0, 0), bool),
        "blocked": np.array(grid.blocked, copy=True)
        if grid is not None
        else np.empty((0, 0), bool),
        "route": np.array(controller.route, copy=True)
        if controller.route is not None
        else np.empty((0, 2), float),
    }
    metadata = {
        "scope": "scoring only; renderer labels never supplied to controller",
        "step": step,
        "capture_time": float(timestamp),
        "simulation_time": float(world.time),
        "measurement_time": float(measurement.timestamp)
        if measurement is not None
        else None,
        "status": controller.status,
        "measurement_status": measurement.status if measurement is not None else None,
        "route_error": getattr(controller, "last_route_error", None),
        "action": np.asarray(action).tolist(),
        "estimate": controller.belief.state.tolist()
        if controller.belief.state is not None
        else None,
        "goal": controller.goal.tolist(),
        "map_sigma": controller.map_sigma,
        "grid": {
            "resolution": grid.resolution,
            "inflation": grid.inflation,
            "extent": grid.extent,
            "n": grid.n,
        }
        if grid is not None
        else None,
        "calibration": {
            "intrinsics": calibration.intrinsics.tolist(),
            "world_to_camera": calibration.world_to_camera.tolist(),
            "resolution": calibration.resolution,
            "extent": calibration.extent,
            "head_height": calibration.head_height,
            "provenance": calibration.provenance,
        },
        "ground_labels": [int(label) for label in ground_labels],
        "self_labels": [int(label) for label in self_labels],
        "segmentation_entity_mapping": {
            str(label): [int(value) for value in key]
            if isinstance(key, tuple)
            else str(key)
            for label, key in label_map.items()
        },
        "controller_rgb_matches_same_state_render": bool(
            np.array_equal(rgb, render_rgb)
        ),
        "npz_arrays": list(arrays),
    }
    return arrays, metadata


def save_camera_audit(prefix: Path, snapshot):
    arrays, metadata = snapshot
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    write_report(prefix.with_suffix(".json"), metadata)
    cv2.imwrite(
        str(prefix.with_suffix(".png")),
        cv2.cvtColor(arrays["rgb"], cv2.COLOR_RGB2BGR),
    )
