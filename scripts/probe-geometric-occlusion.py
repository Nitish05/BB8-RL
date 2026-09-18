"""Actual box occlusion during prescribed Genesis motion; no visual control.

Two fixed cameras keep the trained nominal angles. Taller, wider boxes create
new geometry, so this is a targeted challenge rather than in-distribution proof.
Renderer head masks and states are saved solely for scoring/replay.
"""

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAMERAS = {"A": (2.8, -3.4, 5.2), "B": (-2.8, 3.4, 5.2)}
BOXES = (
    {"position": (0.0, -0.6, 0.6), "size": (1.0, 0.4, 1.2)},
    {"position": (0.0, 0.6, 0.6), "size": (1.0, 0.4, 1.2)},
)
START = (-1.6, 0.0)
GOAL = (1.6, 0.0)
SETTLE_SECONDS = 0.5
DRIVE_SECONDS = 12.0
BRAKE_SECONDS = 1.0
TOTAL_SECONDS = SETTLE_SECONDS + DRIVE_SECONDS + BRAKE_SECONDS


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prescribed_command(timestamp):
    """A fixed time schedule, independent of truth, detections and visibility."""
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Command time must be finite and nonnegative")
    active = SETTLE_SECONDS <= timestamp < SETTLE_SECONDS + DRIVE_SECONDS
    return np.array([0.8 if active else 0.0, 0.0])


def ray_hits_box(camera, point, box):
    """Analytic fixture design only: test the camera-to-point segment in 3D."""
    camera, point = np.asarray(camera, float), np.asarray(point, float)
    position, size = np.asarray(box["position"], float), np.asarray(box["size"], float)
    if any(
        value.shape != (3,) or not np.isfinite(value).all()
        for value in (camera, point, position, size)
    ) or np.any(size <= 0):
        raise ValueError("Invalid ray or box")
    low, high = position - size / 2, position + size / 2
    delta = point - camera
    enter, leave = 0.0, 1.0
    for axis in (0, 1, 2):
        if abs(delta[axis]) < 1e-15:
            if not low[axis] <= camera[axis] <= high[axis]:
                return False
        else:
            a, b = (
                (low[axis] - camera[axis]) / delta[axis],
                (high[axis] - camera[axis]) / delta[axis],
            )
            enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
    return bool(enter <= leave and enter < 1)


def head_cube_hidden(camera, xy, *, radius=0.019, head_height=0.083):
    """All eight bounding-cube corners hidden by one box is a design certificate.

    A convex box's perspective shadow is convex. The certificate covers the
    entire head cube, not merely its center. Native labels remain the scorer.
    """
    center = np.array([*xy, head_height])
    corners = [
        center + radius * np.array(signs) for signs in product((-1, 1), repeat=3)
    ]
    return any(
        all(ray_hits_box(camera, corner, box) for corner in corners) for box in BOXES
    )


def create_fixture(output):
    """Write a BB8-RL-only synthetic world; this function does not run Genesis."""
    import yaml

    from bb8_rl.planner import OccupancyGrid
    from bb8_rl.room import generate_room
    from bb8_rl.world import validate_world

    generate_room(
        REPO / "configs/navigation/bb8-state.yaml",
        output,
        seed=751200,
        side=4,
        obstacles=0,
    )
    project_path = output / "room.genesis.json"
    project = json.loads(project_path.read_text())
    for index, box in enumerate(BOXES):
        project["objects"].append(
            {
                "name": f"room_obstacle_{index}",
                "format": "box",
                "fixed": True,
                "position": box["position"],
                "size": box["size"],
                "material": {"color": [0.35, 0.44, 0.58, 1]},
            }
        )
    project["name"] = "BB8 geometric occlusion challenge — two tall boxes"
    project["sensors"][0]["position"] = CAMERAS["A"]
    write_report(project_path, project)
    config_path = output / "room.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["obstacle_names"] = [f"room_obstacle_{index}" for index in range(len(BOXES))]
    config["notes"] = (
        "Synthetic geometric-occlusion challenge. Nominal M75 camera angles; 1.2m-tall boxes are outside training geometry. Prescribed central-lane motion, no visual navigation."
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    boxes = [[*box["position"][:2], *box["size"][:2]] for box in BOXES]
    route = OccupancyGrid(2, 0.1, 0.147, boxes).route(START, GOAL)
    write_report(
        output / "layout.json",
        {
            "provenance": "synthetic_challenge_scoring_only",
            "start_xy": START,
            "goal_xy": GOAL,
            "boxes": BOXES,
            "cameras": CAMERAS,
            "centerline_box_clearance_m": 0.4,
            "scoring_only_clear_route": route.tolist(),
            "camera_angles_changed": False,
            "geometry_outside_training": "Widths1.0m and heights1.2m exceed training boxes",
        },
    )
    validate_world(config_path)


def visibility_class(count):
    return "fully_hidden" if count == 0 else "weak" if count < 3 else "visible"


def summarize(records):
    views = {}
    for name in CAMERAS:
        observations = [
            dict(record["views"][name], time=record["time"]) for record in records
        ]
        hidden = [row for row in observations if row["head_pixels_scoring_only"] == 0]
        visible = [row for row in observations if row["head_pixels_scoring_only"] >= 3]
        errors = [
            row["error_m_scoring_only"]
            for row in visible
            if row["error_m_scoring_only"] is not None
        ]
        reappearances, pending = [], False
        for index, row in enumerate(observations):
            if row["head_pixels_scoring_only"] == 0:
                pending = True
            elif pending and row["head_pixels_scoring_only"] >= 3:
                recovered = next(
                    (
                        later
                        for later in observations[index:]
                        if later["head_pixels_scoring_only"] >= 3
                        and later["status"] == "visible"
                    ),
                    None,
                )
                reappearances.append(
                    {
                        "first_visible_time": row["time"],
                        "first_detection_time": recovered["time"]
                        if recovered
                        else None,
                        "delay_seconds": recovered["time"] - row["time"]
                        if recovered
                        else None,
                    }
                )
                pending = False
        views[name] = {
            "total_frames": len(observations),
            "fully_hidden_frames": len(hidden),
            "weak_1_to_2_pixel_frames": sum(
                0 < row["head_pixels_scoring_only"] < 3 for row in observations
            ),
            "visible_frames": len(visible),
            "fully_hidden_false_accepts": sum(
                row["status"] == "visible" for row in hidden
            ),
            "visible_head_misses": sum(row["status"] != "visible" for row in visible),
            "visible_localization_p95_m": float(np.quantile(errors, 0.95))
            if errors
            else None,
            "reappearances": reappearances,
        }
    both_hidden = [
        record
        for record in records
        if all(
            record["views"][name]["head_pixels_scoring_only"] == 0 for name in CAMERAS
        )
    ]
    phases = {}
    for record in records:
        phase = ",".join(
            f"{name}:{record['views'][name]['truth_visibility_scoring_only']}"
            for name in CAMERAS
        )
        phases[phase] = phases.get(phase, 0) + 1
    return {
        "views": views,
        "visibility_phase_counts": phases,
        "both_fully_hidden_frames": len(both_hidden),
        "both_hidden_rig_false_accepts": sum(
            record["rig"]["position"] is not None for record in both_hidden
        ),
        "handover_events": sum(record["rig"]["handover"] for record in records),
        "all_requested_observations_retained": True,
    }


def main(args):
    import cv2
    import torch

    from bb8_rl.camera_rig import (
        CameraFrame,
        CameraRig,
        CameraView,
        calibration_from_live_camera,
    )
    from bb8_rl.env import NavigationEnv
    from bb8_rl.vision import LearnedObserver

    if args.output.exists() or args.sample_stride < 1:
        raise ValueError("Use a fresh output and positive sample stride")
    args.output.mkdir(parents=True)
    create_fixture(args.output / "fixture")
    torch.set_num_threads(2)
    source_paths = [
        Path(__file__),
        *[
            REPO / "src/bb8_rl" / name
            for name in (
                "camera_rig.py",
                "vision.py",
                "env.py",
                "world.py",
                "control/genesis_backend.py",
            )
        ],
    ]
    sources = {str(path.relative_to(REPO)): file_hash(path) for path in source_paths}
    source_output = args.output / "sources"
    source_output.mkdir()
    for path in source_paths:
        (source_output / path.name).write_bytes(path.read_bytes())
    manifest = {
        "status": "running",
        "scope": "Prescribed-motion geometric occlusion challenge; no visual control, blind-navigation or real-time acceptance",
        "camera_positions": CAMERAS,
        "camera_count": 2,
        "boxes": BOXES,
        "geometry_outside_training": True,
        "camera_angles_changed": False,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_sha256": sources,
        "initial_position": START,
        "settle_seconds": SETTLE_SECONDS,
        "drive_seconds": DRIVE_SECONDS,
        "brake_seconds": BRAKE_SECONDS,
        "drive_command": [0.8, 0.0],
        "sample_stride_actions": args.sample_stride,
        "truth_boundary": "Only RGB/calibration/capture time enter observers. Renderer masks/state and analytic rays are scoring only; commands depend on time alone.",
        "head_visibility_definitions": {
            "fully_hidden": "zero renderer head pixels",
            "weak": "one or two pixels",
            "visible": "at least three pixels",
        },
    }
    write_report(args.output / "manifest.json", manifest)
    records, commands, calibrations = [], [], {}
    try:
        with NavigationEnv(
            args.output / "fixture/task.yaml",
            render_mode="rgb_array",
            camera_positions=list(CAMERAS.values()),
        ) as env:
            raw, _ = env.reset(
                seed=751201, options={"layout_seed": 1201, "start": START, "goal": GOAL}
            )
            env.world.camera_period = env.world._next_frame = 1e9
            period = env.config.action_steps * env.world.dt
            total_steps = round(TOTAL_SECONDS / period)
            if not np.isclose(total_steps * period, TOTAL_SECONDS):
                raise ValueError("Probe duration must divide action period")
            requested_steps = [
                step
                for step in range(total_steps + 1)
                if step % args.sample_stride == 0 or step == total_steps
            ]
            manifest["requested_observation_steps"] = requested_steps
            manifest["action_period_seconds"] = period
            views = []
            for name, camera in env.world.cameras.items():
                calibration = calibration_from_live_camera(camera, 2)
                calibrations[name] = {
                    "intrinsics": calibration.intrinsics.tolist(),
                    "world_to_camera": calibration.world_to_camera.tolist(),
                    "resolution": calibration.resolution,
                    "head_height": calibration.head_height,
                    "provenance": calibration.provenance,
                }
                views.append(
                    CameraView(
                        name,
                        "geometric-v1",
                        calibration,
                        LearnedObserver(
                            calibration, args.checkpoint, floor_refinement="guided"
                        ),
                    )
                )
            rig = CameraRig(views)
            label_map = env.world.scene.visualizer.segmentation_idx_dict
            head_ids = [
                key
                for key, value in label_map.items()
                if isinstance(value, tuple) and value[0] == env.world.head.idx
            ]
            if not head_ids:
                raise ValueError("Missing explicit renderer head labels")
            write_report(args.output / "calibrations.json", calibrations)
            write_report(args.output / "manifest.json", manifest)
            for step in range(total_steps + 1):
                timestamp = env.world.time
                action = prescribed_command(timestamp)
                if step in requested_steps:
                    frames, scoring = [], {}
                    started = time.perf_counter()
                    for name, camera in env.world.cameras.items():
                        rgb, _, labels, _ = camera.render(
                            rgb=True, segmentation=True, force_render=True
                        )
                        rgb = np.ascontiguousarray(rgb)
                        head = np.isin(labels, head_ids)
                        frames.append(CameraFrame(name, "geometric-v1", rgb, timestamp))
                        rgb_path = args.output / f"frame-{step:04}-{name}-rgb.png"
                        mask_path = (
                            args.output / f"frame-{step:04}-{name}-head-truth.png"
                        )
                        if not cv2.imwrite(
                            str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        ) or not cv2.imwrite(
                            str(mask_path), head.astype(np.uint8) * 255
                        ):
                            raise OSError("Could not save synchronized RGB/head mask")
                        pixels = int(head.sum())
                        scoring[name] = {
                            "rgb_path": rgb_path.name,
                            "rgb_sha256": file_hash(rgb_path),
                            "head_truth_path": mask_path.name,
                            "head_truth_sha256": file_hash(mask_path),
                            "head_pixels_scoring_only": pixels,
                            "truth_visibility_scoring_only": visibility_class(pixels),
                        }
                    capture_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    observation = rig.observe(frames, now=timestamp)
                    inference_seconds = time.perf_counter() - started
                    for name, result in observation.views.items():
                        measurement = result.measurement
                        xy = measurement.xy if measurement is not None else None
                        scoring[name].update(
                            status=result.status,
                            position=xy.tolist() if xy is not None else None,
                            covariance=measurement.covariance.tolist()
                            if xy is not None
                            else None,
                            error_m_scoring_only=float(
                                np.linalg.norm(xy - raw["achieved_goal"][:2])
                            )
                            if xy is not None
                            else None,
                        )
                    row = {
                        "step": step,
                        "time": timestamp,
                        "command_for_next_action": action.tolist()
                        if step < total_steps
                        else None,
                        "truth_position_xy_scoring_only": list(env.state.position),
                        "truth_velocity_xy_scoring_only": list(env.state.velocity),
                        "views": scoring,
                        "rig": {
                            "status": observation.status,
                            "position": observation.xy.tolist()
                            if observation.xy is not None
                            else None,
                            "covariance": observation.covariance.tolist()
                            if observation.covariance is not None
                            else None,
                            "source_ids": observation.source_ids,
                            "handover": observation.handover,
                        },
                        "capture_and_save_wall_seconds": capture_seconds,
                        "inference_wall_seconds": inference_seconds,
                    }
                    records.append(row)
                    with (args.output / "observations.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if step % 20 == 0 or step == total_steps:
                        print(
                            json.dumps(
                                {
                                    "step": step,
                                    "time": timestamp,
                                    "head_pixels": {
                                        name: value["head_pixels_scoring_only"]
                                        for name, value in scoring.items()
                                    },
                                    "rig_status": observation.status,
                                }
                            ),
                            flush=True,
                        )
                if step == total_steps:
                    break
                command = {
                    "step": step,
                    "issue_time": timestamp,
                    "action": action.tolist(),
                }
                raw, _, terminal, truncated, info = env.step(action)
                command.update(
                    applied_time_after_step=env.world.time,
                    applied_request_after_step=list(env.world.backend.applied_request),
                    truth_position_after_step_scoring_only=list(env.state.position),
                    truth_velocity_after_step_scoring_only=list(env.state.velocity),
                    collision_scoring_only=info["collision"],
                    boundary_scoring_only=info["boundary_failure"],
                )
                commands.append(command)
                with (args.output / "commands.jsonl").open("a") as stream:
                    stream.write(json.dumps(command, allow_nan=False) + "\n")
                if terminal or truncated:
                    raise RuntimeError(
                        "Prescribed probe ended early; all recorded frames retained"
                    )
            manifest["contacts"] = env.world.contacts
            manifest["final_position_scoring_only"] = list(env.state.position)
            manifest["final_velocity_scoring_only"] = list(env.state.velocity)
        if len(records) != len(requested_steps) or len(commands) != total_steps:
            raise ValueError("Missing requested observations or commands")
        manifest["status"] = "complete"
        manifest["summary"] = summarize(records)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["observations_recorded"] = len(records)
        manifest["commands_recorded"] = len(commands)
        manifest["checkpoint_unchanged"] = (
            file_hash(args.checkpoint) == manifest["checkpoint_sha256"]
        )
        manifest["source_files_unchanged"] = all(
            file_hash(path) == sources[str(path.relative_to(REPO))]
            for path in source_paths
        )
        write_report(args.output / "manifest.json", manifest)
        write_report(
            args.output / "report.json",
            {
                **manifest,
                "calibrations": calibrations,
                "observations": records,
                "commands": commands,
            },
        )
    print(
        json.dumps(
            {
                k: value
                for k, value in manifest.items()
                if k not in ("source_sha256", "requested_observation_steps")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO / "work/m75/calibrated/model.pt"
    )
    parser.add_argument("--sample-stride", type=int, default=2)
    main(parser.parse_args())
