"""Native scan-once navigation: RGB+registered map control, truth only for scoring.

This is a lockstep simulation experiment. Wall timings are recorded, not claimed
as a real-time hardware guarantee. No simulator pose/velocity, obstacle geometry
or renderer labels enter the controller or its frozen learned policy.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration
from bb8_rl.camera_rig import calibration_from_live_camera
from bb8_rl.env import NavigationEnv
from bb8_rl.guided import TASKS, GuidedSAC
from bb8_rl.mapping.scan_free_space import ScanFreeMemory
from bb8_rl.occluded_control import OccludedController, OcclusionParameters
from bb8_rl.vision import LearnedObserver

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def write(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def append(path, value):
    with path.open("a") as f:
        f.write(json.dumps(serializable(value), allow_nan=False) + "\n")


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh output; retain failed experiments")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    protocol = json.loads(args.protocol.read_text())
    if digest(Path(protocol["scene"])) != protocol["scene_sha256"]:
        raise ValueError("Native authored scene differs from frozen protocol")
    if digest(args.registration) != protocol["registration_report_sha256"]:
        raise ValueError("Registration differs from frozen protocol")
    cases = protocol["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        if not wanted <= {row["id"] for row in cases}:
            raise ValueError("Unknown frozen case ID")
        cases = [row for row in cases if row["id"] in wanted]
    memory = ScanFreeMemory.load(args.memory)
    map_version = digest(args.memory / "manifest.json")
    registration = json.loads(args.registration.read_text())
    query_index = int(protocol["camera_view_id"].removeprefix("scan-"))
    if query_index not in {14, 17, 20, 23}:
        raise ValueError("Fixed camera must be a held-out registered query")
    estimate = next(
        row for row in registration["queries"] if row["query_index"] == query_index
    )
    if estimate["status"] != "candidate":
        raise ValueError("Fixed camera has no accepted RGB PnP candidate")
    scan_meta = json.loads((args.scan_dir / f"scan-{query_index:03d}.json").read_text())
    if not np.allclose(scan_meta["position"], protocol["render_camera_position"]):
        raise ValueError("Protocol camera differs from its held-out source frame")
    calibration_version = (
        f"m77-rgb-registration-query-{query_index}-" + registration["landmarks_sha256"]
    )
    calibration = Calibration(
        np.asarray(scan_meta["intrinsics"]),
        np.asarray(estimate["world_to_camera"]),
        (1280, 960),
        2.0,
        provenance="M77 RGB PnP pose; synthetic intrinsics and metric scan frame",
    )
    observer = LearnedObserver(
        calibration, args.vision_checkpoint, floor_refinement="guided"
    )
    policy = GuidedSAC.load(args.policy, device="cpu")
    parameters = OcclusionParameters()
    sources = [
        Path(__file__),
        ROOT / "src/bb8_rl/occluded_control.py",
        ROOT / "src/bb8_rl/mapping/scan_free_space.py",
        ROOT / "src/bb8_rl/control/genesis_backend.py",
    ]
    (args.output / "source").mkdir()
    for path in sources:
        (args.output / "source" / path.name).write_bytes(path.read_bytes())
    manifest = {
        "status": "running",
        "scope": "Native lockstep synthetic scan-once occluded control development, one fixed camera",
        "protocol_sha256": digest(args.protocol),
        "static_scene_sha256": protocol["scene_sha256"],
        "requested_cases": [row["id"] for row in cases],
        "all_protocol_cases": [row["id"] for row in protocol["cases"]],
        "policy_sha256": digest(args.policy),
        "policy_path": str(args.policy.resolve()),
        "vision_sha256": digest(args.vision_checkpoint),
        "vision_checkpoint_path": str(args.vision_checkpoint.resolve()),
        "map_version": map_version,
        "calibration_version": calibration_version,
        "registration_sha256": digest(args.registration),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "controller_parameters": asdict(parameters),
        "planning_radius_m": args.planning_radius,
        "action_period_seconds": 0.05,
        "continuous_hard_rejection_stop_seconds": 3.0,
        "camera_count": 1,
        "camera_query_index": query_index,
        "control_inputs": [
            "current RGB position/covariance/status",
            "capture time",
            "known synthetic intrinsics + RGB-estimated camera pose",
            "frozen scan RGB free/occupied evidence",
            "map-space target",
            "preceding acknowledged drive command",
            "complete timestamped acknowledged command intervals since previous capture",
        ],
        "truth_to_controller": False,
        "segmentation_to_controller": False,
        "applied_command_note": "Command-only telemetry sampled immediately after each existing backend.advance, before physics advances, supplies completed half-open intervals at the next capture. No measured velocity feedback. Future latency and response-error bounds remain assumptions.",
        "wall_time_note": "Physics pauses while rendering/inference/scoring run; this does not establish real-time operation.",
    }
    write(args.output / "manifest.json", manifest)
    write(args.output / "protocol.json", protocol)
    (args.output / "static-scene.json").write_bytes(
        Path(protocol["scene"]).read_bytes()
    )
    manifest["static_scene_snapshot"] = "static-scene.json"
    write(
        args.output / "calibration.json",
        {
            "intrinsics": calibration.intrinsics,
            "estimated_world_to_camera": calibration.world_to_camera,
            "resolution": calibration.resolution,
            "provenance": calibration.provenance,
        },
    )
    summaries = []
    physics_trace = []
    command_trace = []
    pending_command_ticks = []
    original_world_step = None
    try:
        with NavigationEnv(
            TASKS["authored"],
            split="validation",
            render_mode="rgb_array",
            camera_positions=[scan_meta["position"]],
        ) as env:
            for case in cases:
                out = args.output / case["id"]
                out.mkdir()
                env.task.max_episode_steps = args.max_steps
                # Truth is allowed for fixture initialization and evaluation only.
                env.reset(
                    seed=case["seed"],
                    options={
                        "layout_seed": case["layout_seed"],
                        "start": case["start"],
                        "goal": case["goal"],
                    },
                )
                write(
                    out / "native-drive-parameters.json",
                    asdict(env.world.backend.parameters),
                )
                write(
                    out / "native-static-obstacles.json",
                    {
                        "names": env.config.obstacle_names,
                        "positions": env.authored_positions,
                        "sizes": env.sizes,
                    },
                )
                if not np.isclose(env.config.action_steps * env.world.dt, 0.05):
                    raise ValueError(
                        "Native timestep differs from frozen action period"
                    )
                if original_world_step is None:
                    original_world_step = env.world.step
                    original_advance = env.world.backend.advance

                    def record_acknowledged_command(*, now, _advance=original_advance):
                        result = _advance(now=now)
                        pending_command_ticks.append(
                            {
                                "start": now,
                                "end": now + env.world.dt,
                                "command": list(env.world.backend.applied_request),
                            }
                        )
                        return result

                    env.world.backend.advance = record_acknowledged_command

                    def record_physics_step(
                        *positional, _step=original_world_step, **keywords
                    ):
                        active_before = list(env.world.backend.applied_request)
                        pending_command_ticks.clear()
                        state = _step(*positional, **keywords)
                        # Only commit acknowledged intervals after a successful tick.
                        # This list contains commands and clock values, never state.
                        command_trace.extend(pending_command_ticks)
                        physics_trace.append(
                            {
                                "time": env.world.time,
                                "position": list(state.position),
                                "velocity": list(state.velocity),
                                "command_active_before_tick": active_before,
                                "command_active_after_tick": list(
                                    env.world.backend.applied_request
                                ),
                            }
                        )
                        return state

                    env.world.step = record_physics_step
                env.world.camera_period = env.world._next_frame = 1e9
                camera = env.world.camera
                camera.set_pose(pos=scan_meta["position"], lookat=(0, 0, 0.1))
                live = calibration_from_live_camera(camera, 2.0)
                if (
                    not np.allclose(
                        live.world_to_camera, scan_meta["world_to_camera"], atol=1e-5
                    )
                    or not np.allclose(
                        live.intrinsics, calibration.intrinsics, atol=1e-5
                    )
                    or tuple(live.resolution) != tuple(calibration.resolution)
                ):
                    raise ValueError(
                        "Rendered final camera pose/intrinsics/resolution differs from source scan fixture"
                    )
                labels = env.world.scene.visualizer.segmentation_idx_dict
                head_ids = [
                    key
                    for key, value in labels.items()
                    if isinstance(value, tuple) and value[0] == env.world.head.idx
                ]
                if not head_ids:
                    raise ValueError("Scorer could not identify renderer head labels")
                controller = OccludedController(
                    lambda vector: policy.predict(vector, deterministic=True)[0],
                    np.asarray(case["goal"]),
                    memory.planning_grid(args.planning_radius),
                    memory.segment_free,
                    map_version=map_version,
                    calibration_version=calibration_version,
                    parameters=parameters,
                )
                applied = np.zeros(2)
                preceding_intervals = []
                statuses = Counter()
                ending = "time_limit"
                rows = []
                rejected_since = None
                for step in range(args.max_steps):
                    timestamp = env.world.time
                    tick = time.perf_counter()
                    rgb, _, segmentation, _ = camera.render(
                        rgb=True, segmentation=True, force_render=True
                    )
                    rgb = np.ascontiguousarray(rgb)
                    capture_seconds = time.perf_counter() - tick
                    injected = any(
                        start <= timestamp < end
                        for start, end in case.get("dropout_intervals_seconds", [])
                    )
                    observed_rgb = np.zeros_like(rgb) if injected else rgb
                    before_inference = time.perf_counter()
                    measurement = observer.observe_position(observed_rgb, timestamp)
                    inference_seconds = time.perf_counter() - before_inference
                    changed = case.get("invalidate_memory_at_seconds")
                    version_changed = changed is not None and timestamp >= changed
                    current_map_version = (
                        map_version + ":changed" if version_changed else map_version
                    )
                    before_control = time.perf_counter()
                    action = controller.action(
                        timestamp,
                        measurement,
                        applied_command=applied,
                        applied_command_intervals=preceding_intervals,
                        map_version=current_map_version,
                        calibration_version=calibration_version,
                        now=timestamp,
                    )
                    control_seconds = time.perf_counter() - before_control
                    # Decision is frozen before truth, labels or audit outputs are read.
                    head_mask = np.isin(segmentation, head_ids)
                    pixels = int(head_mask.sum())
                    rgb_name, mask_name = (
                        f"frame-{step:04}-rgb.png",
                        f"frame-{step:04}-head-truth.png",
                    )
                    if not cv2.imwrite(
                        str(out / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ) or not cv2.imwrite(
                        str(out / mask_name), head_mask.astype(np.uint8) * 255
                    ):
                        raise OSError("Failed to save native RGB / scorer mask")
                    diagnostics = controller.diagnostics
                    if callable(diagnostics):
                        diagnostics = diagnostics()
                    row = {
                        "step": step,
                        "time": timestamp,
                        "capture_time": timestamp,
                        "rgb_path": rgb_name,
                        "rgb_sha256": digest(out / rgb_name),
                        "head_mask_path": mask_name,
                        "head_mask_sha256": digest(out / mask_name),
                        "head_pixels_scoring_only": pixels,
                        "dropout_injected": injected,
                        "map_version_changed": version_changed,
                        "measurement": {
                            "status": measurement.status,
                            "xy": measurement.xy,
                            "covariance": measurement.covariance,
                            "timestamp": measurement.timestamp,
                        },
                        "prior_applied_command": applied,
                        "prior_acknowledged_command_intervals": preceding_intervals,
                        "action": action,
                        "controller_status": controller.status,
                        "controller_arrived": controller.arrived,
                        "controller": diagnostics,
                        "truth_before_scoring_only": {
                            "position": env.state.position,
                            "velocity": env.state.velocity,
                        },
                        "wall_seconds": {
                            "capture": capture_seconds,
                            "inference": inference_seconds,
                            "controller": control_seconds,
                        },
                    }
                    physics_trace.clear()
                    command_trace.clear()
                    _, _, terminal, truncated, info = env.step(action)
                    applied = np.asarray(env.world.backend.applied_request)
                    preceding_intervals = list(command_trace)
                    row["after_step"] = {
                        "time": env.world.time,
                        "position_scoring_only": env.state.position,
                        "velocity_scoring_only": env.state.velocity,
                        "applied_command": applied,
                        "acknowledged_command_intervals": preceding_intervals,
                        "command_sampling_note": "End of 50 ms env.step, after any terminal stop; physics trace also records each 5 ms tick before and after pending command activation.",
                        "collision_scoring_only": info["collision"],
                        "boundary_scoring_only": info["boundary_failure"],
                        "dwell_seconds_scoring_only": info["dwell_seconds"],
                        "success_scoring_only": info["is_success"],
                        "physics_trace_scoring_only": list(physics_trace),
                    }
                    append(out / "rows.jsonl", row)
                    rows.append(serializable(row))
                    statuses[controller.status] += 1
                    hard_rejection = controller.status in {
                        "no_proven_route_memory",
                        "route_not_certified",
                        "stopping_envelope_not_certified",
                        "uncertain_prediction",
                        "occlusion_timeout",
                        "reacquisition_rejected",
                        "invalid_applied_command_intervals",
                    } and not np.any(action)
                    rejected_since = (
                        (timestamp if rejected_since is None else rejected_since)
                        if hard_rejection
                        else None
                    )
                    if step % 40 == 0:
                        print(
                            json.dumps(
                                {
                                    "case": case["id"],
                                    "step": step,
                                    "time": timestamp,
                                    "status": controller.status,
                                    "head_pixels": pixels,
                                    "action": action.tolist(),
                                }
                            ),
                            flush=True,
                        )
                    if terminal:
                        ending = "collision_or_boundary"
                        break
                    if controller.arrived and env.world.time >= case.get(
                        "minimum_end_time_seconds", 0
                    ):
                        ending = "controller_arrived"
                        break
                    minimum_end = max(
                        case.get("minimum_end_time_seconds", 0),
                        max(
                            (
                                end + 2.0
                                for _, end in case.get("dropout_intervals_seconds", [])
                            ),
                            default=0,
                        ),
                    )
                    if (
                        rejected_since is not None
                        and timestamp - rejected_since >= 3.0
                        and timestamp >= minimum_end
                    ):
                        ending = "sustained_guard_rejection"
                        break
                    if version_changed and timestamp >= max(
                        changed + 2.0, case.get("minimum_end_time_seconds", 0)
                    ):
                        ending = "map_change_brake_observed"
                        break
                    if truncated:
                        break
                summary = {
                    "id": case["id"],
                    "ending": ending,
                    "steps": len(rows),
                    "statuses": dict(statuses),
                    "contacts": env.world.contacts,
                    "source": "root native harness; independent scoring follows",
                }
                summaries.append(summary)
                write(out / "summary.json", summary)
                print(json.dumps(summary), flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["completed_cases"] = summaries
        manifest["source_files_unchanged"] = all(
            digest(ROOT / path) == value
            for path, value in manifest["source_sha256"].items()
        )
        manifest["policy_unchanged"] = digest(args.policy) == manifest["policy_sha256"]
        manifest["vision_unchanged"] = (
            digest(args.vision_checkpoint) == manifest["vision_sha256"]
        )
        write(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--planning-radius", type=float, default=0.10)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(
            "work/scan/view-probe"
        ),
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "work/m77/registration/superpoint-lightglue/report.json",
    )
    parser.add_argument("--policy", type=Path, default=ROOT / "work/m6/sac-2/model.zip")
    parser.add_argument(
        "--vision-checkpoint", type=Path, default=ROOT / "work/m75/calibrated/model.pt"
    )
    main(parser.parse_args())
