"""M7 development pilot: rendered RGB controls; oracle state only scores errors."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration, CameraController
from bb8_rl.diagnostics import machine_report, write_report
from bb8_rl.env import NavigationEnv
from bb8_rl.guided import TASKS, GuidedSAC
from bb8_rl.training import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--count", type=int, default=4)
parser.add_argument(
    "--case-indices", help="Comma-separated original case IDs (0–11); overrides --count"
)
parser.add_argument("--map-resolution", type=float, default=0.05)
parser.add_argument(
    "--map-method", choices=["legacy", "metric", "continuous"], default="legacy"
)
parser.add_argument(
    "--audit-masks",
    action="store_true",
    help="Save scoring-only synchronized renderer masks",
)
parser.add_argument("--dropout-start-step", type=int, default=-1)
parser.add_argument("--checkpoint", type=Path, default=Path("work/m6/sac-2/model.zip"))
parser.add_argument("--vision-checkpoint", type=Path)
parser.add_argument("--vision-device", choices=["cpu", "mps"], default="cpu")
parser.add_argument("--floor-refinement", choices=["none", "guided"], default="none")
parser.add_argument(
    "--cases", type=Path, default=Path("work/m6/sac-validation-12/episodes")
)
args = parser.parse_args()
if args.output.exists() or not 1 <= args.count <= 12:
    raise ValueError("Use a fresh output and 1–12 development cases")
case_indices = (
    [int(value) for value in args.case_indices.split(",")]
    if args.case_indices is not None
    else list(range(args.count))
)
if (
    not case_indices
    or len(case_indices) != len(set(case_indices))
    or any(index < 0 or index >= 12 for index in case_indices)
):
    raise ValueError("Case indices must be distinct integers from 0 to 11")
if not np.isfinite(args.map_resolution) or args.map_resolution <= 0:
    raise ValueError("Map resolution must be finite and positive")
if args.floor_refinement != "none" and args.vision_checkpoint is None:
    raise ValueError("Floor refinement requires a learned vision checkpoint")
args.output.mkdir(parents=True)
torch.set_num_threads(2)
policy = GuidedSAC.load(args.checkpoint, device="cpu")


def learned_action(vector):
    return policy.predict(vector, deterministic=True)[0]


def annotated(rgb, controller):
    image = rgb.copy()
    c = controller.calibration
    goal = tuple(np.rint(c.to_pixel(controller.goal)).astype(int))
    cv2.drawMarker(image, goal, (255, 100, 150), cv2.MARKER_STAR, 18, 2)
    if controller.belief.state is not None:
        pixel = tuple(np.rint(c.to_pixel(controller.belief.state[:2])).astype(int))
        cv2.circle(image, pixel, 10, (80, 255, 130), 2)
    if controller.route is not None:
        points = np.rint(c.to_pixel(controller.route)).astype(np.int32)
        cv2.polylines(image, [points], False, (255, 200, 70), 2)
    cv2.putText(
        image,
        controller.status,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


manifest = {
    "status": "running",
    "scope": "camera development; not held-out acceptance",
    "observer": "learned-floor-head-gray-proposals-v1"
    if args.vision_checkpoint
    else "synthetic-palette-v1",
    "vision_checkpoint_sha256": file_hash(args.vision_checkpoint)
    if args.vision_checkpoint
    else None,
    "vision_device": args.vision_device if args.vision_checkpoint else None,
    "floor_refinement": args.floor_refinement,
    "calibration": "synthetic_exact",
    "control_inputs": [
        "RGB",
        "capture timestamp",
        "calibration",
        "goal pixel",
        "known robot head height",
    ],
    "oracle_pose_to_controller": False,
    "oracle_map_to_controller": False,
    "depth_or_segmentation_buffer_to_controller": False,
    "controller": "frozen M6 SAC plus tracking/route stop guard",
    "checkpoint_sha256": file_hash(args.checkpoint),
    "cases": len(case_indices),
    "case_indices": case_indices,
    "case_directory": str(args.cases.resolve()),
    "map_resolution_m": args.map_resolution,
    "map_method": args.map_method,
    "continuous_clearance": {
        "sample_step_m": 0.005,
        "connector_radius_m": 0.12,
        "native_pixel_guard": "full projected pixel diameter, maximum over room",
        "interval_refinement": "adaptive bisection; unresolved intervals reject",
        "minimum_sample_step_m": 0.0001,
        "maximum_refinement_depth": 6,
    }
    if args.map_method == "continuous"
    else None,
    "audit_masks": args.audit_masks,
    "audit_timing": "scoring capture excluded from controller and cycle latency"
    if args.audit_masks
    else None,
    "camera_rate_hz": 20,
    "policy_action_rate_hz": 20,
    "dropout_start_step": args.dropout_start_step,
    "dropout_frames": 5 if args.dropout_start_step >= 0 else 0,
}
write_report(args.output / "manifest.json", manifest)
records = []
try:
    with NavigationEnv(
        TASKS["authored"], split="validation", render_mode="rgb_array"
    ) as env:
        env.task.max_episode_steps = 1200
        for index in case_indices:
            specification = json.loads(
                (args.cases / f"authored-validation-{index:03}.json").read_text()
            )
            # Scenario placement and goal are evaluation inputs, never observations.
            raw, info = env.reset(
                seed=18000 + index,
                options={
                    "layout_seed": 11000 + index,
                    "start": specification["start"],
                    "goal": specification["goal"][:2],
                },
            )
            env.world.camera_period = 0.05
            camera = env.world.camera
            calibration = Calibration(
                np.array(camera.intrinsics),
                np.array(camera.extrinsics),
                tuple(camera.res),
                env.config.arena_half_extent,
            )
            observer = None
            if args.vision_checkpoint:
                from bb8_rl.vision import LearnedObserver

                observer = LearnedObserver(
                    calibration,
                    args.vision_checkpoint,
                    device=args.vision_device,
                    floor_refinement=args.floor_refinement,
                )
                manifest["vision_floor_target"] = observer.floor_target
            controller = CameraController(
                calibration,
                learned_action,
                calibration.to_pixel(specification["goal"][:2]),
                observer=observer,
                map_resolution=args.map_resolution,
                map_method=args.map_method,
            )
            if "machine" not in manifest:
                manifest["machine"] = machine_report(env.world_path)
                write_report(
                    args.output / "calibration.json",
                    {
                        "intrinsics": calibration.intrinsics.tolist(),
                        "world_to_camera": calibration.world_to_camera.tolist(),
                        "resolution": calibration.resolution,
                        "extent": calibration.extent,
                        "head_height": calibration.head_height,
                        "provenance": calibration.provenance,
                    },
                )
            rgb = camera.render(rgb=True)[0].copy()
            timestamp = 0.0
            rows, latencies, errors, velocity_errors = [], [], [], []
            cycles = []
            audit_latencies = []
            initial_audit = final_audit = None
            statuses, stopped = Counter(), 0
            ending = "time_limit"
            for step in range(1200):
                tick = time.perf_counter()
                observed_rgb = (
                    np.zeros_like(rgb)
                    if args.dropout_start_step >= 0
                    and args.dropout_start_step <= step < args.dropout_start_step + 5
                    else rgb
                )
                action = controller.action(observed_rgb, timestamp, now=env.world.time)
                latencies.append(time.perf_counter() - tick)
                statuses[controller.status] += 1
                # Read ground truth for scoring only, after the action is chosen.
                estimate = controller.belief.state
                if estimate is not None and controller.belief.status == "visible":
                    errors.append(
                        float(np.linalg.norm(estimate[:2] - raw["achieved_goal"][:2]))
                    )
                    velocity_errors.append(
                        float(np.linalg.norm(estimate[2:] - raw["observation"][2:4]))
                    )
                rows.append(
                    {
                        "time": env.world.time,
                        "capture_time": timestamp,
                        "action": action.tolist(),
                        "dropout_injected": observed_rgb is not rgb,
                        "status": controller.status,
                        "route_error": getattr(controller, "last_route_error", None),
                        "estimate": estimate.tolist() if estimate is not None else None,
                        "sigma_m": controller.belief.sigma
                        if estimate is not None
                        else None,
                        "scoring_truth": raw["achieved_goal"].tolist(),
                    }
                )
                if step == 2:
                    cv2.imwrite(
                        str(args.output / f"case-{index:03}-initial.png"),
                        annotated(rgb, controller),
                    )
                    if controller.grid is not None:
                        np.save(
                            args.output / f"case-{index:03}-camera-map.npy",
                            controller.grid.blocked,
                        )
                stopped = (
                    stopped + 1
                    if controller.status not in ("tracking", "warming_up")
                    else 0
                )
                audit_seconds = 0.0
                if args.audit_masks:
                    from bb8_rl.camera_audit import capture_camera_audit

                    audit_tick = time.perf_counter()
                    final_audit = capture_camera_audit(
                        env.world,
                        controller,
                        observed_rgb,
                        timestamp,
                        step=step,
                        action=action,
                    )
                    if initial_audit is None and controller.grid is not None:
                        initial_audit = final_audit
                    audit_seconds = time.perf_counter() - audit_tick
                    audit_latencies.append(audit_seconds)
                raw, _, terminal, truncated, info = env.step(action)
                cycles.append(time.perf_counter() - tick - audit_seconds)
                frame = env.world.frames[-1]
                rgb, timestamp = frame.rgb, frame.timestamp
                if info["is_success"] or terminal or truncated or stopped >= 20:
                    ending = (
                        "arrival"
                        if info["is_success"]
                        else "collision"
                        if info["collision"]
                        else "boundary"
                        if info["boundary_failure"]
                        else "perception_abort"
                        if stopped >= 20
                        else "time_limit"
                    )
                    break
            env.world.stop()
            if args.audit_masks:
                from bb8_rl.camera_audit import save_camera_audit

                if initial_audit is not None:
                    save_camera_audit(
                        args.output / f"case-{index:03}-audit-initial", initial_audit
                    )
                if final_audit is not None:
                    save_camera_audit(
                        args.output / f"case-{index:03}-audit-final", final_audit
                    )
            cv2.imwrite(
                str(args.output / f"case-{index:03}-final.png"),
                annotated(rgb, controller),
            )
            record = {
                "id": index,
                "scenario_id": specification.get("id", str(index)),
                "ending": ending,
                "success": info["is_success"],
                "collision": info["collision"],
                "sim_time": env.world.time,
                "final_error_m": float(
                    np.linalg.norm(raw["achieved_goal"][:2] - specification["goal"][:2])
                ),
                "final_speed_mps": float(raw["achieved_goal"][2]),
                "dwell_seconds": info["dwell_seconds"],
                "status_counts": dict(statuses),
                "final_route_error": getattr(controller, "last_route_error", None),
                "pixel_guard_m": controller.clearance.pixel_guard_m
                if controller.clearance is not None
                else None,
                "localization_p95_m": float(np.quantile(errors, 0.95))
                if errors
                else None,
                "velocity_error_p95_mps": float(np.quantile(velocity_errors, 0.95))
                if velocity_errors
                else None,
                "observer_policy_latency_p95_seconds": float(
                    np.quantile(latencies, 0.95)
                ),
                "observer_physics_render_cycle_p95_seconds": float(
                    np.quantile(cycles, 0.95)
                ),
                "audit_capture_p95_seconds": float(np.quantile(audit_latencies, 0.95))
                if audit_latencies
                else None,
                "audit_initial_map_saved": initial_audit is not None,
                "audit_final_action_saved": final_audit is not None,
                "trajectory": rows,
            }
            records.append(record)
            write_report(args.output / f"case-{index:03}.json", record)
            print(
                json.dumps({k: v for k, v in record.items() if k != "trajectory"}),
                flush=True,
            )
    manifest["status"] = "complete"
except BaseException as error:
    manifest.update(status="failed", error=repr(error))
    raise
finally:
    write_report(args.output / "manifest.json", manifest)
    write_report(
        args.output / "summary.json",
        {
            "completed": len(records),
            "successes": sum(r["success"] for r in records),
            "collisions": sum(r["collision"] for r in records),
            "perception_aborts": sum(
                r["ending"] == "perception_abort" for r in records
            ),
            "scope": "development-only RGB pilot",
            "cases": [
                {k: v for k, v in r.items() if k != "trajectory"} for r in records
            ],
        },
    )
