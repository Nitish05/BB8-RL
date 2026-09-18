"""Frozen RGB parity audit and separately scheduled native two-camera timing."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration
from bb8_rl.camera_rig import CameraRig, CameraView, calibration_from_live_camera
from bb8_rl.camera_runtime import DeadlineCameraRig
from bb8_rl.diagnostics import write_report
from bb8_rl.training import file_hash
from bb8_rl.vision import LearnedObserver, PositionOnlyObserver
from bb8_rl.vision_data import validate_scene_splits


def summary(values):
    return {
        "count": len(values),
        "p50": float(np.quantile(values, 0.5)) if values else None,
        "p95": float(np.quantile(values, 0.95)) if values else None,
        "max": float(max(values)) if values else None,
    }


def paired_equal(first, second):
    if first.status != second.status or first.timestamp != second.timestamp:
        return False
    if first.xy is None or second.xy is None:
        return (
            first.xy is None
            and second.xy is None
            and first.covariance is None
            and second.covariance is None
        )
    return bool(
        np.array_equal(first.xy, second.xy)
        and np.array_equal(first.covariance, second.covariance)
    )


def offline(args):
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_scene_splits(manifest)
    if (
        manifest.get("status") != "complete"
        or manifest.get("format") != "bb8-multiview-supervision-v1"
    ):
        raise ValueError("Complete native multi-view dataset required")
    reference = json.loads(args.reference_report.read_text())
    if reference["checkpoint_sha256"] != file_hash(args.checkpoint):
        raise ValueError("Historical full-observer report used another checkpoint")
    previous = {
        (row["scene_group"], row["record_index"], row["camera_id"]): row
        for row in reference["per_view_frames"]
    }
    records, consumed = [], {}
    observer = None
    for sequence in manifest["sequences"]:
        if sequence["split"] != "development":
            continue
        path = args.dataset / "scenes" / f"{sequence['id']:03}" / "labels.json"
        if file_hash(path) != sequence["labels_sha256"]:
            raise ValueError("Calibration/labels checksum mismatch")
        consumed[str(path.relative_to(args.dataset))] = file_hash(path)
        labels = json.loads(path.read_text())
        for entry in sequence["native_frames"]:
            path = args.dataset / entry["rgb"]
            if file_hash(path) != entry["sha256"]["rgb"]:
                raise ValueError("Native RGB checksum mismatch")
            consumed[entry["rgb"]] = file_hash(path)
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            c = labels["calibration"]
            calibration = Calibration(
                np.array(c["intrinsics"]),
                np.array(c["world_to_camera"]),
                (rgb.shape[1], rgb.shape[0]),
                2,
            )
            if observer is None:
                observer = LearnedObserver(
                    calibration, args.checkpoint, floor_refinement="guided"
                )
            observer.calibration = calibration
            index = entry["record_index"]
            timestamp = labels["records"][index]["time"]
            measured, timings = {}, {}
            # Alternate order to avoid giving one path all warm-cache benefits.
            for variant in (
                ("full", "fast") if len(records) % 2 == 0 else ("fast", "full")
            ):
                started = time.monotonic()
                measured[variant] = (
                    observer.observe if variant == "full" else observer.observe_position
                )(rgb, timestamp)
                timings[variant] = time.monotonic() - started
            full, fast = measured["full"], measured["fast"]
            before = previous[(sequence["scene_group"], index, sequence["camera_id"])]
            historical_match = before["status"] == full.status and (
                (before["xy"] is None and full.xy is None)
                or (
                    before["xy"] is not None
                    and full.xy is not None
                    and np.array_equal(before["xy"], full.xy)
                )
            )
            no_free_space = not fast.visible_floor.any() and not fast.robot_pixels.any()
            records.append(
                {
                    "scene_group": sequence["scene_group"],
                    "camera_id": sequence["camera_id"],
                    "record_index": index,
                    "status": fast.status,
                    "exact_status_position_covariance_parity": paired_equal(full, fast),
                    "exact_historical_full_status_position_parity": bool(
                        historical_match
                    ),
                    "no_free_space_evidence": bool(no_free_space),
                    "full_seconds": timings["full"],
                    "fast_seconds": timings["fast"],
                }
            )
    if not records or len(records) != len(previous):
        raise ValueError("Historical report and development RGB membership differ")
    return {
        "mode": "offline",
        "scope": "paired identical RGB detection parity; timings exclude capture and are not isolated native throughput",
        "dataset_manifest_sha256": file_hash(manifest_path),
        "historical_report_sha256": file_hash(args.reference_report),
        "consumed_files_sha256": consumed,
        "frames": len(records),
        "exact_parity_count": sum(
            row["exact_status_position_covariance_parity"] for row in records
        ),
        "exact_historical_parity_count": sum(
            row["exact_historical_full_status_position_parity"] for row in records
        ),
        "no_free_space_evidence_count": sum(
            row["no_free_space_evidence"] for row in records
        ),
        "full_seconds": summary([row["full_seconds"] for row in records]),
        "fast_seconds": summary([row["fast_seconds"] for row in records]),
        "records": records,
    }


def native(args):
    # Native work is opt-in and must be scheduled with other heavy jobs paused.
    from bb8_rl.env import NavigationEnv
    from bb8_rl.guided import TASKS

    records = []
    with NavigationEnv(
        TASKS["authored"],
        render_mode="rgb_array",
        camera_positions=[(2.8, -3.4, 5.2), (-2.8, 3.4, 5.2)],
    ) as env:
        env.reset(
            seed=761001,
            options={"layout_seed": 1001, "start": [-1.75, -1.75], "goal": [-1, -1.75]},
        )
        env.world.camera_period = env.world._next_frame = 1e9
        views = {"full": [], "fast": []}
        for name, camera in env.world.cameras.items():
            calibration = calibration_from_live_camera(camera, 2)
            observer = LearnedObserver(
                calibration, args.checkpoint, floor_refinement="guided"
            )
            for variant, registered in views.items():
                registered.append(
                    CameraView(
                        name,
                        "synthetic-v1",
                        calibration,
                        observer
                        if variant == "full"
                        else PositionOnlyObserver(observer),
                    )
                )
        full_rig = CameraRig(views["full"])
        fast_runtime = DeadlineCameraRig(
            CameraRig(views["fast"]), budget_seconds=args.budget
        )
        for index in range(args.warmup + args.iterations):
            for variant in ("full", "fast") if index % 2 == 0 else ("fast", "full"):
                started = time.monotonic()
                frames = env.world.capture_rig()
                captured = time.monotonic()
                if variant == "fast":
                    gated = fast_runtime.process(
                        frames, now=env.world.time, capture_started=started
                    )
                    result = gated.observation
                    elapsed = gated.elapsed_seconds
                    processing = gated.processing_seconds
                    accepted = gated.accepted
                    gate_status = gated.status
                else:
                    result = full_rig.observe(frames, now=env.world.time)
                    finished = time.monotonic()
                    elapsed, processing = finished - started, finished - captured
                    accepted = result.status == "visible"
                    gate_status = "ungated_reference"
                records.append(
                    {
                        "iteration": index,
                        "warmup": index < args.warmup,
                        "variant": variant,
                        "capture_seconds": captured - started,
                        "processing_seconds": processing,
                        "total_seconds": elapsed,
                        "accepted": accepted,
                        "gate_status": gate_status,
                        "rig_status": result.status,
                        "source_ids": list(result.source_ids),
                        "capture_times": {
                            frame.camera_id: frame.capture_time for frame in frames
                        },
                        "xy": result.xy.tolist() if result.xy is not None else None,
                        "no_free_space_evidence": all(
                            item.measurement is None
                            or (
                                not item.measurement.visible_floor.any()
                                and not item.measurement.robot_pixels.any()
                            )
                            for item in result.views.values()
                        )
                        if variant == "fast"
                        else None,
                    }
                )
            # Zero command only advances independent capture timestamps. No rig
            # estimate is connected to control and no drive-policy claim is made.
            _, _, terminal, truncated, _ = env.step([0, 0])
            if terminal or truncated:
                raise RuntimeError("Stationary timing probe ended unexpectedly")
        if env.world.contacts:
            raise RuntimeError(
                "Stationary timing probe unexpectedly contacted an obstacle"
            )
    aggregate = {}
    for variant in ("full", "fast"):
        selected = [r for r in records if r["variant"] == variant and not r["warmup"]]
        aggregate[variant] = {
            "iterations": len(selected),
            **{
                metric: summary([row[metric] for row in selected])
                for metric in ("capture_seconds", "processing_seconds", "total_seconds")
            },
            "accepted": sum(row["accepted"] for row in selected),
            "status_counts": dict(Counter(row["rig_status"] for row in selected)),
            "deadline_misses": sum(
                row["total_seconds"] > args.budget for row in selected
            ),
        }
    return {
        "mode": "native",
        "scope": "isolated stationary two-camera capture + CPU localization + fusion; no navigation or closed-loop rate claim",
        "isolation_asserted_by_caller": True,
        "budget_seconds": args.budget,
        "warmup_iterations_per_variant": args.warmup,
        "aggregate": aggregate,
        "records": records,
        "limitations": [
            "Rendering/inference cannot be preempted; overdue output is rejected after it finishes.",
            "Stationary scene and CPU observer; deadlines require broader moving/cluttered-scene evaluation.",
        ],
    }


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh benchmark output")
    if args.iterations < 1 or args.warmup < 0 or args.budget <= 0:
        raise ValueError("Invalid benchmark length or budget")
    if args.mode == "native" and not args.isolated:
        raise ValueError(
            "Native timing requires --isolated after coordinating heavy jobs"
        )
    checkpoint_hash = file_hash(args.checkpoint)
    selection_path = args.checkpoint.parent / "calibration.json"
    if json.loads(selection_path.read_text())["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Frozen checkpoint/calibration hash mismatch")
    torch.set_num_threads(2)
    report = offline(args) if args.mode == "offline" else native(args)
    if file_hash(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during benchmark")
    source_root = Path(__file__).resolve().parent.parent
    report.update(
        {
            "status": "complete",
            "checkpoint_sha256": checkpoint_hash,
            "calibration_sha256": file_hash(selection_path),
            "torch_threads": torch.get_num_threads(),
            "source_sha256": {
                name: file_hash(source_root / name)
                for name in (
                    "src/bb8_rl/vision.py",
                    "src/bb8_rl/camera_rig.py",
                    "src/bb8_rl/camera_runtime.py",
                    "scripts/benchmark-fast-rig.py",
                )
            },
            "fitting_or_threshold_changes": False,
        }
    )
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("records", "consumed_files_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "native"), required=True)
    parser.add_argument("--dataset", type=Path, default=Path("work/m75/multiview-data"))
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path("work/m75/rig-localization/report.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("work/m75/calibrated/model.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--isolated", action="store_true")
    main(parser.parse_args())
