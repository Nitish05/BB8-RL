"""Offline development re-audit of the twelve static M7.4 captured scenes.

Only captured RGB and calibration enter inference. Truth visibility and authored
obstacles are separate scoring inputs; no simulation, actuation or promotion.
"""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration
from bb8_rl.planner import OccupancyGrid
from bb8_rl.vision import LearnedObserver

EXTENT = 2.0
RESOLUTION = 0.02
CLEARANCE = 0.10
REPO = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def segment_box_distance(start, end, box):
    """Exact Euclidean distance from a closed 2D segment to an axis-aligned box.

    ``box`` is (center_x, center_y, width, height). Contact or crossing gives zero;
    otherwise the closest pair involves a segment endpoint or rectangle corner.
    """
    start, end, box = (np.asarray(value, dtype=float) for value in (start, end, box))
    if (
        start.shape != (2,)
        or end.shape != (2,)
        or box.shape != (4,)
        or not all(np.isfinite(value).all() for value in (start, end, box))
        or np.any(box[2:] <= 0)
    ):
        raise ValueError("Need finite segment endpoints and a positive-size box")
    lower, upper = box[:2] - box[2:] / 2, box[:2] + box[2:] / 2
    delta = end - start
    t0, t1 = 0.0, 1.0
    intersects = True
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-15:
            intersects &= lower[axis] <= start[axis] <= upper[axis]
        else:
            a, b = (
                (lower[axis] - start[axis]) / delta[axis],
                (upper[axis] - start[axis]) / delta[axis],
            )
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    if intersects and t0 <= t1:
        return 0.0
    endpoint_distance = min(
        np.linalg.norm(np.maximum(np.maximum(lower - point, point - upper), 0))
        for point in (start, end)
    )
    length_squared = float(delta @ delta)
    if length_squared == 0:
        return float(endpoint_distance)
    corners = np.array(
        [[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]
    )
    t = np.clip((corners - start) @ delta / length_squared, 0, 1)
    corner_distance = np.linalg.norm(
        corners - (start + t[:, None] * delta), axis=1
    ).min()
    return float(min(endpoint_distance, corner_distance))


def segment_clearance(start, end, boxes, *, extent=EXTENT):
    """Centerline clearance to physical boxes and the arena's inner walls.

    The signed arena term also rejects segments already beyond a wall. For a
    straight segment, maximum absolute coordinate occurs at an endpoint.
    """
    endpoints = np.asarray([start, end], dtype=float)
    if (
        endpoints.shape != (2, 2)
        or not np.isfinite(endpoints).all()
        or not np.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("Invalid segment or arena extent")
    boundary = float(extent - np.max(abs(endpoints)))
    return min([boundary, *[segment_box_distance(start, end, box) for box in boxes]])


def visible_cells(calibration, mask):
    """Original probe rule: every pixel in each projected cell bbox is free."""
    axis = -EXTENT + (np.arange(200) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack((xx, yy), -1).reshape(-1, 2)
    offsets = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * RESOLUTION / 2
    uv = calibration.to_pixel(points[:, None] + offsets)
    low, high = np.floor(uv.min(1)).astype(int), np.ceil(uv.max(1)).astype(int)
    width, height = calibration.resolution
    mask = np.asarray(mask)
    if mask.shape != (height, width) or mask.dtype != np.bool_:
        raise ValueError("Native visibility mask must match calibration")
    inside = (
        (low[:, 0] >= 0)
        & (low[:, 1] >= 0)
        & (high[:, 0] < width)
        & (high[:, 1] < height)
    )
    x0, y0 = np.clip(low[:, 0], 0, width - 1), np.clip(low[:, 1], 0, height - 1)
    x1, y1 = (
        np.clip(high[:, 0], 0, width - 1) + 1,
        np.clip(high[:, 1], 0, height - 1) + 1,
    )
    integral = cv2.integral((~mask).astype(np.uint8), sdepth=cv2.CV_32S)
    count = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return (inside & (count == 0)).reshape(200, 200)


def inflated_grid(free):
    """Match the probe's square-cell metric buffer, including its wall margin."""
    grid = OccupancyGrid(EXTENT, RESOLUTION, CLEARANCE, [])
    radius = int(np.ceil(CLEARANCE / RESOLUTION)) + 1
    a = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(a, a)
    kernel = (
        RESOLUTION * np.hypot(np.maximum(abs(dx) - 1, 0), np.maximum(abs(dy) - 1, 0))
        <= CLEARANCE + 1e-12
    ).astype(np.uint8)
    grid.blocked |= cv2.dilate((~free).astype(np.uint8), kernel).astype(bool)
    return grid


def route_score(free, truth, start, goal, boxes):
    grid, truth_grid = inflated_grid(free), inflated_grid(truth)
    try:
        route = grid.route(start, goal)
    except ValueError as error:
        return {
            "proposed": False,
            "reason": str(error),
            "truth_certified": False,
            "physical_clearance_certified": False,
            "min_physical_clearance_m": None,
        }
    segments = list(pairwise(route)) or [(route[0], route[0])]
    uncertified = [
        i for i, (a, b) in enumerate(segments) if not truth_grid.segment_free(a, b)
    ]
    clearances = [segment_clearance(a, b, boxes) for a, b in segments]
    physical_bad = [
        i for i, value in enumerate(clearances) if value + 1e-12 < CLEARANCE
    ]
    return {
        "proposed": True,
        "points": route.tolist(),
        "length_m": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
        "truth_certified": not uncertified,
        "truth_uncertified_segments": uncertified,
        "physical_clearance_certified": not physical_bad,
        "physical_uncertified_segments": physical_bad,
        "segment_physical_clearance_m": clearances,
        "min_physical_clearance_m": min(clearances),
    }


def audit(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    capture_manifest_path = args.captures / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    if (
        capture_manifest.get("status") != "complete"
        or capture_manifest.get("world_grid_resolution_m") != RESOLUTION
        or capture_manifest.get("clearance_m") != CLEARANCE
    ):
        raise ValueError("Require a complete original 2cm/10cm M7.4 capture")
    project = json.loads(args.world.read_text())
    physical_objects = [
        obj
        for obj in project["objects"]
        if obj["name"].startswith(("room_obstacle", "wall_"))
    ]
    if not physical_objects or any(
        obj.get("euler", [0, 0, 0]) != [0, 0, 0] for obj in physical_objects
    ):
        raise ValueError("Physical scorer requires authored axis-aligned room boxes")
    boxes = [[*obj["position"][:2], *obj["size"][:2]] for obj in physical_objects]
    axis = -EXTENT + (np.arange(200) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(axis, axis)
    physical_centers = np.zeros((200, 200), bool)
    for x, y, width, height in boxes:
        physical_centers |= (abs(xx - x) <= width / 2) & (abs(yy - y) <= height / 2)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    checkpoint_hash = file_hash(args.checkpoint)
    provenance = {"manifest.json": file_hash(capture_manifest_path)}
    cases = []
    observer = None
    for case in range(12):
        case_path = args.captures / f"case-{case:03}.json"
        provenance[case_path.name] = file_hash(case_path)
        record = json.loads(case_path.read_text())
        current, baseline, truths, view_rows = [], [], [], []
        for view in range(1, 4):
            prefix = args.captures / f"case-{case:03}-view-{view}"
            for suffix in (".json", ".png", ".npz"):
                path = prefix.with_suffix(suffix)
                provenance[path.name] = file_hash(path)
            metadata = json.loads(prefix.with_suffix(".json").read_text())
            rgb = cv2.cvtColor(
                cv2.imread(str(prefix.with_suffix(".png"))), cv2.COLOR_BGR2RGB
            )
            calibration = Calibration(
                np.array(metadata["intrinsics"]),
                np.array(metadata["world_to_camera"]),
                (1280, 960),
                EXTENT,
            )
            if observer is None:
                observer = LearnedObserver(
                    calibration, args.checkpoint, floor_refinement="guided"
                )
            observer.calibration = calibration
            # There is deliberately no scorer input in this inference call.
            measurement = observer.observe(rgb, metadata["capture_time"])
            predicted = visible_cells(
                calibration, measurement.visible_floor | measurement.robot_pixels
            )
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as stored:
                truth = visible_cells(calibration, stored["truth_visible"])
                old = visible_cells(calibration, stored["predicted_visible"])
                if not np.array_equal(
                    truth, stored["truth_free_cells"]
                ) or not np.array_equal(old, stored["predicted_free_cells"]):
                    raise ValueError(
                        "Original whole-cell map failed exact reconstruction"
                    )
            current.append(predicted)
            baseline.append(old)
            truths.append(truth)
            view_rows.append(
                {
                    "camera_id": "ABC"[view - 1],
                    "status": measurement.status,
                    "localization_error_m": float(
                        np.linalg.norm(measurement.xy - record["start"])
                    )
                    if measurement.xy is not None
                    else None,
                    "current_false_free_vs_own_truth_cells": int(
                        (predicted & ~truth).sum()
                    ),
                    "current_free_centers_inside_physical_boxes": int(
                        (predicted & physical_centers).sum()
                    ),
                    "baseline_free_centers_inside_physical_boxes": int(
                        (old & physical_centers).sum()
                    ),
                }
            )
        all_truth = np.logical_or.reduce(truths)
        for row, predicted, old in zip(view_rows, current, baseline):
            row["current_false_free_vs_all_three_truth_cells"] = int(
                (predicted & ~all_truth).sum()
            )
            row["baseline_false_free_vs_all_three_truth_cells"] = int(
                (old & ~all_truth).sum()
            )
        rigs = {}
        for count in (1, 2, 3):
            truth = np.logical_or.reduce(truths[:count])
            rigs[str(count)] = {}
            for name, masks in (("current", current), ("baseline", baseline)):
                free = np.logical_or.reduce(masks[:count])
                score = route_score(free, truth, record["start"], record["goal"], boxes)
                score.update(
                    false_free_vs_same_rig_truth_cells=int((free & ~truth).sum()),
                    false_free_vs_all_three_truth_cells=int((free & ~all_truth).sum()),
                    free_centers_inside_physical_boxes=int(
                        (free & physical_centers).sum()
                    ),
                    predicted_visible_fraction=float(free.mean()),
                )
                rigs[str(count)][name] = score
        item = {
            "case": case,
            "start": record["start"],
            "goal": record["goal"],
            "views": view_rows,
            "rigs": rigs,
        }
        write_report(args.output / f"case-{case:03}.json", item)
        np.savez_compressed(
            args.output / f"case-{case:03}-maps.npz",
            current=np.stack(current),
            baseline=np.stack(baseline),
            truth=np.stack(truths),
        )
        cases.append(item)
        print(
            json.dumps(
                {
                    "case": case,
                    "current_routes": {
                        k: v["current"]["proposed"] for k, v in rigs.items()
                    },
                }
            ),
            flush=True,
        )
    summary = {}
    for name in ("current", "baseline"):
        summary[name] = {}
        for count in (1, 2, 3):
            scores = [case["rigs"][str(count)][name] for case in cases]
            summary[name][str(count)] = {
                "proposed_routes": sum(row["proposed"] for row in scores),
                "same_rig_truth_certified_routes": sum(
                    row["truth_certified"] for row in scores
                ),
                "continuous_physical_clearance_certified_routes": sum(
                    row["physical_clearance_certified"] for row in scores
                ),
                "mean_free_centers_inside_physical_boxes": float(
                    np.mean(
                        [row["free_centers_inside_physical_boxes"] for row in scores]
                    )
                ),
                "mean_false_free_vs_same_rig_truth_cells": float(
                    np.mean(
                        [row["false_free_vs_same_rig_truth_cells"] for row in scores]
                    )
                ),
                "physical_uncertified_case_ids": [
                    case["case"]
                    for case, row in zip(cases, scores)
                    if row["proposed"] and not row["physical_clearance_certified"]
                ],
                "truth_uncertified_case_ids": [
                    case["case"]
                    for case, row in zip(cases, scores)
                    if row["proposed"] and not row["truth_certified"]
                ],
            }
    if file_hash(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during offline audit")
    report = {
        "status": "complete",
        "scope": "Original twelve development cases only; offline static routes, not arrivals or driving approval",
        "checkpoint_sha256": checkpoint_hash,
        "capture_baseline_checkpoint_sha256": capture_manifest["vision_sha256"],
        "audit_source_sha256": file_hash(Path(__file__)),
        "physical_world_sha256": file_hash(args.world),
        "capture_file_sha256": provenance,
        "refinement": "guided-radius4-epsilon1e-4",
        "grid_resolution_m": RESOLUTION,
        "total_centerline_clearance_m": CLEARANCE,
        "physical_clearance_rule": "Exact continuous segment-to-axis-aligned-rectangle Euclidean distance plus signed inner-wall clearance; 0.10m is the entire footprint/uncertainty envelope, not extra radius",
        "false_free_rule": "Physical metrics count grid centers inside box footprints, not whole-cell safety. Visibility errors may instead be occlusion. Truth labels and exact endpoints are scoring only.",
        "fusion_rule": "Exploratory OR of whole-cell free evidence; never promoted by this audit",
        "summary": summary,
        "cases": cases,
    }
    write_report(args.output / "report.json", report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--world",
        type=Path,
        default=REPO / "projects/bb8/synthetic-room/room.genesis.json",
    )
    audit(parser.parse_args())
