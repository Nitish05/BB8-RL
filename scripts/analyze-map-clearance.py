"""Offline scoring of estimated-map clearance against authored synthetic boxes.

This scorer is never imported by control. Exact authored geometry is evaluation
data only; results apply to this frozen room, not unknown physical obstacles.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bb8_rl.mapping.scan_free_space import ScanFreeMemory


def analyze(assets):
    memory = ScanFreeMemory.load(assets / "memory")
    scene_path = assets / "scene/room.genesis.json"
    scene = json.loads(scene_path.read_text())
    boxes = [
        obj for obj in scene["objects"]
        if obj.get("fixed") and obj.get("format") == "box"
        and obj.get("collision_group", 0)
        and obj["position"][2] + obj["size"][2] / 2 > 0
        and obj["position"][2] - obj["size"][2] / 2 < 0.074
    ]
    y, x = np.indices(memory.free_mask.shape)
    centers = np.stack([
        -memory.extent + (x + 0.5) * memory.resolution,
        -memory.extent + (y + 0.5) * memory.resolution,
    ], axis=-1)
    distance = np.full(memory.free_mask.shape, np.inf)
    for box in boxes:
        if not np.allclose(box["euler"], 0):
            raise ValueError("This static scorer requires axis-aligned boxes")
        q = np.abs(centers - box["position"][:2]) - np.array(box["size"][:2]) / 2
        signed = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
        distance = np.minimum(distance, signed)
    checks = []
    for uncertainty in (0.027, 0.043, 0.053):
        for margin in (0.04, 0.03, 0.02):
            radius = math.ceil(max(0.1, 0.037 + margin + uncertainty) / 0.02) * 0.02
            valid = ~memory.planning_grid(radius).blocked
            # Signed distance is 1-Lipschitz. Subtract the cell half diagonal,
            # body radius and localization bound for every possible center.
            clearance = distance[valid] - 0.037 - uncertainty - memory.resolution / math.sqrt(2)
            checks.append({
                "margin_m": margin, "position_bound_m": uncertainty,
                "planning_radius_m": radius, "free_planning_cells": int(valid.sum()),
                "minimum_true_surface_gap_with_cell_and_localization_bounds_m": float(clearance.min()),
                "nonpositive_gap_cells": int((clearance <= 0).sum()),
            })
    return {
        "scope": "Offline authored-axis-aligned synthetic collider scoring only. No geometry enters online control. Covers frozen map/cell corners at chosen position bounds, not unseen maps or motion guarantees.",
        "box_count": len(boxes), "raw_free_cells": int(memory.free_mask.sum()),
        "raw_free_centers_inside_true_box": int((memory.free_mask & (distance <= 0)).sum()),
        "raw_free_center_min_true_signed_distance_m": float(distance[memory.free_mask].min()),
        "checks": checks,
        "body_radius_m": 0.037,
        "grid_resolution_m": memory.resolution,
        "position_bounds_m": [0.027, 0.043, 0.053],
        "memory_manifest_sha256": hashlib.sha256((assets / "memory/manifest.json").read_bytes()).hexdigest(),
        "free_grid_sha256": hashlib.sha256((assets / "memory/free-grid.npz").read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.assets)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
