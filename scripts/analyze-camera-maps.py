"""Offline map audit. Oracle visible masks are diagnostics, never control input."""

import argparse
import json
from pathlib import Path

import numpy as np

from bb8_rl.camera import Calibration, VisualMeasurement, camera_grid
from bb8_rl.diagnostics import write_report

parser = argparse.ArgumentParser()
parser.add_argument("--audit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = []
for path in sorted(args.audit.glob("*-audit-*.json")):
    metadata = json.loads(path.read_text())
    c = metadata["calibration"]
    calibration = Calibration(
        np.array(c["intrinsics"]),
        np.array(c["world_to_camera"]),
        tuple(c["resolution"]),
        c["extent"],
        c["head_height"],
    )
    position, goal = np.array(metadata["estimate"][:2]), np.array(metadata["goal"])
    inflation = metadata["grid"]["inflation"]
    scores = []
    with np.load(path.with_suffix(".npz")) as data:
        for source in ("predicted", "truth"):
            measurement = VisualMeasurement(
                metadata["capture_time"],
                position,
                np.eye(2),
                data[f"{source}_floor"],
                data[f"{source}_self"],
                "visible",
            )
            for method, resolution in (
                ("legacy", 0.05),
                ("legacy", 0.025),
                ("metric", 0.025),
                ("metric", 0.02),
            ):
                grid = camera_grid(
                    calibration,
                    measurement,
                    resolution=resolution,
                    inflation=inflation,
                    method=method,
                )
                try:
                    route = grid.route(position, goal)
                    route_error, length = (
                        None,
                        float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
                    )
                except ValueError as error:
                    route_error, length = str(error), None
                scores.append(
                    {
                        "source": source,
                        "method": method,
                        "resolution": resolution,
                        "inflation_m": inflation,
                        "start_free": grid.free(grid.cell(position)),
                        "goal_free": grid.free(grid.cell(goal)),
                        "route_error": route_error,
                        "route_length_m": length,
                    }
                )
    rows.append(
        {
            "snapshot": path.stem,
            "estimate": position.tolist(),
            "goal": goal.tolist(),
            "status": metadata["status"],
            "scores": scores,
        }
    )
    print(json.dumps(rows[-1]), flush=True)
write_report(
    args.output,
    {
        "scope": "scoring-only offline sweep; truth-mask routes are not deployable results",
        "snapshots": rows,
    },
)
