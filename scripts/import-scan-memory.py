"""Import completed learned surface estimates; never declare a free corridor."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bb8_rl.mapping import Bounds3D
from bb8_rl.mapping.surface_memory import import_surfaces


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = np.load(args.reconstruction)
    memory, report = import_surfaces(
        data[args.field],
        data["valid_mask"],
        data["view_ids"].tolist(),
        bounds=Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 1.5)),
        resolution_m=0.05,
        uncertainty_m=0.05,
        scene_version=args.scene_version,
        calibration_version="synthetic-scan-poses-metric-registration",
    )
    memory.save(args.output / "room-memory.json")
    report.update(
        {
            "source_npz_sha256": hashlib.sha256(
                args.reconstruction.read_bytes()
            ).hexdigest(),
            "field": args.field,
            "view_ids": data["view_ids"].tolist(),
            "suitable_for_blind_control": False,
        }
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--scene-version", default="estimated-scan-surfaces")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
