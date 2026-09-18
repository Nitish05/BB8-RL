"""Compare head-only RGB inference with frozen native full-observer occlusion results."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration
from bb8_rl.camera_rig import CameraFrame, CameraRig, CameraView
from bb8_rl.diagnostics import write_report
from bb8_rl.training import file_hash
from bb8_rl.vision import LearnedObserver, PositionOnlyObserver


def same_optional(first, second):
    if first is None or second is None:
        return first is None and second is None
    return bool(np.array_equal(first, second))


def audit(args):
    if args.output.exists():
        raise ValueError("Use a fresh parity output")
    reference_path = args.dataset / "report.json"
    calibration_path = args.dataset / "calibrations.json"
    reference = json.loads(reference_path.read_text())
    calibrations = json.loads(calibration_path.read_text())
    if reference["status"] != "complete" or reference["calibrations"] != calibrations:
        raise ValueError("Incomplete probe or mismatched calibration metadata")
    checkpoint_hash = file_hash(args.checkpoint)
    if reference["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint differs from the frozen native probe")
    torch.set_num_threads(2)
    views = []
    for name, c in calibrations.items():
        calibration = Calibration(
            np.array(c["intrinsics"]),
            np.array(c["world_to_camera"]),
            tuple(c["resolution"]),
            2,
            head_height=c["head_height"],
            provenance=c["provenance"],
        )
        observer = LearnedObserver(
            calibration, args.checkpoint, floor_refinement="guided"
        )
        views.append(
            CameraView(
                name, "geometric-v1", calibration, PositionOnlyObserver(observer)
            )
        )
    rig = CameraRig(views)
    records, rig_records, consumed = [], [], {}
    for row in reference["observations"]:
        frames = []
        for name, recorded in row["views"].items():
            path = args.dataset / recorded["rgb_path"]
            if file_hash(path) != recorded["rgb_sha256"]:
                raise ValueError("Native RGB checksum mismatch")
            consumed[recorded["rgb_path"]] = recorded["rgb_sha256"]
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            frames.append(CameraFrame(name, "geometric-v1", rgb, row["time"]))
        # Only RGB, calibrated camera geometry and sensor times enter the rig.
        result = rig.observe(frames, now=row["time"])
        for name, observed in result.views.items():
            recorded = row["views"][name]
            measurement = observed.measurement
            xy = measurement.xy if measurement is not None else None
            covariance = measurement.covariance if measurement is not None else None
            records.append(
                {
                    "step": row["step"],
                    "time": row["time"],
                    "camera_id": name,
                    "recorded_status": recorded["status"],
                    "fast_status": observed.status,
                    "exact_status_parity": observed.status == recorded["status"],
                    "exact_position_parity": same_optional(xy, recorded["position"]),
                    "exact_covariance_parity": same_optional(
                        covariance, recorded["covariance"]
                    ),
                    "no_free_space_evidence": measurement is None
                    or (
                        not measurement.visible_floor.any()
                        and not measurement.robot_pixels.any()
                    ),
                    "truth_visibility_scoring_only": recorded[
                        "truth_visibility_scoring_only"
                    ],
                    "head_pixels_scoring_only": recorded["head_pixels_scoring_only"],
                }
            )
        expected = row["rig"]
        rig_records.append(
            {
                "step": row["step"],
                "time": row["time"],
                "recorded_status": expected["status"],
                "fast_status": result.status,
                "exact_status_parity": result.status == expected["status"],
                "exact_position_parity": same_optional(result.xy, expected["position"]),
                "exact_covariance_parity": same_optional(
                    result.covariance, expected["covariance"]
                ),
                "exact_source_ids_parity": list(result.source_ids)
                == expected["source_ids"],
                "exact_handover_parity": result.handover == expected["handover"],
                "source_ids": list(result.source_ids),
            }
        )
    if file_hash(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during audit")
    view_keys = [
        "exact_status_parity",
        "exact_position_parity",
        "exact_covariance_parity",
        "no_free_space_evidence",
    ]
    rig_keys = [
        "exact_status_parity",
        "exact_position_parity",
        "exact_covariance_parity",
        "exact_source_ids_parity",
        "exact_handover_parity",
    ]
    by_visibility = {}
    for visibility in sorted({row["truth_visibility_scoring_only"] for row in records}):
        selected = [
            row for row in records if row["truth_visibility_scoring_only"] == visibility
        ]
        by_visibility[visibility] = {
            "frames": len(selected),
            "status_counts": dict(Counter(row["fast_status"] for row in selected)),
            **{key: sum(row[key] for row in selected) for key in view_keys},
        }
    report = {
        "status": "complete",
        "scope": "offline exact detection/fusion parity on saved native geometric occlusion RGB; no timing or driving claim",
        "checkpoint_sha256": checkpoint_hash,
        "reference_report_sha256": file_hash(reference_path),
        "calibrations_sha256": file_hash(calibration_path),
        "rgb_sha256": consumed,
        "script_sha256": file_hash(Path(__file__)),
        "vision_source_sha256": file_hash(
            Path(__file__).resolve().parent.parent / "src/bb8_rl/vision.py"
        ),
        "fitting_or_threshold_changes": False,
        "head_truth_masks_used_as_input": False,
        "view_frames": len(records),
        "view_parity_counts": {
            key: sum(row[key] for row in records) for key in view_keys
        },
        "status_counts": dict(Counter(row["fast_status"] for row in records)),
        "by_truth_visibility_scoring_only": by_visibility,
        "rig_frames": len(rig_records),
        "rig_parity_counts": {
            key: sum(row[key] for row in rig_records) for key in rig_keys
        },
        "all_exact": all(all(row[key] for key in view_keys) for row in records)
        and all(all(row[key] for key in rig_keys) for row in rig_records),
        "view_records": records,
        "rig_records": rig_records,
    }
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("view_records", "rig_records", "rgb_sha256")
            },
            indent=2,
        )
    )
    if not report["all_exact"]:
        raise RuntimeError("Fast occlusion parity failed; inspect the saved report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("work/m76/geometric-occlusion")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("work/m75/calibrated/model.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/m76/fast-occlusion-parity")
    )
    audit(parser.parse_args())
