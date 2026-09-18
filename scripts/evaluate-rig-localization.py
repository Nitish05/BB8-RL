"""Frozen, paired static RGB localization audit; never a navigation/tracking test."""

import argparse
import hashlib
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
from bb8_rl.vision import LearnedObserver
from bb8_rl.vision_data import validate_scene_splits


class CachedRGBObserver:
    """Replay only the exact same RGB-derived measurement, never scorer labels."""

    def __init__(self, calibration, rgb, measurement):
        self.calibration = calibration
        self.rgb_digest = hashlib.sha256(rgb.tobytes()).digest()
        self.measurement = measurement

    def observe(self, rgb, timestamp):
        if (
            timestamp != self.measurement.timestamp
            or hashlib.sha256(rgb.tobytes()).digest() != self.rgb_digest
        ):
            raise ValueError("Cached measurement belongs to another RGB capture")
        return self.measurement


def error_summary(values):
    return {
        "count": len(values),
        "p50": float(np.quantile(values, 0.5)) if values else None,
        "p95": float(np.quantile(values, 0.95)) if values else None,
        "max": float(max(values)) if values else None,
    }


def evaluate(args):
    if args.output.exists():
        raise ValueError("Use a fresh evaluation output")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("format") != "bb8-multiview-supervision-v1"
    ):
        raise ValueError("A complete native multi-view dataset is required")
    validate_scene_splits(manifest)
    calibration_path = args.checkpoint.parent / "calibration.json"
    checkpoint_hash = file_hash(args.checkpoint)
    selection = json.loads(calibration_path.read_text())
    if selection["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint differs from its frozen validation calibration")
    groups = {}
    consumed = {}
    for sequence in manifest["sequences"]:
        if sequence["split"] != "development":
            continue
        group = groups.setdefault(sequence["scene_group"], {})
        if sequence["camera_id"] in group:
            raise ValueError("Duplicate camera in development room")
        labels_path = args.dataset / "scenes" / f"{sequence['id']:03}" / "labels.json"
        digest = file_hash(labels_path)
        if digest != sequence["labels_sha256"]:
            raise ValueError("Native labels/calibration checksum mismatch")
        consumed[str(labels_path.relative_to(args.dataset))] = digest
        labels = json.loads(labels_path.read_text())
        if len(labels["records"]) != sequence["frames"]:
            raise ValueError("Native record count differs from manifest")
        group[sequence["camera_id"]] = (sequence, labels)
    if not groups or any(set(group) != set("ABC") for group in groups.values()):
        raise ValueError("Every development room must contain cameras A, B and C")

    torch.set_num_threads(2)
    observer = None
    rows, view_rows = [], []
    for room, group in sorted(groups.items()):
        entry_maps = {
            name: {entry["record_index"]: entry for entry in sequence["native_frames"]}
            for name, (sequence, _) in group.items()
        }
        indices = set(entry_maps["A"])
        if any(set(entries) != indices for entries in entry_maps.values()):
            raise ValueError("Camera reset positions are not paired")
        if any(len(indices) != seq["frames"] for seq, _ in group.values()):
            raise ValueError("Duplicate or missing native reset record")
        for index in sorted(indices):
            cached, frames = {}, {}
            # Inference consumes only RGB, fixed calibration and capture time.
            for name in "ABC":
                sequence, labels = group[name]
                entry = entry_maps[name][index]
                rgb_path = args.dataset / entry["rgb"]
                digest = file_hash(rgb_path)
                if digest != entry["sha256"]["rgb"]:
                    raise ValueError("Native RGB checksum mismatch")
                consumed[entry["rgb"]] = digest
                bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Cannot decode native RGB")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                c = labels["calibration"]
                calibration = Calibration(
                    np.array(c["intrinsics"]),
                    np.array(c["world_to_camera"]),
                    (rgb.shape[1], rgb.shape[0]),
                    args.extent,
                    provenance="synthetic_exact_live_transform",
                )
                if observer is None:
                    observer = LearnedObserver(
                        calibration, args.checkpoint, floor_refinement="guided"
                    )
                observer.calibration = calibration
                capture_time = labels["records"][index]["time"]
                measurement = observer.observe(rgb, capture_time)
                cached[name] = CachedRGBObserver(calibration, rgb, measurement)
                frames[name] = CameraFrame(name, "synthetic-v1", rgb, capture_time)
            if len({frame.capture_time for frame in frames.values()}) != 1:
                raise ValueError("Static native captures are not synchronized")
            outcomes = {}
            for names in ("A", "AB", "ABC"):
                # Every static reset gets fresh rig state. Repeated 0.05 s reset
                # timestamps must not be represented as a fabricated trajectory.
                rig = CameraRig(
                    [
                        CameraView(
                            name, "synthetic-v1", cached[name].calibration, cached[name]
                        )
                        for name in names
                    ]
                )
                outcomes[names] = rig.observe(
                    [frames[name] for name in names], now=frames["A"].capture_time
                )

            # Truth is inspected only after all RGB inference/fusion is complete.
            truth = np.array(group["A"][1]["records"][index]["position_label"][:2])
            head_pixels = {}
            for name in "ABC":
                record = group[name][1]["records"][index]
                if not np.array_equal(record["position_label"][:2], truth):
                    raise ValueError("Camera labels do not describe the same reset")
                head_pixels[name] = record["head_visible_pixels"]
                measurement = cached[name].measurement
                view_rows.append(
                    {
                        "scene_group": room,
                        "record_index": index,
                        "camera_id": name,
                        "capture_time": frames[name].capture_time,
                        "status": measurement.status,
                        "head_visible_pixels": head_pixels[name],
                        "xy": measurement.xy.tolist()
                        if measurement.xy is not None
                        else None,
                        "localization_error_m": float(
                            np.linalg.norm(measurement.xy - truth)
                        )
                        if measurement.xy is not None
                        else None,
                    }
                )
            for names, result in outcomes.items():
                rows.append(
                    {
                        "scene_group": room,
                        "record_index": index,
                        "rig": names,
                        "timestamp": result.timestamp,
                        "status": result.status,
                        "source_ids": list(result.source_ids),
                        "xy": result.xy.tolist() if result.xy is not None else None,
                        "covariance": result.covariance.tolist()
                        if result.covariance is not None
                        else None,
                        "truth_xy_scoring_only": truth.tolist(),
                        "head_visible_pixels_scoring_only": {
                            name: head_pixels[name] for name in names
                        },
                        "localization_error_m": float(np.linalg.norm(result.xy - truth))
                        if result.xy is not None
                        else None,
                        "views": {
                            name: {"status": item.status, "reason": item.reason}
                            for name, item in result.views.items()
                        },
                        "handover": result.handover,
                    }
                )
        print(json.dumps({"room": room, "paired_positions": len(indices)}), flush=True)

    aggregate = {}
    for names in ("A", "AB", "ABC"):
        selected = [row for row in rows if row["rig"] == names]
        errors = [
            r["localization_error_m"]
            for r in selected
            if r["localization_error_m"] is not None
        ]
        aggregate[names] = {
            "positions": len(selected),
            "available": len(errors),
            "availability": len(errors) / len(selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "view_status_counts": {
                name: dict(Counter(row["views"][name]["status"] for row in selected))
                for name in names
            },
            "source_id_counts": dict(
                Counter("".join(row["source_ids"]) for row in selected)
            ),
            "any_head_physically_visible_positions": sum(
                any(v > 0 for v in row["head_visible_pixels_scoring_only"].values())
                for row in selected
            ),
            "error_m_given_available": error_summary(errors),
        }
    by_view = {}
    for name in "ABC":
        selected = [row for row in view_rows if row["camera_id"] == name]
        by_view[name] = {
            "positions": len(selected),
            "physically_visible_head_positions": sum(
                row["head_visible_pixels"] > 0 for row in selected
            ),
            "fully_occluded_head_positions": sum(
                row["head_visible_pixels"] == 0 for row in selected
            ),
            "head_visible_pixel_counts": [
                row["head_visible_pixels"] for row in selected
            ],
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "error_m_given_available": error_summary(
                [
                    row["localization_error_m"]
                    for row in selected
                    if row["localization_error_m"] is not None
                ]
            ),
        }
    if file_hash(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation")
    report = {
        "status": "complete",
        "test": "paired heldout static RGB localization, not tracking or navigation",
        "split": "development",
        "rooms": sorted(groups),
        "fitting_or_threshold_selection": False,
        "fresh_rig_per_static_reset": True,
        "ground_truth_use": "posthoc scoring only; observer sees RGB and fixed camera calibration",
        "refinement": "guided-radius4-epsilon1e-4",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "calibration_sha256": file_hash(calibration_path),
        "dataset_manifest_sha256": file_hash(manifest_path),
        "consumed_files_sha256": consumed,
        "consumed_file_manifest_sha256": hashlib.sha256(
            json.dumps(consumed, sort_keys=True).encode()
        ).hexdigest(),
        "script_sha256": file_hash(Path(__file__)),
        "rig_source_sha256": file_hash(
            Path(__file__).resolve().parent.parent / "src/bb8_rl/camera_rig.py"
        ),
        "rig_configuration": {
            "max_age": 0.1,
            "max_skew": 0.05,
            "velocity_bound": 0.35,
            "identity_gate": 16.0,
            "max_disagreement": 0.1,
            "fusion": "equal-weight covariance intersection",
        },
        "room_extent": args.extent,
        "aggregate": aggregate,
        "per_view": by_view,
        "frames": rows,
        "per_view_frames": view_rows,
        "limitations": [
            "Static resets do not establish temporal handover, blackout safety or goal-arrival reliability.",
            "Error percentiles are conditional on an available fused estimate.",
            "Synthetic exact calibration; observer covariance is heuristic, not a calibrated safety bound.",
            "Floor masks are inferred but are not fused or used for planning in this audit.",
        ],
    }
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extent", type=float, default=2.0)
    evaluate(parser.parse_args())
