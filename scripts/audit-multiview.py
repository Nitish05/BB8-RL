"""Calibrate on whole validation rooms, then audit locked new rooms at native size."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration
from bb8_rl.diagnostics import write_report
from bb8_rl.training import file_hash
from bb8_rl.vision import LearnedObserver, guided_floor_probability
from bb8_rl.vision_data import validate_scene_splits

THRESHOLDS = (0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.995, 0.999)


def samples(dataset, manifest, split):
    for sequence in manifest["sequences"]:
        if sequence["split"] != split:
            continue
        labels_path = dataset / "scenes" / f"{sequence['id']:03}" / "labels.json"
        if file_hash(labels_path) != sequence["labels_sha256"]:
            raise ValueError("Native labels checksum mismatch")
        labels = json.loads(labels_path.read_text())
        if len(labels["records"]) != sequence["frames"]:
            raise ValueError("Native record count differs from manifest")
        c = labels["calibration"]
        calibration = Calibration(
            np.array(c["intrinsics"]), np.array(c["world_to_camera"]), (1280, 960), 2
        )
        roi = np.zeros((960, 1280), np.uint8)
        cv2.fillPoly(
            roi,
            [
                np.rint(
                    calibration.to_pixel([[-2, -2], [2, -2], [2, 2], [-2, 2]])
                ).astype(np.int32)
            ],
            1,
        )
        for entry in sequence["native_frames"]:
            images = {}
            for name in ("rgb", "floor", "self"):
                path = dataset / entry[name]
                if file_hash(path) != entry["sha256"][name]:
                    raise ValueError("Native image checksum mismatch")
                images[name] = cv2.imread(
                    str(path),
                    cv2.IMREAD_COLOR if name == "rgb" else cv2.IMREAD_GRAYSCALE,
                )
            rgb = cv2.cvtColor(images["rgb"], cv2.COLOR_BGR2RGB)
            floor, own = images["floor"] > 127, images["self"] > 127
            yield (
                sequence,
                entry,
                calibration,
                rgb,
                floor | own,
                own,
                roi.astype(bool),
                labels["records"][entry["record_index"]],
            )


def counts(predicted, truth, own, roi):
    return np.array(
        [
            (predicted & truth & roi).sum(),
            (predicted & ~truth & roi).sum(),
            (truth & roi).sum(),
            (~truth & roi).sum(),
            (~predicted & own).sum(),
            own.sum(),
        ],
        dtype=np.int64,
    )


def rates(count):
    tp, fp, positive, negative, own_miss, own_total = (int(v) for v in count)
    recall, false_free, self_miss = (
        tp / max(1, positive),
        fp / max(1, negative),
        own_miss / max(1, own_total),
    )
    return {
        "counts": [tp, fp, positive, negative, own_miss, own_total],
        "floor_recall": recall,
        "false_free_fraction_of_blocked_pixels": false_free,
        "self_pixel_miss_fraction": self_miss,
        "selection_loss": 20 * false_free + 1 - recall + 2 * self_miss,
    }


def probability(observer, rgb):
    small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
    p = cv2.resize(
        observer.probabilities([small])[0], (1280, 960), interpolation=cv2.INTER_LINEAR
    )
    return guided_floor_probability(rgb, p)


def audit(args):
    if args.output.exists():
        raise ValueError("Use a fresh audit output")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    validate_scene_splits(manifest)
    for checkpoint in (args.checkpoint, args.baseline):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("floor_target") != "traversable":
            raise ValueError(
                "This audit requires visible-floor/self target checkpoints"
            )
    if manifest.get("format") != "bb8-multiview-supervision-v1":
        raise ValueError("Native multi-view dataset required")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    totals = {t: {name: np.zeros(6, np.int64) for name in "ABC"} for t in THRESHOLDS}
    observer = None
    validation_ids = []
    for seq, entry, c, rgb, truth, own, roi, record in samples(
        args.dataset, manifest, "validation"
    ):
        if observer is None:
            observer = LearnedObserver(c, args.checkpoint, floor_refinement="guided")
        p = probability(observer, rgb)
        for threshold in THRESHOLDS:
            totals[threshold][seq["camera_id"]] += counts(
                p >= threshold, truth, own, roi
            )
        validation_ids.append([seq["id"], entry["record_index"]])
    scores = [
        {
            "threshold": t,
            "by_view": {name: rates(count) for name, count in views.items()},
            "selection_loss": max(
                rates(count)["selection_loss"] for count in views.values()
            ),
        }
        for t, views in totals.items()
    ]
    selected = min(scores, key=lambda row: row["selection_loss"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state["floor_threshold"] = selected["threshold"]
    state["calibration_refinement"] = "guided-radius4-epsilon1e-4"
    torch.save(state, args.output / "model.pt")
    frozen_hash = file_hash(args.output / "model.pt")
    write_report(
        args.output / "calibration.json",
        {
            "selection_rule": "minimize worst-camera validation selection loss; no driving approval",
            "validation_samples": validation_ids,
            "scores": scores,
            "selected": selected,
            "checkpoint_sha256": frozen_hash,
        },
    )
    rows = []
    observers = None
    for seq, entry, c, rgb, truth, own, roi, record in samples(
        args.dataset, manifest, "development"
    ):
        if observers is None:
            observers = {
                "new": LearnedObserver(
                    c, args.output / "model.pt", floor_refinement="guided"
                ),
                "baseline": LearnedObserver(
                    c, args.baseline, floor_refinement="guided"
                ),
            }
        for label, current in observers.items():
            current.calibration = c
            measurement = current.observe(rgb, record["time"])
            row = {
                "model": label,
                "scene_group": seq["scene_group"],
                "camera_id": seq["camera_id"],
                "record_index": entry["record_index"],
                **rates(counts(measurement.visible_floor, truth, own, roi)),
                "deployed_free_mask": rates(
                    counts(
                        measurement.visible_floor | measurement.robot_pixels,
                        truth,
                        own,
                        roi,
                    )
                ),
                "status": measurement.status,
                "head_visible_pixels": record["head_visible_pixels"],
                "localization_error_m": float(
                    np.linalg.norm(
                        measurement.xy - np.array(record["position_label"][:2])
                    )
                )
                if measurement.xy is not None
                else None,
            }
            rows.append(row)
        if entry["record_index"] == 0:
            print(
                json.dumps(
                    {
                        "room": seq["scene_group"],
                        "view": seq["camera_id"],
                        "new": rows[-2]["false_free_fraction_of_blocked_pixels"],
                        "baseline": rows[-1]["false_free_fraction_of_blocked_pixels"],
                    }
                ),
                flush=True,
            )
    aggregates = {}
    for label in ("new", "baseline"):
        aggregates[label] = {}
        for name in "ABC":
            subset = [r for r in rows if r["model"] == label and r["camera_id"] == name]
            errors = [
                r["localization_error_m"]
                for r in subset
                if r["localization_error_m"] is not None
            ]
            aggregates[label][name] = {
                **rates(np.sum([r["counts"] for r in subset], axis=0)),
                "deployed_free_mask": rates(
                    np.sum([r["deployed_free_mask"]["counts"] for r in subset], axis=0)
                ),
                "frames": len(subset),
                "localized_frames": len(errors),
                "head_present_definition": "at least three native renderer head pixels; scorer only",
                "visible_head_frames": sum(
                    r["head_visible_pixels"] >= 3 for r in subset
                ),
                "visible_head_misses": sum(
                    r["head_visible_pixels"] >= 3 and r["localization_error_m"] is None
                    for r in subset
                ),
                "hidden_head_false_accepts": sum(
                    r["head_visible_pixels"] < 3
                    and r["localization_error_m"] is not None
                    for r in subset
                ),
                "ambiguous_frames": sum(r["status"] == "ambiguous" for r in subset),
                "localization_error_p95_m": float(np.quantile(errors, 0.95))
                if errors
                else None,
                "worst_frame_false_free": max(
                    r["false_free_fraction_of_blocked_pixels"] for r in subset
                ),
                "by_room": {
                    str(room): rates(
                        np.sum(
                            [r["counts"] for r in subset if r["scene_group"] == room],
                            axis=0,
                        )
                    )
                    for room in sorted({r["scene_group"] for r in subset})
                },
            }
    assert file_hash(args.output / "model.pt") == frozen_hash
    write_report(
        args.output / "report.json",
        {
            "status": "complete",
            "scope": "locked new-room perception only; no route or driving acceptance",
            "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
            "input_checkpoint_sha256": file_hash(args.checkpoint),
            "checkpoint_sha256": frozen_hash,
            "baseline_sha256": file_hash(args.baseline),
            "refinement": "guided-radius4-epsilon1e-4",
            "development_used_for_threshold_selection": False,
            "aggregates": aggregates,
            "frames": rows,
        },
    )
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path("work/m72/calibrated-2/model.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
