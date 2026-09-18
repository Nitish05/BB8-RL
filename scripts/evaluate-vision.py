"""Scene-wise offline development audit; renderer labels are scoring-only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration, SyntheticPaletteObserver
from bb8_rl.diagnostics import write_report
from bb8_rl.training import file_hash
from bb8_rl.vision import LearnedObserver

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Use a fresh audit output")
args.output.mkdir(parents=True)
torch.set_num_threads(2)
manifest = json.loads((args.dataset / "manifest.json").read_text())
if manifest["status"] != "complete":
    raise ValueError("Require a complete dataset")


def metrics(prediction, truth, roi):
    tp = int((prediction & truth & roi).sum())
    fp = int((prediction & ~truth & roi).sum())
    fn = int((~prediction & truth & roi).sum())
    return {
        "floor_iou": tp / max(1, tp + fp + fn),
        "floor_recall": tp / max(1, tp + fn),
        "false_free_fraction_of_blocked_pixels": fp / max(1, int((~truth & roi).sum())),
    }


rows = []
for entry in manifest["sequences"]:
    if entry["split"] not in ("validation", "development"):
        continue
    path = args.dataset / entry["path"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    scene = args.dataset / "scenes" / f"{entry['id']:03}"
    labels = json.loads((scene / "labels.json").read_text())
    c = labels["calibration"]
    calibration = Calibration(
        np.array(c["intrinsics"]), np.array(c["world_to_camera"]), (1280, 960), 2
    )
    observer = LearnedObserver(calibration, args.checkpoint)
    rgb = cv2.cvtColor(cv2.imread(str(scene / "sample-rgb.png")), cv2.COLOR_BGR2RGB)
    truth = cv2.imread(str(scene / "sample-floor.png"), 0) > 127
    roi = np.zeros(truth.shape, np.uint8)
    corners = calibration.to_pixel([[-2, -2], [2, -2], [2, 2], [-2, 2]])
    cv2.fillPoly(roi, [np.rint(corners).astype(np.int32)], 1)
    self_path = scene / "sample-self.png"
    if self_path.exists():
        # Compare visible floor independently of how either observer represents
        # the robot's own occupied pixels in its traversability map.
        roi[cv2.imread(str(self_path), 0) > 127] = 0
    learned = observer.observe(rgb, 0)
    palette = SyntheticPaletteObserver(calibration).observe(rgb, 0)
    row = {
        "sequence": entry["id"],
        "split": entry["split"],
        "sample_learned": metrics(learned.visible_floor, truth, roi.astype(bool)),
        "sample_palette": metrics(palette.visible_floor, truth, roi.astype(bool)),
        "sample_localization_status": learned.status,
        "self_pixels_excluded_from_floor_scoring": self_path.exists(),
        "sample_localization_error_m": float(
            np.linalg.norm(
                learned.xy - np.array(labels["records"][0]["position_label"][:2])
            )
        )
        if learned.xy is not None
        else None,
    }
    overlay = rgb.copy()
    overlay[learned.visible_floor] = (
        0.5 * overlay[learned.visible_floor] + np.array([0, 110, 30])
    ).clip(0, 255)
    cv2.imwrite(
        str(args.output / f"scene-{entry['id']:03}-floor.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    with np.load(path) as data:
        patches, targets = data["patches"], data["head"].astype(bool)
        probabilities = np.concatenate(
            [
                observer.probabilities(patches[i : i + 32], head=True)
                for i in range(0, len(patches), 32)
            ]
        )
        masks = probabilities >= observer.head_threshold
        counts = masks.sum((1, 2))
        detected, present = (counts >= 3) & (counts <= 120), targets.sum((1, 2)) >= 3
        row["head_patches"] = {
            "count": len(patches),
            "true_positive": int((detected & present).sum()),
            "false_positive": int((detected & ~present).sum()),
            "false_negative": int((~detected & present).sum()),
            "true_negative": int((~detected & ~present).sum()),
        }
    rows.append(row)
    print(json.dumps(row), flush=True)

write_report(
    args.output / "report.json",
    {
        "status": "complete",
        "checkpoint_sha256": file_hash(args.checkpoint),
        "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
        "scope": "Offline synthetic audit, whole-scene split; not closed-loop or hardware acceptance",
        "floor_sample_rule": "One native RGB frame per sequence, both observers on identical inputs",
        "head_sample_rule": "Every stored patch; correlated samples, not independent trials",
        "sequences": rows,
    },
)
