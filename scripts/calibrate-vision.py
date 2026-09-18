"""Select deployed-resolution floor threshold using validation scenes only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bb8_rl.camera import Calibration
from bb8_rl.diagnostics import write_report
from bb8_rl.training import file_hash
from bb8_rl.vision import LearnedObserver

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Use a fresh calibration output")
args.output.mkdir(parents=True)
torch.set_num_threads(2)
manifest = json.loads((args.dataset / "manifest.json").read_text())
if manifest["status"] != "complete":
    raise ValueError("Require a complete dataset")
native_inputs = {}
counts = {
    threshold: np.zeros(6) for threshold in (0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.995)
}
sequences = []
for sequence in manifest["sequences"]:
    if sequence["split"] != "validation":
        continue
    sequences.append(sequence["id"])
    scene = args.dataset / "scenes" / f"{sequence['id']:03}"
    for name in (
        "labels.json",
        "sample-rgb.png",
        "sample-floor.png",
        "sample-self.png",
    ):
        path = scene / name
        if path.exists():
            native_inputs[str(path.relative_to(args.dataset))] = file_hash(path)
    c = json.loads((scene / "labels.json").read_text())["calibration"]
    calibration = Calibration(
        np.array(c["intrinsics"]), np.array(c["world_to_camera"]), (1280, 960), 2
    )
    observer = LearnedObserver(calibration, args.checkpoint)
    rgb = cv2.cvtColor(cv2.imread(str(scene / "sample-rgb.png")), cv2.COLOR_BGR2RGB)
    truth = cv2.imread(str(scene / "sample-floor.png"), 0) > 127
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    own = np.zeros(truth.shape, bool)
    if state.get("floor_target") == "traversable":
        own = cv2.imread(str(scene / "sample-self.png"), 0) > 127
        truth |= own
    probability = cv2.resize(
        observer.probabilities(
            [cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)]
        )[0],
        (1280, 960),
        interpolation=cv2.INTER_LINEAR,
    )
    roi = np.zeros(truth.shape, np.uint8)
    cv2.fillPoly(
        roi,
        [
            np.rint(calibration.to_pixel([[-2, -2], [2, -2], [2, 2], [-2, 2]])).astype(
                np.int32
            )
        ],
        1,
    )
    roi = roi.astype(bool)
    for threshold, count in counts.items():
        predicted = probability >= threshold
        count += [
            (predicted & truth & roi).sum(),
            (predicted & ~truth & roi).sum(),
            (truth & roi).sum(),
            (~truth & roi).sum(),
            (~predicted & own).sum(),
            own.sum(),
        ]
if not sequences:
    raise ValueError("Need native validation scenes")
scores = []
for threshold, (tp, fp, positive, negative, own_miss, own_count) in counts.items():
    recall, false_free = tp / max(1, positive), fp / max(1, negative)
    self_miss = own_miss / max(1, own_count)
    scores.append(
        {
            "threshold": threshold,
            "floor_recall": recall,
            "false_free_fraction_of_blocked_pixels": false_free,
            "self_pixel_miss_fraction": self_miss,
            "selection_loss": 20 * false_free + 1 - recall + 2 * self_miss,
        }
    )
selected = min(scores, key=lambda row: row["selection_loss"])
state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
previous_threshold = state["floor_threshold"]
state["floor_threshold"] = selected["threshold"]
torch.save(state, args.output / "model.pt")
write_report(
    args.output / "manifest.json",
    {
        "status": "complete",
        "scope": "native-resolution validation threshold refinement; no weight updates",
        "validation_sequences": sequences,
        "native_input_sha256": native_inputs,
        "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
        "source_checkpoint_sha256": file_hash(args.checkpoint),
        "previous_threshold": previous_threshold,
        "selected": selected,
        "scores": scores,
        "checkpoint_sha256": file_hash(args.output / "model.pt"),
        "development_used_for_threshold_selection": False,
        "development_failure_motivated_native_validation": True,
    },
)
print(json.dumps(selected))
