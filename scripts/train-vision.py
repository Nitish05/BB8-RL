"""Train two-scale perception with scene-disjoint validation; no simulator import."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bb8_rl.training import file_hash
from bb8_rl.vision import VisionStudent
from bb8_rl.vision_data import validate_scene_splits

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
parser.add_argument("--floor-target", choices=["floor", "traversable"], default="floor")
parser.add_argument("--head-checkpoint", type=Path)
parser.add_argument("--freeze-head", action="store_true")
args = parser.parse_args()
if args.steps < 1:
    raise ValueError("Training steps must be positive")
if args.output.exists():
    raise ValueError("Use fresh training output")
if args.freeze_head and not args.head_checkpoint:
    raise ValueError("Freezing the head requires trained head weights")
manifest = json.loads((args.dataset / "manifest.json").read_text())
if manifest["status"] != "complete":
    raise ValueError("Require a complete immutable dataset")
validate_scene_splits(manifest)
args.output.mkdir(parents=True)
torch.manual_seed(17)
torch.set_num_threads(2)
rng = np.random.default_rng(17)
device = args.device
if device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("Requested MPS unavailable; no silent fallback")


def load(split):
    parts = {k: [] for k in ("rgb", "floor", "roi", "patches", "head", "self")}
    for sequence in manifest["sequences"]:
        if sequence["split"] != split:
            continue
        path = args.dataset / sequence["path"]
        if file_hash(path) != sequence["sha256"]:
            raise ValueError("Dataset checksum mismatch")
        with np.load(path) as data:
            patch_indices = np.random.default_rng(17000 + sequence["id"]).choice(
                len(data["patches"]),
                size=max(1, len(data["patches"]) // 8),
                replace=False,
            )
            for key, values in parts.items():
                if key == "self":
                    source = (
                        (data["traversable"] > data["floor"])
                        if args.floor_target == "traversable"
                        else np.zeros_like(data["floor"])
                    )
                else:
                    source = data[args.floor_target] if key == "floor" else data[key]
                stride = 1 if split == "train" or "scene_group" in sequence else 16
                # Fixed-stride crop sampling can repeatedly select the same
                # proposal slot and exclude the head. Sample paired crops once.
                values.append(
                    source[patch_indices]
                    if split != "train" and key in ("patches", "head")
                    else source[::stride]
                )
    return {k: np.concatenate(v) for k, v in parts.items()}


train, validation = load("train"), load("validation")
positive = np.flatnonzero(train["head"].sum((1, 2)) >= 3)
negative = np.flatnonzero(train["head"].sum((1, 2)) == 0)
if not len(positive) or not len(negative):
    raise ValueError("Need positive and negative RGB head proposals")
model = VisionStudent().to(device)
if args.head_checkpoint:
    pretrained = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
    model.head.load_state_dict(
        {
            k.removeprefix("head."): v
            for k, v in pretrained["model"].items()
            if k.startswith("head.")
        }
    )
if args.freeze_head:
    model.head.requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
weights_floor = torch.tensor([3.0, 1.0], device=device)
weights_head = torch.tensor([1.0, 30.0], device=device)


def tensor(images):
    return (
        torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device).float() / 255
    )


def augment(images):
    brightness = torch.rand((len(images), 1, 1, 1), device=device) * 0.4 + 0.8
    channels = torch.rand((len(images), 3, 1, 1), device=device) * 0.2 + 0.9
    return (images * brightness * channels + torch.randn_like(images) * 0.005).clamp(
        0, 1
    )


def evaluate():
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(validation["rgb"]), 4):
            probabilities.append(
                model.floor(tensor(validation["rgb"][start : start + 4]))
                .softmax(1)[:, 1]
                .cpu()
                .numpy()
            )
    probability = np.concatenate(probabilities)
    y, roi = validation["floor"].astype(bool), validation["roi"].astype(bool)
    scores = []
    for threshold in (0.5, 0.7, 0.85, 0.95, 0.98, 0.995):
        p = probability >= threshold
        false_free = float((p & ~y & roi).sum() / max(1, (~y & roi).sum()))
        recall = float((p & y & roi).sum() / max(1, (y & roi).sum()))
        own = validation["self"].astype(bool)
        self_miss = float((~p & own).sum() / max(1, own.sum()))
        scores.append(
            {
                "threshold": threshold,
                "false_free_fraction_of_blocked_pixels": false_free,
                "floor_recall": recall,
                "self_pixel_miss_fraction": self_miss,
                "selection_loss": 20 * false_free + 1 - recall + 2 * self_miss,
            }
        )
    selected = min(scores, key=lambda s: s["selection_loss"])
    with torch.inference_mode():
        heads = (
            np.concatenate(
                [
                    model.head(tensor(validation["patches"][start : start + 32]))
                    .softmax(1)[:, 1]
                    .cpu()
                    .numpy()
                    for start in range(0, len(validation["patches"]), 32)
                ]
            )
            >= 0.5
        )
    count = heads.sum((1, 2))
    detected = (count >= 3) & (count <= 120)
    visible = validation["head"].sum((1, 2)) >= 3
    miss = float((~detected & visible).sum() / max(1, visible.sum()))
    false_detection = float((detected & ~visible).sum() / max(1, (~visible).sum()))
    selected = dict(selected)
    selected.update(
        head_patch_miss_fraction=miss,
        head_patch_false_detection_fraction=false_detection,
    )
    selected["selection_loss"] += miss + false_detection
    model.train()
    return selected, scores


started = time.perf_counter()
best = float("inf")
run = {
    "status": "running",
    "seed": 17,
    "device": device,
    "steps_requested": args.steps,
    "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
    "training_sequences": [
        s["id"] for s in manifest["sequences"] if s["split"] == "train"
    ],
    "validation_sequences": [
        s["id"] for s in manifest["sequences"] if s["split"] == "validation"
    ],
    "development_sequences_used_in_training": False,
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "trainable_parameter_count": sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ),
    "label_source": "synthetic renderer",
    "sam_or_tap_weights_used": False,
    "floor_target": args.floor_target,
    "head_initialization_sha256": file_hash(args.head_checkpoint)
    if args.head_checkpoint
    else None,
    "head_frozen": args.freeze_head,
    "self_pixel_loss_multiplier": 100 if args.floor_target == "traversable" else 1,
    "selection_rule": "20*floor_false_free + 1-floor_recall + 2*self_pixel_miss + head_patch_miss + head_patch_false_detection",
}
(args.output / "manifest.json").write_text(json.dumps(run, indent=2))
try:
    for step in range(1, args.steps + 1):
        indices = rng.integers(len(train["rgb"]), size=4)
        images = augment(tensor(train["rgb"][indices]))
        targets = torch.as_tensor(
            train["floor"][indices], dtype=torch.long, device=device
        )
        roi = torch.as_tensor(train["roi"][indices], dtype=torch.float32, device=device)
        self_pixels = torch.as_tensor(
            train["self"][indices], dtype=torch.float32, device=device
        )
        floor_loss = (
            F.cross_entropy(
                model.floor(images), targets, weight=weights_floor, reduction="none"
            )
            * (0.1 + 0.9 * roi)
            * (1 + 99 * self_pixels)
        ).mean()
        indices = np.r_[rng.choice(positive, 8), rng.choice(negative, 8)]
        patches = tensor(train["patches"][indices])
        target = torch.as_tensor(
            train["head"][indices], dtype=torch.long, device=device
        )
        shift = tuple(int(x) for x in rng.integers(-12, 13, 2))
        patches, target = (
            torch.roll(patches, shift, (-2, -1)),
            torch.roll(target, shift, (-2, -1)),
        )
        logits = model.head(augment(patches))
        probability = logits.softmax(1)[:, 1]
        head_loss = F.cross_entropy(logits, target, weight=weights_head)
        head_loss += (
            1
            - (
                (2 * (probability * target).sum((1, 2)) + 1)
                / (probability.sum((1, 2)) + target.sum((1, 2)) + 1)
            ).mean()
        )
        loss = floor_loss + head_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite vision training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            selected, scores = evaluate()
            row = {
                "step": step,
                "floor_loss": float(floor_loss.detach()),
                "head_loss": float(head_loss.detach()),
                "validation": selected,
                "thresholds": scores,
            }
            with (args.output / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            if selected["selection_loss"] < best:
                best = selected["selection_loss"]
                torch.save(
                    {
                        "format": "bb8-vision-v1",
                        "model": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "floor_threshold": selected["threshold"],
                        "head_threshold": 0.5,
                        "floor_target": args.floor_target,
                        "step": step,
                    },
                    args.output / "model.pt",
                )
                run["selected_step"], run["selected_validation"] = step, selected
            print(
                json.dumps({k: v for k, v in row.items() if k != "thresholds"}),
                flush=True,
            )
    run.update(
        status="complete",
        steps=args.steps,
        elapsed_seconds=time.perf_counter() - started,
        checkpoint_sha256=file_hash(args.output / "model.pt"),
    )
except BaseException as error:
    run.update(status="failed", error=repr(error))
    raise
finally:
    (args.output / "manifest.json").write_text(json.dumps(run, indent=2) + "\n")
