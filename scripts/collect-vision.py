"""Render disjoint scene sequences; all privileged fields are training labels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.baseline import RouteController
from bb8_rl.camera import Calibration
from bb8_rl.diagnostics import write_report
from bb8_rl.env import NavigationEnv
from bb8_rl.room import generate_room
from bb8_rl.training import file_hash
from bb8_rl.vision import candidates, crop

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sequences", type=int, default=40)
parser.add_argument("--frames", type=int, default=128)
parser.add_argument("--frame-stride", type=int, default=1)
args = parser.parse_args()
if min(args.sequences, args.frames, args.frame_stride) < 1:
    raise ValueError("Sequence count, frame count and frame stride must be positive")
if args.output.exists():
    raise ValueError("Use a fresh dataset directory")
args.output.mkdir(parents=True)
manifest = {
    "status": "running",
    "label_source": "Genesis segmentation and pose; training/scoring only",
    "sequences": [],
    "frames_per_sequence": args.frames,
    "frame_stride": args.frame_stride,
    "traversable_label": "visible ground or visible BB8 body/head; other occluded pixels stay blocked",
}
write_report(args.output / "manifest.json", manifest)
try:
    for sequence in range(args.sequences):
        split = (
            "train"
            if sequence < 24
            else "validation"
            if sequence < 32
            else "development"
        )
        seed = 310000 + sequence
        rng = np.random.default_rng(seed)
        scene = args.output / "scenes" / f"{sequence:03}"
        generate_room(
            Path("configs/navigation/bb8-state.yaml"),
            scene,
            seed=seed,
            side=4,
            obstacles=int(rng.integers(3, 9)),
        )
        project_path = scene / "room.genesis.json"
        project = json.loads(project_path.read_text())
        camera = project["sensors"][0]
        camera["position"] = (
            np.array([2.8, -3.4, 5.2])
            + rng.uniform([-0.5, -0.5, -0.6], [0.5, 0.5, 0.7])
        ).tolist()
        camera["lookat"] = [*rng.uniform(-0.15, 0.15, 2), 0]
        camera["fov"] = float(rng.uniform(54, 62))
        project["environment"]["ambient_light"] = [float(rng.uniform(0.25, 0.45))] * 3
        project["environment"]["light_intensity"] = float(rng.uniform(3, 6))
        for obj in project["objects"]:
            if obj["name"].startswith("room_obstacle") and sequence % 2:
                obj["material"]["color"] = [*rng.uniform(0.2, 0.7, 3), 1]
        write_report(project_path, project)
        frames, floors, traversable, rois, patches, heads, records = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with NavigationEnv(scene / "task.yaml", render_mode="rgb_array") as env:
            raw, _ = env.reset(seed=seed, options={"layout_seed": sequence})
            env.world.camera_period = 1e9
            env.world._next_frame = 1e9
            camera = env.world.camera
            calibration = Calibration(
                np.array(camera.intrinsics),
                np.array(camera.extrinsics),
                tuple(camera.res),
                2,
            )
            controller = RouteController(env.task, env.config.drive.parameters())
            controller.reset(env.grid, raw)
            label_map = env.world.scene.visualizer.segmentation_idx_dict
            ground = env.world.scene.get_entity(name="ground").idx
            ground_labels = [
                label
                for label, key in label_map.items()
                if isinstance(key, tuple) and key[0] == ground
            ]
            head_labels = [
                label
                for label, key in label_map.items()
                if isinstance(key, tuple) and key[0] == env.world.head.idx
            ]
            self_labels = [
                label
                for label, key in label_map.items()
                if isinstance(key, tuple)
                and key[0] in (env.world.head.idx, env.world.body.idx)
            ]
            if not ground_labels or not head_labels:
                raise ValueError("Missing explicit renderer label mapping")
            roi = np.zeros((240, 320), np.uint8)
            corners = calibration.to_pixel([[-2, -2], [2, -2], [2, 2], [-2, 2]]) / 4
            cv2.fillPoly(roi, [np.rint(corners).astype(np.int32)], 1)
            for index in range(args.frames):
                for _ in range(args.frame_stride if index else 1):
                    raw, _, terminal, truncated, _ = env.step(controller.action(raw))
                    if terminal or truncated:
                        raise RuntimeError(
                            "Collection ended before its bounded sequence"
                        )
                rgb, _, labels, _ = camera.render(rgb=True, segmentation=True)
                rgb = np.ascontiguousarray(rgb)
                floor = np.isin(labels, ground_labels).astype(np.float32)
                head = np.isin(labels, head_labels).astype(np.uint8)
                self_mask = np.isin(labels, self_labels)
                traversable.append(
                    (
                        cv2.resize(
                            (floor.astype(bool) | self_mask).astype(np.float32),
                            (320, 240),
                            interpolation=cv2.INTER_AREA,
                        )
                        > 0.99
                    ).astype(np.uint8)
                )
                frames.append(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA))
                floors.append(
                    (
                        cv2.resize(floor, (320, 240), interpolation=cv2.INTER_AREA)
                        > 0.99
                    ).astype(np.uint8)
                )
                rois.append(roi)
                # Crop proposals come from RGB, never from the ground-truth pose.
                proposals = candidates(rgb)[:6]
                proposals += [rng.uniform([40, 40], [1240, 920]) for _ in range(2)]
                for center in proposals:
                    patch, origin = crop(rgb, center)
                    target, _ = crop(head, center)
                    patches.append(patch)
                    heads.append(target)
                records.append(
                    {
                        "time": env.world.time,
                        "position_label": raw["achieved_goal"].tolist(),
                        "head_visible_pixels": int(head.sum()),
                        "proposal_count": len(proposals) - 2,
                    }
                )
                if index == 0:
                    cv2.imwrite(
                        str(scene / "sample-rgb.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(scene / "sample-floor.png"), (floor * 255).astype(np.uint8)
                    )
                    cv2.imwrite(
                        str(scene / "sample-self.png"), self_mask.astype(np.uint8) * 255
                    )
                if terminal or truncated:
                    raise RuntimeError(
                        "Training collection episode ended before its bounded sequence"
                    )
        output = args.output / f"sequence-{sequence:03}.npz"
        np.savez_compressed(
            output,
            rgb=np.stack(frames),
            floor=np.stack(floors),
            traversable=np.stack(traversable),
            roi=np.stack(rois),
            patches=np.stack(patches),
            head=np.stack(heads),
        )
        write_report(
            scene / "labels.json",
            {
                "records": records,
                "calibration": {
                    "intrinsics": calibration.intrinsics.tolist(),
                    "world_to_camera": calibration.world_to_camera.tolist(),
                },
            },
        )
        manifest["sequences"].append(
            {
                "id": sequence,
                "split": split,
                "seed": seed,
                "path": output.name,
                "sha256": file_hash(output),
                "frames": len(frames),
                "patches": len(patches),
            }
        )
        write_report(args.output / "manifest.json", manifest)
        print(json.dumps(manifest["sequences"][-1]), flush=True)
    manifest["status"] = "complete"
except BaseException as error:
    manifest.update(status="failed", error=repr(error))
    raise
finally:
    write_report(args.output / "manifest.json", manifest)
