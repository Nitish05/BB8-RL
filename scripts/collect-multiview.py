"""Collect room-disjoint RGB supervision with one sequence per room and view."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.camera_rig import calibration_from_live_camera
from bb8_rl.diagnostics import write_report
from bb8_rl.env import NavigationEnv
from bb8_rl.room import generate_room
from bb8_rl.training import file_hash
from bb8_rl.vision import candidates, crop


def collect(args):
    if (
        min(
            args.train_rooms, args.validation_rooms, args.development_rooms, args.frames
        )
        < 1
    ):
        raise ValueError("Each room split and frame count must be positive")
    if args.output.exists():
        raise ValueError("Use a fresh dataset output")
    args.output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "format": "bb8-multiview-supervision-v1",
        "label_source": "Genesis segmentation and pose; training/scoring only",
        "split_unit": "whole room including every camera and reset position",
        "seed_base": args.seed_base,
        "camera_count": 3,
        "frames_per_sequence": args.frames,
        "capture": "Three fixed native cameras rendered at the same paused physics time",
        "traversable_label": "visible ground or visible self; hidden pixels remain blocked",
        "sequences": [],
    }
    write_report(args.output / "manifest.json", manifest)
    total = args.train_rooms + args.validation_rooms + args.development_rooms
    try:
        for room in range(total):
            split = (
                "train"
                if room < args.train_rooms
                else "validation"
                if room < args.train_rooms + args.validation_rooms
                else "development"
            )
            seed = args.seed_base + room
            rng = np.random.default_rng(seed)
            scene = args.output / "rooms" / f"{room:03}"
            generate_room(
                Path("configs/navigation/bb8-state.yaml"),
                scene,
                seed=seed,
                side=4,
                obstacles=int(rng.integers(3, 9)),
            )
            project_path = scene / "room.genesis.json"
            project = json.loads(project_path.read_text())
            positions = np.array(
                [[2.8, -3.4, 5.2], [-2.8, 3.4, 5.2], [-3.4, -2.8, 5.2]]
            )
            positions += rng.uniform([-0.4, -0.4, -0.5], [0.4, 0.4, 0.6], (3, 3))
            project["sensors"][0]["lookat"] = [*rng.uniform(-0.15, 0.15, 2), 0]
            project["sensors"][0]["fov"] = float(rng.uniform(54, 62))
            project["environment"]["ambient_light"] = [
                float(rng.uniform(0.25, 0.45))
            ] * 3
            project["environment"]["light_intensity"] = float(rng.uniform(3, 6))
            for obj in project["objects"]:
                if obj["name"].startswith("room_obstacle") and room % 2:
                    obj["material"]["color"] = [*rng.uniform(0.2, 0.7, 3), 1]
            write_report(project_path, project)
            banks = {
                name: {
                    key: []
                    for key in (
                        "rgb",
                        "floor",
                        "traversable",
                        "roi",
                        "patches",
                        "head",
                        "records",
                    )
                }
                for name in "ABC"
            }
            native_frames = {name: [] for name in "ABC"}
            with NavigationEnv(
                scene / "task.yaml",
                render_mode="rgb_array",
                camera_positions=positions.tolist(),
            ) as env:
                raw, _ = env.reset(seed=seed, options={"layout_seed": room})
                env.world.camera_period = env.world._next_frame = 1e9
                label_map = env.world.scene.visualizer.segmentation_idx_dict
                ground = env.world.scene.get_entity(name="ground").idx
                ids = {
                    "floor": [
                        k
                        for k, v in label_map.items()
                        if isinstance(v, tuple) and v[0] == ground
                    ],
                    "head": [
                        k
                        for k, v in label_map.items()
                        if isinstance(v, tuple) and v[0] == env.world.head.idx
                    ],
                    "self": [
                        k
                        for k, v in label_map.items()
                        if isinstance(v, tuple)
                        and v[0] in (env.world.body.idx, env.world.head.idx)
                    ],
                }
                if any(not value for value in ids.values()):
                    raise ValueError("Missing explicit renderer label mapping")
                calibrations = {
                    name: calibration_from_live_camera(camera, 2)
                    for name, camera in env.world.cameras.items()
                }
                for index in range(args.frames):
                    if index:
                        raw, _ = env.reset(
                            seed=seed + 10000 * index, options={"layout_seed": room}
                        )
                    env.world._next_frame = 1e9
                    # Only episode reset sets the body position. One genuine
                    # physics step settles the parked robot; no driving policy.
                    raw, _, _, _, _ = env.step(np.zeros(2))
                    for vi, (name, camera) in enumerate(env.world.cameras.items()):
                        bank, calibration = banks[name], calibrations[name]
                        rgb, _, labels, _ = camera.render(
                            rgb=True, segmentation=True, force_render=True
                        )
                        rgb = np.ascontiguousarray(rgb)
                        floor = np.isin(labels, ids["floor"])
                        own = np.isin(labels, ids["self"])
                        head = np.isin(labels, ids["head"]).astype(np.uint8)
                        roi = np.zeros((240, 320), np.uint8)
                        corners = (
                            calibration.to_pixel([[-2, -2], [2, -2], [2, 2], [-2, 2]])
                            / 4
                        )
                        cv2.fillPoly(roi, [np.rint(corners).astype(np.int32)], 1)
                        bank["rgb"].append(
                            cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
                        )
                        for key, mask in (
                            ("floor", floor),
                            ("traversable", floor | own),
                        ):
                            bank[key].append(
                                (
                                    cv2.resize(
                                        mask.astype(np.float32),
                                        (320, 240),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                    > 0.99
                                ).astype(np.uint8)
                            )
                        bank["roi"].append(roi)
                        proposals = candidates(rgb)[:64]
                        proposals += [
                            rng.uniform([40, 40], [1240, 920]) for _ in range(2)
                        ]
                        for center in proposals:
                            bank["patches"].append(crop(rgb, center)[0])
                            bank["head"].append(crop(head, center)[0])
                        bank["records"].append(
                            {
                                "time": env.world.time,
                                "reset_index": index,
                                "position_label": raw["achieved_goal"].tolist(),
                                "head_visible_pixels": int(head.sum()),
                                "proposal_count": len(proposals) - 2,
                            }
                        )
                        sample = args.output / "scenes" / f"{room * 3 + vi:03}"
                        sample.mkdir(parents=True, exist_ok=True)
                        saved = {"record_index": index, "sha256": {}}
                        for suffix, pixels in (
                            ("rgb", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                            ("floor", floor.astype(np.uint8) * 255),
                            ("self", own.astype(np.uint8) * 255),
                        ):
                            path = sample / f"frame-{index:03}-{suffix}.png"
                            if not cv2.imwrite(str(path), pixels):
                                raise OSError(f"Cannot save {path}")
                            saved[suffix] = str(path.relative_to(args.output))
                            saved["sha256"][suffix] = file_hash(path)
                            if index == 0:
                                cv2.imwrite(
                                    str(sample / f"sample-{suffix}.png"), pixels
                                )
                        native_frames[name].append(saved)
                for vi, name in enumerate("ABC"):
                    bank, c = banks[name], calibrations[name]
                    identifier = room * 3 + vi
                    output = args.output / f"sequence-{identifier:03}.npz"
                    np.savez_compressed(
                        output,
                        **{k: np.stack(v) for k, v in bank.items() if k != "records"},
                    )
                    sample = args.output / "scenes" / f"{identifier:03}"
                    write_report(
                        sample / "labels.json",
                        {
                            "records": bank["records"],
                            "calibration": {
                                "intrinsics": c.intrinsics.tolist(),
                                "world_to_camera": c.world_to_camera.tolist(),
                            },
                            "camera_id": name,
                            "scene_group": room,
                        },
                    )
                    manifest["sequences"].append(
                        {
                            "id": identifier,
                            "scene_group": room,
                            "camera_id": name,
                            "split": split,
                            "seed": seed,
                            "path": output.name,
                            "sha256": file_hash(output),
                            "frames": len(bank["rgb"]),
                            "patches": len(bank["patches"]),
                            "native_frames": native_frames[name],
                            "labels_sha256": file_hash(sample / "labels.json"),
                        }
                    )
            write_report(args.output / "manifest.json", manifest)
            print(
                json.dumps(
                    {
                        "room": room,
                        "split": split,
                        "sequences": len(manifest["sequences"]),
                    }
                ),
                flush=True,
            )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_report(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rooms", type=int, default=24)
    parser.add_argument("--validation-rooms", type=int, default=6)
    parser.add_argument("--development-rooms", type=int, default=6)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=740000)
    collect(parser.parse_args())
