"""Provision four frozen held-out scenes from new RGB-only metric scans.

Prepare is data-only. Capture is explicitly native and must be serialized by the
operator. Registration uses local pinned weights, never a query camera transform.
Metric acquisition poses are known synthetic priors, not unknown-pose SLAM.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/interactive/navigation-heldout-v1.json"
PROTOCOL_SHA256 = "82ccfbe9e65c8a65b1ef454b69317087994a4b6257a56cf4a07af9f6ddd875fe"
QUERIES = (14, 17, 20, 23)
CAMERA_QUERIES = {"A": 20, "B": 23, "C": 14}
MAP_AUDIT_INPUTS = (
    "scene/room.genesis.json",
    "scan-plan.json",
    "scan/manifest.json",
    "registration/report.json",
    "registration/landmarks.npz",
    "registration/tracks.json",
    "memory/manifest.json",
    "memory/room-memory.json",
    "memory/free-grid.npz",
)
MEMORY_SETTINGS = {
    "extent": 2.0,
    "resolution": 0.02,
    "body_height_m": 0.12,
    "projection_margin_m": 0.05,
    "pixel_guard": 1,
    "minimum_views": 3,
    "minimum_baseline_m": 0.5,
    "minimum_angle_degrees": 15.0,
    "projection_shape": "convex_hull",
    "evidence_mode": "full_prism",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def module(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), ROOT / "scripts" / name
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Held-out protocol differs from pre-tuning freeze")
    return read(PROTOCOL)


def family_scene(original, spec, name):
    family = spec["families"][name]
    layout = spec["layouts"][family["layout"]]
    colors = spec["appearances"][family["appearance"]]
    result = copy.deepcopy(original)
    result["name"] = f"BB8 frozen heldout {name}"
    result["seed"] = layout["seed"]
    result["objects"] = [
        o for o in result["objects"] if not o["name"].startswith("room_obstacle_")
    ]
    for obj in result["objects"]:
        if obj["name"] in colors:
            obj["material"]["color"] = colors[obj["name"]]
    for obstacle in layout["obstacles"]:
        result["objects"].append(
            {
                **copy.deepcopy(obstacle),
                "visible": True,
                "collision_group": 65535,
                "collision_mask": 65535,
                "material": {
                    "friction": 1.0,
                    "density": 1000.0,
                    "color": colors["room_obstacle_*"],
                },
            }
        )
    camera = spec["camera_sets"][family["camera_set"]]["A"]
    for sensor in result["sensors"]:
        if sensor["name"] == "navigation_camera":
            sensor.update(
                position=camera["render_position"],
                lookat=camera["render_lookat"],
                resolution=camera["resolution"],
                fov=camera["vertical_fov_degrees"],
            )
    return result


def acquisition_up(position, lookat):
    """A pose is independent of preceding camera views, including its roll."""
    if (
        len(position) != 3
        or len(lookat) != 3
        or any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (*position, *lookat)
        )
    ):
        raise ValueError("Need finite three-dimensional camera position/lookat")
    direction = [b - a for a, b in zip(position, lookat, strict=True)]
    length = math.sqrt(sum(v * v for v in direction))
    if length == 0:
        raise ValueError("Camera position and lookat must differ")
    # Frozen overhead views tilt 0.05 m and use Z-up. The deterministic Y-up
    # fallback is only for an exactly/numerically vertical sightline.
    horizontal = math.hypot(direction[0], direction[1]) / length
    return [0.0, 1.0, 0.0] if horizontal <= 1e-12 else [0.0, 0.0, 1.0]


def set_acquisition_pose(camera, view):
    camera.set_pose(
        pos=view["position"],
        lookat=view["lookat"],
        up=acquisition_up(view["position"], view["lookat"]),
    )


def acquisition_plan(spec, name):
    family = spec["families"][name]
    layout = spec["layouts"][family["layout"]]
    cameras = spec["camera_sets"][family["camera_set"]]
    views = []
    for index in range(24):
        angle = math.tau * index / 24
        position = [
            4.4 * math.cos(angle),
            4.4 * math.sin(angle),
            3.2 if index % 2 == 0 else 4.0,
        ]
        lookat = [0, 0, 0.1]
        for camera_id, query in CAMERA_QUERIES.items():
            if query == index:
                position, lookat = (
                    cameras[camera_id]["render_position"],
                    cameras[camera_id]["render_lookat"],
                )
        views.append(
            {
                "id": f"scan-{index:03}",
                "index": index,
                "position": position,
                "lookat": lookat,
                "role": "query" if index in QUERIES else "mapping",
            }
        )
    positions = [[x, y, 3.2] for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
    positions += [[-1.5, -1.5, 4.0], [1.5, -1.5, 4.0], [0, 1.5, 4.0]]
    for index, pos in enumerate(positions, 24):
        views.append(
            {
                "id": f"scan-{index:03}",
                "index": index,
                "position": pos,
                "lookat": [pos[0], pos[1] + 0.05, 0],
                "role": "mapping",
            }
        )
    return {
        "schema": "bb8.heldout-rgb-acquisition.v1",
        "family": name,
        "renderer_orientation": {
            "schema": "bb8.explicit-camera-up.v1",
            "coordinate_frame": "world",
            "normal_up": [0.0, 0.0, 1.0],
            "vertical_fallback_up": [0.0, 1.0, 0.0],
            "vertical_relative_horizontal_max": 1e-12,
            "scope": "Full render orientation is independent of capture order. Position/lookat are unchanged; query scan metadata still excludes every pose field.",
        },
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": layout["seed"],
        "layout_seed": layout["seed"],
        "parked_robot_xy": [-1.75, -1.75],
        "reset_goal": [-1.0, -1.5],
        "resolution": [1280, 960],
        "fov": 58.0,
        "views": views,
        "query_ids": [f"scan-{i:03}" for i in QUERIES],
        "camera_query_indices": CAMERA_QUERIES,
        "memory_settings": MEMORY_SETTINGS,
        "occupied_landmark_uncertainty_m": 0.05,
        "occupied_height_interval_m": [0.04, 0.3],
        "floor_evidence": "native palette AND fixed planar photometric agreement AND unchanged learned floor mask",
        "registration": "pinned local SuperPoint/LightGlue, 20 known-pose mapping views, query RGB+K only",
        "query_transform_export": "scoring-only file outside scan inputs; omitted from query metadata",
        "scope": "Known synthetic metric scan K/poses are acquisition priors; no depth, segmentation, object geometry or runtime query pose is reconstruction input. Not unknown-pose SLAM.",
        "depth_used": False,
        "segmentation_used": False,
        "original_map_reused": False,
    }


def prepare(args):
    from bb8_rl.demo_assets import validate_assets

    spec = protocol()
    validate_assets(args.assets)
    for name, expected in spec["baseline_assets"].items():
        if digest(args.assets / name) != expected:
            raise ValueError(f"Frozen baseline input differs: {name}")
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root / "models").mkdir()
    for name in ("policy.zip", "vision.pt"):
        shutil.copyfile(args.assets / name, args.root / "models" / name)
    write(args.root / "protocol.json", spec)
    shutil.copyfile(
        args.assets / "scene/room.genesis.json", args.root / "baseline-project.json"
    )
    original = read(args.assets / "scene/room.genesis.json")
    prepared = {}
    for name, family in spec["families"].items():
        path = args.root / name
        (path / "scene").mkdir(parents=True)
        write(path / "scene/room.genesis.json", family_scene(original, spec, name))
        world = yaml.safe_load((args.assets / "scene/room.yaml").read_text())
        world["obstacle_names"] = [
            o["name"] for o in spec["layouts"][family["layout"]]["obstacles"]
        ]
        world["project"] = "room.genesis.json"
        (path / "scene/room.yaml").write_text(yaml.safe_dump(world, sort_keys=False))
        task = yaml.safe_load((args.assets / "scene/task.yaml").read_text())
        task["world_config"] = "room.yaml"
        (path / "scene/task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
        write(path / "scan-plan.json", acquisition_plan(spec, name))
        prepared[name] = {
            str(p.relative_to(path)): digest(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
    manifest = {
        "schema": "bb8.heldout-preparation.v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "family_inputs": prepared,
        "model_sha256": {
            n: digest(args.root / "models" / n) for n in ("policy.zip", "vision.pt")
        },
        "source_sha256": digest(__file__),
        "all_families_separately_captured": True,
        "heldout_requests_unexecuted": 48,
        "settings_frozen_before_capture": True,
    }
    write(args.root / "preparation.json", manifest)
    print(json.dumps(manifest, indent=2))


def prepared_family(root, name):
    spec = protocol()
    if name not in spec["families"]:
        raise ValueError("Unknown frozen family")
    preparation = read(root / "preparation.json")
    if (
        preparation["protocol_sha256"] != PROTOCOL_SHA256
        or read(root / "protocol.json") != spec
    ):
        raise ValueError("Preparation protocol mismatch")
    path = root / name
    for relative, expected in preparation["family_inputs"][name].items():
        if digest(path / relative) != expected:
            raise ValueError(f"Prepared scene or scan plan changed: {relative}")
    if read(path / "scan-plan.json") != acquisition_plan(spec, name):
        raise ValueError("Acquisition settings differ from frozen design")
    for model, expected in preparation["model_sha256"].items():
        if (
            digest(root / "models" / model) != expected
            or expected != spec["baseline_assets"][model]
        ):
            raise ValueError("Frozen policy or vision bytes changed")
    return spec, path, read(path / "scan-plan.json")


def scan_metadata(view, intrinsics, acquisition_transform):
    """Query files expose RGB intrinsics and identity only, never pose fixtures."""
    result = {key: view[key] for key in ("id", "index", "role")}
    result.update(
        intrinsics=intrinsics,
        calibration="query_rgb_intrinsics_only",
        depth_used=False,
        segmentation_used=False,
    )
    if view["role"] == "mapping":
        result.update(
            position=view["position"],
            lookat=view["lookat"],
            world_to_camera=acquisition_transform,
            calibration="synthetic_exact_acquisition",
        )
    return result


def capture(args):
    _, path, plan = prepared_family(args.root, args.family)
    output = path / "scan"
    output.mkdir(exist_ok=False)
    import cv2
    import torch

    from bb8_rl.camera_rig import calibration_from_live_camera
    from bb8_rl.env import NavigationEnv
    from bb8_rl.run_identity import capture_run_identity, verify_run_identity

    torch.set_num_threads(2)
    manifest = {
        "schema": "bb8.heldout-rgb-capture.v1",
        "status": "running",
        "family": args.family,
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_sha256": digest(path / "scan-plan.json"),
        "project_sha256": digest(path / "scene/room.genesis.json"),
        "source_sha256": digest(__file__),
        "evaluation_split": "heldout",
        "seed": plan["seed"],
        "layout_seed": plan["layout_seed"],
        "input_sha256": {},
        "mapping_input_view_ids": [
            v["id"] for v in plan["views"] if v["role"] == "mapping"
        ],
        "query_input_view_ids": plan["query_ids"],
        "depth_used": False,
        "segmentation_used": False,
        "original_map_reused": False,
        "query_transforms_in_scan_metadata": False,
    }
    write(output / "manifest.json", manifest)
    truth = {}
    try:
        with NavigationEnv(
            path / "scene/task.yaml", split="heldout", render_mode="rgb_array"
        ) as env:
            env.reset(
                seed=plan["seed"],
                options={
                    "layout_seed": plan["layout_seed"],
                    "start": plan["parked_robot_xy"],
                    "goal": plan["reset_goal"],
                },
            )
            manifest["run_identity"] = capture_run_identity(
                path / "capture-identity",
                asset_dir=args.assets,
                task_path=path / "scene/task.yaml",
            )
            manifest["identity_baseline_assets_are_mapping_inputs"] = False
            write(output / "manifest.json", manifest)
            env.world.camera_period = env.world._next_frame = 1e9
            camera = env.world.camera
            for view in plan["views"]:
                set_acquisition_pose(camera, view)
                rgb = camera.render(rgb=True, force_render=True)[0].copy()
                calibration = calibration_from_live_camera(camera, 2)
                if list(calibration.resolution) != plan["resolution"]:
                    raise ValueError("Native scan resolution differs from plan")
                meta = scan_metadata(
                    view,
                    calibration.intrinsics.tolist(),
                    calibration.world_to_camera.tolist(),
                )
                if view["role"] == "query":
                    truth[view["id"]] = {
                        "world_to_camera": calibration.world_to_camera.tolist(),
                        "position": view["position"],
                        "lookat": view["lookat"],
                    }
                target = output / view["id"]
                if not cv2.imwrite(
                    str(target.with_suffix(".png")),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise OSError("Failed to save captured RGB")
                write(target.with_suffix(".json"), meta)
                manifest["input_sha256"][view["id"]] = {
                    suffix: digest(target.with_suffix(suffix))
                    for suffix in (".png", ".json")
                }
                write(output / "manifest.json", manifest)
                print(
                    json.dumps({"captured": view["id"], "family": args.family}),
                    flush=True,
                )
        write(path / "query-truth-scoring-only.json", truth)
        manifest["run_identity_verification"] = verify_run_identity(
            manifest["run_identity"]
        )
        if manifest["run_identity_verification"]["unchanged"] is not True:
            raise ValueError("Native capture inputs changed during run")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "manifest.json", manifest)


def checked_capture(path, plan):
    manifest = read(path / "scan/manifest.json")
    expected = {v["id"] for v in plan["views"]}
    if (
        manifest.get("status") != "complete"
        or manifest.get("plan_sha256") != digest(path / "scan-plan.json")
        or manifest.get("project_sha256") != digest(path / "scene/room.genesis.json")
        or manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or manifest.get("family") != plan["family"]
        or manifest.get("evaluation_split") != "heldout"
        or manifest.get("seed") != plan["seed"]
        or manifest.get("layout_seed") != plan["layout_seed"]
        or set(manifest.get("input_sha256", {})) != expected
        or manifest.get("query_transforms_in_scan_metadata") is not False
        or manifest.get("depth_used") is not False
        or manifest.get("segmentation_used") is not False
    ):
        raise ValueError("Need complete separately captured frozen RGB inputs")
    for view in plan["views"]:
        stem = path / "scan" / view["id"]
        if {
            suffix: digest(stem.with_suffix(suffix)) for suffix in (".png", ".json")
        } != manifest["input_sha256"][view["id"]]:
            raise ValueError("Captured scan checksum mismatch")
        meta = read(stem.with_suffix(".json"))
        expected = scan_metadata(
            view, meta.get("intrinsics"), meta.get("world_to_camera")
        )
        if meta != expected or (
            view["role"] == "mapping" and meta.get("world_to_camera") is None
        ):
            raise ValueError("Scan metadata pose/role or query isolation mismatch")
    if manifest.get("run_identity_verification", {}).get("unchanged") is not True:
        raise ValueError("Capture identity did not remain unchanged")
    return manifest


def registration_ancestry(registration, captured):
    expected_map = [f"scan-{i:03}" for i in range(24) if i not in QUERIES]
    expected_query = [f"scan-{i:03}" for i in QUERIES]
    if (
        registration.get("status") != "complete"
        or registration.get("mapping_input_view_ids") != expected_map
        or registration.get("query_indices") != list(QUERIES)
        or registration.get("query_pose_consumed") is not False
        or registration.get("map_frozen_before_query_matching") is not True
        or set(registration.get("mapping_inputs", {})) != set(expected_map)
        or set(registration.get("query_inputs", {})) != set(expected_query)
    ):
        raise ValueError("Registration ancestry or query isolation mismatch")
    for category in ("mapping_inputs", "query_inputs"):
        for name, record in registration[category].items():
            if (
                record["rgb_sha256"] != captured["input_sha256"][name][".png"]
                or record["metadata_sha256"] != captured["input_sha256"][name][".json"]
            ):
                raise ValueError("Registration does not use this family's captured RGB")


def registration_interpreter(runtime, selected=None):
    # Resolving the executable symlink discards pyvenv.cfg discovery and escapes
    # the preserved matching environment into its dependency-free base Python.
    interpreter = selected or runtime / "venv/bin/python"
    if not interpreter.is_file():
        raise ValueError(
            "Need the preserved local registration interpreter or --registration-python"
        )
    return interpreter.absolute()


def reconstruct(args):
    _, path, plan = prepared_family(args.root, args.family)
    checked_capture(path, plan)
    registration = path / "registration"
    interpreter = registration_interpreter(
        args.registration_runtime, args.registration_python
    )
    subprocess.run(
        [
            str(interpreter),
            str(ROOT / "scripts/refine-scan-registration.py"),
            "--runtime-root",
            str(args.registration_runtime),
            "--scan-dir",
            str(path / "scan"),
            "--output",
            str(registration),
        ],
        check=True,
    )
    build_memory(args.root, path, plan)


def build_memory(root, path, plan):
    import cv2
    import numpy as np

    from bb8_rl.mapping import (
        Bounds3D,
        EvidenceSource,
        Provenance,
        RoomMemory,
        SpaceState,
    )
    from bb8_rl.mapping.scan_free_space import (
        ScanRGBView,
        build_scan_free_memory,
        floor_homography_masks,
    )

    capture_manifest = checked_capture(path, plan)
    cv2.setNumThreads(2)
    views = []
    for view in plan["views"]:
        if view["role"] != "mapping":
            continue
        meta = read(path / "scan" / f"{view['id']}.json")
        rgb = cv2.cvtColor(
            cv2.imread(str(path / "scan" / f"{view['id']}.png")), cv2.COLOR_BGR2RGB
        )
        views.append(
            ScanRGBView(
                view["id"],
                rgb,
                np.asarray(meta["intrinsics"]),
                np.asarray(meta["world_to_camera"]),
            )
        )
    masks, _ = floor_homography_masks(views)
    floor_module = module("build-extended-scan-memory.py")
    learned, learned_report = floor_module.learned_floor_masks(
        views,
        root / "models/vision.pt",
        protocol()["baseline_assets"]["vision.pt"],
        2.0,
    )
    masks = [a & b for a, b in zip(masks, learned, strict=True)]
    registration = read(path / "registration/report.json")
    registration_ancestry(registration, capture_manifest)
    if digest(path / "registration/landmarks.npz") != registration.get(
        "landmarks_sha256"
    ):
        raise ValueError("RGB landmarks differ from registration report")
    tracks = read(path / "registration/tracks.json")["tracks"]
    with np.load(path / "registration/landmarks.npz", allow_pickle=False) as data:
        points = data["points"]
    if len(points) != len(tracks):
        raise ValueError("RGB landmark support mismatch")
    occupied = RoomMemory(
        Bounds3D((-2, -2, 0), (2, 2, 0.3)),
        0.02,
        scene_version="heldout-" + plan["family"],
        calibration_version="known-synthetic-acquisition-"
        + digest(path / "scan-plan.json"),
    )
    count = 0
    for index, (point, track) in enumerate(zip(points, tracks, strict=True)):
        if np.max(abs(point[:2])) >= 2 or not 0.04 <= point[2] <= 0.3:
            continue
        ids = tuple(sorted({row[0] for row in track["observations"]}))
        if len(ids) < 3 or any(int(v[5:]) in QUERIES for v in ids):
            raise ValueError("Invalid occupied RGB support")
        source = EvidenceSource(
            f"rgb-landmark-{index}", ids, 0, Provenance.RGB_RECONSTRUCTION, 0.05
        )
        occupied.observe_volume(
            Bounds3D(tuple(point - 0.001), tuple(point + 0.001)),
            SpaceState.OCCUPIED,
            source,
        )
        count += 1
    memory = build_scan_free_memory(
        views, occupied_memory=occupied, floor_masks=masks, **plan["memory_settings"]
    )
    memory.metadata.update(
        family=plan["family"],
        protocol_sha256=PROTOCOL_SHA256,
        capture_manifest_sha256=digest(path / "scan/manifest.json"),
        capture_input_sha256=capture_manifest["input_sha256"],
        scan_plan_sha256=digest(path / "scan-plan.json"),
        registration_report_sha256=digest(path / "registration/report.json"),
        landmark_sha256=digest(path / "registration/landmarks.npz"),
        tracks_sha256=digest(path / "registration/tracks.json"),
        occupied_rgb_landmarks=count,
        original_map_reused=False,
        truth_inputs_used=False,
        query_inputs_used=False,
        learned_mask_intersection=learned_report,
        thresholds_selected_using_truth=False,
        geometry_provenance="RGB triangulation with known synthetic acquisition priors; query RGB+K PnP",
        registered_query_candidates={
            str(q["query_index"]): q["status"] for q in registration["queries"]
        },
    )
    memory.save(path / "memory")
    np.savez_compressed(
        path / "memory/floor-masks.npz",
        masks=np.asarray(masks),
        view_ids=np.asarray([v.view_id for v in views]),
    )
    manifest = read(path / "memory/manifest.json")
    manifest["artifact_sha256"]["floor-masks.npz"] = digest(
        path / "memory/floor-masks.npz"
    )
    write(path / "memory/manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "free_cells",
                    "occupied_cells",
                    "unknown_cells",
                    "registered_query_candidates",
                )
            },
            indent=2,
        )
    )


def map_admission_checks(full_volume, robot, evidence):
    """Reject unsafe/incomplete maps; never turn scorer truth into map evidence."""
    return {
        "nonempty_free_volume": full_volume.get("free_prisms", 0) > 0,
        "no_false_free_voxels": full_volume.get("false_free_solid_intersection_voxels")
        == 0,
        "no_false_free_prisms": full_volume.get("free_solid_intersection_prisms") == 0,
        "within_room": full_volume.get("free_voxels_outside_room_or_below_floor") == 0,
        "rgb_sources_only": full_volume.get("free_sources_rgb_only") is True,
        "query_exclusion": full_volume.get("heldout_query_in_map") == [],
        "parked_robot_excluded": robot.get("pass") is True,
        "evidence_complete": bool(evidence)
        and all(v is True for v in evidence.values()),
    }


def audit_map(args):
    """Independent authored-geometry scorer. Its output cannot repair a map."""
    import numpy as np

    from bb8_rl.mapping.scan_free_space import ScanFreeMemory

    _, path, plan = prepared_family(args.root, args.family)
    captured = checked_capture(path, plan)
    registration = read(path / "registration/report.json")
    registration_ancestry(registration, captured)
    output = path / "map-audit"
    output.mkdir(exist_ok=False)
    memory = ScanFreeMemory.load(path / "memory")
    scene = read(path / "scene/room.genesis.json")
    scorer = module("audit-map-improvement.py")
    full_volume = scorer.LEGACY.audit_memory(path / "memory", scene)
    # The reused legacy scorer has a 20-view membership requirement. This family
    # instead has exactly the predeclared 32 mapping views, checked separately.
    full_volume["legacy_twenty_view_status"] = full_volume.pop("status")
    parts = {
        o["name"]: o for o in scene["objects"] if o["name"] in ("bb8_body", "bb8_head")
    }
    robot_fixture = {"position_xy": plan["parked_robot_xy"]}
    for part in ("body", "head"):
        obj = parts[f"bb8_{part}"]
        if obj["format"] != "sphere":
            raise ValueError("Parked-robot scorer requires declared spheres")
        robot_fixture[f"{part}_radius_m"] = obj["radius"]
        robot_fixture[f"{part}_center_z_m"] = obj["position"][2]
    robot = scorer.scan_robot_checks(memory, robot_fixture)
    meta = memory.metadata
    counts = np.array([int(v).bit_count() for v in memory.support_bits.flat]).reshape(
        memory.free_mask.shape
    )
    expected_ids = [v["id"] for v in plan["views"] if v["role"] == "mapping"]
    evidence = {
        "valid_memory": memory.memory.valid,
        "exact_frozen_mapping_views": list(memory.view_ids) == expected_ids,
        "minimum_frozen_support": bool(
            np.all(counts[memory.free_mask] >= plan["memory_settings"]["minimum_views"])
        ),
        "full_prism": meta.get("free_evidence_mode") == "full_prism",
        "frozen_settings": all(
            meta.get(key) == value
            for key, value in {
                "extent_m": MEMORY_SETTINGS["extent"],
                "resolution_m": MEMORY_SETTINGS["resolution"],
                "requested_body_height_m": MEMORY_SETTINGS["body_height_m"],
                **{
                    k: v
                    for k, v in MEMORY_SETTINGS.items()
                    if k
                    not in ("extent", "resolution", "body_height_m", "evidence_mode")
                },
            }.items()
        ),
        "capture_bound": meta.get("capture_manifest_sha256")
        == digest(path / "scan/manifest.json")
        and meta.get("capture_input_sha256") == captured["input_sha256"],
        "registration_bound": meta.get("registration_report_sha256")
        == digest(path / "registration/report.json"),
        "landmarks_bound": meta.get("landmark_sha256")
        == registration.get("landmarks_sha256")
        == digest(path / "registration/landmarks.npz"),
        "tracks_bound": meta.get("tracks_sha256")
        == digest(path / "registration/tracks.json"),
        "models_unchanged": meta.get("learned_mask_intersection", {}).get(
            "checkpoint_sha256"
        )
        == protocol()["baseline_assets"]["vision.pt"],
        "no_truth_or_original_map": all(
            meta.get(key) is False
            for key in (
                "truth_inputs_used",
                "query_inputs_used",
                "original_map_reused",
                "thresholds_selected_using_truth",
                "missing_geometry_creates_free",
            )
        ),
    }
    with np.load(path / "memory/free-grid.npz", allow_pickle=False) as arrays:
        evidence.update(
            scorer.support_storage_checks(
                arrays["support_bits"], memory.support_bits, len(memory.view_ids)
            )
        )
    checks = map_admission_checks(full_volume, robot, evidence)
    report = {
        "schema": "bb8.heldout-map-audit.v1",
        "family": args.family,
        "protocol_sha256": PROTOCOL_SHA256,
        "native_trial_gate_pass": all(checks.values()),
        "scope": "Independent whole-volume synthetic geometry and parked-robot scoring only. No oracle repair; static safety does not establish route success or physical reliability.",
        "controller_map_modified": False,
        "checks": checks,
        "evidence": evidence,
        "full_volume": full_volume,
        "parked_robot": robot,
        "input_sha256": {name: digest(path / name) for name in MAP_AUDIT_INPUTS},
        "scorer_sha256": {
            "provisioner": digest(__file__),
            "geometry": digest(ROOT / "scripts/audit-occluded-control.py"),
            "parked_robot": digest(ROOT / "scripts/audit-map-improvement.py"),
        },
    }
    write(output / "report.json", report)
    print(
        json.dumps(
            {
                "family": args.family,
                "native_trial_gate_pass": report["native_trial_gate_pass"],
                "checks": checks,
            },
            indent=2,
        )
    )
    return report


def checked_map_audit(path):
    report = read(path / "map-audit/report.json")
    if (
        report.get("schema") != "bb8.heldout-map-audit.v1"
        or report.get("protocol_sha256") != PROTOCOL_SHA256
        or report.get("native_trial_gate_pass") is not True
        or report.get("controller_map_modified") is not False
        or set(report.get("input_sha256", {})) != set(MAP_AUDIT_INPUTS)
        or not report.get("checks")
        or not all(v is True for v in report["checks"].values())
        or report.get("checks")
        != map_admission_checks(
            report.get("full_volume", {}),
            report.get("parked_robot", {}),
            report.get("evidence", {}),
        )
    ):
        raise ValueError("Held-out map failed independent whole-volume safety gate")
    if any(
        digest(path / name) != expected
        for name, expected in report["input_sha256"].items()
    ):
        raise ValueError("Audited map inputs changed")
    return report


def bundle(args):
    spec, path, plan = prepared_family(args.root, args.family)
    checked_map_audit(path)
    capture_manifest = checked_capture(path, plan)
    registration = read(path / "registration/report.json")
    memory = read(path / "memory/manifest.json")
    registration_ancestry(registration, capture_manifest)
    if (
        registration.get("status") != "complete"
        or registration.get("query_pose_consumed") is not False
        or registration.get("map_frozen_before_query_matching") is not True
        or memory.get("capture_manifest_sha256") != digest(path / "scan/manifest.json")
        or memory.get("registration_report_sha256")
        != digest(path / "registration/report.json")
        or memory.get("original_map_reused") is not False
    ):
        raise ValueError("New map/registration ancestry is incomplete")
    for query in CAMERA_QUERIES.values():
        candidates = [q for q in registration["queries"] if q["query_index"] == query]
        if len(candidates) != 1 or candidates[0]["status"] != "candidate":
            raise ValueError(
                f"RGB registration failed for runtime query {query}; no truth substitution"
            )
    cases = [
        dict(c, id=i)
        for i, c in spec["cases"].items()
        if c["family"] == args.family and c["intent"] == "arrival"
    ]
    selected = {
        "scene_sha256": digest(path / "scene/room.genesis.json"),
        "registration_report_sha256": digest(path / "registration/report.json"),
        "cases": cases,
    }
    write(path / "bundle-protocol.json", selected)
    render_metadata = path / "render-fixture-metadata"
    render_metadata.mkdir(exist_ok=False)
    for name, index in CAMERA_QUERIES.items():
        meta = read(path / "scan" / f"scan-{index:03}.json")
        fixture = spec["camera_sets"][spec["families"][args.family]["camera_set"]][name]
        write(
            render_metadata / f"camera-{name}.json",
            {
                "index": index,
                "intrinsics": meta["intrinsics"],
                "position": fixture["render_position"],
                "lookat": fixture["render_lookat"],
                "up": acquisition_up(
                    fixture["render_position"], fixture["render_lookat"]
                ),
                "scope": "Frozen renderer fixture only; excluded from RGB registration inputs",
                "query_metadata_sha256": digest(
                    path / "scan" / f"scan-{index:03}.json"
                ),
            },
        )

    def record(p):
        return {"path": str(p.relative_to(args.root)), "sha256": digest(p)}

    inputs = {
        "policy": args.root / "models/policy.zip",
        "vision": args.root / "models/vision.pt",
        "task": path / "scene/task.yaml",
        "world": path / "scene/room.yaml",
        "project": path / "scene/room.genesis.json",
        "registration": path / "registration/report.json",
        "protocol": path / "bundle-protocol.json",
        "memory_manifest": path / "memory/manifest.json",
    }
    manifest = {
        "schema": "bb8.interactive-asset-inputs.v1",
        "root": "..",
        "inputs": {k: record(p) for k, p in inputs.items()},
        "protocol_case": 0,
        "memory_artifacts": {
            k: record(path / "memory" / k) for k in memory["artifact_sha256"]
        },
        "cameras": {
            name: {
                "query_index": index,
                "metadata": record(render_metadata / f"camera-{name}.json"),
                "lookat": spec["camera_sets"][
                    spec["families"][args.family]["camera_set"]
                ][name]["render_lookat"],
                "resolution": [1280, 960],
            }
            for name, index in CAMERA_QUERIES.items()
        },
    }
    write(path / "asset-inputs.json", manifest)
    builder = module("build-interactive-assets.py")
    builder.main(
        SimpleNamespace(
            inputs=path / "asset-inputs.json",
            input_root=args.root,
            validate_inputs_only=False,
            write_input_manifest=None,
            output=path / "raw-assets",
            scan_dir=None,
            memory=None,
        )
    )
    output = path / "bundle"
    shutil.copytree(path / "raw-assets", output)
    evidence = output / "heldout-evidence"
    evidence.mkdir()
    for source, name in (
        (path / "scan-plan.json", "scan-plan.json"),
        (path / "scan/manifest.json", "capture.json"),
        (path / "registration/report.json", "registration.json"),
        (path / "memory/manifest.json", "memory.json"),
        (path / "map-audit/report.json", "map-audit.json"),
        (args.root / "preparation.json", "preparation.json"),
        (args.root / "baseline-project.json", "baseline-project.json"),
    ):
        shutil.copyfile(source, evidence / name)
    write(
        output / "heldout-provisioning.json",
        {
            "schema": "bb8.heldout-bundle.v1",
            "family": args.family,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_project_sha256": digest(path / "scene/room.genesis.json"),
            "source_capture_sha256": digest(path / "scan/manifest.json"),
            "source_map_sha256": digest(path / "memory/manifest.json"),
            "source_map_audit_sha256": digest(path / "map-audit/report.json"),
            "source_registration_sha256": digest(path / "registration/report.json"),
            "scan_input_count": len(capture_manifest["input_sha256"]),
            "original_map_reused": False,
            "query_extrinsics_rgb_estimated": True,
            "source_family_directory": str(path.resolve()),
            "scope": plan["scope"],
        },
    )
    demo = read(output / "demo.json")
    for record in demo["cameras"].values():
        record["up"] = acquisition_up(record["position"], record["lookat"])
    demo["scope"] = (
        "Frozen held-out family "
        + args.family
        + "; new RGB map and RGB-estimated query registration; synthetic metric acquisition priors."
    )
    write(output / "demo.json", demo)
    write(
        output / "bundle.json",
        {
            "schema": "bb8.interactive-assets.v1",
            "sha256": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "bundle.json"
            },
        },
    )
    validate_bundle(output, args.family)
    print(
        json.dumps(
            {
                "family": args.family,
                "bundle": str(output),
                "bundle_sha256": digest(output / "bundle.json"),
            },
            indent=2,
        )
    )


def validate_bundle(assets, family):
    from bb8_rl.demo_assets import validate_assets

    spec = protocol()
    demo = validate_assets(assets)
    provision = read(assets / "heldout-provisioning.json")
    if (
        provision.get("schema") != "bb8.heldout-bundle.v1"
        or provision.get("family") != family
        or provision.get("protocol_sha256") != PROTOCOL_SHA256
        or provision.get("original_map_reused") is not False
        or provision.get("query_extrinsics_rgb_estimated") is not True
    ):
        raise ValueError("Held-out provisioning does not match frozen family")
    evidence = assets / "heldout-evidence"
    plan = read(evidence / "scan-plan.json")
    capture_manifest = read(evidence / "capture.json")
    memory = read(evidence / "memory.json")
    audit = read(evidence / "map-audit.json")
    if (
        digest(evidence / "map-audit.json") != provision.get("source_map_audit_sha256")
        or audit.get("native_trial_gate_pass") is not True
        or audit.get("family") != family
        or audit.get("protocol_sha256") != PROTOCOL_SHA256
        or audit.get("controller_map_modified") is not False
        or audit.get("input_sha256", {}).get("memory/manifest.json")
        != provision["source_map_sha256"]
        or audit.get("input_sha256", {}).get("scene/room.genesis.json")
        != provision["source_project_sha256"]
        or audit.get("checks")
        != map_admission_checks(
            audit.get("full_volume", {}),
            audit.get("parked_robot", {}),
            audit.get("evidence", {}),
        )
        or not all(v is True for v in audit["checks"].values())
    ):
        raise ValueError("Held-out map safety audit missing or invalid")
    for name, expected in memory["artifact_sha256"].items():
        if digest(assets / "memory" / name) != expected:
            raise ValueError(
                "Bundled memory artifact differs from independently audited source"
            )
    for name in ("room-memory.json", "free-grid.npz"):
        if (
            audit.get("input_sha256", {}).get("memory/" + name)
            != memory["artifact_sha256"][name]
        ):
            raise ValueError("Map audit does not bind actual source artifacts")
    registration = read(evidence / "registration.json")
    registration_ancestry(registration, capture_manifest)
    if (
        plan != acquisition_plan(spec, family)
        or capture_manifest["status"] != "complete"
        or capture_manifest["plan_sha256"] != digest(evidence / "scan-plan.json")
        or capture_manifest.get("evaluation_split") != "heldout"
        or capture_manifest.get("seed") != plan["seed"]
        or capture_manifest.get("layout_seed") != plan["layout_seed"]
        or capture_manifest.get("run_identity_verification", {}).get("unchanged")
        is not True
    ):
        raise ValueError("Held-out capture plan is not frozen and complete")
    if (
        digest(evidence / "capture.json") != provision["source_capture_sha256"]
        or digest(evidence / "memory.json") != provision["source_map_sha256"]
        or digest(evidence / "registration.json")
        != provision["source_registration_sha256"]
        or memory["capture_manifest_sha256"] != provision["source_capture_sha256"]
        or memory["registration_report_sha256"]
        != provision["source_registration_sha256"]
        or read(assets / "memory/manifest.json")["portable_source_manifest_sha256"]
        != provision["source_map_sha256"]
    ):
        raise ValueError("Held-out map ancestry mismatch")
    if (
        digest(assets / "scene/room.genesis.json") != provision["source_project_sha256"]
        or capture_manifest["project_sha256"] != provision["source_project_sha256"]
    ):
        raise ValueError("Held-out scene does not match its scan")
    if (
        digest(evidence / "baseline-project.json")
        != spec["baseline_assets"]["scene/room.genesis.json"]
    ):
        raise ValueError("Frozen baseline scene template differs")
    original = read(evidence / "baseline-project.json")
    if read(assets / "scene/room.genesis.json") != family_scene(original, spec, family):
        raise ValueError("Held-out scene differs from frozen geometry/appearance")
    if (
        registration.get("query_pose_consumed") is not False
        or registration.get("map_frozen_before_query_matching") is not True
    ):
        raise ValueError("Registration query isolation missing")
    for name, index in CAMERA_QUERIES.items():
        query = next(
            (q for q in registration["queries"] if q["query_index"] == index), None
        )
        expected = spec["camera_sets"][spec["families"][family]["camera_set"]][name]
        actual = demo["cameras"][name]
        if (
            query is None
            or query["status"] != "candidate"
            or actual["world_to_camera"] != query["world_to_camera"]
            or actual["position"] != expected["render_position"]
            or actual["lookat"] != expected["render_lookat"]
            or actual.get("up")
            != acquisition_up(expected["render_position"], expected["render_lookat"])
            or actual["resolution"] != expected["resolution"]
        ):
            raise ValueError("Camera fixture or RGB-estimated transform differs")
    for name in ("policy.zip", "vision.pt"):
        if digest(assets / name) != spec["baseline_assets"][name]:
            raise ValueError("Policy or vision changed")
    if (
        digest(assets / "memory/manifest.json")
        == spec["baseline_assets"]["memory/manifest.json"]
        or digest(assets / "memory/room-memory.json")
        == spec["baseline_assets"]["memory/room-memory.json"]
    ):
        raise ValueError("Original map is inadmissible for held-out layout")
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "capture",
            "reconstruct",
            "memory",
            "audit-map",
            "bundle",
            "validate",
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT / "work/continuation-20260926-visual-heldout"
    )
    parser.add_argument("--family", choices=tuple(protocol()["families"]))
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument(
        "--registration-runtime", type=Path, default=ROOT / "work/m77/registration"
    )
    parser.add_argument(
        "--registration-python",
        type=Path,
        help="Existing local matching interpreter; defaults to registration-runtime/venv/bin/python",
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.assets = args.assets.resolve()
    if args.phase != "prepare" and args.family is None:
        parser.error("phase requires --family")
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "capture":
        capture(args)
    elif args.phase == "reconstruct":
        reconstruct(args)
    elif args.phase == "memory":
        _, path, plan = prepared_family(args.root, args.family)
        build_memory(args.root, path, plan)
    elif args.phase == "audit-map":
        audit_map(args)
    elif args.phase == "bundle":
        bundle(args)
    else:
        validate_bundle(args.root / args.family / "bundle", args.family)


if __name__ == "__main__":
    main()
