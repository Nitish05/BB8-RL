"""Seeded synthetic room and exact simulator-camera calibration artifacts."""

from pathlib import Path

import numpy as np
import yaml
from genesis_studio.models import BoxObjectConfig

from .diagnostics import machine_report, write_report
from .planner import OccupancyGrid
from .task import TaskConfig
from .world import NavigationWorld, validate_world


def generate_room(base_config, output, *, seed=42, side=4.0, obstacles=7):
    output = Path(output)
    if not 3 <= side <= 6 or not np.isfinite(side) or not 0 <= obstacles <= 8:
        raise ValueError("Choose a 3–6 m square room and 0–8 obstacles")
    if not 0 <= seed < 2**31:
        raise ValueError("Seed must lie in [0, 2**31)")
    if not np.isclose(side / 0.1, round(side / 0.1)):
        raise ValueError("Room side must be a multiple of 0.1 m")
    if any(
        (output / name).exists()
        for name in ("room.genesis.json", "room.yaml", "layout.json", "task.yaml")
    ):
        raise ValueError("Room already exists; choose a fresh output directory")
    config, project, body, head, camera = validate_world(Path(base_config))
    # The known empty-floor fixture supplies appearance and drive parameters.
    if len(project.objects) != 6 or config.obstacle_names:
        raise ValueError("Generate rooms from the six-object empty-floor fixture")
    extent = side / 2
    start, goal = np.array([-0.325 * side] * 2), np.array([0.325 * side] * 2)
    for obj in project.objects:
        if obj.name.startswith("wall_"):
            axis = 0 if obj.name.startswith("wall_0") else 1
            sign = -1 if obj.name.endswith("minus") else 1
            size = [side + 0.1, side + 0.1, 0.15]
            size[axis] = 0.05
            pos = [0, 0, 0.075]
            pos[axis] = sign * (extent + 0.025)
            obj.size, obj.position = tuple(size), tuple(pos)
    body.position = (*start.tolist(), body.radius)
    head.position = (*start.tolist(), head.position[2])
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(2000):
        if len(boxes) == obstacles:
            break
        width, depth, height = rng.uniform([0.25, 0.25, 0.15], [0.6, 0.6, 0.45])
        x, y = rng.uniform(-extent + 0.55, extent - 0.55, 2)
        if any(
            abs(x - bx) < (width + bw) / 2 + 0.18
            and abs(y - by) < (depth + bd) / 2 + 0.18
            for bx, by, bw, bd, _ in boxes
        ):
            continue
        candidate = [
            *boxes,
            (float(x), float(y), float(width), float(depth), float(height)),
        ]
        grid = OccupancyGrid(
            extent, 0.1, body.radius + 0.11, [b[:4] for b in candidate]
        )
        try:
            grid.route(start, goal)
        except ValueError:
            continue
        boxes = candidate
    if len(boxes) != obstacles:
        raise ValueError("Could not place all obstacles with a connected route")
    for i, (x, y, width, depth, height) in enumerate(boxes):
        project.objects.append(
            BoxObjectConfig(
                name=f"room_obstacle_{i}",
                fixed=True,
                position=(x, y, height / 2),
                size=(width, depth, height),
                material={"color": (0.3 + 0.05 * (i % 3), 0.44, 0.58, 1)},
            )
        )
    grid = OccupancyGrid(extent, 0.1, body.radius + 0.11, [b[:4] for b in boxes])
    route = grid.route(start, goal)
    camera.position = (0.7 * side, -0.85 * side, 1.3 * side)
    camera.lookat, camera.resolution, camera.fov = (0, 0, 0), (1280, 960), 58
    project.environment.camera_position = camera.position
    project.environment.camera_lookat = camera.lookat
    project.environment.camera_fov = camera.fov
    project.seed, project.name = seed, f"BB8 synthetic {side:g} m room — seed {seed}"
    config.project, config.arena_half_extent = "room.genesis.json", extent
    config.obstacle_names = tuple(f"room_obstacle_{i}" for i in range(obstacles))
    config.provenance = "synthetic"
    config.notes = (
        "User-authorized arbitrary room dimensions and seeded random obstacles. "
        "Fixed elevated pinhole camera; exact simulator geometry, not measured calibration. "
        "Low perimeter walls are a cutaway floor boundary. BB8 geometry and dynamics are synthetic. "
        "Authored geometry is shared by all episodes using this room's task.yaml."
    )
    output.mkdir(parents=True, exist_ok=True)
    write_report(output / "room.genesis.json", project.model_dump(mode="json"))
    (output / "room.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    task = TaskConfig(
        world_config="room.yaml",
        layout_mode="authored",
        max_goal_distance=side * 0.9,
        max_episode_steps=1200,
    )
    (output / "task.yaml").write_text(
        yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False)
    )
    write_report(
        output / "layout.json",
        {
            "schema_version": 1,
            "provenance": "synthetic",
            "seed": seed,
            "room_size_m": [side, side],
            "obstacles_xy_width_depth_height": boxes,
            "start_xy": start.tolist(),
            "goal_xy": goal.tolist(),
            "route_xy": route.tolist(),
            "grid_resolution_m": 0.1,
            "footprint_inflation_m": body.radius + 0.11,
            "camera_position_m": camera.position,
            "camera_lookat_m": camera.lookat,
            "camera_vertical_fov_degrees": camera.fov,
            "resolution_wh": camera.resolution,
        },
    )
    validate_world(output / "room.yaml")
    return {
        "status": "passed",
        "config": str(output / "room.yaml"),
        "seed": seed,
        "obstacles": obstacles,
    }


def project_points(intrinsics, world_to_camera, points):
    points = np.asarray(points, dtype=float)
    camera = np.c_[points, np.ones(len(points))] @ np.asarray(world_to_camera).T
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError("Projection points must be finite and in front of the camera")
    pixels = camera[:, :3] @ np.asarray(intrinsics).T
    return pixels[:, :2] / pixels[:, 2:3]


def preview_room(config_path, output, *, backend="cpu"):
    from PIL import Image

    output = Path(output)
    if (output / "camera-report.json").exists():
        raise ValueError(
            "Camera report already exists; choose a fresh output directory"
        )
    report = machine_report(Path(config_path))
    with NavigationWorld(Path(config_path), backend=backend, render=True) as world:
        world.step(2)
        rgb, _, segmentation, _ = world.camera.render(rgb=True, segmentation=True)
        intrinsics, extrinsics = world.camera.intrinsics, world.camera.extrinsics
        e = world.config.arena_half_extent
        corners = np.array([[-e, -e, 0], [e, -e, 0], [e, e, 0], [-e, e, 0]])
        pixels = project_points(intrinsics, extrinsics, corners)
        width, height = world.camera.res
        in_frame = bool(np.all((pixels >= 0) & (pixels < [width, height])))
        entity_ids = {world.body.idx, world.head.idx}
        labels = [
            label
            for label, key in world.scene.visualizer.segmentation_idx_dict.items()
            if (key[0] if isinstance(key, tuple) else key) in entity_ids
        ]
        ys, xs = np.where(np.isin(segmentation, labels))
        visible = len(xs)
        homography = intrinsics @ extrinsics[:3, [0, 1, 3]]
        inverse = np.linalg.inv(homography)
        restored = np.c_[pixels, np.ones(4)] @ inverse.T
        residual = float(
            np.max(
                np.linalg.norm(
                    restored[:, :2] / restored[:, 2:] - corners[:, :2], axis=1
                )
            )
        )
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output / "camera.png")
        np.save(output / "segmentation.npy", segmentation)
        if visible:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            crop = rgb[
                max(0, bbox[1] - 20) : min(height, bbox[3] + 21),
                max(0, bbox[0] - 20) : min(width, bbox[2] + 21),
            ]
            Image.fromarray(crop).resize((384, 384)).save(output / "bb8-detail.png")
        else:
            bbox = None
        write_report(
            output / "calibration.json",
            {
                "schema_version": 1,
                "provenance": "synthetic_exact",
                "model": "ideal_pinhole",
                "resolution_wh": [width, height],
                "intrinsics": intrinsics.tolist(),
                "world_to_camera": extrinsics.tolist(),
                "distortion_k1_k2_p1_p2_k3": [0] * 5,
                "floor_to_pixel_homography": homography.tolist(),
                "pixel_to_floor_homography": inverse.tolist(),
                "floor_z_m": 0,
                "world_axes": "X/Y floor, Z up; metres",
                "camera_axes": "OpenCV: X right, Y down, Z forward; pixels u right, v down",
                "scope": "Maps floor pixels only. Object surfaces and occlusions require depth/masking. No physical calibration claim.",
                "floor_corners_world": corners.tolist(),
                "floor_corners_pixels": pixels.tolist(),
                "numerical_roundtrip_error_m": residual,
            },
        )
        report.update(
            status="passed"
            if in_frame and visible >= 8 and residual < 1e-5
            else "failed",
            resolved_backend=world.gs.backend.name,
            all_floor_corners_in_frame=in_frame,
            bb8_visible_pixels=visible,
            bb8_bbox_xyxy=bbox,
            numerical_roundtrip_error_m=residual,
            scope="Initial-pose visibility and geometric frustum coverage; not all floor pixels are observable behind obstacles.",
        )
    write_report(output / "camera-report.json", report)
    return {
        k: report[k]
        for k in (
            "status",
            "resolved_backend",
            "all_floor_corners_in_frame",
            "bb8_visible_pixels",
            "bb8_bbox_xyxy",
        )
    }
