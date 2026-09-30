"""Freeze then capture a stationary native RGB fixture preflight.

--freeze writes a protocol without importing Genesis or loading any weights.
Native execution is separate and must be serialized with other Genesis runs.
Manual gauge settings test rendering/perception, not action-dependent learning.
"""

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bb8_rl.demo_assets import validate_assets
from bb8_rl.visual_fixture import (
    DEFAULT_PANELS,
    VisualCamera,
    VisualFixture,
    VisualPanel,
)
from bb8_rl.visual_interaction import GAUGE_BINS, ID_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/preflight-visual-fixture.py",
    "src/bb8_rl/visual_fixture.py",
    "src/bb8_rl/visual_interaction.py",
    "src/bb8_rl/world.py",
    "src/bb8_rl/env.py",
    "src/bb8_rl/vision.py",
    "src/bb8_rl/camera.py",
    "src/bb8_rl/camera_rig.py",
    "src/bb8_rl/scene_validity.py",
    "src/bb8_rl/control/genesis_backend.py",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def environment_kwargs(asset_dir, demo, cameras, fixture):
    """Bind the installed development fixture to its original seed domain."""
    from bb8_rl.task import require_split_seed

    require_split_seed("validation", demo["layout_seed"])
    return {
        "task_path": asset_dir / demo["task"],
        "split": "validation",
        "render_mode": "rgb_array",
        "camera_positions": [tuple(cameras[name]["position"]) for name in "ABC"],
        "visual_fixture": fixture,
    }


def make_protocol(asset_dir):
    asset_dir = asset_dir.resolve()
    demo = validate_assets(asset_dir)
    cameras = demo["cameras"]
    environment_kwargs(asset_dir, demo, cameras, None)
    panels = (
        *DEFAULT_PANELS,
        VisualPanel("marker-01", DEFAULT_PANELS[1].center_xy),
        VisualPanel("marker-02", DEFAULT_PANELS[0].center_xy),
        VisualPanel("marker-01", (0.6, 1.25)),
    )
    cases = [
        {
            "id": f"warmup-{i}",
            "kind": "warmup",
            "level": 8,
            "visible_panels": [0, 1],
            "occluded": False,
        }
        for i in range(3)
    ]
    cases.extend(
        {
            "id": f"level-{level:02d}",
            "kind": "level",
            "level": level,
            "visible_panels": [0, 1],
            "occluded": False,
        }
        for level in range(GAUGE_BINS + 1)
    )
    for name, indices, occluded in (
        ("absent", [], False),
        ("occluded", [0, 1], True),
        ("duplicate", [0, 1, 4], False),
        ("reversed", [2, 3], False),
    ):
        cases.append(
            {
                "id": name,
                "kind": name,
                "level": 16,
                "visible_panels": indices,
                "occluded": occluded,
            }
        )
    protocol = {
        "schema": "bb8.visual-fixture-preflight.v2",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stationary normal native RGB rendering/perception gate; manual resource levels, no interaction-learning claim.",
        "asset_dir": str(asset_dir),
        "evaluation_split": "validation",
        "bundle_sha256": file_hash(asset_dir / "bundle.json"),
        "sources_sha256": {name: file_hash(ROOT / name) for name in SOURCES},
        "fixture": VisualFixture(panels).rendering_contract(),
        "patterns": ID_PATTERNS,
        "semantic_camera_ids": ["B", "C"],
        "semantic_camera_selection": "Registered B/C retain both floor panels in view; development v1 camera A was physically occluded by an existing obstacle.",
        "semantic_resolution_scale": 2,
        "navigation_camera_ids": ["A", "B", "C"],
        "registered_cameras": cameras,
        "capture_step_simulation_s": 0.5,
        "cases": cases,
        "occluder": {
            "position": [0.15, 0.7, 0.75],
            "size": [3.3, 2.4, 0.06],
            "scope": "Opaque noncolliding native box; visual vertices hidden below ground except occlusion case.",
        },
        "pass_criteria": {
            "all_level_frames_decode_both_correct_ids_and_levels": True,
            "anchor_error_within_reported_radius": True,
            "absent_and_occluded_frames_have_no_readings": True,
            "duplicate_ids_abstain": True,
            "reversed_ids_follow_pixels": True,
            "normal_navigation_observer_visible_for_all_levels": True,
            "max_navigation_position_error_m": 0.06,
            "unchanged_scene_guard_stable_for_all_levels": True,
            "no_requested_motion": True,
            "max_body_displacement_m": 0.005,
            "no_contacts_or_boundary_failure": True,
        },
        "limitations": [
            "A renderer probe does not prove an authorized action caused an observed outcome or updated memory.",
            "Dedicated semantic images use scaled registered intrinsics, never live simulator extrinsics.",
            "RGB marker decoding is fixture-specific and nonphysical; it is not generic semantic understanding.",
            "Manual all-level images are development preflight; later independent challenge captures remain required.",
        ],
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    return protocol


def validate_protocol(protocol, asset_dir):
    record = dict(protocol)
    recorded_hash = record.pop("protocol_sha256", None)
    if (
        record.get("schema") != "bb8.visual-fixture-preflight.v2"
        or canonical_hash(record) != recorded_hash
    ):
        raise ValueError("Frozen preflight protocol checksum mismatch")
    if (
        str(asset_dir.resolve()) != record["asset_dir"]
        or file_hash(asset_dir / "bundle.json") != record["bundle_sha256"]
    ):
        raise ValueError("Asset bundle differs from frozen preflight")
    for name, digest in record["sources_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Source changed after preflight freeze: {name}")
    validate_assets(asset_dir)


class PreflightFixture(VisualFixture):
    """An ordinary opaque native occluder, manipulated only as visual geometry."""

    def __init__(self, protocol, **kwargs):
        panels = tuple(
            VisualPanel(p["visual_id"], tuple(p["center_xy"]))
            for p in protocol["fixture"]["panels_scoring_only"]
        )
        super().__init__(panels, **kwargs)
        self.occluder_spec = protocol["occluder"]

    def attach(self, scene, gs):
        super().attach(scene, gs)
        self.occluder = scene.add_entity(
            name="visual_preflight_opaque_cover",
            morph=gs.morphs.Box(
                pos=self.occluder_spec["position"],
                size=self.occluder_spec["size"],
                fixed=True,
                collision=False,
                visualization=True,
                enable_custom_vverts=True,
            ),
            surface=gs.surfaces.Default(color=(0.18, 0.18, 0.18, 1.0), roughness=1.0),
        )

    def initialize_visuals(self):
        super().initialize_visuals()
        self.cover_vertices = self.occluder.get_vverts().detach().cpu().numpy().copy()
        self.set_occluded(False)

    def set_occluded(self, enabled):
        vertices = self.cover_vertices.copy()
        if not enabled:
            vertices[:, 2] -= 2.0
        self.occluder.set_vverts(vertices)


def calibration(record, scale):
    from bb8_rl.camera import Calibration

    intrinsics = np.array(record["intrinsics"], dtype=float)
    intrinsics[:2] *= scale
    return Calibration(
        intrinsics,
        np.array(record["world_to_camera"], dtype=float),
        tuple(scale * value for value in record["resolution"]),
        2.0,
        provenance="RGB-estimated registered pose; scaled synthetic intrinsics; no live extrinsics",
    )


def score_pixels(observation, case, panels):
    if case["kind"] in ("absent", "occluded", "duplicate"):
        return {
            "expected_abstention": not observation.readings
            and observation.status in ("missing", "ambiguous")
        }
    expected = {
        panels[i].visual_id: panels[i].center_xy for i in case["visible_panels"]
    }
    observed = {reading.marker_id: reading for reading in observation.readings}
    return {
        "both_identities": observation.status == "visible"
        and set(expected) == set(observed),
        "gauge_correct": bool(observed)
        and all(reading.gauge_level == case["level"] for reading in observed.values()),
        "anchor_within_declared_radius": bool(observed)
        and all(
            name in expected
            and np.linalg.norm(np.array(reading.center_xy) - expected[name])
            <= reading.radius_m
            for name, reading in observed.items()
        ),
    }


def capture(args, protocol):
    # Native/model imports deliberately occur only after protocol/asset validation.
    import cv2
    import torch

    from bb8_rl.camera_rig import CameraFrame, CameraRig, CameraView
    from bb8_rl.env import NavigationEnv
    from bb8_rl.run_identity import capture_run_identity, verify_run_identity
    from bb8_rl.scene_validity import SceneValidityGuard
    from bb8_rl.vision import LearnedObserver, PositionOnlyObserver
    from bb8_rl.visual_interaction import PixelVisualObserver

    torch.set_num_threads(2)
    asset_dir = args.asset_dir.resolve()
    demo = validate_assets(asset_dir)
    cameras = protocol["registered_cameras"]
    scale = protocol["semantic_resolution_scale"]
    camera_specs = [
        VisualCamera(
            f"V{name}",
            tuple(cameras[name]["position"]),
            tuple(cameras[name]["lookat"]),
            tuple(scale * value for value in cameras[name]["resolution"]),
        )
        for name in protocol["semantic_camera_ids"]
    ]
    fixture = PreflightFixture(protocol, camera_specs=camera_specs)
    report = {
        "schema": "bb8.visual-fixture-preflight-result.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "running",
        "scope": protocol["scope"],
        "motion_commands_requested": 0,
        "cases": [],
        "limitations": protocol["limitations"],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    env = None
    try:
        env = NavigationEnv(**environment_kwargs(asset_dir, demo, cameras, fixture))
        env.reset(
            seed=demo["seed"],
            options={
                "layout_seed": demo["layout_seed"],
                "start": demo["start"],
                "goal": demo["goal"],
            },
        )
        world = env.world
        world.camera_period = world._next_frame = 1e9
        # Registration belongs to the declared pose, not the Project sensor's
        # look-at default. Establish all poses before acquiring any reference.
        for name, camera in world.cameras.items():
            camera.set_pose(
                pos=cameras[name]["position"], lookat=cameras[name]["lookat"]
            )
        fixture.set_visible_panels((0, 1))
        fixture.set_occluded(False)
        nav_cal = {name: calibration(cameras[name], 1) for name in "ABC"}
        versions = {name: cameras[name]["calibration_version"] for name in "ABC"}
        views = [
            CameraView(
                name,
                versions[name],
                nav_cal[name],
                PositionOnlyObserver(
                    LearnedObserver(
                        nav_cal[name],
                        asset_dir / demo["vision"],
                        floor_refinement="guided",
                    ),
                ),
            )
            for name in "ABC"
        ]
        rig = CameraRig(views)
        guard = SceneValidityGuard(nav_cal, versions)
        observers = {
            name: PixelVisualObserver(
                calibration(cameras[name], scale),
                f"V{name}",
                versions[name] + "-scaled2",
            )
            for name in protocol["semantic_camera_ids"]
        }
        initial = np.array(world.backend.read_planar_state(now=world.time).position)
        identity = capture_run_identity(
            args.output / "identity",
            asset_dir=asset_dir,
            task_path=asset_dir / demo["task"],
        )
        report["run_identity"] = identity
        report["loaded_setup"] = {
            "project": world.project.model_dump(mode="json"),
            "fixture": fixture.rendering_contract(),
            "camera_specs": [asdict(spec) for spec in camera_specs],
        }
        report["body_initial_xy_scoring_only"] = initial.tolist()
        for case in protocol["cases"]:
            fixture.set_visible_panels(case["visible_panels"])
            fixture.set_resource(case["level"] / GAUGE_BINS)
            fixture.set_occluded(case["occluded"])
            state = world.step(
                max(1, round(protocol["capture_step_simulation_s"] / world.dt))
            )
            now = world.time
            result = {
                "case": case["id"],
                "kind": case["kind"],
                "capture_time": now,
                "semantic": {},
                "navigation": {},
                "body_xy_scoring_only": list(state.position),
            }
            for name, observer in observers.items():
                rgb = fixture.cameras[f"V{name}"].render(rgb=True, force_render=True)[0]
                frame = CameraFrame(f"V{name}", versions[name] + "-scaled2", rgb, now)
                path = args.output / f"{case['id']}-V{name}-rgb.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError("Could not save native semantic RGB")
                observation = observer.observe(frame, now=now)
                result["semantic"][name] = {
                    "image": path.name,
                    "image_sha256": file_hash(path),
                    "observation": asdict(observation),
                    "checks": score_pixels(observation, case, fixture.panels),
                }
            frames = []
            for name, camera in world.cameras.items():
                rgb = camera.render(rgb=True, force_render=True)[0]
                frames.append(CameraFrame(name, versions[name], rgb, now))
                path = args.output / f"{case['id']}-{name}-nav-rgb.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError("Could not save native navigation RGB")
                result["navigation"][name] = {
                    "image": path.name,
                    "image_sha256": file_hash(path),
                }
            observed = rig.observe(frames, now=now)
            validity = guard.observe(frames, observed.views, now=now)
            result.update(
                navigation_status=observed.status,
                scene_validity=validity,
                navigation_position=None
                if observed.xy is None
                else observed.xy.tolist(),
                navigation_error_m=None
                if observed.xy is None
                else float(np.linalg.norm(observed.xy - state.position)),
                navigation_view_status={
                    name: view.status for name, view in observed.views.items()
                },
            )
            report["cases"].append(result)
            (args.output / "progress.json").write_text(
                json.dumps(
                    {
                        "last_case": case["id"],
                        "completed": len(report["cases"]),
                        "requested": len(protocol["cases"]),
                    },
                    indent=2,
                )
                + "\n"
            )
        final = np.array(world.backend.read_planar_state(now=world.time).position)
        report["body_displacement_m"] = float(np.linalg.norm(final - initial))
        report["contacts"] = world.contacts
        report["boundary_failure"] = world.boundary_failure
        report["run_identity_verification"] = verify_run_identity(identity)
        levels = [row for row in report["cases"] if row["kind"] == "level"]
        report["checks"] = {
            "all_pixel_checks": all(
                all(view["checks"].values())
                for row in report["cases"]
                for view in row["semantic"].values()
            ),
            "all_level_navigation_visible": all(
                row["navigation_status"] == "visible" for row in levels
            ),
            "all_level_navigation_position_accurate": all(
                row["navigation_error_m"] is not None
                and row["navigation_error_m"]
                <= protocol["pass_criteria"]["max_navigation_position_error_m"]
                for row in levels
            ),
            "all_level_scene_guard_stable": all(
                row["scene_validity"]["status"] == "stable" for row in levels
            ),
            "stationary": report["body_displacement_m"]
            <= protocol["pass_criteria"]["max_body_displacement_m"],
            "no_contacts_or_boundary": not world.contacts
            and not world.boundary_failure,
            "identity_unchanged": report["run_identity_verification"]["unchanged"],
        }
        report["passed"] = all(report["checks"].values())
        report["status"] = "complete"
    except BaseException:
        report.update(status="failed", passed=False, error=traceback.format_exc())
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except BaseException:
            report.update(teardown_error=traceback.format_exc(), passed=False)
            raise
        finally:
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "passed", "checks")}, indent=2
        )
    )
    return 0 if report["passed"] else 1


def main(args):
    if args.freeze:
        if args.protocol or args.output:
            raise ValueError("Freeze and capture are separate commands")
        protocol = make_protocol(args.asset_dir)
        args.freeze.parent.mkdir(parents=True, exist_ok=True)
        with args.freeze.open("x") as handle:
            handle.write(json.dumps(protocol, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "protocol": str(args.freeze),
                    "sha256": protocol["protocol_sha256"],
                    "native_run": False,
                    "semantic_frames": len(protocol["cases"]) * 2,
                }
            )
        )
        return 0
    if not args.protocol or not args.output:
        raise ValueError("Capture requires a frozen --protocol and fresh --output")
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol, args.asset_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    return capture(args, protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir", type=Path, default=ROOT / "work/interactive-assets"
    )
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    raise SystemExit(main(parser.parse_args()))
