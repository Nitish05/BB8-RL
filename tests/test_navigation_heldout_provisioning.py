"""Held-out provisioner keeps frozen scene inputs and query isolation explicit."""

import builtins
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heldout_provisioning", ROOT / "scripts/provision-navigation-heldout.py"
)
provision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provision)


def test_import_plan_and_scene_are_data_only(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "genesis",
            "genesis_studio",
            "torch",
            "gymnasium",
            "stable_baselines3",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    spec = module.protocol()
    for family in spec["families"]:
        assert len(module.acquisition_plan(spec, family)["views"]) == 36


@pytest.mark.parametrize("family", tuple(provision.protocol()["families"]))
def test_scene_changes_match_frozen_geometry_and_palette_without_mutating_template(
    family,
):
    spec = provision.protocol()
    original = {
        "schema_version": 4,
        "name": "fixture",
        "seed": 42,
        "objects": [
            {
                "name": "bb8_body",
                "format": "sphere",
                "radius": 0.037,
                "fixed": False,
                "material": {"color": [0.1, 0.2, 0.3, 1], "density": 940},
            },
            {
                "name": "bb8_head",
                "format": "sphere",
                "radius": 0.019,
                "fixed": True,
                "material": {"color": [0.7, 0.7, 0.7, 1]},
            },
            {"name": "wall_0", "format": "box", "position": [-2, 0, 0.1]},
            {"name": "floor_0", "format": "box", "position": [0, 0, 0]},
            {"name": "room_obstacle_99", "format": "box", "position": [1, 1, 1]},
        ],
        "sensors": [{"name": "navigation_camera", "type": "camera", "fov": 58}],
    }
    original_copy = copy.deepcopy(original)
    result = provision.family_scene(original, spec, family)
    layout = spec["layouts"][spec["families"][family]["layout"]]
    colors = spec["appearances"][spec["families"][family]["appearance"]]
    obstacles = [o for o in result["objects"] if o["name"].startswith("room_obstacle_")]
    assert len(obstacles) == 3
    for actual, expected in zip(obstacles, layout["obstacles"], strict=True):
        assert {k: actual[k] for k in expected} == expected
        assert actual["material"]["color"] == colors["room_obstacle_*"]
    for obj in result["objects"]:
        if obj["name"].startswith("wall_") or obj["name"].startswith("floor_"):
            assert obj == next(
                o for o in original["objects"] if o["name"] == obj["name"]
            )
        if obj["name"] in ("bb8_body", "bb8_head"):
            base = next(o for o in original["objects"] if o["name"] == obj["name"])
            assert {k: v for k, v in obj.items() if k != "material"} == {
                k: v for k, v in base.items() if k != "material"
            }
            assert obj["material"]["color"] == colors[obj["name"]]
    assert original == original_copy


@pytest.mark.parametrize("family", tuple(provision.protocol()["families"]))
def test_plan_excludes_all_queries_and_uses_frozen_runtime_camera_poses(family):
    spec = provision.protocol()
    plan = provision.acquisition_plan(spec, family)
    assert len({v["id"] for v in plan["views"]}) == 36
    assert sum(v["role"] == "mapping" for v in plan["views"]) == 32
    assert sum(v["role"] == "query" for v in plan["views"]) == 4
    for name, index in provision.CAMERA_QUERIES.items():
        camera = spec["camera_sets"][spec["families"][family]["camera_set"]][name]
        view = next(v for v in plan["views"] if v["index"] == index)
        assert (
            view["role"] == "query"
            and view["position"] == camera["render_position"]
            and view["lookat"] == camera["render_lookat"]
        )
    assert plan["memory_settings"] == provision.MEMORY_SETTINGS
    assert (
        plan["depth_used"]
        is plan["segmentation_used"]
        is plan["original_map_reused"]
        is False
    )


def fake_capture(tmp_path):
    plan = provision.acquisition_plan(provision.protocol(), "north_oblique_cool")
    (tmp_path / "scan").mkdir()
    (tmp_path / "scene").mkdir()
    provision.write(tmp_path / "scan-plan.json", plan)
    provision.write(
        tmp_path / "scene/room.genesis.json", {"fixture": "not a native scene"}
    )
    manifest = {
        "status": "complete",
        "family": plan["family"],
        "evaluation_split": "heldout",
        "seed": plan["seed"],
        "layout_seed": plan["layout_seed"],
        "protocol_sha256": provision.PROTOCOL_SHA256,
        "plan_sha256": provision.digest(tmp_path / "scan-plan.json"),
        "project_sha256": provision.digest(tmp_path / "scene/room.genesis.json"),
        "input_sha256": {},
        "query_transforms_in_scan_metadata": False,
        "depth_used": False,
        "segmentation_used": False,
        "run_identity_verification": {"unchanged": True},
    }
    for view in plan["views"]:
        stem = tmp_path / "scan" / view["id"]
        stem.with_suffix(".png").write_bytes(b"opaque test rgb")
        meta = provision.scan_metadata(
            view,
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        provision.write(stem.with_suffix(".json"), meta)
        manifest["input_sha256"][view["id"]] = {
            suffix: provision.digest(stem.with_suffix(suffix))
            for suffix in (".png", ".json")
        }
    provision.write(tmp_path / "scan/manifest.json", manifest)
    return plan, manifest


@pytest.mark.parametrize(
    "defect",
    (
        "rgb",
        "query_transform",
        "query_position",
        "query_lookat",
        "role",
        "project",
        "status",
        "identity",
        "split",
        "seed",
        "layout_seed",
    ),
)
def test_capture_gate_rejects_changed_evidence_or_query_pose_leak(tmp_path, defect):
    plan, manifest = fake_capture(tmp_path)
    assert provision.checked_capture(tmp_path, plan) == manifest
    if defect == "rgb":
        (tmp_path / "scan/scan-000.png").write_bytes(b"changed")
    elif defect in ("query_transform", "query_position", "query_lookat", "role"):
        p = tmp_path / "scan/scan-020.json"
        meta = json.loads(p.read_text())
        if defect == "query_transform":
            meta["world_to_camera"] = [[1] * 4] * 4
        elif defect == "query_position":
            meta["position"] = [4, -3, 3]
        elif defect == "query_lookat":
            meta["lookat"] = [0, 0, 0.1]
        else:
            meta["role"] = "mapping"
        provision.write(p, meta)
        manifest["input_sha256"]["scan-020"][".json"] = provision.digest(p)
        provision.write(tmp_path / "scan/manifest.json", manifest)
    elif defect == "project":
        (tmp_path / "scene/room.genesis.json").write_text("{}")
    else:
        if defect == "status":
            manifest["status"] = "failed"
        elif defect == "split":
            manifest["evaluation_split"] = "validation"
        elif defect in ("seed", "layout_seed"):
            manifest[defect] = 12345
        else:
            manifest["run_identity_verification"]["unchanged"] = False
        provision.write(tmp_path / "scan/manifest.json", manifest)
    with pytest.raises(ValueError):
        provision.checked_capture(tmp_path, plan)


@pytest.mark.parametrize(
    "defect", ("rgb", "metadata", "query_consumed", "map_member", "map_unfrozen")
)
def test_registration_must_use_new_family_images_and_exclude_queries(tmp_path, defect):
    _, capture = fake_capture(tmp_path)
    registration = {
        "status": "complete",
        "mapping_input_view_ids": [
            f"scan-{i:03}" for i in range(24) if i not in provision.QUERIES
        ],
        "query_indices": list(provision.QUERIES),
        "query_pose_consumed": False,
        "map_frozen_before_query_matching": True,
    }
    for key, names in (
        ("mapping_inputs", registration["mapping_input_view_ids"]),
        ("query_inputs", [f"scan-{i:03}" for i in provision.QUERIES]),
    ):
        registration[key] = {
            name: {
                "rgb_sha256": capture["input_sha256"][name][".png"],
                "metadata_sha256": capture["input_sha256"][name][".json"],
            }
            for name in names
        }
    provision.registration_ancestry(registration, capture)
    if defect == "rgb":
        registration["mapping_inputs"]["scan-000"]["rgb_sha256"] = "0" * 64
    elif defect == "metadata":
        registration["query_inputs"]["scan-020"]["metadata_sha256"] = "0" * 64
    elif defect == "query_consumed":
        registration["query_pose_consumed"] = True
    elif defect == "map_member":
        registration["mapping_input_view_ids"].append("scan-020")
    else:
        registration["map_frozen_before_query_matching"] = False
    with pytest.raises(ValueError):
        provision.registration_ancestry(registration, capture)


def test_query_scan_metadata_contains_no_recoverable_render_pose():
    plan = provision.acquisition_plan(provision.protocol(), "north_oblique_cool")
    for view in plan["views"]:
        if view["role"] != "query":
            continue
        metadata = provision.scan_metadata(
            view, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], object()
        )
        assert set(metadata) == {
            "id",
            "index",
            "role",
            "intrinsics",
            "calibration",
            "depth_used",
            "segmentation_used",
        }
        assert not {
            "position",
            "lookat",
            "euler",
            "world_to_camera",
            "camera_to_world",
        } & set(metadata)


def safe_map_report():
    full = {
        "free_prisms": 4,
        "false_free_solid_intersection_voxels": 0,
        "free_solid_intersection_prisms": 0,
        "free_voxels_outside_room_or_below_floor": 0,
        "free_sources_rgb_only": True,
        "heldout_query_in_map": [],
    }
    return full, {"pass": True}, {"hashes": True, "support": True}


@pytest.mark.parametrize(
    "defect",
    (
        "voxels",
        "prisms",
        "outside",
        "empty",
        "query",
        "source",
        "robot",
        "evidence",
        "absent",
    ),
)
def test_independent_map_gate_fails_closed_without_repairing_evidence(defect):
    full, robot, evidence = safe_map_report()
    assert all(provision.map_admission_checks(full, robot, evidence).values())
    if defect in ("voxels", "prisms", "outside", "empty"):
        field = {
            "voxels": "false_free_solid_intersection_voxels",
            "prisms": "free_solid_intersection_prisms",
            "outside": "free_voxels_outside_room_or_below_floor",
            "empty": "free_prisms",
        }[defect]
        full[field] = 0 if defect == "empty" else 1
    elif defect == "query":
        full["heldout_query_in_map"] = ["scan-020"]
    elif defect == "source":
        full["free_sources_rgb_only"] = False
    elif defect == "robot":
        robot["pass"] = False
    elif defect == "absent":
        full.pop("false_free_solid_intersection_voxels")
    else:
        evidence["hashes"] = False
    before = copy.deepcopy((full, robot, evidence))
    assert not all(provision.map_admission_checks(full, robot, evidence).values())
    assert (full, robot, evidence) == before


def test_map_gate_receipt_rejects_changed_map_and_forged_pass(tmp_path):
    full, robot, evidence = safe_map_report()
    (tmp_path / "map-audit").mkdir()
    for name in provision.MAP_AUDIT_INPUTS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"frozen evidence")
    report = {
        "schema": "bb8.heldout-map-audit.v1",
        "protocol_sha256": provision.PROTOCOL_SHA256,
        "native_trial_gate_pass": True,
        "controller_map_modified": False,
        "full_volume": full,
        "parked_robot": robot,
        "evidence": evidence,
        "checks": provision.map_admission_checks(full, robot, evidence),
        "input_sha256": {
            name: provision.digest(tmp_path / name)
            for name in provision.MAP_AUDIT_INPUTS
        },
    }
    provision.write(tmp_path / "map-audit/report.json", report)
    assert provision.checked_map_audit(tmp_path) == report
    (tmp_path / "memory/free-grid.npz").write_bytes(b"modified map")
    with pytest.raises(ValueError, match="inputs changed"):
        provision.checked_map_audit(tmp_path)
    report["full_volume"]["false_free_solid_intersection_voxels"] = 1
    provision.write(tmp_path / "map-audit/report.json", report)
    with pytest.raises(ValueError, match="safety gate"):
        provision.checked_map_audit(tmp_path)


def test_registration_interpreter_preserves_virtual_environment_symlink(tmp_path):
    base = tmp_path / "base/python"
    base.parent.mkdir()
    base.write_bytes(b"interpreter fixture")
    runtime = tmp_path / "matching"
    selected = runtime / "venv/bin/python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(base)
    assert provision.registration_interpreter(runtime) == selected.absolute()
    assert provision.registration_interpreter(runtime) != base.resolve()
    assert provision.registration_interpreter(runtime, selected) == selected
    with pytest.raises(ValueError, match="preserved local"):
        provision.registration_interpreter(tmp_path / "missing")


def test_capture_camera_up_is_explicit_and_independent_of_view_order():
    import numpy as np

    class Camera:
        def __init__(self):
            self.up = np.array([0.0, 0.0, 1.0])
            self.calls = []

        def set_pose(self, *, pos, lookat, up=None):
            self.calls.append(up)
            forward = np.array(lookat) - np.array(pos)
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, self.up if up is None else up)
            right /= np.linalg.norm(right)
            self.up = np.cross(right, forward)
            self.rotation = np.stack((right, -self.up, forward))

    plan = provision.acquisition_plan(provision.protocol(), "north_oblique_cool")
    sequence = Camera()
    for view in plan["views"]:
        provision.set_acquisition_pose(sequence, view)
        fresh = Camera()
        provision.set_acquisition_pose(fresh, view)
        np.testing.assert_allclose(sequence.rotation, fresh.rotation, atol=1e-12)
        assert sequence.calls[-1] == [0.0, 0.0, 1.0]
    assert plan["renderer_orientation"]["normal_up"] == [0.0, 0.0, 1.0]
    # This negative control models Genesis's retained orthogonalized _up.
    retained = Camera()
    for view in plan["views"][:21]:
        retained.set_pose(pos=view["position"], lookat=view["lookat"])
    fresh = Camera()
    provision.set_acquisition_pose(fresh, plan["views"][20])
    assert np.linalg.norm(retained.rotation - fresh.rotation) > 0.1


def test_vertical_camera_up_fallback_does_not_change_query_isolation():
    assert provision.acquisition_up([0, 0, 3], [0, 0, 0]) == [0.0, 1.0, 0.0]
    assert provision.acquisition_up([0, 0, 3], [0, 0.05, 0]) == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        provision.acquisition_up([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        provision.acquisition_up([0, 0, float("nan")], [0, 0, 0])
    query = {
        "id": "scan-020",
        "index": 20,
        "role": "query",
        "position": [4, -2.8, 3.3],
        "lookat": [0, 0, 0.1],
    }
    metadata = provision.scan_metadata(
        query, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], object()
    )
    assert not {"position", "lookat", "up", "world_to_camera"} & metadata.keys()
