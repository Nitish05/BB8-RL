"""Portable asset reconstruction with opaque, deliberately unloadable model bytes."""

import builtins
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from bb8_rl.demo_assets import sha256, validate_assets

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "asset_builder", ROOT / "scripts/build-interactive-assets.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return {"path": relative, "sha256": sha256(path)}


def write_json(root, relative, value):
    return write(root, relative, json.dumps(value))


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "source"
    records = {
        "policy": write(root, "models/policy.zip", b"opaque policy, not a zip"),
        "vision": write(root, "models/vision.pt", b"opaque vision, not a pickle"),
        "task": write(root, "config/task.yaml", "world_config: ../world/room.yaml\n"),
        "world": write(
            root,
            "world/room.yaml",
            "project: ../scene/room.genesis.json\nnotes: fixture\n",
        ),
        "project": write_json(
            root, "scene/room.genesis.json", {"schema_version": 4, "name": "fixture"}
        ),
    }
    transform = [[int(row == col) for col in range(4)] for row in range(4)]
    records["registration"] = write_json(
        root,
        "evidence/registration.json",
        {
            "landmarks_sha256": "f" * 64,
            "queries": [
                {
                    "query_index": query,
                    "status": "candidate",
                    "world_to_camera": transform,
                }
                for query in range(3)
            ],
        },
    )
    records["protocol"] = write_json(
        root,
        "evidence/protocol.json",
        {
            "scene_sha256": records["project"]["sha256"],
            "registration_report_sha256": records["registration"]["sha256"],
            "cases": [{"start": [0, 0], "goal": [0, 1], "seed": 1, "layout_seed": 2}],
        },
    )
    artifacts = {
        name: write(root, f"memory/{name}", content)
        for name, content in {
            "room-memory.json": b"{}",
            "free-grid.npz": b"opaque arrays, never loaded",
            "evidence/floor-mask.npz": b"opaque extra evidence",
        }.items()
    }
    records["memory_manifest"] = write_json(
        root,
        "memory/manifest.json",
        {
            "artifact_sha256": {
                name: record["sha256"] for name, record in artifacts.items()
            },
            "scan_dir": "/old/machine/source-scans",
            "free_evidence_semantics": "synthetic fixture only",
        },
    )
    cameras = {}
    for query, camera_id in enumerate("ABC"):
        metadata = write_json(
            root,
            f"calibration/scan-{query:03d}.json",
            {
                "index": query,
                "position": [query, 0, 1],
                "intrinsics": [[32, 0, 32], [0, 32, 24], [0, 0, 1]],
            },
        )
        cameras[camera_id] = {
            "query_index": query,
            "metadata": metadata,
            "lookat": [0, 0, 0.1],
            "resolution": [64, 48],
        }
    manifest = {
        "schema": builder.INPUT_SCHEMA,
        "root": ".",
        "inputs": records,
        "memory_artifacts": artifacts,
        "cameras": cameras,
        "protocol_case": 0,
    }
    write_json(root, "inputs.json", manifest)
    return root, manifest


def args(root, output, **overrides):
    return SimpleNamespace(
        **{
            "inputs": root / "inputs.json",
            "input_root": None,
            "scan_dir": None,
            "memory": None,
            "output": output,
            "write_input_manifest": None,
            "validate_inputs_only": False,
            **overrides,
        }
    )


def save_manifest(root, manifest):
    (root / "inputs.json").write_text(json.dumps(manifest))


def set_pixel_convention(root, manifest, value):
    for record in [
        *(camera["metadata"] for camera in manifest["cameras"].values()),
        *(
            manifest["inputs"][name]
            for name in ("memory_manifest", "registration", "protocol")
        ),
    ]:
        data = json.loads((root / record["path"]).read_text())
        data["pixel_coordinates"] = value
        record["sha256"] = write_json(root, record["path"], data)["sha256"]
    protocol = manifest["inputs"]["protocol"]
    data = json.loads((root / protocol["path"]).read_text())
    data["registration_report_sha256"] = manifest["inputs"]["registration"]["sha256"]
    protocol["sha256"] = write_json(root, protocol["path"], data)["sha256"]
    save_manifest(root, manifest)


def test_explicit_pixel_convention_preserved_without_mutating_estimated_intrinsics(
    inputs, tmp_path
):
    root, manifest = inputs
    set_pixel_convention(root, manifest, "opencv_integer_center")
    result = builder.validate_inputs(manifest, root)
    for name, camera in result["cameras"].items():
        original = json.loads(
            (root / manifest["cameras"][name]["metadata"]["path"]).read_text()
        )
        assert camera["pixel_coordinates"] == "opencv_integer_center"
        assert camera["intrinsics"] == original["intrinsics"]
    output = tmp_path / "tagged"
    builder.main(args(root, output))
    assert all(
        camera["pixel_coordinates"] == "opencv_integer_center"
        for camera in validate_assets(output)["cameras"].values()
    )


@pytest.mark.parametrize(
    "target,value",
    [
        ("A", "genesis_viewport"),
        ("B", "unknown"),
        ("C", None),
        ("memory_manifest", "genesis_viewport"),
        ("registration", "genesis_viewport"),
        ("protocol", "genesis_viewport"),
    ],
)
def test_unknown_or_mixed_pixel_convention_rejected_before_any_output(
    inputs, tmp_path, target, value
):
    root, manifest = inputs
    set_pixel_convention(root, manifest, "opencv_integer_center")
    record = (
        manifest["cameras"][target]["metadata"]
        if target in "ABC"
        else manifest["inputs"][target]
    )
    data = json.loads((root / record["path"]).read_text())
    data["pixel_coordinates"] = value
    record["sha256"] = write_json(root, record["path"], data)["sha256"]
    save_manifest(root, manifest)
    output = tmp_path / "not-created" / "bundle"
    with pytest.raises(ValueError, match="pixel_coordinates"):
        builder.main(args(root, output))
    assert not output.parent.exists()


def test_legacy_absence_adds_no_pixel_coordinate_fields(inputs):
    root, manifest = inputs
    result = builder.validate_inputs(manifest, root)
    assert all(
        "pixel_coordinates" not in camera for camera in result["cameras"].values()
    )
    assert all(
        "pixel_coordinates" not in value
        for value in [result["map_manifest"], result["case"]]
    )


def test_relocated_inputs_rebuild_identical_archives_without_loading_weights(
    inputs, tmp_path, monkeypatch
):
    root, _ = inputs
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    os.utime(relocated / "models/policy.zip", (0, 0))
    original_import = builtins.__import__

    def guarded_import(name, *positional, **keyword):
        assert name.split(".")[0] not in {
            "torch",
            "numpy",
            "genesis",
            "genesis_studio",
            "genesis_studio_desktop",
            "stable_baselines3",
            "cv2",
        }
        return original_import(name, *positional, **keyword)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    first, second = tmp_path / "first", tmp_path / "second"
    builder.main(args(root, first))
    builder.main(args(root, second, input_root=relocated))
    assert sha256(first.with_suffix(".zip")) == sha256(second.with_suffix(".zip"))
    demo = validate_assets(first)
    assert demo["task"] == "scene/task.yaml"
    assert demo["cameras"]["C"]["query_index"] == 2
    assert (first / "policy.zip").read_bytes() == b"opaque policy, not a zip"
    assert "/old/machine" not in (first / "memory/manifest.json").read_text()
    assert (
        json.loads((first / "input-manifest.json").read_text())["root"]
        == "<INPUT_ROOT>"
    )


@pytest.mark.parametrize("name", sorted(builder.INPUT_NAMES) + ["artifact", "camera"])
def test_any_source_checksum_failure_leaves_no_output(inputs, tmp_path, name):
    root, manifest = inputs
    record = (
        manifest["memory_artifacts"]["free-grid.npz"]
        if name == "artifact"
        else manifest["cameras"]["A"]["metadata"]
        if name == "camera"
        else manifest["inputs"][name]
    )
    (root / record["path"]).write_bytes(b"changed after input manifest capture")
    output = tmp_path / "fresh-parent" / "bundle"
    with pytest.raises(ValueError, match="checksum"):
        builder.main(args(root, output))
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "defect", ["escape", "absolute", "symlink", "missing", "noncanonical"]
)
def test_source_paths_are_contained_regular_files(inputs, tmp_path, defect):
    root, manifest = inputs
    record = manifest["inputs"]["policy"]
    if defect == "escape":
        record["path"] = "../outside.zip"
    elif defect == "absolute":
        record["path"] = str(root / record["path"])
    elif defect == "symlink":
        (root / record["path"]).unlink()
        (root / record["path"]).symlink_to(root / "models/vision.pt")
    elif defect == "missing":
        (root / record["path"]).unlink()
    else:
        record["path"] = "models/../models/policy.zip"
    save_manifest(root, manifest)
    expected = {
        "escape": "inside",
        "absolute": "inside",
        "symlink": "Links",
        "missing": "Missing",
        "noncanonical": "canonical",
    }
    with pytest.raises(ValueError, match=expected[defect]):
        builder.main(args(root, tmp_path / "output"))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "defect",
    [
        "graph",
        "missing_artifact",
        "wrong_case",
        "rejected_camera",
        "wrong_scan",
        "primitive_world",
        "protocol_scene",
    ],
)
def test_semantic_input_mismatch_is_rejected_before_output(inputs, tmp_path, defect):
    root, manifest = inputs
    if defect == "graph":
        manifest["inputs"]["task"] = write(
            root, "config/task.yaml", "world_config: ../other.yaml\n"
        )
    elif defect == "missing_artifact":
        del manifest["memory_artifacts"]["free-grid.npz"]
    elif defect == "wrong_case":
        manifest["protocol_case"] = 10
    elif defect == "rejected_camera":
        data = json.loads(
            (root / manifest["inputs"]["registration"]["path"]).read_text()
        )
        data["queries"][0]["status"] = "rejected"
        manifest["inputs"]["registration"] = write_json(
            root, "evidence/registration.json", data
        )
        protocol = json.loads((root / "evidence/protocol.json").read_text())
        protocol["registration_report_sha256"] = manifest["inputs"]["registration"][
            "sha256"
        ]
        manifest["inputs"]["protocol"] = write_json(
            root, "evidence/protocol.json", protocol
        )
    elif defect == "wrong_scan":
        manifest["cameras"]["A"]["query_index"] = 1
    elif defect == "primitive_world":
        manifest["inputs"]["project"] = write_json(
            root,
            "scene/room.genesis.json",
            {
                "schema_version": 4,
                "name": "invalid external mesh",
                "objects": [
                    {"name": "mesh", "format": "mesh", "source": "elsewhere.obj"}
                ],
            },
        )
    else:
        protocol = json.loads((root / "evidence/protocol.json").read_text())
        protocol["scene_sha256"] = "0" * 64
        manifest["inputs"]["protocol"] = write_json(
            root, "evidence/protocol.json", protocol
        )
    save_manifest(root, manifest)
    expected = {
        "graph": "configuration graph",
        "missing_artifact": "exactly cover",
        "wrong_case": "case index",
        "rejected_camera": "registration candidate",
        "wrong_scan": "metadata index",
        "primitive_world": "primitive-only",
        "protocol_scene": "scene_sha256",
    }
    with pytest.raises(ValueError, match=expected[defect]):
        builder.main(args(root, tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_validation_only_can_capture_and_reuse_manifest_without_bundle(
    inputs, tmp_path
):
    root, manifest = inputs
    captured = tmp_path / "captured.json"
    output = tmp_path / "unused-output"
    builder.main(
        args(root, output, write_input_manifest=captured, validate_inputs_only=True)
    )
    assert not output.exists() and not output.with_suffix(".zip").exists()
    saved = json.loads(captured.read_text())
    assert saved["inputs"] == manifest["inputs"]
    builder.main(args(root, output, inputs=captured, validate_inputs_only=True))
    assert not output.exists()


def test_existing_bundle_or_archive_is_never_overwritten(inputs, tmp_path):
    root, _ = inputs
    output = tmp_path / "existing"
    archive = output.with_suffix(".zip")
    archive.write_bytes(b"keep existing evidence")
    with pytest.raises(ValueError, match="already exist"):
        builder.main(args(root, output))
    assert archive.read_bytes() == b"keep existing evidence"
    assert not output.exists()


def test_legacy_invocation_captures_explicit_reusable_inputs(
    inputs, tmp_path, monkeypatch
):
    source, manifest = inputs
    project_root = tmp_path / "legacy-project"
    destinations = {
        "policy": "work/m6/sac-2/model.zip",
        "vision": "work/m75/calibrated/model.pt",
        "task": "projects/bb8/synthetic-room/task.yaml",
        "world": "projects/bb8/synthetic-room/room.yaml",
        "project": "projects/bb8/synthetic-room/room.genesis.json",
        "registration": "work/m77/registration/superpoint-lightglue/report.json",
        "protocol": "work/m78/supplemental-fixture/v6-camera20-r20-short/protocol.json",
    }
    for name, relative in destinations.items():
        write(
            project_root,
            relative,
            (source / manifest["inputs"][name]["path"]).read_bytes(),
        )
    write(project_root, destinations["task"], "world_config: room.yaml\n")
    write(
        project_root,
        destinations["world"],
        "project: room.genesis.json\nnotes: fixture\n",
    )
    registration = json.loads((project_root / destinations["registration"]).read_text())
    scan_dir = tmp_path / "external-scans"
    for row, camera, query in zip(
        registration["queries"], "ABC", (20, 23, 14), strict=True
    ):
        row["query_index"] = query
        metadata = json.loads(
            (source / manifest["cameras"][camera]["metadata"]["path"]).read_text()
        )
        metadata["index"] = query
        write_json(scan_dir, f"scan-{query:03d}.json", metadata)
    record = write_json(project_root, destinations["registration"], registration)
    protocol = json.loads((project_root / destinations["protocol"]).read_text())
    protocol["registration_report_sha256"] = record["sha256"]
    write_json(project_root, destinations["protocol"], protocol)
    shutil.copytree(source / "memory", project_root / "memory")
    monkeypatch.setattr(builder, "ROOT", project_root)
    captured = tmp_path / "captured-legacy.json"
    builder.main(
        args(
            source,
            None,
            inputs=None,
            scan_dir=scan_dir,
            memory=project_root / "memory",
            validate_inputs_only=True,
            write_input_manifest=captured,
        )
    )
    saved = json.loads(captured.read_text())
    assert set(saved["inputs"]) == builder.INPUT_NAMES
    assert saved["cameras"]["A"]["query_index"] == 20
    builder.main(args(source, None, inputs=captured, validate_inputs_only=True))
