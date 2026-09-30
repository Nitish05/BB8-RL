import builtins
import hashlib
import json
import stat
import zipfile

import pytest

from bb8_rl.demo_assets import checked_path, install_bundle, sha256, validate_assets


def write_bundle(root):
    """Complete dependency graph with deliberately non-deserializable weights."""
    demo = {
        "task": "scene/task.yaml",
        "memory": "memory",
        "policy": "policy.zip",
        "vision": "vision.pt",
    }
    artifacts = {
        "room-memory.json": b"{}",
        "free-grid.npz": b"array bytes, never loaded",
        "floor-masks.npz": b"extra declared evidence, never loaded",
    }
    files = {
        "demo.json": json.dumps(demo).encode(),
        "policy.zip": b"policy bytes, never deserialized",
        "vision.pt": b"vision bytes, never deserialized",
        "scene/task.yaml": b"world_config: room.yaml\n",
        "scene/room.yaml": b"project: room.genesis.json\nnotes: test fixture\n",
        "scene/room.genesis.json": b'{"schema_version":4,"name":"test fixture"}',
        "memory/manifest.json": json.dumps(
            {
                "artifact_sha256": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in artifacts.items()
                },
                # Historical inputs are descriptive, not live runtime inputs.
                "retained_baseline_memory_path": "<BB8-RL>/work/old-memory",
            }
        ).encode(),
        **{f"memory/{name}": content for name, content in artifacts.items()},
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": "bb8.interactive-assets.v1",
                "sha256": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in files.items()
                },
            }
        )
    )
    return demo


def change_asset(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    manifest = json.loads((root / "bundle.json").read_text())
    manifest["sha256"][name] = sha256(path)
    (root / "bundle.json").write_text(json.dumps(manifest))


def omit_checksum(root, name):
    manifest = json.loads((root / "bundle.json").read_text())
    del manifest["sha256"][name]
    (root / "bundle.json").write_text(json.dumps(manifest))


def test_checked_asset_paths_cannot_escape(tmp_path):
    for name in ("../outside", "/absolute", "sub/../../outside"):
        with pytest.raises(ValueError):
            checked_path(tmp_path, name)
    assert checked_path(tmp_path, "memory/grid.npz") == tmp_path / "memory/grid.npz"


def test_bad_release_hash_and_zip_traversal_leave_no_installation(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", "not an asset")
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="checksum"):
        install_bundle(archive, destination, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="inside"):
        install_bundle(archive, destination, expected_sha256=sha256(archive))
    assert not destination.exists() and not (tmp_path / "escape").exists()


def test_validation_does_not_import_model_or_native_loaders(tmp_path, monkeypatch):
    demo = write_bundle(tmp_path)
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch",
            "numpy",
            "genesis",
            "genesis_studio",
            "genesis_studio_desktop",
            "stable_baselines3",
            "cv2",
        }, f"Validation imported a model/native loader: {name}"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert validate_assets(tmp_path) == demo


def test_changed_model_is_rejected_before_deserialization(tmp_path):
    write_bundle(tmp_path)
    (tmp_path / "vision.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_assets(tmp_path)


@pytest.mark.parametrize("field", ["task", "policy", "vision"])
def test_alternate_unmanifested_runtime_input_is_rejected(tmp_path, field):
    demo = write_bundle(tmp_path)
    demo[field] = "unverified-input"
    change_asset(tmp_path, "demo.json", json.dumps(demo))
    (tmp_path / "unverified-input").write_bytes(b"present but never verified")
    with pytest.raises(ValueError, match=f"Referenced {field}"):
        validate_assets(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "scene/room.yaml",
        "scene/room.genesis.json",
        "memory/room-memory.json",
        "memory/free-grid.npz",
        "memory/floor-masks.npz",
    ],
)
@pytest.mark.parametrize("failure", ["missing", "unlisted", "directory"])
def test_every_runtime_dependency_needs_a_checksummed_regular_file(
    tmp_path, name, failure
):
    write_bundle(tmp_path)
    if failure == "unlisted":
        omit_checksum(tmp_path, name)
    else:
        (tmp_path / name).unlink()
        if failure == "directory":
            (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="Missing|manifest"):
        validate_assets(tmp_path)


@pytest.mark.parametrize(
    "name,content",
    [
        ("scene/task.yaml", "world_config: ../../unchecked-world.yaml\n"),
        ("scene/room.yaml", "project: ../../unchecked-project.json\nnotes: test\n"),
    ],
)
def test_nested_configuration_cannot_reference_outside_bundle(tmp_path, name, content):
    root = tmp_path / "bundle"
    write_bundle(root)
    (tmp_path / "unchecked-world.yaml").write_text("project: unchecked-project.json")
    (tmp_path / "unchecked-project.json").write_text('{"name":"external"}')
    change_asset(root, name, content)
    with pytest.raises(ValueError, match="inside"):
        validate_assets(root)


@pytest.mark.parametrize("level", ["task", "world"])
def test_absolute_configuration_reference_is_rejected_even_inside_bundle(
    tmp_path, level
):
    write_bundle(tmp_path)
    name, content = (
        ("scene/task.yaml", f"world_config: {tmp_path}/scene/room.yaml\n")
        if level == "task"
        else (
            "scene/room.yaml",
            f"project: {tmp_path}/scene/room.genesis.json\nnotes: test\n",
        )
    )
    change_asset(tmp_path, name, content)
    with pytest.raises(ValueError, match="relative"):
        validate_assets(tmp_path)


def test_configuration_paths_resolve_relative_to_each_containing_file(tmp_path):
    demo = write_bundle(tmp_path)
    change_asset(tmp_path, "scene/task.yaml", "world_config: ../worlds/room.yaml\n")
    change_asset(
        tmp_path,
        "worlds/room.yaml",
        "project: ../scene/room.genesis.json\nnotes: test fixture\n",
    )
    assert validate_assets(tmp_path) == demo


@pytest.mark.parametrize(
    "name",
    [
        "bundle.json",
        "demo.json",
        "policy.zip",
        "vision.pt",
        "scene/task.yaml",
        "scene/room.yaml",
        "scene/room.genesis.json",
        "memory/manifest.json",
        "memory/room-memory.json",
        "memory/free-grid.npz",
        "memory/floor-masks.npz",
        "scene",
        "memory",
    ],
)
@pytest.mark.parametrize("external", [False, True])
def test_symlinked_files_and_directories_are_rejected(tmp_path, name, external):
    root = tmp_path / "bundle"
    write_bundle(root)
    original = root / name
    target = (tmp_path if external else root) / "link-target"
    original.rename(target)
    original.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="Links"):
        validate_assets(root)


def test_symlinked_bundle_root_is_rejected(tmp_path):
    root = tmp_path / "bundle"
    write_bundle(root)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="Links"):
        validate_assets(link)


def test_symlink_cannot_be_hidden_by_parent_traversal(tmp_path):
    write_bundle(tmp_path)
    (tmp_path / "scene/link").symlink_to(tmp_path / "memory", target_is_directory=True)
    change_asset(tmp_path, "scene/task.yaml", "world_config: link/../scene/room.yaml\n")
    with pytest.raises(ValueError, match="Links"):
        validate_assets(tmp_path)


@pytest.mark.parametrize("name", ["room-memory.json", "free-grid.npz"])
def test_map_manifest_must_cover_implicitly_loaded_artifacts(tmp_path, name):
    write_bundle(tmp_path)
    manifest = json.loads((tmp_path / "memory/manifest.json").read_text())
    del manifest["artifact_sha256"][name]
    change_asset(tmp_path, "memory/manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="required artifact"):
        validate_assets(tmp_path)


@pytest.mark.parametrize("path", ["../policy.zip", "/absolute", "../../outside"])
def test_map_artifact_paths_must_stay_within_memory(tmp_path, path):
    write_bundle(tmp_path)
    manifest = json.loads((tmp_path / "memory/manifest.json").read_text())
    manifest["artifact_sha256"][path] = sha256(tmp_path / "policy.zip")
    change_asset(tmp_path, "memory/manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="inside"):
        validate_assets(tmp_path)


def test_map_and_bundle_hashes_must_agree(tmp_path):
    write_bundle(tmp_path)
    manifest = json.loads((tmp_path / "memory/manifest.json").read_text())
    manifest["artifact_sha256"]["free-grid.npz"] = "f" * 64
    change_asset(tmp_path, "memory/manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="Map artifact missing"):
        validate_assets(tmp_path)


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
def test_supported_studio_project_schema_migrations_remain_accepted(tmp_path, schema):
    demo = write_bundle(tmp_path)
    change_asset(
        tmp_path,
        "scene/room.genesis.json",
        json.dumps({"schema_version": schema, "name": "legacy primitive world"}),
    )
    assert validate_assets(tmp_path) == demo


def test_unknown_project_schema_fails_closed(tmp_path):
    write_bundle(tmp_path)
    change_asset(
        tmp_path,
        "scene/room.genesis.json",
        json.dumps({"schema_version": 5, "name": "unknown future format"}),
    )
    with pytest.raises(ValueError, match="schema"):
        validate_assets(tmp_path)


@pytest.mark.parametrize("kind", ["mesh", "terrain", "robot"])
def test_project_runtime_boundary_rejects_unsupported_external_assets(tmp_path, kind):
    write_bundle(tmp_path)
    project = {"name": "unsupported world", "schema_version": 4}
    if kind == "robot":
        project["robot"] = {
            "name": "robot",
            "source": "unchecked.urdf",
            "browser_url": "/robot",
        }
    else:
        project["objects"] = [{"name": "object", "format": kind}]
        if kind == "mesh":
            project["objects"][0]["source"] = "unchecked.obj"
    change_asset(tmp_path, "scene/room.genesis.json", json.dumps(project))
    with pytest.raises(ValueError, match="primitive-only"):
        validate_assets(tmp_path)


def test_valid_bundle_installs_atomically_without_deserializing_models(tmp_path):
    source = tmp_path / "source"
    demo = write_bundle(source)
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(source))
    destination = tmp_path / "installed"
    assert (
        install_bundle(archive, destination, expected_sha256=sha256(archive))
        == destination
    )
    assert validate_assets(destination) == demo


def test_archive_symlink_leaves_no_installation(tmp_path):
    archive = tmp_path / "links.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(link, "target")
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="Links"):
        install_bundle(archive, destination, expected_sha256=sha256(archive))
    assert not destination.exists()
