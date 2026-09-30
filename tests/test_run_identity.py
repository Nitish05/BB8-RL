import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bb8_rl.run_identity import capture_run_identity, verify_run_identity


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def commit(root):
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Identity Test",
        "-c",
        "user.email=identity@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )


def bundle(root):
    demo = {
        "task": "scene/task.yaml",
        "memory": "memory",
        "policy": "policy.zip",
        "vision": "vision.pt",
    }
    files = {
        "demo.json": json.dumps(demo),
        "scene/task.yaml": "world_config: room.yaml\n",
        "scene/room.yaml": "project: room.json\nnotes: identity fixture\n",
        "scene/room.json": '{"name":"fixture","schema_version":4}',
        "memory/room-memory.json": "{}",
        "memory/free-grid.npz": "opaque map bytes; never deserialize",
        "policy.zip": "opaque policy bytes; never deserialize",
        "vision.pt": "opaque vision bytes; never deserialize",
    }
    files["memory/manifest.json"] = json.dumps(
        {
            "artifact_sha256": {
                name: hashlib.sha256(files[f"memory/{name}"].encode()).hexdigest()
                for name in ("room-memory.json", "free-grid.npz")
            }
        }
    )
    for name, content in files.items():
        write(root / name, content)
    write(
        root / "bundle.json",
        json.dumps(
            {
                "schema": "bb8.interactive-assets.v1",
                "sha256": {
                    name: hashlib.sha256(content.encode()).hexdigest()
                    for name, content in files.items()
                },
            }
        ),
    )
    return demo


@pytest.fixture
def inputs(tmp_path):
    bb8, studio, assets = (tmp_path / name for name in ("bb8", "studio", "assets"))
    write(bb8 / "src/bb8_rl/planning/planner.py", "# planner version one\n")
    write(
        bb8 / "src/bb8_rl/control/planner.py",
        "# different module with the same basename\n",
    )
    write(bb8 / "src/bb8_rl/web/app.js", "// frontend source\n")
    write(bb8 / "configs/default.yaml", "mode: development\n")
    write(bb8 / "pyproject.toml", '[project]\nname = "fixture"\n')
    write(studio / "apps/api/src/genesis_studio/models.py", "# Studio model source\n")
    write(
        studio / "apps/desktop/src/genesis_studio_desktop/scene_builder.py",
        "# Studio scene builder\n",
    )
    commit(bb8)
    commit(studio)
    bundle(assets)
    return {"project_root": bb8, "studio_root": studio, "asset_dir": assets}


def capture(tmp_path, inputs, name="snapshot", **kwargs):
    return capture_run_identity(tmp_path / name, **inputs, **kwargs)


def test_complete_namespaced_sources_assets_and_configs_are_preserved(tmp_path, inputs):
    identity = capture(tmp_path, inputs)
    files = identity["source_files"]
    assert "bb8/src/bb8_rl/planning/planner.py" in files
    assert "bb8/src/bb8_rl/control/planner.py" in files
    assert "bb8/src/bb8_rl/web/app.js" in files
    assert "bb8/configs/default.yaml" in files
    assert "studio/apps/api/src/genesis_studio/models.py" in files
    assert {"task", "world", "project"} == set(identity["configuration"])
    assert {
        "policy.zip",
        "vision.pt",
        "bundle.json",
        "memory/free-grid.npz",
    } <= identity["assets"].keys()
    for record in files.values():
        assert (tmp_path / "snapshot" / record["snapshot"]).read_bytes() == Path(
            record["path"]
        ).read_bytes()
    assert verify_run_identity(identity)["unchanged"]
    assert json.loads((tmp_path / "snapshot/identity.json").read_text()) == identity
    assert not any(
        b"opaque policy bytes" in path.read_bytes()
        for path in (tmp_path / "snapshot").rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize("name", ["planning/planner.py", "control/planner.py"])
def test_planner_changes_alter_identity_and_end_verification(tmp_path, inputs, name):
    before = capture(tmp_path, inputs)
    write(inputs["project_root"] / "src/bb8_rl" / name, "# planner version two\n")
    verified = verify_run_identity(before)
    assert not verified["unchanged"]
    assert f"source_files/bb8/src/bb8_rl/{name}" in verified["changed_files"]
    assert "bb8" in verified["git_changes"]
    after = capture(tmp_path, inputs, "after")
    assert after["identity_sha256"] != before["identity_sha256"]


def test_actual_external_task_override_graph_is_captured(tmp_path, inputs):
    task = tmp_path / "harness/task.yaml"
    write(task, "world_config: sub/world.yaml\n")
    write(
        task.parent / "sub/world.yaml",
        "project: ../project.json\nnotes: harness override\n",
    )
    project = task.parent / "project.json"
    write(project, '{"name":"harness world"}')
    before = capture(tmp_path, inputs, task_path=task)
    assert before["configuration"]["project"]["resolved_path"] == str(project)
    assert verify_run_identity(before)["unchanged"]
    write(project, '{"name":"changed harness world"}')
    verified = verify_run_identity(before)
    assert "configuration/project" in verified["changed_files"]
    after = capture(tmp_path, inputs, "after", task_path=task)
    assert after["identity_sha256"] != before["identity_sha256"]


def test_world_dependency_change_updates_hash_even_with_same_policy(tmp_path, inputs):
    before = capture(tmp_path, inputs)
    root = inputs["asset_dir"]
    world = root / "scene/room.yaml"
    write(world, "project: room.json\nnotes: changed world\narena_half_extent: 2\n")
    manifest = json.loads((root / "bundle.json").read_text())
    manifest["sha256"]["scene/room.yaml"] = hashlib.sha256(
        world.read_bytes()
    ).hexdigest()
    write(root / "bundle.json", json.dumps(manifest))
    verified = verify_run_identity(before)
    assert "configuration/world" in verified["changed_files"]
    after = capture(tmp_path, inputs, "after")
    assert after["identity_sha256"] != before["identity_sha256"]
    assert after["assets"]["policy.zip"] == before["assets"]["policy.zip"]


def test_git_dirty_changes_and_untracked_source_are_saved_without_unrelated_files(
    tmp_path, inputs
):
    bb8, studio = inputs["project_root"], inputs["studio_root"]
    write(bb8 / "src/bb8_rl/planning/planner.py", "# dirty planner\n")
    write(bb8 / "src/bb8_rl/new_runtime.py", "# untracked executable source\n")
    write(
        studio / "apps/desktop/src/genesis_studio_desktop/new_feature.py",
        "# untracked Studio source\n",
    )
    write(bb8 / "work/private.json", '{"secret":"must not be copied"}')
    write(bb8 / ".env", "SECRET=must-not-be-copied\n")
    write(bb8 / "src/bb8_rl/.env", "SECRET=must-not-be-copied\n")
    identity = capture(tmp_path, inputs)
    assert "bb8/src/bb8_rl/new_runtime.py" in identity["source_files"]
    assert (
        "studio/apps/desktop/src/genesis_studio_desktop/new_feature.py"
        in identity["source_files"]
    )
    assert "new_runtime.py" in identity["git"]["bb8"]["status"]
    diff = tmp_path / "snapshot" / identity["git"]["bb8"]["diff_snapshot"]
    assert "dirty planner" in diff.read_text()
    assert not any(
        b"must-not-be-copied" in path.read_bytes()
        for path in (tmp_path / "snapshot").rglob("*")
        if path.is_file()
    )
    assert verify_run_identity(identity)["unchanged"]


@pytest.mark.parametrize("operation", ["add", "delete", "rename"])
def test_source_inventory_changes_are_detected(tmp_path, inputs, operation):
    identity = capture(tmp_path, inputs)
    source = inputs["project_root"] / "src/bb8_rl/planning/planner.py"
    if operation == "add":
        write(source.with_name("new.py"), "# new dependency\n")
    elif operation == "delete":
        source.unlink()
    else:
        source.rename(source.with_name("renamed.py"))
    assert not verify_run_identity(identity)["unchanged"]


def test_studio_source_changes_are_detected(tmp_path, inputs):
    identity = capture(tmp_path, inputs)
    write(
        inputs["studio_root"] / "apps/api/src/genesis_studio/models.py",
        "# changed Studio\n",
    )
    verified = verify_run_identity(identity)
    assert (
        "source_files/studio/apps/api/src/genesis_studio/models.py"
        in verified["changed_files"]
    )
    assert "studio" in verified["git_changes"]


def test_loaded_module_origins_versions_and_bytes_are_independent_of_metadata(
    tmp_path, inputs, monkeypatch
):
    path = tmp_path / "shadowed-numpy.py"
    write(path, "# stand-in source, not imported\n")
    module = SimpleNamespace(__file__=str(path), __version__="synthetic-shadow-version")
    monkeypatch.setitem(sys.modules, "numpy", module)
    identity = capture(tmp_path, inputs)
    loaded = identity["runtime"]["loaded_modules"]["numpy"]
    assert loaded["file"] == str(path)
    assert loaded["version"] == "synthetic-shadow-version"
    assert identity["runtime"]["packages"]["numpy"] != loaded["version"]
    write(path, "# changed installed source\n")
    assert "numpy" in verify_run_identity(identity)["module_changes"]


def test_asset_mutation_is_detected_without_deserialization(tmp_path, inputs):
    identity = capture(tmp_path, inputs)
    write(inputs["asset_dir"] / "policy.zip", "tampered opaque model bytes")
    verified = verify_run_identity(identity)
    assert "assets/policy.zip" in verified["changed_files"]
    assert "assets/dependency_graph" in verified["changed_files"]
    with pytest.raises(ValueError, match="checksum"):
        capture(tmp_path, inputs, "invalid")
    assert not (tmp_path / "invalid").exists()


def test_snapshot_directory_cannot_overwrite_evidence(tmp_path, inputs):
    capture(tmp_path, inputs)
    with pytest.raises(ValueError, match="fresh"):
        capture(tmp_path, inputs)


def test_non_git_install_records_explicit_limitation_and_package_sources(
    tmp_path, inputs
):
    package_root = tmp_path / "installed"
    write(package_root / "bb8_rl/control/backend.py", "# installed source\n")
    studio_root = tmp_path / "installed-studio"
    write(studio_root / "genesis_studio/models.py", "# installed Studio source\n")
    identity = capture(
        tmp_path,
        {**inputs, "project_root": package_root, "studio_root": studio_root},
    )
    assert not identity["git"]["bb8"]["available"]
    assert not identity["git"]["studio"]["available"]
    assert "bb8/bb8_rl/control/backend.py" in identity["source_files"]
    assert "studio/genesis_studio/models.py" in identity["source_files"]
    assert verify_run_identity(identity)["unchanged"]


def test_imported_studio_overrides_mismatched_environment_root(
    tmp_path, inputs, monkeypatch
):
    actual = inputs["studio_root"]
    origin = actual / "apps/api/src/genesis_studio/models.py"
    monkeypatch.setitem(
        sys.modules,
        "genesis_studio",
        SimpleNamespace(__file__=str(origin), __version__="fixture"),
    )
    monkeypatch.setenv("GENESIS_STUDIO_ROOT", str(tmp_path / "wrong-studio"))
    identity = capture(tmp_path, {**inputs, "studio_root": None})
    assert identity["roots"]["studio"] == str(actual)
    assert identity["git"]["studio"]["head"] == git(actual, "rev-parse", "HEAD").strip()
    assert identity["studio_resolution"]["configured_matches_selected"] is False
    assert identity["studio_resolution"]["package_origin"] == str(origin)
