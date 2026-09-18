import hashlib
import json
import zipfile

import pytest

from bb8_rl.demo_assets import checked_path, install_bundle, sha256, validate_assets


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


def test_changed_model_is_rejected_before_deserialization(tmp_path):
    demo = {
        "task": "task.yaml",
        "memory": "memory",
        "policy": "policy.zip",
        "vision": "vision.pt",
    }
    files = {
        "demo.json": json.dumps(demo).encode(),
        "policy.zip": b"policy",
        "vision.pt": b"vision",
        "task.yaml": b"world_config: room.yaml",
        "memory/manifest.json": b"{}",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "bundle.json").write_text(
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
    assert validate_assets(tmp_path) == demo
    (tmp_path / "vision.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_assets(tmp_path)


def test_manifest_cannot_omit_an_artifact_required_by_the_map(tmp_path):
    demo = {
        "task": "task.yaml",
        "memory": "memory",
        "policy": "policy.zip",
        "vision": "vision.pt",
    }
    missing_hash = "f" * 64
    files = {
        "demo.json": json.dumps(demo).encode(),
        "policy.zip": b"policy",
        "vision.pt": b"vision",
        "task.yaml": b"world_config: room.yaml",
        "memory/manifest.json": json.dumps(
            {"artifact_sha256": {"floor-masks.npz": missing_hash}}
        ).encode(),
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "bundle.json").write_text(
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
    with pytest.raises(ValueError, match="Map artifact missing"):
        validate_assets(tmp_path)


@pytest.mark.parametrize("field", ["task", "policy", "vision"])
def test_alternate_unmanifested_runtime_input_is_rejected(tmp_path, field):
    demo = {
        "task": "task.yaml",
        "memory": "memory",
        "policy": "policy.zip",
        "vision": "vision.pt",
    }
    demo[field] = "unverified-input"
    files = {
        "demo.json": json.dumps(demo).encode(),
        "policy.zip": b"policy",
        "vision.pt": b"vision",
        "task.yaml": b"task",
        "memory/manifest.json": b"{}",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "unverified-input").write_bytes(b"present but never verified")
    (tmp_path / "bundle.json").write_text(
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
    with pytest.raises(ValueError, match=f"Referenced {field}"):
        validate_assets(tmp_path)
