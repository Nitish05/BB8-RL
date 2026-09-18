import json
import subprocess
import sys
from pathlib import Path

import pytest
from genesis_studio.models import Project

from bb8_rl.cli import run
from bb8_rl.config import NavigationConfig
from bb8_rl.world import validate_world

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/navigation/bb8-state.yaml"


def test_world_roundtrip_and_cli():
    _config, project, body, head, camera = validate_world(CONFIG)
    assert Project.model_validate_json(project.model_dump_json()) == project
    assert project.schema_version == 4
    assert body.radius == 0.037
    assert head.collision_mask == head.collision_group == 0
    assert camera.parent is None
    assert run(["world", "validate", "--config", str(CONFIG)]) == 0
    assert run(["simulate", "--seconds", "nan"]) == 1


def test_unknown_task_fields_fail():
    with pytest.raises(ValueError):
        NavigationConfig(project="p", notes="test", unknown=True)


def test_drive_capability_is_owned_by_bb8_not_studio():
    import genesis_studio.control.contract as studio_contract

    from bb8_rl.control.contract import PlanarDriveCommand
    from bb8_rl.control.genesis_backend import GenesisPlanarDriveBackend

    assert PlanarDriveCommand.__module__.startswith("bb8_rl.")
    assert GenesisPlanarDriveBackend.__module__.startswith("bb8_rl.")
    assert not hasattr(studio_contract, "PlanarDriveCommand")


def test_explicit_camera_failure_returns_nonzero(monkeypatch):
    monkeypatch.setattr(
        "bb8_rl.cli.doctor",
        lambda *a: {
            "simulation_gate": "passed",
            "hardware_gate": "pending",
            "camera": {"status": "failed"},
        },
    )
    assert run(["doctor", "--camera-index", "0"]) == 1


def test_world_rejects_missing_physical_boundary(tmp_path):
    config, project, *_ = validate_world(CONFIG)
    project.objects.pop()
    (tmp_path / "world.json").write_text(project.model_dump_json())
    data = config.model_dump()
    data["project"] = "world.json"
    task = tmp_path / "task.yaml"
    task.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="four colliding walls"):
        validate_world(task)


def test_import_is_passive_without_genesis_or_hardware():
    script = """
import sys
import bb8_rl
import bb8_rl.cli
assert 'genesis' not in sys.modules
assert 'bleak' not in sys.modules
assert 'cv2' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_world_rejects_head_collision_and_invalid_action_period(tmp_path):
    config, project, *_ = validate_world(CONFIG)
    path = tmp_path / "world.json"
    task = tmp_path / "task.yaml"
    data = config.model_dump()
    data["project"] = "world.json"
    task.write_text(json.dumps(data))
    project.objects[1].collision_mask = 65535
    path.write_text(project.model_dump_json())
    with pytest.raises(ValueError, match="head"):
        validate_world(task)
    project.objects[1].collision_mask = 0
    path.write_text(project.model_dump_json())
    data["action_steps"] = 100
    task.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="lifetime"):
        validate_world(task)
