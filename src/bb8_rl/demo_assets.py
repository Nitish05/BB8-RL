"""Explicit, checksum-verified demo installation. Import has no side effects."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Asset path must be a nonempty relative string")
    if Path(root).is_symlink():
        raise ValueError("Links are not allowed in the demo bundle")
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError("Asset path must stay inside its bundle")
    # Inspect the unresolved path too: resolving first conceals links, including
    # links whose destination happens to be another checksummed bundle file.
    candidate = root
    for part in Path(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("Links are not allowed in the demo bundle")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset path must stay inside its bundle")
    return path


def _json_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")  # noqa: TRY004
    return value


def _require_file(path):
    if not path.is_file():
        raise ValueError(f"Missing demo asset file: {path.name}")
    return path


def validate_project_dependencies(path):
    """Close the primitive world's file-loading boundary without importing Studio.

    Studio migrates project schemas 1--3 to 4. Its full semantic validation still
    belongs to NavigationWorld; this shared install/build gate rejects unknown
    schemas and any object that could introduce another runtime file dependency.
    """
    project = _json_object(path)
    if project.get("schema_version", 4) not in (1, 2, 3, 4):
        raise ValueError("Unrecognized demo world project schema")
    objects = project.get("objects", [])
    if (
        project.get("robot") is not None
        or project.get("robots")
        or not isinstance(objects, list)
        or any(
            not isinstance(item, dict)
            or item.get("format") not in ("sphere", "box", "cylinder")
            for item in objects
        )
    ):
        raise ValueError("Demo runtime requires a primitive-only world without robots")
    return project


def validate_assets(directory):
    """Check the runtime dependency graph without loading models or map arrays.

    YAML references follow the runtime loaders' containing-directory semantics.
    The graph ends at primitive-only Studio projects and the map loader's fixed
    JSON/NPZ inputs plus its declared artifacts. Provenance/source-history paths
    in map metadata are not opened by the runtime and are not dependencies.
    """
    manifest_path = checked_path(directory, "bundle.json")
    directory = manifest_path.parent
    manifest = _json_object(_require_file(manifest_path))
    if manifest.get("schema") != "bb8.interactive-assets.v1":
        raise ValueError("Unrecognized demo asset bundle")
    files = manifest.get("sha256", {})
    if (
        not isinstance(files, dict)
        or not {"demo.json", "policy.zip", "vision.pt", "memory/manifest.json"}
        <= files.keys()
    ):
        raise ValueError("Demo asset manifest is incomplete")
    for relative, expected in files.items():
        path = checked_path(directory, relative)
        if relative != path.relative_to(directory).as_posix():
            raise ValueError("Bundle manifest paths must be canonical relative paths")
        if sha256(_require_file(path)) != expected:
            raise ValueError(f"Demo asset checksum failed: {relative}")

    def reference(base, relative, label):
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError(f"Referenced {label} must be a nonempty relative path")
        path = checked_path(directory, str(base.relative_to(directory) / relative))
        if path.relative_to(directory).as_posix() not in files:
            raise ValueError(
                f"Referenced {label} file is absent from the checksum manifest"
            )
        return path

    demo = _json_object(directory / "demo.json")
    inputs = {}
    for name in ("task", "memory", "policy", "vision"):
        path = checked_path(directory, demo.get(name))
        referenced = path / "manifest.json" if name == "memory" else path
        inputs[name] = reference(
            directory, str(referenced.relative_to(directory)), name
        )

    # These are the data-only schemas used by load_task/load_config. Importing
    # the world or native/model loaders here would cross this gate.
    from .config import NavigationConfig
    from .task import TaskConfig

    task_path = inputs["task"]
    task = TaskConfig.model_validate(yaml.safe_load(task_path.read_text()))
    world_path = reference(task_path.parent, task.world_config, "world_config")
    world = NavigationConfig.model_validate(yaml.safe_load(world_path.read_text()))
    project_path = reference(world_path.parent, world.project, "project")
    validate_project_dependencies(project_path)

    memory = inputs["memory"].parent
    map_manifest = _json_object(inputs["memory"])
    artifacts = map_manifest.get("artifact_sha256", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Map artifact checksums must be an object")  # noqa: TRY004
    for name, expected in artifacts.items():
        # The map loader interprets every artifact name relative to memory.
        path = checked_path(memory, name)
        relative = path.relative_to(directory).as_posix()
        if relative not in files or files[relative] != expected:
            raise ValueError(f"Map artifact missing from bundle manifest: {relative}")
    # ScanFreeMemory.load and map_payload open these regardless of the artifact
    # list, so a checksummed but incomplete map manifest cannot bypass coverage.
    for name in ("room-memory.json", "free-grid.npz"):
        reference(memory, name, "map")
        if name not in artifacts:
            raise ValueError(f"Map manifest is missing a required artifact: {name}")
    return demo


def install_bundle(archive, destination, *, expected_sha256):
    """Install a trusted release archive atomically; never unpickle a download here."""
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != expected_sha256:
        raise ValueError("Release archive checksum mismatch")
    if destination.exists():
        raise ValueError("Destination exists; choose a fresh asset directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "assets"
        staging.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            if sum(item.file_size for item in bundle.infolist()) > 200_000_000:
                raise ValueError("Demo archive exceeds the supported unpacked size")
            for item in bundle.infolist():
                target = checked_path(staging, item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Links are not allowed in the demo bundle")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        validate_assets(staging)
        staging.rename(destination)
    return destination
