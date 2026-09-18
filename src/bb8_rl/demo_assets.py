"""Explicit, checksum-verified demo installation. Import has no side effects."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Asset path must be a nonempty relative string")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Asset path must stay inside its bundle")
    return path


def validate_assets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest.get("schema") != "bb8.interactive-assets.v1":
        raise ValueError("Unrecognized demo asset bundle")
    files = manifest.get("sha256", {})
    if (
        not {"demo.json", "policy.zip", "vision.pt", "memory/manifest.json"}
        <= files.keys()
    ):
        raise ValueError("Demo asset manifest is incomplete")
    for relative, expected in files.items():
        if sha256(checked_path(directory, relative)) != expected:
            raise ValueError(f"Demo asset checksum failed: {relative}")
    demo = json.loads((directory / "demo.json").read_text())
    for name in ("task", "memory", "policy", "vision"):
        path = checked_path(directory, demo[name])
        referenced = path / "manifest.json" if name == "memory" else path
        relative = str(referenced.relative_to(directory.resolve()))
        if relative not in files:
            raise ValueError(
                f"Referenced {name} file is absent from the checksum manifest"
            )
    memory = checked_path(directory, demo["memory"])
    map_manifest = json.loads((memory / "manifest.json").read_text())
    for name, expected in map_manifest.get("artifact_sha256", {}).items():
        path = checked_path(memory, name)
        relative = str(path.relative_to(directory.resolve()))
        if relative not in files or files[relative] != expected:
            raise ValueError(f"Map artifact missing from bundle manifest: {relative}")
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
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Links are not allowed in the demo bundle")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        validate_assets(staging)
        staging.rename(destination)
    return destination
