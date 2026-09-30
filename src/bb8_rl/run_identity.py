"""Data-only, local run provenance; never imports simulation or loads weights.

Snapshots preserve source namespaces, including untracked first-party modules.
Asset bytes are hashed, not copied. Capture and end verification detect changes;
they are not an atomic filesystem freeze or a signature from a trusted publisher.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

from .config import NavigationConfig
from .demo_assets import checked_path, validate_assets
from .task import TaskConfig

_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".mjs",
    ".css",
    ".html",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}
_EXCLUDED_PARTS = {
    "__pycache__",
    "node_modules",
    ".git",
    "work",
    "outputs",
    ".env",
    ".venv",
    ".venv-dreamer",
    ".venv-genesis",
}
_PACKAGES = (
    "bb8-rl",
    "genesis-studio-api",
    "genesis-world",
    "torch",
    "stable-baselines3",
    "numpy",
    "PyYAML",
    "pydantic",
    "opencv-python",
    "gymnasium",
    "scipy",
)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git(root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _safe_source(path, root):
    relative = path.relative_to(root)
    return (
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root)
        and not any(
            part in _EXCLUDED_PARTS or part.startswith(".venv")
            for part in relative.parts
        )
        and path.suffix in _SOURCE_SUFFIXES
    )


def _bb8_sources(root):
    files = set()
    for name in ("src", "scripts", "configs"):
        directory = root / name
        if directory.is_dir():
            files.update(
                path for path in directory.rglob("*") if _safe_source(path, root)
            )
    # Installed wheels have their package directly below the supplied root.
    if not (root / "src/bb8_rl").is_dir() and (root / "bb8_rl").is_dir():
        files.update(
            path for path in (root / "bb8_rl").rglob("*") if _safe_source(path, root)
        )
    if (root / "pyproject.toml").is_file():
        files.add(root / "pyproject.toml")
    return sorted(files)


def _studio_sources(root):
    if root is None:
        return []
    listed = _git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
    )
    if listed is not None:
        candidates = (root / name for name in listed.split("\0") if name)
    else:
        # Fallback for an installed Studio wheel: package sources only, never the
        # surrounding environment or other third-party distributions.
        candidates = (
            path
            for name in ("genesis_studio", "genesis_studio_desktop")
            for path in (root / name).rglob("*.py")
        )
    return sorted({path for path in candidates if _safe_source(path, root)})


def _studio_origin():
    module = sys.modules.get("genesis_studio")
    if module is not None and getattr(module, "__file__", None):
        return Path(module.__file__).resolve()
    try:
        spec = importlib.util.find_spec("genesis_studio")
    except (ImportError, ValueError):
        spec = None
    return Path(spec.origin).resolve() if spec and spec.origin else None


def _default_roots(project_root, studio_root):
    package = Path(__file__).resolve().parent
    if project_root is None:
        candidate = package.parents[1]
        project_root = (
            candidate if (candidate / "src/bb8_rl").is_dir() else package.parent
        )
    project_root = Path(project_root).resolve()
    if studio_root is None:
        configured = os.environ.get("GENESIS_STUDIO_ROOT")
        source = _studio_origin()
        # PYTHONPATH/import resolution can select a different Studio checkout
        # than GENESIS_STUDIO_ROOT. Fingerprint the dependency actually loaded.
        if source is not None:
            studio_root = next(
                (parent for parent in source.parents if (parent / ".git").exists()),
                source.parent.parent,
            )
        elif configured:
            studio_root = Path(configured)
        elif (project_root.parent / "Genesis-Studio").is_dir():
            studio_root = project_root.parent / "Genesis-Studio"
    return project_root, Path(studio_root).resolve() if studio_root else None


def _record(path, *, snapshot_root=None, snapshot_name=None):
    path = Path(path).absolute()
    if not path.is_file():
        raise ValueError(f"Run identity input is not a regular file: {path}")
    data = path.read_bytes() if snapshot_root is not None else None
    record = {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest()
        if data is not None
        else _digest(path),
        "bytes": len(data) if data is not None else path.stat().st_size,
    }
    if snapshot_root is not None:
        target = snapshot_root / snapshot_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record["snapshot"] = snapshot_name
    return record


def _config_dependencies(task_path, snapshot_root=None):
    requested = Path(task_path).absolute()
    paths = {"task": requested}
    records = {}
    task = TaskConfig.model_validate(yaml.safe_load(requested.read_text()))
    paths["world"] = (requested.resolve().parent / task.world_config).absolute()
    world = NavigationConfig.model_validate(yaml.safe_load(paths["world"].read_text()))
    paths["project"] = (paths["world"].resolve().parent / world.project).absolute()
    for label, path in paths.items():
        records[label] = _record(
            path,
            snapshot_root=snapshot_root,
            snapshot_name=f"configuration/{label}{path.suffix}",
        )
    return records


def _git_identity(root, files, snapshot_root=None, name=None):
    if root is None:
        return {"available": False, "reason": "Studio source root unavailable"}
    top = _git(root, "rev-parse", "--show-toplevel")
    head = _git(root, "rev-parse", "HEAD")
    if top is None or head is None:
        return {
            "available": False,
            "root": str(root),
            "reason": "Git checkout unavailable",
        }
    repository = Path(top.strip()).resolve()
    relative = sorted(
        {
            str(path.relative_to(repository))
            for path in files
            if path.is_relative_to(repository)
        }
    )
    # Empty path lists must not accidentally collect unrelated files or secrets.
    status = (
        _git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *relative,
        )
        if relative
        else ""
    )
    diff = (
        _git(
            repository,
            "diff",
            "HEAD",
            "--no-ext-diff",
            "--no-textconv",
            "--",
            *relative,
        )
        if relative
        else ""
    )
    if status is None or diff is None:
        raise ValueError(f"Unable to capture relevant Git changes: {repository}")
    result = {
        "available": True,
        "root": str(repository),
        "head": head.strip(),
        "branch": (
            _git(repository, "symbolic-ref", "--short", "-q", "HEAD") or ""
        ).strip(),
        "status": status,
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "tracked_paths": relative,
    }
    if snapshot_root is not None:
        target = snapshot_root / "git" / f"{name}.patch"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(diff)
        result["diff_snapshot"] = str(target.relative_to(snapshot_root))
    return result


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime():
    # Distribution metadata is read without importing optional/native packages.
    mapping = importlib.metadata.packages_distributions()
    imported = {name.split(".")[0] for name in sys.modules}
    distributions = {
        distribution for name in imported for distribution in mapping.get(name, [])
    }
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _version(name) for name in sorted(set(_PACKAGES) | distributions)
        },
        "imported_distributions": sorted(distributions),
        "loaded_modules": {
            name: _module_record(module)
            for name, module in tuple(sys.modules.items())
            if module is not None
            and (
                name
                in {
                    "genesis",
                    "torch",
                    "stable_baselines3",
                    "numpy",
                    "yaml",
                    "pydantic",
                    "cv2",
                    "gymnasium",
                    "scipy",
                }
                or name.split(".")[0]
                in {"bb8_rl", "genesis_studio", "genesis_studio_desktop"}
            )
        },
    }


def _module_record(module):
    source = getattr(module, "__file__", None)
    path = Path(source).resolve() if source else None
    version = getattr(module, "__version__", None)
    return {
        "file": str(path) if path else None,
        "version": str(version) if version is not None else None,
        "sha256": _digest(path) if path and path.is_file() else None,
    }


def capture_run_identity(
    snapshot_dir,
    *,
    asset_dir,
    demo=None,
    project_root=None,
    studio_root=None,
    task_path=None,
):
    """Capture executed inputs; task_path records a harness's actual override.

    Snapshot directories must be fresh. Configuration snapshots are small text
    inputs; model/map archives are only hashed. Caller records effective in-memory
    config/project changes after reset separately from these on-disk inputs.
    """
    snapshot_root = Path(snapshot_dir).resolve()
    if snapshot_root.exists():
        raise ValueError("Run identity requires a fresh snapshot directory")
    asset_root = Path(asset_dir).resolve()
    checked_demo = validate_assets(asset_root)
    if demo is not None and demo != checked_demo:
        raise ValueError(
            "Run identity demo differs from the validated asset configuration"
        )
    demo = checked_demo
    explicit_studio_root = studio_root is not None
    project_root, studio_root = _default_roots(project_root, studio_root)
    task_path = Path(task_path) if task_path is not None else asset_root / demo["task"]
    snapshot_root.mkdir(parents=True)
    sources = {}
    paths = {"bb8": _bb8_sources(project_root), "studio": _studio_sources(studio_root)}
    for namespace, root in (("bb8", project_root), ("studio", studio_root)):
        for path in paths[namespace]:
            name = f"{namespace}/{path.relative_to(root).as_posix()}"
            sources[name] = _record(
                path, snapshot_root=snapshot_root, snapshot_name=f"source/{name}"
            )
    configuration = _config_dependencies(task_path, snapshot_root)
    manifest = json.loads((asset_root / "bundle.json").read_text())
    assets = {"bundle.json": _record(asset_root / "bundle.json")}
    for relative, expected in manifest["sha256"].items():
        assets[relative] = _record(checked_path(asset_root, relative))
        if assets[relative]["sha256"] != expected:
            raise ValueError(f"Asset changed while recording run identity: {relative}")
    relevant_bb8 = paths["bb8"] + [
        Path(record["resolved_path"]) for record in configuration.values()
    ]
    identity = {
        "schema": "bb8.run-identity.v1",
        "roots": {
            "bb8": str(project_root),
            "studio": str(studio_root) if studio_root else None,
        },
        "asset_root": str(asset_root),
        "requested_task_path": str(task_path.absolute()),
        "source_files": sources,
        "configuration": configuration,
        "assets": assets,
        "git": {
            "bb8": _git_identity(project_root, relevant_bb8, snapshot_root, "bb8"),
            "studio": _git_identity(
                studio_root, paths["studio"], snapshot_root, "studio"
            ),
        },
        "runtime": _runtime(),
        "studio_resolution": {
            "explicit_root_override": explicit_studio_root,
            "package_origin": str(_studio_origin()) if _studio_origin() else None,
            "configured_root": str(Path(os.environ["GENESIS_STUDIO_ROOT"]).resolve())
            if os.environ.get("GENESIS_STUDIO_ROOT")
            else None,
            "configured_matches_selected": Path(
                os.environ["GENESIS_STUDIO_ROOT"]
            ).resolve()
            == studio_root
            if os.environ.get("GENESIS_STUDIO_ROOT")
            else None,
        },
        "scope": "On-disk inputs at capture; asset bytes hashed only; effective runtime overrides recorded by the caller. No atomic freeze or publisher authentication.",
    }
    identity["identity_sha256"] = _json_hash(identity)
    (snapshot_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    return identity


def verify_run_identity(identity):
    """Recheck the captured inventory, new source files, config graph and versions."""
    changes = []
    for category in ("source_files", "configuration", "assets"):
        for name, previous in identity[category].items():
            try:
                current = _record(previous["path"])
            except (OSError, ValueError):
                current = None
            if current is None or any(
                current[key] != previous[key]
                for key in ("resolved_path", "sha256", "bytes")
            ):
                changes.append(f"{category}/{name}")
    roots = identity["roots"]
    bb8 = Path(roots["bb8"])
    studio = Path(roots["studio"]) if roots["studio"] else None
    paths = {"bb8": _bb8_sources(bb8), "studio": _studio_sources(studio)}
    current_names = {
        f"{namespace}/{path.relative_to(root).as_posix()}"
        for namespace, root in (("bb8", bb8), ("studio", studio))
        for path in paths[namespace]
    }
    changes.extend(
        f"source_files/{name}"
        for name in current_names ^ identity["source_files"].keys()
    )
    try:
        config = _config_dependencies(identity["requested_task_path"])
        for label, current in config.items():
            if any(
                current[key] != identity["configuration"][label][key]
                for key in ("resolved_path", "sha256", "bytes")
            ):
                changes.append(f"configuration/{label}")
    except (OSError, ValueError):
        changes.append("configuration/dependency_graph")
    try:
        validate_assets(identity["asset_root"])
    except (OSError, ValueError):
        changes.append("assets/dependency_graph")
    git_changes = []
    relevant = paths["bb8"] + [
        Path(record["resolved_path"]) for record in identity["configuration"].values()
    ]
    for namespace, root, files in (
        ("bb8", bb8, relevant),
        ("studio", studio, paths["studio"]),
    ):
        current = _git_identity(root, files)
        previous = {
            key: value
            for key, value in identity["git"][namespace].items()
            if key != "diff_snapshot"
        }
        if current != previous:
            git_changes.append(namespace)
    package_changes = [
        name
        for name, version in identity["runtime"]["packages"].items()
        if _version(name) != version
    ]
    module_changes = [
        name
        for name, previous in identity["runtime"]["loaded_modules"].items()
        if (module := sys.modules.get(name)) is None
        or _module_record(module) != previous
    ]
    return {
        "unchanged": not (changes or git_changes or package_changes or module_changes),
        "changed_files": sorted(set(changes)),
        "git_changes": git_changes,
        "package_changes": package_changes,
        "module_changes": module_changes,
    }
