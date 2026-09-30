"""Package this project's frozen demo inputs; never fetch or train models."""

import argparse
import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path

import yaml

from bb8_rl.demo_assets import (
    checked_path,
    sha256,
    validate_assets,
    validate_project_dependencies,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = "bb8.interactive-asset-inputs.v1"
INPUT_NAMES = frozenset(
    {
        "policy",
        "vision",
        "task",
        "world",
        "project",
        "registration",
        "protocol",
        "memory_manifest",
    }
)
PIXEL_COORDINATES = frozenset({"genesis_viewport", "opencv_integer_center"})


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def portable(value, roots):
    if isinstance(value, dict):
        return {portable(k, roots): portable(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(item, roots) for item in value]
    if isinstance(value, str):
        if Path(value).is_absolute():
            # Historical metadata paths are not runtime inputs. Keep their role
            # while avoiding machine-specific prefixes in rebuilt bundles.
            return f"<SOURCE_PATH>/{Path(value).name}"
        for root, replacement in roots:
            value = value.replace(str(root), replacement)
    return value


def _json(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")  # noqa: TRY004
    return data


def _array(value, shape, label):
    if not shape:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid finite numeric {label}")
    elif not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"Invalid shape for {label}")
    else:
        for item in value:
            _array(item, shape[1:], label)


def validate_inputs(manifest, root):
    """Validate all selected source bytes and relationships before any output write.

    Paths are relative to an explicit input root; only JSON/YAML data schemas are
    loaded. Policies, vision weights and map arrays are hashed as opaque bytes.
    """
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("Unrecognized asset input manifest schema")
    if set(manifest.get("inputs", {})) != INPUT_NAMES:
        raise ValueError("Asset input manifest must declare every named input")

    def checked(record):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("Each asset input requires path and sha256")
        path = checked_path(root, record["path"])
        if record["path"] != path.relative_to(Path(root).resolve()).as_posix():
            raise ValueError("Input paths must be canonical relative paths")
        expected = record["sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid input sha256")
        if not path.is_file():
            raise ValueError(f"Missing asset input: {record['path']}")
        if sha256(path) != expected:
            raise ValueError(f"Asset input checksum failed: {record['path']}")
        return path

    paths = {name: checked(record) for name, record in manifest["inputs"].items()}
    map_manifest = _json(paths["memory_manifest"])
    if "free_evidence_mode" in map_manifest and not isinstance(
        map_manifest.get("free_evidence_semantics"), str
    ):
        raise ValueError("Map evidence mode requires explicit semantics")
    artifacts = manifest.get("memory_artifacts", {})
    expected_artifacts = map_manifest.get("artifact_sha256", {})
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(expected_artifacts)
        or not {"room-memory.json", "free-grid.npz"} <= set(artifacts)
    ):
        raise ValueError(
            "Input memory artifacts must exactly cover the source manifest"
        )
    memory_paths = {}
    for name, record in artifacts.items():
        path = checked(record)
        if path != checked_path(paths["memory_manifest"].parent, name):
            raise ValueError("Memory artifact paths must match the source manifest")
        if record["sha256"] != expected_artifacts[name]:
            raise ValueError("Memory artifact checksum disagrees with source manifest")
        memory_paths[name] = path
    cameras = manifest.get("cameras", {})
    if not isinstance(cameras, dict) or set(cameras) != set("ABC"):
        raise ValueError("Input manifest must explicitly declare cameras A, B and C")
    metadata = {
        name: _json(checked(camera["metadata"])) for name, camera in cameras.items()
    }

    # Only after all bytes are verified, parse the same data-only configuration
    # task/world schemas used by the runtime. Studio's full scene semantics are
    # deferred to NavigationWorld; this preflight checks its file dependencies.
    from bb8_rl.config import NavigationConfig
    from bb8_rl.task import TaskConfig

    task_data = yaml.safe_load(paths["task"].read_text())
    world_data = yaml.safe_load(paths["world"].read_text())
    task = TaskConfig.model_validate(task_data)
    world = NavigationConfig.model_validate(world_data)
    for base, relative, target in (
        (paths["task"].parent, task.world_config, paths["world"]),
        (paths["world"].parent, world.project, paths["project"]),
    ):
        if Path(relative).is_absolute():
            raise ValueError("Source configuration references must be relative")
        resolved = checked_path(
            root, str(base.relative_to(Path(root).resolve()) / relative)
        )
        if resolved != target:
            raise ValueError(
                "Source configuration graph disagrees with selected inputs"
            )
    validate_project_dependencies(paths["project"])
    registration, protocol = _json(paths["registration"]), _json(paths["protocol"])
    # A convention tag describes already-exported numeric K. Never infer it from
    # matrix values or shift external OpenCV estimates while packaging assets.
    records = [*metadata.values(), map_manifest, registration, protocol]
    conventions = [
        record.get("pixel_coordinates", "genesis_viewport") for record in records
    ]
    if any(
        not isinstance(value, str) or value not in PIXEL_COORDINATES
        for value in conventions
    ):
        raise ValueError("Unknown camera pixel_coordinates convention")
    if len(set(conventions)) != 1:
        raise ValueError("Mixed camera/map/registration/protocol pixel_coordinates")
    for key, name in (
        ("scene_sha256", "project"),
        ("registration_report_sha256", "registration"),
    ):
        if key in protocol and protocol[key] != manifest["inputs"][name]["sha256"]:
            raise ValueError(f"Protocol {key} disagrees with selected inputs")
    case_index = manifest.get("protocol_case", 0)
    if (
        isinstance(case_index, bool)
        or not isinstance(case_index, int)
        or not 0 <= case_index < len(protocol.get("cases", []))
    ):
        raise ValueError("Invalid protocol case index")
    case = protocol["cases"][case_index]
    for field in ("start", "goal"):
        _array(case[field], (2,), field)
    for field in ("seed", "layout_seed"):
        if isinstance(case[field], bool) or not isinstance(case[field], int):
            raise TypeError(f"Invalid protocol {field}")
    built_cameras = {}
    for name, camera in cameras.items():
        query = camera["query_index"]
        if isinstance(query, bool) or not isinstance(query, int) or query < 0:
            raise ValueError("Invalid camera query index")
        matches = [
            row for row in registration["queries"] if row["query_index"] == query
        ]
        if len(matches) != 1 or matches[0]["status"] != "candidate":
            raise ValueError("Camera needs exactly one accepted registration candidate")
        scan = metadata[name]
        if scan.get("index") != query:
            raise ValueError("Camera metadata index differs from registration query")
        for value, shape, label in (
            (scan["position"], (3,), "camera position"),
            (scan["intrinsics"], (3, 3), "camera intrinsics"),
            (matches[0]["world_to_camera"], (4, 4), "camera transform"),
            (camera["lookat"], (3,), "camera lookat"),
        ):
            _array(value, shape, label)
        resolution = camera["resolution"]
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 1
                for x in resolution
            )
        ):
            raise ValueError("Invalid camera resolution")
        built_cameras[name] = {
            "query_index": query,
            "position": scan["position"],
            "lookat": camera["lookat"],
            "intrinsics": scan["intrinsics"],
            "world_to_camera": matches[0]["world_to_camera"],
            "resolution": resolution,
            "calibration_version": f"query-{query}-{registration['landmarks_sha256']}",
            "provenance": "RGB-estimated pose; synthetic intrinsics/metric scan frame. Render fixture position is excluded from control.",
        }
        if "pixel_coordinates" in scan:
            built_cameras[name]["pixel_coordinates"] = scan["pixel_coordinates"]
    if len({camera["query_index"] for camera in cameras.values()}) != 3:
        raise ValueError("Camera queries must be distinct")
    return {
        "paths": paths,
        "memory_paths": memory_paths,
        "map_manifest": map_manifest,
        "task": task_data,
        "world": world_data,
        "case": case,
        "cameras": built_cameras,
    }


def legacy_inputs(scan_dir, memory):
    """Capture legacy defaults once; future rebuilds can use the emitted manifest."""
    paths = {
        "policy": ROOT / "work/m6/sac-2/model.zip",
        "vision": ROOT / "work/m75/calibrated/model.pt",
        "task": ROOT / "projects/bb8/synthetic-room/task.yaml",
        "world": ROOT / "projects/bb8/synthetic-room/room.yaml",
        "project": ROOT / "projects/bb8/synthetic-room/room.genesis.json",
        "registration": ROOT / "work/m77/registration/superpoint-lightglue/report.json",
        "protocol": ROOT
        / "work/m78/supplemental-fixture/v6-camera20-r20-short/protocol.json",
        "memory_manifest": memory / "manifest.json",
    }
    artifacts = {
        name: checked_path(memory, name)
        for name in _json(paths["memory_manifest"])["artifact_sha256"]
    }
    scans = {
        name: scan_dir / f"scan-{query:03d}.json"
        for name, query in zip("ABC", (20, 23, 14), strict=True)
    }
    root = Path(
        os.path.commonpath(
            [
                str(path.absolute().parent)
                for path in [*paths.values(), *artifacts.values(), *scans.values()]
            ]
        )
    )

    def record(path):
        relative = path.absolute().relative_to(root).as_posix()
        checked_path(root, relative)
        return {"path": relative, "sha256": sha256(path)}

    manifest = {
        "schema": INPUT_SCHEMA,
        "inputs": {name: record(path) for name, path in paths.items()},
        "memory_artifacts": {name: record(path) for name, path in artifacts.items()},
        "protocol_case": 0,
        "cameras": {
            name: {
                "query_index": query,
                "metadata": record(scans[name]),
                "lookat": [0, 0, 0.1],
                "resolution": [1280, 960],
            }
            for name, query in zip("ABC", (20, 23, 14), strict=True)
        },
    }
    return manifest, root


def main(args):
    if args.inputs:
        if args.scan_dir or args.memory:
            raise ValueError("Use either --inputs or legacy --scan-dir/--memory")
        manifest = _json(args.inputs)
        declared_root = manifest.get("root", ".")
        if not isinstance(declared_root, str) or Path(declared_root).is_absolute():
            raise ValueError(
                "Manifest root must be relative; use --input-root to relocate"
            )
        root = args.input_root or args.inputs.parent / declared_root
    else:
        if args.input_root:
            raise ValueError("--input-root requires --inputs")
        if args.scan_dir is None:
            raise ValueError("Provide --inputs or legacy --scan-dir")
        manifest, root = legacy_inputs(
            args.scan_dir,
            args.memory or ROOT / "work/m78/scan-free-memory-v6-hull-fine",
        )
    prepared = validate_inputs(manifest, root)
    if args.write_input_manifest:
        captured = {
            **manifest,
            "root": os.path.relpath(root, args.write_input_manifest.parent),
        }
        args.write_input_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.write_input_manifest.open("x") as target:
            target.write(json.dumps(captured, indent=2, allow_nan=False) + "\n")
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "schema": INPUT_SCHEMA,
                    "validated": True,
                    "files": len(manifest["inputs"])
                    + len(manifest["memory_artifacts"])
                    + 3,
                }
            )
        )
        return
    if args.output is None:
        raise ValueError("Provide --output or --validate-inputs-only")
    output = args.output.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise ValueError("Output assets or archive already exist")
    paths, case = prepared["paths"], prepared["case"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "memory").mkdir()
    for name, source in prepared["memory_paths"].items():
        destination = output / "memory" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    map_manifest = portable(
        prepared["map_manifest"],
        [(Path(root).resolve(), "<INPUT_ROOT>")],
    )
    # Clarify legacy wording without changing any evidence, arrays or assumptions.
    map_manifest["assumptions"] = [
        "synthetic palette candidates overlap obstacle colors and require learned/geometric intersection"
        if text == "distinct synthetic floor palette"
        else text
        for text in map_manifest.get("assumptions", [])
    ]
    if "free_evidence_mode" not in map_manifest:
        map_manifest["free_evidence_semantics"] = (
            "Expanded whole-body prisms require three separated RGB supports; unknown remains blocked."
        )
    map_manifest["portable_source_manifest_sha256"] = sha256(paths["memory_manifest"])
    write(output / "memory/manifest.json", map_manifest)
    shutil.copy2(paths["policy"], output / "policy.zip")
    shutil.copy2(paths["vision"], output / "vision.pt")
    (output / "scene").mkdir()
    for name, field, relative in (
        ("task", "world_config", "room.yaml"),
        ("world", "project", "room.genesis.json"),
    ):
        # Normalize only the dependency path; retain source hashes in provenance.
        data = {**prepared[name], field: relative}
        destination = (
            output / "scene" / ("task.yaml" if name == "task" else "room.yaml")
        )
        if prepared[name][field] == relative:
            shutil.copy2(paths[name], destination)
        else:
            destination.write_text(yaml.safe_dump(data, sort_keys=False))
    shutil.copy2(paths["project"], output / "scene/room.genesis.json")
    write(output / "input-manifest.json", {**manifest, "root": "<INPUT_ROOT>"})
    write(
        output / "demo.json",
        {
            "task": "scene/task.yaml",
            "memory": "memory",
            "policy": "policy.zip",
            "vision": "vision.pt",
            "start": case["start"],
            "goal": case["goal"],
            "seed": case["seed"],
            "layout_seed": case["layout_seed"],
            "planning_radius": 0.10,
            "cameras": prepared["cameras"],
            "scope": "Selected synthetic development route; known metric scan poses; lockstep simulation; no hardware.",
        },
    )
    write(
        output / "provenance.json",
        {
            "source_map_manifest_sha256": sha256(paths["memory_manifest"]),
            "source_registration_sha256": sha256(paths["registration"]),
            "source_protocol_sha256": sha256(paths["protocol"]),
            "input_manifest": "input-manifest.json",
            "input_manifest_sha256": sha256(output / "input-manifest.json"),
            "free_evidence_mode": map_manifest.get("free_evidence_mode", "full_prism"),
            "free_evidence_semantics": map_manifest["free_evidence_semantics"],
            "model_origin": "Explicit checksummed policy and vision inputs; training and licensing provenance must be reviewed separately.",
            "map_origin": "Explicit checksummed map artifacts; reconstruction and registration evidence described by source manifests.",
            "limits": [
                "Metric scan poses known",
                "Static grounded opaque geometry",
                "Unknown map cells blocked",
                "Heuristic uncertainty",
                "Selected short development route",
            ],
        },
    )
    write(
        output / "bundle.json",
        {
            "schema": "bb8.interactive-assets.v1",
            "sha256": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    validate_assets(output)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                # Source tree mtimes and host file modes do not define an input.
                info = zipfile.ZipInfo(path.relative_to(output).as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with path.open("rb") as source, bundle.open(info, "w") as target:
                    shutil.copyfileobj(source, target)
    print(
        json.dumps(
            {
                "assets": str(output),
                "archive": str(archive),
                "sha256": sha256(archive),
                "bytes": archive.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, help="Versioned checksummed input manifest"
    )
    parser.add_argument("--input-root", type=Path, help="Relocated input tree root")
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Verify source bytes/configuration without building or loading weights",
    )
    parser.add_argument(
        "--write-input-manifest",
        type=Path,
        help="Capture selected inputs at a fresh path for portable rebuilds",
    )
    parser.add_argument("--scan-dir", type=Path, help="Legacy source capture directory")
    parser.add_argument(
        "--memory",
        type=Path,
        help="Independently audited scan-memory directory to package",
    )
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
