"""Package this project's frozen demo inputs; never fetch or train models."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from bb8_rl.demo_assets import sha256, validate_assets

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def portable(value, roots):
    if isinstance(value, dict):
        return {portable(k, roots): portable(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(item, roots) for item in value]
    if isinstance(value, str):
        for root, replacement in roots:
            value = value.replace(str(root), replacement)
    return value


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    memory = args.memory.resolve()
    registration_path = ROOT / "work/m77/registration/superpoint-lightglue/report.json"
    registration = json.loads(registration_path.read_text())
    protocol_path = (
        ROOT / "work/m78/supplemental-fixture/v6-camera20-r20-short/protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    case = protocol["cases"][0]
    (output / "memory").mkdir()
    map_manifest = json.loads((memory / "manifest.json").read_text())
    for name in map_manifest["artifact_sha256"]:
        shutil.copy2(memory / name, output / "memory" / name)
    map_manifest = portable(
        map_manifest,
        [(ROOT, "<BB8-RL>"), (args.scan_dir.parent.parent, "<CAPTURE_ROOT>")],
    )
    # Clarify legacy wording without changing any evidence, arrays or assumptions.
    map_manifest["assumptions"] = [
        "synthetic palette candidates overlap obstacle colors and require learned/geometric intersection"
        if text == "distinct synthetic floor palette"
        else text
        for text in map_manifest["assumptions"]
    ]
    if "free_evidence_mode" not in map_manifest:
        map_manifest["free_evidence_semantics"] = (
            "Expanded whole-body prisms require three separated RGB supports; unknown remains blocked."
        )
    map_manifest["portable_source_manifest_sha256"] = sha256(memory / "manifest.json")
    write(output / "memory/manifest.json", map_manifest)
    shutil.copy2(ROOT / "work/m6/sac-2/model.zip", output / "policy.zip")
    shutil.copy2(ROOT / "work/m75/calibrated/model.pt", output / "vision.pt")
    (output / "scene").mkdir()
    for name in ("task.yaml", "room.yaml", "room.genesis.json"):
        shutil.copy2(
            ROOT / "projects/bb8/synthetic-room" / name, output / "scene" / name
        )
    cameras = {}
    for camera_id, query in zip("ABC", (20, 23, 14), strict=True):
        metadata = json.loads((args.scan_dir / f"scan-{query:03d}.json").read_text())
        estimate = next(
            row for row in registration["queries"] if row["query_index"] == query
        )
        assert estimate["status"] == "candidate"
        cameras[camera_id] = {
            "query_index": query,
            "position": metadata["position"],
            "lookat": [0, 0, 0.1],
            "intrinsics": metadata["intrinsics"],
            "world_to_camera": estimate["world_to_camera"],
            "resolution": [1280, 960],
            "calibration_version": f"m77-query-{query}-{registration['landmarks_sha256']}",
            "provenance": "RGB-estimated pose; synthetic intrinsics/metric scan frame. Render fixture position is excluded from control.",
        }
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
            "cameras": cameras,
            "scope": "Selected synthetic development route; known metric scan poses; lockstep simulation; no hardware.",
        },
    )
    write(
        output / "provenance.json",
        {
            "source_map_manifest_sha256": sha256(memory / "manifest.json"),
            "source_registration_sha256": sha256(registration_path),
            "source_protocol_sha256": sha256(protocol_path),
            "free_evidence_mode": map_manifest.get("free_evidence_mode", "full_prism"),
            "free_evidence_semantics": map_manifest["free_evidence_semantics"],
            "model_origin": "Project-trained SAC and compact vision models, unchanged from M7.8.",
            "third_party_weights_included": False,
            "map_origin": "Project synthetic RGB scan and project photometric/free-volume reconstruction; third-party matching outputs used for camera registration.",
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
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
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
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument(
        "--memory", type=Path,
        default=ROOT / "work/m78/scan-free-memory-v6-hull-fine",
        help="Independently audited scan-memory directory to package",
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
