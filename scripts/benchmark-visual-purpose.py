"""Freeze, run and independently audit bounded native RGB interaction cases.

No native module is imported until an explicitly selected execution starts.
Fresh/copy-on-run databases isolate every case from the user's memories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import queue
import random
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from bb8_rl.camera import Calibration
from bb8_rl.camera_rig import CameraFrame
from bb8_rl.visual_interaction import PixelVisualObserver, validate_visual_outcome

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "src/bb8_rl").is_dir()
)
SOURCES = (
    "scripts/benchmark-visual-purpose.py",
    "src/bb8_rl/visual_interaction.py",
    "src/bb8_rl/visual_purpose_runtime.py",
    "src/bb8_rl/visual_fixture.py",
    "src/bb8_rl/purpose.py",
    "src/bb8_rl/purpose_runtime.py",
    "src/bb8_rl/interactive_runtime.py",
    "src/bb8_rl/interaction_environment.py",
    "src/bb8_rl/control/genesis_backend.py",
    "src/bb8_rl/visual_evidence.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def validate_preflight(path):
    """Admit only the actual B path; do not relabel C failures as successes."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    cases = report.get("cases", [])
    checks = report.get("checks", {})
    required = (
        "all_level_navigation_visible",
        "all_level_navigation_position_accurate",
        "all_level_scene_guard_stable",
        "stationary",
        "no_contacts_or_boundary",
        "identity_unchanged",
    )
    if (
        report.get("status") != "complete"
        or len(cases) != 40
        or len({item.get("case") for item in cases}) != 40
        or not all(checks.get(key) is True for key in required)
        or any(
            not item.get("semantic", {}).get("B", {}).get("checks")
            or not all(item["semantic"]["B"]["checks"].values())
            for item in cases
        )
    ):
        raise ValueError("The live semantic B/native navigation preflight did not pass")
    for item in cases:
        image = item["semantic"]["B"]
        image_path = (path.parent / image["image"]).resolve()
        if (
            not image_path.is_relative_to(path.parent)
            or digest(image_path) != image["image_sha256"]
        ):
            raise ValueError("B preflight native pixels were changed")
    protocol = json.loads((path.parent / "protocol.json").read_text())
    for source in (
        "src/bb8_rl/visual_interaction.py",
        "src/bb8_rl/visual_fixture.py",
        "src/bb8_rl/vision.py",
        "src/bb8_rl/camera.py",
    ):
        if digest(ROOT / source) != protocol["sources_sha256"][source]:
            raise ValueError("Renderer or decoder changed since native B preflight")
    return {
        "report_path": str(path),
        "report_sha256": digest(path),
        "protocol_sha256": report["protocol_sha256"],
        "live_B_cases": 40,
        "full_B_and_C_passed": report["passed"],
        "scope": "B semantic path passes; retained C limitations remain excluded from live execution",
    }


def external_history_records(base, *, bundle_sha256):
    """Freeze previously audited donors without retraining or changing their DBs."""
    base = Path(base).resolve()
    original = json.loads((base / "protocol.json").read_text())
    payload = dict(original)
    expected = payload.pop("protocol_sha256")
    if (
        hashlib.sha256(canonical(payload).encode()).hexdigest() != expected
        or original["bundle_sha256"] != bundle_sha256
    ):
        raise ValueError("External donor protocol/bundle differs")
    for source, value in original["source_sha256"].items():
        if (
            source != "scripts/benchmark-visual-purpose.py"
            and digest(ROOT / source) != value
        ):
            raise ValueError("Runtime changed since donor history execution")
    records = {}
    for name in ("train-a", "train-b"):
        directory = base / name
        audit = json.loads((directory / "audit.json").read_text())
        if not audit["passed"] or audit["admission"]["useful_outcomes"] < 2:
            raise ValueError(
                "External donor lacks independently audited native history"
            )
        records[name] = {
            "directory": str(directory),
            "protocol_sha256": expected,
            "memory_sha256": digest(directory / "memory.sqlite3"),
            "audit_sha256": digest(directory / "audit.json"),
            "snapshot_sha256": hashlib.sha256(
                canonical(memory_snapshot(directory / "memory.sqlite3")).encode()
            ).hexdigest(),
        }
    return records


def history_directory(protocol, name, output):
    record = protocol.get("external_histories", {}).get(name)
    if record is None:
        return output / name
    directory = Path(record["directory"])
    if (
        digest(directory / "memory.sqlite3") != record["memory_sha256"]
        or digest(directory / "audit.json") != record["audit_sha256"]
    ):
        raise ValueError("Frozen external donor evidence changed")
    if (
        hashlib.sha256(
            canonical(memory_snapshot(directory / "memory.sqlite3")).encode()
        ).hexdigest()
        != record["snapshot_sha256"]
    ):
        raise ValueError("Frozen external donor contents changed")
    return directory


def make_protocol(assets, preflight, *, supplemental_history_base=None):
    from bb8_rl.demo_assets import validate_assets

    assets = Path(assets).resolve()
    preflight_gate = validate_preflight(preflight)
    validate_assets(assets)
    cases = [
        ("train-a", 3, None, {"marker-01": 0.45, "marker-02": 0}, "learn"),
        ("train-b", 3, None, {"marker-01": 0, "marker-02": 0.45}, "learn"),
        (
            "history-a",
            1,
            "train-a",
            {"marker-01": 0.45, "marker-02": 0.45},
            "first_outcome",
        ),
        (
            "history-b",
            1,
            "train-b",
            {"marker-01": 0.45, "marker-02": 0.45},
            "first_outcome",
        ),
        (
            "disabled-restart",
            2,
            "train-b",
            {"marker-01": 0, "marker-02": 0.45},
            "disabled",
        ),
        ("reversal", 3, "train-b", {"marker-01": 0.45, "marker-02": 0}, "learn"),
        ("useless", 2, None, {"marker-01": 0, "marker-02": 0}, "useless_idle"),
        (
            "stop-pending",
            3,
            "train-b",
            {"marker-01": 0, "marker-02": 0.45},
            "stop_pending",
        ),
        (
            "hidden-post",
            3,
            "train-b",
            {"marker-01": 0, "marker-02": 0.45},
            "hidden_post",
        ),
    ]
    result = {
        "schema": "bb8.visual-purpose-protocol.v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing-room synthetic native RGB interaction development; not held-out navigation or generic object understanding.",
        "asset_dir": str(assets),
        "bundle_sha256": digest(assets / "bundle.json"),
        "source_sha256": {name: digest(ROOT / name) for name in SOURCES},
        "preflight": preflight_gate,
        "navigation_camera_modes": [1, 2, 3],
        "semantic_camera": "Dedicated registered B pose at 2560x1920, in addition to 1/2/3 navigation cameras",
        "initial_resource": 0.25,
        "sim_seconds": 120,
        "wall_seconds": 600,
        "stop_tail_seconds": 1.0,
        "terminal_cancellation_tail_seconds": 2.0,
        "terminal_cancellation_policy": "After observed enable, unexpected evidence/authority loss or guarded stop sends Stop immediately, retains two simulated seconds, and fails without retry/re-enable; planned Stop/hidden-post are scored separately.",
        "inactive_window_seconds": 2.0,
        "cases": [
            {
                "id": name,
                "mode": mode,
                "history": history,
                "effects": effects,
                "intent": intent,
            }
            for name, mode, history, effects, intent in cases
        ],
        "requested_native_cases": len(cases),
        "offline_control_seeds": [0, 1, 2],
        "offline_controls": [
            "learned",
            "memory_ablation",
            "random",
            "nearest",
            "fixed_marker_01",
            "fixed_marker_02",
        ],
        "offline_scope": "Choices on one recorded native RGB observation with copied native outcome histories; no new native interactions and no independent episode denominator.",
        "pass_criteria": {
            "complete_physics_and_semantic_images": True,
            "no_contacts_or_unauthorized_motion": True,
            "every_new_event_matches_redecoded_rgb_and_durable_store": True,
            "history_probes_choose_different_retained_useful_entities": True,
            "restart_retains_knowledge_without_authority": True,
            "stop_and_hidden_post_do_not_train": True,
            "useless_stops_probing_without_external_intervention": True,
        },
        "limitations": [
            "B-only native preflight must pass before execution; C has four retained boundary-sampling abstentions and is not the live semantic view.",
            "Successful stops/rejections do not count as arrivals or useful outcomes.",
            "Nine correlated development cases cannot establish broad navigation reliability.",
            "Hidden-post is a declared native indicator-visibility fault, separate from physical obstacle occlusion.",
        ],
    }
    if supplemental_history_base is not None:
        result["cases"] = [
            {**case, "mode": 3}
            for case in result["cases"]
            if case["id"] in ("history-a", "history-b", "hidden-post")
        ]
        result["requested_native_cases"] = 3
        result["scope"] = (
            "Separate supplemental three-navigation-camera plus dedicated B semantic RGB history comparison and corrected hidden-post harness; original mode1 failures and the prematurely stopped hidden-post case remain failed."
        )
        result["navigation_camera_modes"] = [3]
        result["external_histories"] = external_history_records(
            supplemental_history_base, bundle_sha256=result["bundle_sha256"]
        )
        result["pass_criteria"] = {
            key: value
            for key, value in result["pass_criteria"].items()
            if key
            in (
                "complete_physics_and_semantic_images",
                "no_contacts_or_unauthorized_motion",
                "every_new_event_matches_redecoded_rgb_and_durable_store",
                "history_probes_choose_different_retained_useful_entities",
            )
        }
        result["pass_criteria"][
            "hidden_post_world_completion_then_pixel_abstention"
        ] = True
        result["limitations"] = [
            "This is a separately frozen supplemental development experiment, not a retry credited to the original two failed mode1 requests or prematurely stopped hidden-post case.",
            "Only navigation camera count changes versus the original paired probe; renderer, decoder, weights, stationary-start setup, effects and original donor histories stay fixed.",
            "Three correlated development cases do not establish broad navigation or semantic reliability.",
        ]
    result["protocol_sha256"] = hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def validate_protocol(protocol):
    frozen = dict(protocol)
    expected = frozen.pop("protocol_sha256", None)
    if (
        frozen.get("schema") != "bb8.visual-purpose-protocol.v1"
        or hashlib.sha256(canonical(frozen).encode()).hexdigest() != expected
    ):
        raise ValueError("Visual protocol was changed after freeze")
    if (
        len({case["id"] for case in frozen["cases"]})
        != frozen["requested_native_cases"]
    ):
        raise ValueError("Invalid all-request denominator")
    if any(
        digest(ROOT / name) != value for name, value in frozen["source_sha256"].items()
    ):
        raise ValueError("Visual execution source changed after freeze")
    if digest(Path(frozen["asset_dir"]) / "bundle.json") != frozen["bundle_sha256"]:
        raise ValueError("Visual bundle changed after freeze")
    if validate_preflight(frozen["preflight"]["report_path"]) != frozen["preflight"]:
        raise ValueError("Rendering preflight changed after freeze")
    for name in protocol.get("external_histories", {}):
        history_directory(protocol, name, Path("."))
    return protocol


def memory_snapshot(path):
    path = Path(path)
    if not path.exists():
        return {"events": [], "entities": [], "scopes": [], "stations": []}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        return {
            key: [dict(row) for row in database.execute(f"SELECT * FROM {table}")]
            if table in tables
            else []
            for key, table in (
                ("events", "purpose_events"),
                ("entities", "visual_entities"),
                ("scopes", "purpose_scopes"),
                ("stations", "purpose_stations"),
            )
        }


def copy_memory(source, destination):
    if Path(destination).exists():
        raise ValueError("Refusing to overwrite an experiment memory")
    with (
        closing(
            sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
        ) as old,
        closing(sqlite3.connect(destination)) as new,
    ):
        old.backup(new)


def matching_redecoded_evidence(actual, expected):
    """Keep native evidence exact; tolerate only spatial rederivation roundoff."""
    if (
        not isinstance(actual, dict)
        or not isinstance(expected, dict)
        or set(actual) != set(expected)
    ):
        return False
    spatial = {"center_xy", "radius_m"}
    if any(
        actual[key] != expected[key] or type(actual[key]) is not type(expected[key])
        for key in expected
        if key not in spatial
    ):
        return False
    left, right = actual.get("center_xy"), expected.get("center_xy")
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != 2
        or len(right) != 2
    ):
        return False
    for x, y in zip(
        [*left, actual.get("radius_m")], [*right, expected.get("radius_m")], strict=True
    ):
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in (x, y)
        ):
            return False
        if abs(x - y) > 1e-10:
            return False
    return True


def navigation_observation_metrics(rows):
    """Arrival assertions are frames; visits are distinct rising edges."""
    errors, arrivals, prior = [], 0, None
    for row in rows:
        arrived = bool(row.get("controller_arrived"))
        signature = (
            (row.get("generation"), row.get("commanded_goal")) if arrived else None
        )
        if signature is not None and signature != prior:
            arrivals += 1
        prior = signature
        localization = row.get("localization", {})
        pose = localization.get("pose")
        truth = row.get("truth_before_scoring_only", {}).get("position")
        if (
            localization.get("localization_valid")
            and localization.get("localization_status") == "measured"
            and isinstance(pose, list)
            and isinstance(truth, list)
            and len(pose) == len(truth) == 2
        ):
            error = math.dist(pose, truth)
            if math.isfinite(error):
                errors.append(error)
    return {
        "controller_arrival_entries": arrivals,
        "measured_pose_error_m_scoring_only": {
            "samples": len(errors),
            "maximum": max(errors) if errors else None,
            "p95": float(np.percentile(errors, 95)) if errors else None,
        },
    }


def score_receipts(rows, decoded, before_store, after_store):
    """Independent raw-pixel/authority/store admission parity, without inference."""
    errors, receipts = [], {}
    prior = {item["id"]: item for item in before_store["events"]}
    durable = {item["id"]: item for item in after_store["events"]}
    for row in rows:
        receipt = row.get("agency", {}).get("visual_outcome")
        if receipt is None:
            continue
        try:
            receipt = validate_visual_outcome(receipt)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"invalid_receipt:{error}")
            continue
        identifier = receipt["request_id"]
        if identifier in receipts:
            if receipt != receipts[identifier]:
                errors.append(f"changed_receipt:{identifier}")
            continue
        receipts[identifier] = receipt
        if identifier in prior:
            errors.append(f"old_request_reused:{identifier}")
        if row["time"] != receipt["timestamp"]:
            errors.append(f"receipt_not_current:{identifier}")
        for evidence in (
            receipt["before_frames"]
            + receipt["during_frames"]
            + receipt["after_frames"]
        ):
            key = (evidence["timestamp"], evidence["camera_id"], evidence["marker_id"])
            actual = decoded.get(key)
            if not matching_redecoded_evidence(actual, evidence):
                errors.append(
                    f"pixel_evidence_mismatch:{identifier}:{evidence['timestamp']}"
                )
        active = [
            sample
            for sample in rows
            if receipt["started_at"] < sample["time"] <= receipt["timestamp"]
        ]
        if not active or any(
            sample.get("generation") != receipt["generation"]
            or not sample.get("agency", {}).get("enabled")
            or not sample.get("localization", {}).get("localization_valid")
            or sample.get("localization", {}).get("localization_status") != "measured"
            or (
                sample["time"] < receipt["timestamp"]
                and not sample.get("controller_arrived")
            )
            or any(abs(value) > 1e-7 for value in sample["action"])
            for sample in active
        ):
            errors.append(f"unauthorized_or_nonstationary_evidence:{identifier}")
        entity = next(
            (
                item
                for item in after_store["entities"]
                if item["entity"] == receipt["entity_id"]
                and item["map"] == receipt["map_version"]
            ),
            None,
        )
        if (
            entity is None
            or entity["calibration"] != receipt["calibration_version"]
            or entity["marker"] != receipt["marker_id"]
            or [entity["x"], entity["y"]] != receipt["anchor_xy"]
            or entity["radius"] != receipt["anchor_radius_m"]
        ):
            errors.append(f"persistent_entity_binding_mismatch:{identifier}")
        record = durable.get(identifier)
        payload = json.loads(record["payload"]) if record else {}
        if (
            payload.get("source") != "native_scene_rgb"
            or payload.get("visual_evidence") != receipt
            or payload.get("station_id") != receipt["entity_id"]
            or any(
                payload.get(key) != receipt[other]
                for key, other in (
                    ("before", "before"),
                    ("after", "after"),
                    ("now", "timestamp"),
                )
            )
        ):
            errors.append(f"durable_event_mismatch:{identifier}")
    if set(durable) - set(prior) != set(receipts):
        errors.append("sqlite_new_event_set_mismatch")
    if any(durable.get(identifier) != record for identifier, record in prior.items()):
        errors.append("prior_history_was_changed")
    for entity in before_store["entities"]:
        if entity not in after_store["entities"]:
            errors.append("prior_entity_binding_changed")
    return {
        "passed": not errors,
        "errors": errors,
        "new_outcomes": len(receipts),
        "useful_outcomes": sum(
            r["classification"] == "useful" for r in receipts.values()
        ),
        "receipts": list(receipts.values()),
    }


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        "visual_auditor_" + name.replace("-", "_"), ROOT / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_images(directory, rows, record, *, extent=2.0):
    """Replay all recorded semantic RGB without consulting world outcome values."""
    calibration = Calibration(
        np.asarray(record["intrinsics"]),
        np.asarray(record["world_to_camera"]),
        tuple(record["resolution"]),
        extent,
    )
    observer = PixelVisualObserver(
        calibration, record["id"], record["calibration_version"]
    )
    decoded, errors, scenes = {}, [], []
    for row in rows:
        image = row.get("images", {}).get(record["id"])
        if not isinstance(image, dict):
            errors.append(f"missing_semantic_image:{row['time']}")
            continue
        try:
            path = (directory / image["rgb_path"]).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or digest(path) != image["rgb_sha256"]
            ):
                raise ValueError("PNG hash or path differs")
            rgb = cv2.imread(str(path))
            if rgb is None:
                raise ValueError("Image cannot be decoded")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            raw = hashlib.sha256(rgb.tobytes()).hexdigest()
            if (
                raw != image["raw_rgb_sha256"]
                or image["timestamp"] != row["capture_time"]
                or image["step"] != row["step"]
                or image.get("retained") is not True
            ):
                raise ValueError("Raw RGB hash or capture metadata differs")
            frame = CameraFrame(
                record["id"], record["calibration_version"], rgb, image["timestamp"]
            )
            scene = observer.observe(frame, now=row["time"])
            scenes.append(
                {
                    "timestamp": scene.timestamp,
                    "status": scene.status,
                    "reason": scene.reason,
                    "frame_sha256": scene.frame_sha256,
                }
            )
            for reading in scene.readings:
                decoded[(reading.timestamp, reading.camera_id, reading.marker_id)] = (
                    reading.evidence()
                )
        except (KeyError, OSError, ValueError, TypeError) as error:
            errors.append(f"semantic_image:{row['time']}:{error}")
    return decoded, errors, scenes


def verify_archives(directory, receipts):
    errors = []
    for receipt in receipts:
        try:
            path = (
                directory
                / "visual-evidence"
                / f"outcome-{receipt['evidence_sha256']}.json"
            )
            archive = json.loads(path.read_text())
            expected = {
                frame["frame_sha256"]
                for field in ("before_frames", "during_frames", "after_frames")
                for frame in receipt[field]
            }
            if archive["evidence"] != receipt or set(archive["images"]) != expected:
                raise ValueError("Archive receipt or pixel set differs")
            for raw_hash, record in archive["images"].items():
                image = (directory / record["rgb_path"]).resolve()
                if (
                    not image.is_relative_to(directory.resolve())
                    or digest(image) != record["rgb_sha256"]
                ):
                    raise ValueError("Archived PNG differs")
                rgb = cv2.imread(str(image))
                if (
                    rgb is None
                    or hashlib.sha256(
                        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).tobytes()
                    ).hexdigest()
                    != raw_hash
                ):
                    raise ValueError("Archived raw RGB differs")
        except (KeyError, OSError, ValueError, TypeError) as error:
            errors.append(f"outcome_archive:{receipt['request_id']}:{error}")
    return errors


def score_intent(rows, commands, case, admission, stop, scenes, before):
    """Separate a completed interaction from an abstention or inactive startup."""
    outcomes = admission["new_outcomes"]
    intent = case["intent"]
    stop_at = stop["accepted_sim_time"]
    active = [r for r in rows if stop_at is None or r["time"] < stop_at]
    requests = [r for r in active if r.get("agency", {}).get("interaction_request")]
    world_completions = [
        r
        for r in active
        if (r.get("interaction_world_scoring_only") or {}).get("outcome")
    ]
    # Outcome values stay exclusively in the native mechanism scorer, never decode/admission.
    idle = []
    for row in active:
        if row.get("agency", {}).get("status") == "idle" and row["agency"].get(
            "enabled"
        ):
            idle.append(row)
        else:
            idle = []
    idle_seconds = idle[-1]["time"] - idle[0]["time"] if idle else 0
    no_enable = not any(
        c.get("action") == "autonomy"
        and c.get("enabled")
        and c.get("outcome") == "accepted"
        for c in commands
    )
    if intent == "learn":
        passed = admission["useful_outcomes"] >= (2 if case["history"] is None else 1)
    elif intent == "first_outcome":
        passed = outcomes == 1 and admission["useful_outcomes"] == 1
    elif intent == "disabled":
        passed = bool(
            before["events"]
            and rows
            and no_enable
            and outcomes == 0
            and all(
                not r["agency"].get("enabled")
                and r["commanded_goal"] is None
                and all(abs(a) < 1e-7 for a in r["action"])
                for r in rows
            )
        )
    elif intent == "useless_idle":
        passed = (
            outcomes >= 2
            and admission["useful_outcomes"] == 0
            and idle_seconds >= 2.0 - 1e-8
            and all(
                r["agency"].get("interaction_request") is None
                and r["commanded_goal"] is None
                and all(abs(a) < 1e-7 for a in r["action"])
                for r in idle
            )
        )
    elif intent == "stop_pending":
        passed = bool(
            requests
            and stop_at is not None
            and 0 <= stop_at - requests[-1]["time"] <= 0.15
            and not world_completions
            and outcomes == 0
            and stop["passed"]
        )
    elif intent == "hidden_post":
        hidden_after = [
            scene
            for scene in scenes
            if world_completions
            and scene["timestamp"]
            >= world_completions[0]["interaction_world_scoring_only"]["outcome"][
                "timestamp"
            ]
            and scene["status"] != "visible"
        ]
        paused_after = [
            r
            for r in active
            if world_completions
            and r["time"] >= world_completions[0]["time"]
            and not r["agency"].get("enabled")
        ]
        passed = bool(
            requests
            and world_completions
            and hidden_after
            and paused_after
            and outcomes == 0
        )
    else:
        raise ValueError("Unknown case intent")
    return {
        "passed": bool(passed),
        "idle_seconds": idle_seconds,
        "requested_frames": len(requests),
        "world_completion_frames_scoring_only": len(world_completions),
    }


def score_case(directory, case, protocol, *, report_directory=None):
    worker = directory / "worker"
    report_directory = directory if report_directory is None else Path(report_directory)
    manifest = json.loads((worker / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Only complete native manifests may be scored")
    rows, commands = (
        read_jsonl(worker / "rows.jsonl"),
        read_jsonl(worker / "commands.jsonl"),
    )
    auditor = _load_script("audit-interactive.py")
    physics = auditor.score_rows(rows)
    provenance = auditor.verify_manifest(worker, manifest)
    provenance.update(auditor.verify_recording(rows, manifest))
    navigation_rows = [
        {
            **row,
            "images": {
                name: record
                for name, record in row.get("images", {}).items()
                if name in row.get("views", {})
            },
        }
        for row in rows
    ]
    images = auditor.verify_images(worker, navigation_rows)
    decoded, pixel_errors, scenes = decode_images(
        worker, rows, manifest["interaction_fixture"]["visual_camera"]
    )
    before = json.loads((directory / "memory-before.json").read_text())
    after = memory_snapshot(directory / "memory.sqlite3")
    admission = score_receipts(rows, decoded, before, after)
    stop = _load_script("benchmark-purpose-navigation.py").score_stop(rows, commands)
    initial = rows[0]["agency"] if rows else {}
    intended = score_intent(rows, commands, case, admission, stop, scenes, before)
    archive_errors = verify_archives(worker, admission["receipts"])
    preflight = json.loads(
        (
            Path(protocol["preflight"]["report_path"]).parent / "protocol.json"
        ).read_text()
    )
    registered = preflight["registered_cameras"]["B"]
    live_camera = manifest["interaction_fixture"]["visual_camera"]
    intrinsics = np.asarray(registered["intrinsics"]).copy()
    intrinsics[:2] *= 2
    if (
        live_camera["id"] != "V"
        or not np.array_equal(intrinsics, live_camera["intrinsics"])
        or not np.array_equal(
            registered["world_to_camera"], live_camera["world_to_camera"]
        )
        or list(live_camera["resolution"]) != [2 * v for v in registered["resolution"]]
    ):
        pixel_errors.append("semantic_camera_differs_from_native_B_preflight")
    if any(
        set(row.get("images", {})) != set(row.get("views", {})) | {"V"} for row in rows
    ):
        pixel_errors.append("recorded_camera_set_mismatch")
    if any(row.get("interaction_telemetry") is not None for row in rows):
        pixel_errors.append("numeric_telemetry_exposed_in_visual_mode")
    passed = bool(
        rows
        and physics["integrity_pass"]
        and not physics["collision"]
        and not physics["boundary"]
        and all(provenance.values())
        and not images["failures"]
        and images["unavailable_frames"] == 0
        and not pixel_errors
        and admission["passed"]
        and stop["passed"]
        and intended["passed"]
        and not archive_errors
    )
    result = {
        "schema": "bb8.visual-purpose-case-audit.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "denominators": {
            "requested_native_case": 1,
            "initial_goal_ready": int(
                any(
                    r["localization"].get("localization_valid")
                    and r.get("scene_validity", {}).get("navigation_allowed")
                    and r["agency"].get("visual_status") == "fresh_rgb"
                    for r in rows
                )
            ),
            "autonomy_enable_accepted": sum(
                c.get("action") == "autonomy"
                and c.get("enabled") is True
                and c.get("outcome") == "accepted"
                for c in commands
            ),
            "scored_arrival_frames": physics["valid_arrivals"],
            "controller_arrival_entries": navigation_observation_metrics(rows)[
                "controller_arrival_entries"
            ],
            "rgb_admitted_outcomes": admission["new_outcomes"],
        },
        "case": case["id"],
        "passed": passed,
        "scope": "Native case with independent recorded-RGB redecoding and SQLite parity",
        "physics": physics,
        "navigation_observation_metrics": navigation_observation_metrics(rows),
        "audit_spatial_roundoff_tolerance_m": 1e-10,
        "provenance": provenance,
        "navigation_images": images,
        "pixel_errors": pixel_errors,
        "archive_errors": archive_errors,
        "decoded_scenes": scenes,
        "admission": admission,
        "stop": stop,
        "intended": intended,
        "initial_episodes": initial.get("episodes"),
        "final_episodes": rows[-1]["agency"].get("episodes") if rows else None,
        "memory_after_sha256": digest(directory / "memory.sqlite3"),
        "input_sha256": {
            name: digest(worker / name)
            for name in ("manifest.json", "rows.jsonl", "commands.jsonl")
        },
    }
    harness_path = directory / "harness.json"
    harness = json.loads(harness_path.read_text()) if harness_path.exists() else {}
    result["harness_errors"] = harness.get("errors", ["missing_harness_completion"])
    if result["harness_errors"] or harness.get("exitcode") != 0:
        result["passed"] = False
    write_json(report_directory / "audit.json", result)
    write_json(report_directory / "memory-after.json", after)
    return result


def retained_useful_markers(snapshot):
    mapping = {row["entity"]: row["marker"] for row in snapshot["entities"]}
    return {
        mapping.get(event["station"])
        for event in snapshot["events"]
        if event["response"]
    }


def score_history_comparison(output):
    """Require opposite observed outcomes to yield opposite live probe choices."""
    results = {}
    for suffix in ("a", "b"):
        directory = output / f"history-{suffix}"
        audit = json.loads((directory / "audit.json").read_text())
        before = json.loads((directory / "memory-before.json").read_text())
        receipts = audit["admission"]["receipts"]
        useful = retained_useful_markers(before)
        results[suffix] = {
            "passed": bool(
                audit["passed"]
                and len(receipts) == 1
                and receipts[0]["marker_id"] in useful
            ),
            "chosen_marker": receipts[0]["marker_id"] if receipts else None,
            "prior_useful_markers": sorted(useful),
        }
    return {
        "passed": all(v["passed"] for v in results.values())
        and results["a"]["chosen_marker"] != results["b"]["chosen_marker"],
        "histories": results,
    }


def replay_choices(source, destination, snapshot, observations, *, pose, now, seed):
    """One fixed recorded RGB input; copied history and explicit comparison policies."""
    from bb8_rl.purpose import PurposeEngine, PurposeStore, Station
    from bb8_rl.visual_interaction import INTERACTION_OFFSET

    if len(observations) != 2 or len({item["marker_id"] for item in observations}) != 2:
        raise ValueError("Controls require two independently decoded current panels")
    bindings = {row["marker"]: row for row in snapshot["entities"]}
    stations = []
    markers = {}
    resources = []
    for reading in observations:
        if reading["timestamp"] != now:
            raise ValueError("Control observation is stale")
        binding = bindings[reading["marker_id"]]
        anchor = (binding["x"], binding["y"])
        if (
            binding["calibration"] != reading["calibration_version"]
            or math.dist(anchor, reading["center_xy"])
            > binding["radius"] + reading["radius_m"] + 0.01
        ):
            raise ValueError(
                "Native control observation differs from saved identity anchor"
            )
        station = Station(
            binding["entity"],
            tuple(x + d for x, d in zip(anchor, INTERACTION_OFFSET)),
            reading["marker_id"],
        )
        stations.append(station)
        markers[station.id] = reading["marker_id"]
        resources.append(reading["interval"][0])
    if max(resources) - min(resources) > 1e-12:
        raise ValueError("Control gauges disagree")
    stations.sort(key=lambda station: markers[station.id])
    random.Random(seed).shuffle(stations)
    scope = snapshot["scopes"]
    if len(scope) != 1 or scope[0]["source"] != "native_scene_rgb":
        raise ValueError("Controls require one source-bound memory scope")
    destination.mkdir(exist_ok=False)
    copy_memory(source, destination / "learned.sqlite3")
    choices = {}
    for kind in ("learned", "memory_ablation"):
        store = PurposeStore(
            destination
            / ("learned.sqlite3" if kind == "learned" else "ablated.sqlite3"),
            map_version=scope[0]["map"],
            agent_id=scope[0]["agent"],
        )
        try:
            intention = PurposeEngine(store).select(
                stations, resource=resources[0], pose=tuple(pose), now=now
            )
            choices[kind] = (
                None if intention is None else markers[intention["candidate_id"]]
            )
        finally:
            store.close()
    canonical_stations = sorted(stations, key=lambda station: markers[station.id])
    choices.update(
        random=markers[random.Random(seed).choice(canonical_stations).id],
        nearest=markers[
            min(
                canonical_stations,
                key=lambda station: (math.dist(station.xy, pose), markers[station.id]),
            ).id
        ],
        fixed_marker_01="marker-01",
        fixed_marker_02="marker-02",
    )
    return {
        "choices": choices,
        "observed_resource_lower": resources[0],
        "pose": list(pose),
        "observed_rgb_hashes": sorted({item["frame_sha256"] for item in observations}),
    }


def offline_controls(protocol, output):
    """No native calls: all conditions receive exactly the same recorded pixels."""
    directory = output / "history-a"
    manifest = json.loads((directory / "worker/manifest.json").read_text())
    rows = read_jsonl(directory / "worker/rows.jsonl")
    decoded, errors, _ = decode_images(
        directory / "worker", rows, manifest["interaction_fixture"]["visual_camera"]
    )
    if errors:
        raise ValueError("Cannot run controls on unverified recorded pixels")
    choices = [
        (row, [reading for key, reading in decoded.items() if key[0] == row["time"]])
        for row in rows
        if row["localization"].get("localization_valid")
        and not row["agency"].get("enabled")
    ]
    sample, observations = next(
        (row, readings) for row, readings in choices if len(readings) == 2
    )
    if min(item["interval"][0] for item in observations) >= 0.8:
        raise ValueError("Recorded control resource is already satisfied")
    output_dir = output / "offline-controls"
    output_dir.mkdir(exist_ok=False)
    results = []
    for history in ("train-a", "train-b"):
        donor = history_directory(protocol, history, output)
        audit = json.loads((donor / "audit.json").read_text())
        if not audit["passed"]:
            raise ValueError("Control history was not independently verified")
        snapshot = memory_snapshot(donor / "memory.sqlite3")
        for seed in protocol["offline_control_seeds"]:
            result = replay_choices(
                donor / "memory.sqlite3",
                output_dir / f"{history}-seed{seed}",
                snapshot,
                observations,
                pose=sample["localization"]["pose"],
                now=sample["time"],
                seed=seed,
            )
            results.append({"history": history, "seed": seed, **result})
    learned = {
        history: {r["choices"]["learned"] for r in results if r["history"] == history}
        for history in ("train-a", "train-b")
    }
    ablated = {r["choices"]["memory_ablation"] for r in results}
    passed = (
        learned["train-a"] == {"marker-01"}
        and learned["train-b"] == {"marker-02"}
        and len(ablated) == 1
    )
    report = {
        "schema": "bb8.visual-purpose-offline-controls.v1",
        "scope": protocol["offline_scope"],
        "passed": passed,
        "independent_native_episodes": 0,
        "results": results,
        "notes": "Seed varies input ordering/random baseline only. Native histories are fixed. Euclidean nearest does not assert route feasibility.",
    }
    write_json(output_dir / "report.json", report)
    return report


def hidden_post_cancelled(agency, *, enabled_observed):
    """A queued enable is not evidence that this case ever acquired authority."""
    return bool(enabled_observed and agency.get("enabled") is False)


def terminal_cancellation_reason(state, intent, enabled_observed):
    if not enabled_observed or intent in ("stop_pending", "hidden_post", "disabled"):
        return None
    agency = state.get("agency", {})
    if (
        agency.get("enabled") is not True
        or state.get("controller_status") == "guarded_stop"
    ):
        return str(
            agency.get("message")
            or state.get("controller_status")
            or "unexpected authority loss"
        )
    return None


def native_worker(config, commands, events, stop, log_path):
    with open(log_path, "a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        from bb8_rl.interactive_runtime import worker_main

        worker_main(config, commands, events, stop)


def send(channel, value):
    try:
        channel.put_nowait(value)
        return True
    except queue.Full:
        return False


def run_case(protocol, case, output):
    directory = output / case["id"]
    directory.mkdir(exist_ok=False)
    memory = directory / "memory.sqlite3"
    if case["history"]:
        donor = history_directory(protocol, case["history"], output)
        audit = json.loads((donor / "audit.json").read_text())
        if not audit["passed"] or audit["admission"]["useful_outcomes"] < 1:
            raise ValueError("Required native history did not pass independent audit")
        donor_snapshot = memory_snapshot(donor / "memory.sqlite3")
        donor_hash = digest(donor / "memory.sqlite3")
        copy_memory(donor / "memory.sqlite3", memory)
        if (
            memory_snapshot(memory) != donor_snapshot
            or digest(donor / "memory.sqlite3") != donor_hash
        ):
            raise ValueError("Donor copy changed the source or its contents")
        write_json(
            directory / "history-copy.json",
            {
                "source_directory": str(donor.resolve()),
                "source_memory_sha256": donor_hash,
                "source_audit_sha256": digest(donor / "audit.json"),
                "copied_snapshot_sha256": hashlib.sha256(
                    canonical(donor_snapshot).encode()
                ).hexdigest(),
                "source_unchanged": True,
                "copy_contents_equal": True,
            },
        )
    write_json(directory / "memory-before.json", memory_snapshot(memory))
    config = {
        "asset_dir": protocol["asset_dir"],
        "run_dir": str(directory / "worker"),
        "agency_memory": str(memory),
        "agency_mode": "visual",
        "mode": case["mode"],
        "visual_initial_resource": protocol["initial_resource"],
        "visual_effects": case["effects"],
        "visual_hide_after_completion": case["intent"] == "hidden_post",
        "recording_mode": "audit",
        "save_frames": True,
        "save_every": 1,
        "scoring_labels": True,
        "generation": 0,
        "max_steps": int(protocol["sim_seconds"] / 0.05) + 100,
        "control_profile": "reserve-3cm",
        "visibility_planning": True,
        "scene_validity": True,
    }
    write_json(directory / "config.json", config)
    write_json(directory / "case.json", case)
    write_json(
        directory / "protocol-reference.json",
        {"protocol_sha256": protocol["protocol_sha256"]},
    )
    context = multiprocessing.get_context("spawn")
    commands, events, stop = context.Queue(32), context.Queue(32), context.Event()
    process = context.Process(
        target=native_worker,
        args=(config, commands, events, stop, str(directory / "native.log")),
    )
    process.start()
    write_json(
        directory / "native-start.json",
        {"pid": process.pid, "protocol_sha256": protocol["protocol_sha256"]},
    )
    started, heartbeat = time.monotonic(), -math.inf
    ready = inactive = stop_time = request_time = None
    enabled = stopped = enabled_observed = False
    cancellation = None
    state, initial_episodes, errors, ending = None, None, [], None
    try:
        while process.is_alive():
            wall = time.monotonic()
            if wall - heartbeat >= 0.2:
                send(
                    commands, {"action": "heartbeat", "generation": 2 if stopped else 1}
                )
                heartbeat = wall
            if wall - started >= protocol["wall_seconds"]:
                errors.append("wall_limit")
                break
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("type") == "error":
                errors.append(event)
            if event.get("type") != "state":
                continue
            state = event["state"]
            sim, agency = state.get("sim_time", 0), state.get("agency", {})
            if initial_episodes is None:
                initial_episodes = agency.get("episodes", 0)
            if (
                ready is None
                and state.get("localization_valid")
                and state.get("scene_validity", {}).get("navigation_allowed") is True
                and agency.get("visual_status") == "fresh_rgb"
                and agency.get("visual_entities")
            ):
                ready = sim
            if (
                ready is not None
                and case["intent"] != "disabled"
                and not enabled
                and not stopped
            ):
                enabled = send(
                    commands, {"action": "autonomy", "enabled": True, "generation": 1}
                )
            enabled_observed = enabled_observed or agency.get("enabled") is True
            if cancellation is None and not stopped:
                reason = terminal_cancellation_reason(
                    state, case["intent"], enabled_observed
                )
                if reason is not None:
                    cancellation = {"time": sim, "reason": reason}
                    errors.append({"terminal_cancellation": cancellation})
            if cancellation is not None and not stopped:
                stopped = send(commands, {"action": "stop", "generation": 2})
                ending = "terminal_cancellation"
            request = agency.get("interaction_request")
            if request is not None and request_time is None:
                request_time = sim
            quiet = agency.get("status") in ("satisfied", "idle")
            inactive = (sim if inactive is None else inactive) if quiet else None
            elapsed = sim - ready if ready is not None else 0
            finished = (
                case["intent"] == "disabled"
                and ready is not None
                and elapsed >= 2
                or case["intent"] == "first_outcome"
                and agency.get("episodes", 0) > initial_episodes
                or case["intent"] in ("learn", "useless_idle")
                and inactive is not None
                and sim - inactive >= 2
                or case["intent"] == "stop_pending"
                and request_time is not None
                and sim - request_time >= 0.3
                or case["intent"] == "hidden_post"
                and hidden_post_cancelled(agency, enabled_observed=enabled_observed)
                or sim >= protocol["sim_seconds"]
            )
            if finished and not stopped and cancellation is None:
                stopped = send(commands, {"action": "stop", "generation": 2})
                ending = (
                    "case_gate" if sim < protocol["sim_seconds"] else "simulation_limit"
                )
            if stopped and state.get("generation", 0) >= 2:
                stop_time = sim if stop_time is None else stop_time
                if (
                    sim - stop_time
                    >= protocol[
                        "terminal_cancellation_tail_seconds"
                        if cancellation is not None
                        else "stop_tail_seconds"
                    ]
                ):
                    break
    finally:
        stop.set()
        deadline = time.monotonic() + 30
        while process.is_alive() and time.monotonic() < deadline:
            try:
                events.get(timeout=0.1)
            except queue.Empty:
                pass
            process.join(0.1)
        if process.is_alive():
            errors.append("cleanup_timeout")
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        for channel in (commands, events):
            channel.cancel_join_thread()
            channel.close()
    write_json(
        directory / "harness.json",
        {
            "errors": errors,
            "exitcode": process.exitcode,
            "last_state": state,
            "enabled": enabled,
            "enabled_observed": enabled_observed,
            "terminal_cancellation": cancellation,
            "ending": ending,
            "wall_seconds": time.monotonic() - started,
        },
    )
    result = score_case(directory, case, protocol)
    if errors or process.exitcode:
        result["passed"] = False
        result["harness_errors"] = errors
        write_json(directory / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--offline-controls", action="store_true")
    parser.add_argument("--supplemental-history-base", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--case", default="all")
    args = parser.parse_args()
    if args.freeze:
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / "protocol.json"
        if destination.exists():
            parser.error("Refusing to replace a frozen protocol")
        if args.preflight is None:
            parser.error(
                "Freeze requires the retained native B rendering --preflight report"
            )
        write_json(
            destination,
            make_protocol(
                args.assets,
                args.preflight,
                supplemental_history_base=args.supplemental_history_base,
            ),
        )
        print(destination)
        return
    if args.protocol is None:
        parser.error("Native execution requires --protocol from an explicit --freeze")
    protocol = validate_protocol(json.loads(args.protocol.read_text()))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.offline_controls:
        write_json(
            args.output / "history-comparison.json",
            score_history_comparison(args.output),
        )
        offline_controls(protocol, args.output)
        return
    if args.case != "all" and args.case not in {
        case["id"] for case in protocol["cases"]
    }:
        parser.error("Unknown native case")
    results = []
    for case in protocol["cases"]:
        existing = args.output / case["id"] / "audit.json"
        if existing.exists():
            record = json.loads(existing.read_text())
            if record.get("protocol_sha256") != protocol["protocol_sha256"]:
                raise ValueError("Existing result belongs to another frozen protocol")
            results.append(
                {
                    "case": case["id"],
                    "status": "executed",
                    "passed": record["passed"],
                    "denominators": record.get("denominators", {}),
                }
            )
            continue
        if args.case not in ("all", case["id"]):
            results.append(
                {"case": case["id"], "status": "not_selected", "passed": False}
            )
            continue
        try:
            result = run_case(protocol, case, args.output)
            results.append(
                {
                    "case": case["id"],
                    "status": "executed",
                    "passed": result["passed"],
                    "denominators": result.get("denominators", {}),
                }
            )
        except Exception as error:  # noqa: BLE001 — retain every requested failure.
            directory = args.output / case["id"]
            failure_path = directory / "failure.json"
            if failure_path.exists():
                failure = json.loads(failure_path.read_text())
            else:
                failure = {
                    "case": case["id"],
                    "status": "attempted_failed"
                    if (directory / "native-start.json").exists()
                    else "blocked",
                    "passed": False,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "error": f"{type(error).__name__}: {error}",
                }
                if directory.exists():
                    write_json(failure_path, failure)
            results.append(failure)
        write_json(args.output / "progress.json", results)
    write_json(
        args.output / "summary.json",
        {
            "schema": "bb8.visual-purpose-summary.v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "requested": protocol["requested_native_cases"],
            "native_attempted": sum(
                r["status"] in ("executed", "attempted_failed") for r in results
            ),
            "completed_audits": sum(r["status"] == "executed" for r in results),
            "blocked": sum(r["status"] == "blocked" for r in results),
            "passed": sum(r["passed"] for r in results),
            "native_cases_all_passed": all(r["passed"] for r in results),
            "full_history_and_control_claim": "Requires separately generated history-comparison.json and offline-controls/report.json",
            "cases": results,
        },
    )


if __name__ == "__main__":
    main()
