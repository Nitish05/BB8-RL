"""Negative controls for independent RGB/authority/durable-history auditing."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/bb8_rl").is_dir())
sys.path.insert(0, str(ROOT / "tests"))
from test_visual_interaction import calibration
from test_visual_purpose_runtime import VisualLoop

from bb8_rl.visual_evidence import VisualEvidenceArchive


@pytest.fixture
def auditor():
    path = Path(__file__).with_name("benchmark-visual-purpose.py")
    if not path.exists():
        path = ROOT / "scripts/benchmark-visual-purpose.py"
    spec = importlib.util.spec_from_file_location("visual_purpose_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recorded(tmp_path, auditor):
    loop = VisualLoop(tmp_path / "memory.sqlite3")
    archive = VisualEvidenceArchive(tmp_path, audit=True)
    loop.agency.evidence_retainer = archive.retain_outcome
    before = auditor.memory_snapshot(tmp_path / "memory.sqlite3")
    rows = []
    original = loop.step

    def capture(**kwargs):
        t, step = round(loop.index * 0.05, 6), loop.index
        rgb = kwargs.get("rgb", loop.after if kwargs.get("changed") else loop.before)
        image = archive.capture(rgb, step=step, timestamp=t)
        action = original(**kwargs)
        state = loop.control.state()
        rows.append(
            {
                "step": step,
                "time": t,
                "capture_time": t,
                "generation": state["generation"],
                "agency": copy.deepcopy(state["agency"]),
                "localization": {
                    key: state[key]
                    for key in ("localization_valid", "localization_status")
                },
                "action": action.tolist(),
                "controller_arrived": bool(
                    loop.control.controller and loop.control.controller.arrived
                ),
                "commanded_goal": loop.control.goal,
                "images": {"semantic-A": copy.deepcopy(image)},
            }
        )
        return action

    loop.step = capture
    loop.start_request().finish()
    cal = calibration()
    record = {
        "id": "semantic-A",
        "intrinsics": cal.intrinsics.tolist(),
        "world_to_camera": cal.world_to_camera.tolist(),
        "resolution": list(cal.resolution),
        "calibration_version": "cal-v1",
    }
    after = auditor.memory_snapshot(tmp_path / "memory.sqlite3")
    loop.store.close()
    return tmp_path, rows, before, after, record


def test_independent_raw_rgb_replay_and_sqlite_admission_agree(auditor, recorded):
    path, rows, before, after, record = recorded
    decoded, errors, scenes = auditor.decode_images(path, rows, record)
    assert errors == [] and all(scene["status"] == "visible" for scene in scenes)
    score = auditor.score_receipts(rows, decoded, before, after)
    assert score["passed"], score["errors"]
    assert score["new_outcomes"] == score["useful_outcomes"] == 1
    assert auditor.verify_archives(path, score["receipts"]) == []


@pytest.mark.parametrize(
    "fault", ["rgb", "raw_hash", "png_hash", "timestamp", "step", "missing", "outside"]
)
def test_semantic_image_or_capture_substitution_fails(auditor, recorded, fault):
    path, rows, _before, _after, record = recorded
    rows = copy.deepcopy(rows)
    image = rows[8]["images"]["semantic-A"]
    if fault == "rgb":
        pixels = np.zeros((768, 1024, 3), np.uint8)
        cv2.imwrite(str(path / "other.png"), pixels)
        image["rgb_path"] = "other.png"
        image["rgb_sha256"] = hashlib.sha256(
            (path / "other.png").read_bytes()
        ).hexdigest()
    elif fault == "missing":
        rows[8]["images"] = {}
    elif fault == "outside":
        image["rgb_path"] = "../escape.png"
    else:
        key = {"raw_hash": "raw_rgb_sha256", "png_hash": "rgb_sha256"}.get(fault, fault)
        image[key] = "0" * 64 if "hash" in fault else image[key] + 1
    assert auditor.decode_images(path, rows, record)[1]


@pytest.mark.parametrize(
    "fault",
    [
        "pixels",
        "missing_event",
        "forged_payload",
        "extra_event",
        "authority",
        "motion",
        "generation",
        "prior_edit",
    ],
)
def test_plausible_learner_claim_cannot_replace_pixel_authority_store_evidence(
    auditor, recorded, fault
):
    path, rows, before, after, record = recorded
    decoded, errors, _ = auditor.decode_images(path, rows, record)
    assert not errors
    rows, before, after = copy.deepcopy((rows, before, after))
    receipt = next(
        r["agency"]["visual_outcome"] for r in rows if r["agency"]["visual_outcome"]
    )
    if fault == "pixels":
        evidence = receipt["during_frames"][3]
        decoded[(evidence["timestamp"], evidence["camera_id"], evidence["marker_id"])][
            "gauge_level"
        ] += 1
    elif fault == "missing_event":
        after["events"] = []
    elif fault == "forged_payload":
        payload = json.loads(after["events"][0]["payload"])
        payload["after"] -= 0.2
        after["events"][0]["payload"] = json.dumps(payload)
    elif fault == "extra_event":
        after["events"].append({**after["events"][0], "id": "unobserved-event"})
    elif fault == "prior_edit":
        before["events"] = [{**after["events"][0], "id": "old-event"}]
        after["events"].append({**before["events"][0], "gain": -1})
    else:
        sample = next(r for r in rows if r["time"] > receipt["started_at"])
        if fault == "authority":
            sample["agency"]["enabled"] = False
        elif fault == "motion":
            sample["action"] = [0.1, 0]
        else:
            sample["generation"] += 1
    assert not auditor.score_receipts(rows, decoded, before, after)["passed"]


def test_missing_retained_outcome_file_fails_even_when_receipt_valid(auditor, recorded):
    path, rows, before, after, record = recorded
    decoded, _, _ = auditor.decode_images(path, rows, record)
    score = auditor.score_receipts(rows, decoded, before, after)
    receipt = score["receipts"][0]
    (path / "visual-evidence" / f"outcome-{receipt['evidence_sha256']}.json").unlink()
    assert auditor.verify_archives(path, score["receipts"])


def test_no_op_run_cannot_pass_stop_or_hidden_pixel_intent(auditor):
    rows = [
        {
            "time": 0,
            "agency": {"enabled": False},
            "commanded_goal": None,
            "action": [0, 0],
        }
    ]
    admission = {"new_outcomes": 0, "useful_outcomes": 0}
    stop = {"accepted_sim_time": 0, "passed": True}
    scenes = [{"timestamp": 0, "status": "missing"}]
    for intent in ("stop_pending", "hidden_post"):
        assert not auditor.score_intent(
            rows, [], {"intent": intent}, admission, stop, scenes, {"events": []}
        )["passed"]


def create_history(path, useful_marker):
    """Synthetic evidence tests the audit policy, never counts as native history."""
    from test_visual_interaction import reading, scene

    from bb8_rl.purpose import PurposeEngine, PurposeStore
    from bb8_rl.visual_interaction import VisualEntityRegistry, VisualOutcomeGate

    store = PurposeStore(path, "test-map")
    engine = PurposeEngine(store)
    registry = VisualEntityRegistry(store, "cal-v1")
    centers = {"marker-01": (0.0, 1.25), "marker-02": (-0.5, 1.25)}
    tick = 0

    def samples(level):
        nonlocal tick
        t = round(tick * 0.05, 6)
        tick += 1
        values = tuple(
            reading(t, level, code, center) for code, center in centers.items()
        )
        return t, registry.observe(scene(t, *values))

    for _ in range(3):
        now, entities = samples(8)
    engine.select([e.station for e in entities], resource=0.25, pose=(0, 0), now=now)
    for code in centers:
        for attempt in range(4):
            gate = VisualOutcomeGate("test-map", "cal-v1")
            gate.enable(1)
            for _ in range(3):
                now, entities = samples(8)
                assert (
                    gate.observe(
                        entities, now=now, generation=1, measured=True, stationary=True
                    )
                    is None
                )
            entity = next(e for e in entities if e.reading.marker_id == code)
            request = f"{code}-{attempt}"
            assert gate.begin(request, entity.entity_id, now=now, generation=1)
            completed = round(now + 1, 6)
            receipt = None
            for _ in range(25):
                next_time = round(tick * 0.05, 6)
                level = 23 if code == useful_marker and next_time >= completed else 8
                now, entities = samples(level)
                completion = (
                    {
                        "request_id": request,
                        "entity_id": entity.entity_id,
                        "timestamp": completed,
                        "generation": 1,
                    }
                    if now >= completed
                    else None
                )
                receipt = gate.observe(
                    entities,
                    now=now,
                    generation=1,
                    measured=True,
                    stationary=True,
                    completion=completion,
                )
                if receipt:
                    break
            assert receipt
            engine.record_outcome(
                request,
                entity.entity_id,
                before=receipt["before"],
                after=receipt["after"],
                now=receipt["timestamp"],
                source="native_scene_rgb",
                visual_evidence=receipt,
            )
    store.close()
    return [
        reading(100, 8, code, center).evidence() for code, center in centers.items()
    ]


def test_same_pixels_opposite_histories_change_choice_and_ablation_removes_difference(
    auditor, tmp_path
):
    choices = {}
    for code in ("marker-01", "marker-02"):
        path = tmp_path / f"{code}.sqlite3"
        observations = create_history(path, code)
        original_hash = auditor.digest(path)
        snapshot = auditor.memory_snapshot(path)
        choices[code] = []
        for seed in (0, 1, 2):
            result = auditor.replay_choices(
                path,
                tmp_path / f"{code}-{seed}",
                snapshot,
                observations,
                pose=(0, 0),
                now=100,
                seed=seed,
            )
            choices[code].append(result["choices"])
            assert result["choices"]["learned"] == code
            assert result["choices"]["memory_ablation"] == "marker-01"
            assert result["observed_rgb_hashes"] == ["a" * 64]
        assert auditor.digest(path) == original_hash
    assert len({x["memory_ablation"] for group in choices.values() for x in group}) == 1


def test_offline_controls_reject_stale_or_moved_entity_evidence(auditor, tmp_path):
    path = tmp_path / "history.sqlite3"
    observations = create_history(path, "marker-01")
    snapshot = auditor.memory_snapshot(path)
    for fault in ("stale", "moved"):
        bad = copy.deepcopy(observations)
        bad[0]["timestamp" if fault == "stale" else "center_xy"] = (
            99 if fault == "stale" else [1.0, 1.0]
        )
        with pytest.raises(ValueError):
            auditor.replay_choices(
                path, tmp_path / fault, snapshot, bad, pose=(0, 0), now=100, seed=0
            )


def test_hidden_pixel_case_requires_a_completed_request_then_observed_abstention(
    auditor,
):
    rows = [
        {
            "time": 0.5,
            "agency": {"enabled": True, "interaction_request": {"request_id": "r"}},
        },
        {
            "time": 1.5,
            "agency": {"enabled": False},
            "interaction_world_scoring_only": {"outcome": {"timestamp": 1.5}},
        },
    ]
    admission = {"new_outcomes": 0, "useful_outcomes": 0}
    stop = {"accepted_sim_time": 1.55, "passed": True}
    case = {"intent": "hidden_post"}
    before = {"events": []}
    scenes = [{"timestamp": 1.5, "status": "missing"}]
    assert auditor.score_intent(rows, [], case, admission, stop, scenes, before)[
        "passed"
    ]
    assert not auditor.score_intent(
        rows, [], case, admission, stop, [{"timestamp": 0, "status": "missing"}], before
    )["passed"]
    assert not auditor.score_intent(
        rows[1:], [], case, admission, stop, scenes, before
    )["passed"]


def test_protocol_tampering_is_rejected_before_any_execution(auditor):
    protocol = {
        "schema": "bb8.visual-purpose-protocol.v1",
        "cases": [{"id": "case"}],
        "requested_native_cases": 1,
        "sim_seconds": 120,
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        auditor.canonical(protocol).encode()
    ).hexdigest()
    protocol["sim_seconds"] = 500
    with pytest.raises(ValueError, match="changed after freeze"):
        auditor.validate_protocol(protocol)


def test_duplicate_memory_destination_preserves_original_file(auditor, recorded):
    path = recorded[0] / "memory.sqlite3"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        auditor.copy_memory(path, path)
    assert path.read_bytes() == original


def test_native_failure_and_unprovisioned_dependency_keep_separate_denominators(
    auditor, tmp_path, monkeypatch
):
    protocol = {
        "protocol_sha256": "frozen",
        "requested_native_cases": 2,
        "cases": [{"id": "attempt"}, {"id": "blocked"}],
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(auditor, "validate_protocol", lambda value: value)

    def fail(_protocol, case, output):
        directory = output / case["id"]
        directory.mkdir()
        if case["id"] == "attempt":
            (directory / "native-start.json").write_text('{"pid":123}')
        raise ValueError("retained first failure")

    monkeypatch.setattr(auditor, "run_case", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--output", str(tmp_path), "--protocol", str(protocol_path)],
    )
    auditor.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["requested"] == 2
    assert summary["native_attempted"] == summary["blocked"] == 1
    assert summary["completed_audits"] == summary["passed"] == 0
    assert not summary["native_cases_all_passed"]
    auditor.main()  # A repeat must preserve the original error rather than overwrite evidence.
    for name in ("attempt", "blocked"):
        assert (
            json.loads((tmp_path / name / "failure.json").read_text())["error"]
            == "ValueError: retained first failure"
        )


@pytest.mark.parametrize("intent", ["learn", "first_outcome", "useless_idle"])
def test_unexpected_cancellation_requires_observed_enable_and_is_terminal(
    auditor, intent
):
    paused = {
        "agency": {"enabled": False, "message": "Missing RGB evidence"},
        "controller_status": "paused",
    }
    assert auditor.terminal_cancellation_reason(paused, intent, False) is None
    assert (
        auditor.terminal_cancellation_reason(paused, intent, True)
        == "Missing RGB evidence"
    )
    assert auditor.terminal_cancellation_reason(
        {"agency": {"enabled": True}, "controller_status": "guarded_stop"}, intent, True
    )
    assert (
        auditor.terminal_cancellation_reason(
            {"agency": {"enabled": True}, "controller_status": "localized"},
            intent,
            True,
        )
        is None
    )


@pytest.mark.parametrize("intent", ["stop_pending", "hidden_post", "disabled"])
def test_deliberate_fault_controls_use_their_own_admission_criteria(auditor, intent):
    state = {"agency": {"enabled": False}, "controller_status": "guarded_stop"}
    assert auditor.terminal_cancellation_reason(state, intent, True) is None


def test_spatial_roundoff_does_not_change_hash_or_discrete_evidence_contract(auditor):
    from test_visual_interaction import reading

    original = reading(1).evidence()
    for key, value in (
        ("radius_m", original["radius_m"] + 9e-17),
        ("center_xy", [original["center_xy"][0] + 5e-11, original["center_xy"][1]]),
    ):
        assert auditor.matching_redecoded_evidence({**original, key: value}, original)
    for key, value in (
        ("radius_m", original["radius_m"] + 1.01e-10),
        ("center_xy", [original["center_xy"][0] + 1.01e-10, original["center_xy"][1]]),
        ("radius_m", float("nan")),
        ("radius_m", True),
        ("center_xy", [float("inf"), 0]),
        ("center_xy", [0]),
        ("frame_sha256", "b" * 64),
        ("marker_id", "marker-02"),
        ("gauge_level", original["gauge_level"] + 1),
        ("gauge_level", float(original["gauge_level"])),
        ("interval", [original["interval"][0] + 1e-13, original["interval"][1]]),
        ("timestamp", original["timestamp"] + 1e-13),
    ):
        assert not auditor.matching_redecoded_evidence(
            {**original, key: value}, original
        )
    assert not auditor.matching_redecoded_evidence(None, original)
    assert not auditor.matching_redecoded_evidence(
        {**original, "unexpected": 1}, original
    )


def test_arrival_frames_and_entries_have_distinct_counts(auditor):
    row = {
        "generation": 1,
        "commanded_goal": [0, 1],
        "controller_arrived": True,
        "localization": {
            "localization_valid": True,
            "localization_status": "measured",
            "pose": [0, 0],
        },
        "truth_before_scoring_only": {"position": [0.03, 0.04]},
    }
    rows = [row, row, {**row, "controller_arrived": False}, row]
    metrics = auditor.navigation_observation_metrics(rows)
    assert metrics["controller_arrival_entries"] == 2
    assert sum(r["controller_arrived"] for r in rows) == 3
    assert metrics["measured_pose_error_m_scoring_only"] == {
        "samples": 4,
        "maximum": 0.05,
        "p95": 0.05,
    }


def test_external_history_reference_detects_any_mutated_evidence(
    auditor, recorded, tmp_path
):
    directory = recorded[0]
    (directory / "audit.json").write_text('{"passed":true}')
    record = {
        "directory": str(directory),
        "memory_sha256": auditor.digest(directory / "memory.sqlite3"),
        "audit_sha256": auditor.digest(directory / "audit.json"),
        "snapshot_sha256": hashlib.sha256(
            auditor.canonical(
                auditor.memory_snapshot(directory / "memory.sqlite3")
            ).encode()
        ).hexdigest(),
    }
    protocol = {"external_histories": {"train-a": record}}
    assert (
        auditor.history_directory(protocol, "train-a", tmp_path / "new-output")
        == directory
    )
    (directory / "audit.json").write_text('{"passed":false}')
    with pytest.raises(ValueError, match="donor evidence changed"):
        auditor.history_directory(protocol, "train-a", tmp_path / "new-output")


def test_supplemental_freeze_has_three_separate_mode3_requests_and_fixed_probe_world(
    auditor, monkeypatch, tmp_path
):
    from bb8_rl import demo_assets

    monkeypatch.setattr(demo_assets, "validate_assets", lambda _assets: None)
    monkeypatch.setattr(
        auditor, "validate_preflight", lambda _path: {"live_B_cases": 40}
    )
    monkeypatch.setattr(auditor, "digest", lambda _path: "f" * 64)
    monkeypatch.setattr(
        auditor,
        "external_history_records",
        lambda _base, **_kw: {
            "train-a": {"directory": "old-a"},
            "train-b": {"directory": "old-b"},
        },
    )
    original = auditor.make_protocol(tmp_path, tmp_path / "preflight")
    supplemental = auditor.make_protocol(
        tmp_path, tmp_path / "preflight", supplemental_history_base=tmp_path / "old"
    )
    assert original["requested_native_cases"] == 9
    assert supplemental["requested_native_cases"] == 3
    assert supplemental["navigation_camera_modes"] == [3]
    for case in supplemental["cases"]:
        previous = next(c for c in original["cases"] if c["id"] == case["id"])
        assert case == {**previous, "mode": 3}
        if case["intent"] == "first_outcome":
            assert case["effects"] == {"marker-01": 0.45, "marker-02": 0.45}
    assert supplemental["initial_resource"] == original["initial_resource"]
    assert "original mode1 failures" in supplemental["scope"]
    assert "hidden-post case remain failed" in supplemental["scope"]


def test_hidden_post_waits_for_observed_authority_before_accepting_cancellation(
    auditor,
):
    # The launch iteration can enqueue enable while the current snapshot is paused.
    snapshots = [
        ({"enabled": False}, False),
        ({"enabled": True}, True),
        ({"enabled": False}, True),
    ]
    assert [
        auditor.hidden_post_cancelled(agency, enabled_observed=observed)
        for agency, observed in snapshots
    ] == [False, False, True]
