import importlib.util
import json
from pathlib import Path

import pytest

from bb8_rl.session_recording import RotatingJsonl, SessionRecorder, finish_recording


def sample(step, **updates):
    row = {
        "step": step,
        "time": step * 0.05,
        "action": [0, 0],
        "controller_status": "stopped",
        "generation": 0,
        "localization": {"localization_status": "measured", "pose": [0, 1]},
        "agency": {"enabled": False, "status": "paused", "resource": 0.35},
        "after_step": {"physics_trace_scoring_only": [{"large": "x" * 9000}]},
    }
    row.update(updates)
    return row


def read_all(directory, stem):
    return [
        json.loads(line)
        for path in directory.glob(f"{stem}*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_long_stopped_session_bounds_disk_and_labels_sampling(tmp_path):
    log = SessionRecorder(tmp_path, segment_bytes=2048, segments=2)
    for step in range(20_000):  # 1000 simulated seconds, including noisy resource.
        log.row(
            sample(
                step,
                agency={
                    "enabled": False,
                    "status": "paused",
                    "resource": 0.35 + step % 2 * 0.0001,
                },
            )
        )
        log.command({"action": "heartbeat", "sim_time": step * 0.05})
    result = log.finish(status="complete")
    assert result["sampled_rows"] == 1000
    assert result["heartbeat_commands_summarized"] == 20_000
    assert result["telemetry"]["history_truncated"]
    assert result["events"]["records_written"] == 2
    assert sum(p.stat().st_size for p in tmp_path.glob("*.jsonl")) <= 8192
    assert not (tmp_path / "rows.jsonl").exists()
    assert (
        "physics_trace_scoring_only" not in (tmp_path / "telemetry.jsonl").read_text()
    )
    assert json.loads((tmp_path / "recording.json").read_text()) == result


def test_unsampled_transition_outcome_stop_and_fault_are_recorded(tmp_path):
    log = SessionRecorder(tmp_path)
    log.row(sample(0))
    outcome = {"request_id": "receipt-1", "before": 0.35, "after": 0.8}
    for step in (1, 2):
        log.row(
            sample(step, after_step={"interaction_telemetry": {"outcome": outcome}})
        )
    log.command({"action": "stop", "generation": 7})
    log.event("fault", error="camera unavailable")
    events = read_all(tmp_path, "events")
    assert len([e for e in events if e.get("outcome") == outcome]) == 1
    assert events[-2]["action"] == "stop"
    assert events[-1]["error"] == "camera unavailable"
    assert len(read_all(tmp_path, "telemetry")) == 1


def test_event_flood_rotates_with_explicit_eviction_counts(tmp_path):
    log = SessionRecorder(tmp_path, segment_bytes=1024, segments=2)
    for generation in range(100):
        log.command({"action": "stop", "generation": generation})
    result = log.finish()
    retained = read_all(tmp_path, "events")
    assert any(e.get("generation") == 99 for e in retained)
    assert result["events"]["records_evicted"] + len(retained) == 101
    assert result["events"]["history_truncated"]
    assert result["events"]["retained_bytes"] <= 2048


def test_full_audit_is_exact_unsampled_and_unrotated(tmp_path):
    log = SessionRecorder(tmp_path, mode="audit", segment_bytes=1024, segments=1)
    expected = [sample(i) for i in range(25)]
    for row in expected:
        log.row(row)
    command = {"action": "heartbeat", "sim_time": 0}
    log.command(command)
    result = log.finish(status="complete")
    assert read_all(tmp_path, "rows") == expected
    assert read_all(tmp_path, "commands") == [command]
    assert (tmp_path / "rows.jsonl").stat().st_size > 1024 * 25
    assert result["complete_physics_rows"] and result["rows_seen"] == 25
    assert result["sampled_rows"] == 0


def test_oversized_summary_is_labeled_and_counted(tmp_path):
    log = RotatingJsonl(tmp_path / "events.jsonl", segment_bytes=1024, segments=1)
    log.append({"type": "fault", "error": "x" * 2000})
    assert log.summary()["oversized_records"] == 1
    assert log.summary()["history_truncated"]
    assert read_all(tmp_path, "events")[0]["record_type"] == "fault"


def test_oversized_marker_cannot_overflow_quota_itself(tmp_path):
    log = RotatingJsonl(tmp_path / "events.jsonl", segment_bytes=1024, segments=1)
    log.append({"type": "x" * 16_000, "time": "y" * 16_000})
    assert (tmp_path / "events.jsonl").stat().st_size <= 1024
    assert log.summary()["oversized_records"] == 1


def test_recording_finalization_failure_preserves_original_error(tmp_path, monkeypatch):
    log = SessionRecorder(tmp_path, mode="audit")

    def broken_event(*_args, **_kwargs):
        raise OSError("event stream unavailable")

    monkeypatch.setattr(log, "event", broken_event)
    manifest = {"status": "failed", "error": "original fault"}
    finish_recording(log, manifest)
    assert manifest["error"] == "original fault"
    assert "event stream unavailable" in manifest["recording"]["finalization_error"]
    manifest = {"status": "complete"}
    finish_recording(log, manifest)
    assert manifest["status"] == "failed"


def test_independent_auditor_rejects_explicit_telemetry_recording(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/audit-interactive.py"
    spec = importlib.util.spec_from_file_location("recording_audit_test", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    checks = audit.verify_manifest(
        tmp_path,
        {
            "status": "complete",
            "source_files_unchanged": True,
            "recording": {"complete_physics_rows": False},
            "config": {"recording_mode": "telemetry"},
        },
    )
    assert checks["complete_recording"] is False
    checks = audit.verify_manifest(tmp_path, {"schema": "bb8.interactive-run.v2"})
    assert checks["identity:required"] is False
    manifest = {
        "schema": "bb8.interactive-run.v2",
        "completed_steps": 3,
        "recording": {"rows_seen": 3, "mode": "audit", "complete_physics_rows": True},
    }
    rows = [sample(i) for i in range(3)]
    assert all(audit.verify_recording(rows, manifest).values())
    assert not audit.verify_recording(rows[:-1], manifest)["recording:row_count"]
    assert not audit.verify_recording(rows[1:], manifest)["recording:step_sequence"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"segment_bytes": 0},
        {"segments": True},
        {"sample_seconds": float("nan")},
        {"sample_seconds": 0},
    ],
)
def test_invalid_recording_configuration_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, **kwargs)


def test_visual_outcome_keeps_compact_provenance_without_world_telemetry(tmp_path):
    log = SessionRecorder(tmp_path)
    log.row(sample(0))
    evidence = {
        "request_id": "visual-1",
        "entity_id": "observed-uuid",
        "source": "native_scene_rgb",
        "evidence_sha256": "a" * 64,
        "classification": "useful",
        "gain_interval": [0.4, 0.5],
        "before_frames": [{"large": "x" * 9000}],
    }
    for step in (1, 2):
        log.row(
            sample(
                step,
                agency={
                    "enabled": True,
                    "status": "remembering",
                    "resource": 0.7,
                    "resource_source": "native_scene_rgb",
                    "visual_outcome": evidence,
                },
                after_step={
                    "interaction_telemetry": None,
                    "interaction_world_scoring_only": {"resource": 0.75},
                },
            )
        )
    outcomes = [e for e in read_all(tmp_path, "events") if e.get("outcome")]
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"]["evidence_sha256"] == "a" * 64
    assert outcomes[0]["resource"] == 0.7
    assert "before_frames" not in outcomes[0]["outcome"]
