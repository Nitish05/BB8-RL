"""Bounded everyday telemetry, or explicitly complete independent-audit rows.

Everyday logs are diagnostic summaries, never physics benchmark evidence. Each
stream keeps a bounded recent history; eviction and oversized records are counted
in a small sidecar after every write. Full audit mode never rotates or samples.
"""

import json
import math
from pathlib import Path


def _encoded(record):
    return (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()


class RotatingJsonl:
    def __init__(self, path, *, segment_bytes=1_048_576, segments=4):
        if type(segment_bytes) is not int or segment_bytes < 1024:
            raise ValueError(
                "Log segment size must be an integer of at least 1024 bytes"
            )
        if type(segments) is not int or not 1 <= segments <= 16:
            raise ValueError("Log segment count must be between 1 and 16")
        self.path = Path(path)
        self.segment_bytes, self.segments = segment_bytes, segments
        self.sizes, self.counts = [0] * segments, [0] * segments
        self.written = self.evicted = self.oversized = 0

    def _path(self, index):
        return self.path if index == 0 else self.path.with_suffix(f".{index}.jsonl")

    def append(self, record):
        data = _encoded(record)
        if len(data) > self.segment_bytes:
            self.oversized += 1
            data = _encoded(
                {
                    "type": "oversized_record",
                    "original_bytes": len(data),
                    "record_type": str(record.get("type", "unknown"))[:64],
                    "detail_truncated": True,
                }
            )
        if self.sizes[0] + len(data) > self.segment_bytes:
            self.evicted += self.counts[-1]
            self._path(self.segments - 1).unlink(missing_ok=True)
            for index in range(self.segments - 1, 0, -1):
                if self._path(index - 1).exists():
                    self._path(index - 1).rename(self._path(index))
                self.sizes[index], self.counts[index] = (
                    self.sizes[index - 1],
                    self.counts[index - 1],
                )
            self.sizes[0] = self.counts[0] = 0
        with self.path.open("ab") as stream:
            stream.write(data)
        self.sizes[0] += len(data)
        self.counts[0] += 1
        self.written += 1

    def summary(self):
        return {
            "quota_bytes": self.segment_bytes * self.segments,
            "retained_bytes": sum(self.sizes),
            "records_written": self.written,
            "records_evicted": self.evicted,
            "oversized_records": self.oversized,
            "history_truncated": bool(self.evicted or self.oversized),
            "files_newest_first": [
                self._path(i).name for i in range(self.segments) if self.counts[i]
            ],
        }


class SessionRecorder:
    def __init__(
        self,
        directory,
        *,
        mode="telemetry",
        segment_bytes=1_048_576,
        segments=4,
        sample_seconds=1.0,
    ):
        if mode not in ("telemetry", "audit"):
            raise ValueError("Recording mode must be telemetry or audit")
        if not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError("Telemetry sampling interval must be finite and positive")
        self.directory, self.mode = Path(directory), mode
        self.sample_seconds = sample_seconds
        self.next_sample = -math.inf
        self.last_transition = None
        self.rows_seen = self.sampled_rows = self.heartbeats = 0
        self.telemetry = RotatingJsonl(
            self.directory / "telemetry.jsonl",
            segment_bytes=segment_bytes,
            segments=segments,
        )
        self.events = RotatingJsonl(
            self.directory / "events.jsonl",
            segment_bytes=segment_bytes,
            segments=segments,
        )
        self._persist()

    def _full(self, filename, record):
        with (self.directory / filename).open("ab") as stream:
            stream.write(_encoded(record))

    def command(self, record):
        if self.mode == "audit":
            self._full("commands.jsonl", record)
        elif record.get("action") == "heartbeat":
            self.heartbeats += 1
        else:
            self.events.append({"type": "command", **record})
            self._persist()

    def row(self, row):
        self.rows_seen += 1
        if self.mode == "audit":
            self._full("rows.jsonl", row)
            return
        agency = row.get("agency") or {}
        after = row.get("after_step") or {}
        localization = row.get("localization") or {}
        telemetry = after.get("interaction_telemetry") or {}
        outcome = telemetry.get("outcome")
        if agency.get("resource_source") == "native_scene_rgb":
            visual = agency.get("visual_outcome")
            outcome = (
                None
                if visual is None
                else {
                    key: visual[key]
                    for key in (
                        "request_id",
                        "entity_id",
                        "source",
                        "evidence_sha256",
                        "classification",
                        "gain_interval",
                    )
                }
            )
        scene = row.get("scene_validity") or {}
        summary = {
            "step": row["step"],
            "time": row["time"],
            "generation": row.get("generation"),
            "mode": row.get("mode"),
            "action": row.get("action"),
            "goal": row.get("commanded_goal"),
            "controller_status": row.get("controller_status"),
            "localization_status": localization.get("localization_status"),
            "localization_valid": localization.get("localization_valid"),
            "scene_status": {
                key: scene.get(key)
                for key in (
                    "enabled",
                    "ready",
                    "navigation_allowed",
                    "invalidated",
                    "reason",
                )
            },
            "scene_validity": scene,
            "pose": localization.get("pose"),
            "position_radius": localization.get("position_radius"),
            "autonomy_enabled": agency.get("enabled"),
            "autonomy_status": agency.get("status"),
            "resource": telemetry.get("resource", agency.get("resource")),
            "outcome": outcome,
            "collision_scoring_only": after.get("collision_scoring_only"),
            "boundary_scoring_only": after.get("boundary_scoring_only"),
        }
        # Exclude noisy per-frame estimates/resource values from transition keys.
        transition = {
            key: summary[key]
            for key in (
                "generation",
                "mode",
                "goal",
                "controller_status",
                "localization_status",
                "localization_valid",
                "scene_status",
                "autonomy_enabled",
                "autonomy_status",
                "outcome",
                "collision_scoring_only",
                "boundary_scoring_only",
            )
        }
        fingerprint = _encoded(transition)
        changed = False
        if fingerprint != self.last_transition:
            self.events.append({"type": "transition", **summary})
            self.last_transition = fingerprint
            changed = True
        if row["time"] + 1e-9 >= self.next_sample:
            self.telemetry.append({"type": "sample", **summary})
            self.next_sample = row["time"] + self.sample_seconds
            self.sampled_rows += 1
            changed = True
        if changed:
            self._persist()

    def event(self, kind, **detail):
        record = {"type": kind, **detail}
        if self.mode == "audit":
            self._full("events.jsonl", record)
        else:
            self.events.append(record)
        self._persist()

    def summary(self):
        return {
            "mode": self.mode,
            "complete_physics_rows": self.mode == "audit",
            "rows_seen": self.rows_seen,
            "sampled_rows": self.sampled_rows,
            "heartbeat_commands_summarized": self.heartbeats,
            "sample_seconds": self.sample_seconds if self.mode == "telemetry" else None,
            "scope": "Complete audit rows; no sampling or rotation"
            if self.mode == "audit"
            else "Bounded diagnostic history; not independent physics audit evidence",
            "telemetry": self.telemetry.summary(),
            "events": self.events.summary(),
        }

    def _persist(self):
        path = self.directory / "recording.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary(), indent=2) + "\n")
        temporary.replace(path)

    def finish(self, **detail):
        self.event("session_end", **detail)
        return self.summary()


def finish_recording(recorder, manifest):
    """Keep a failed event stream from preventing the separate final manifest."""
    try:
        manifest["recording"] = recorder.finish(status=manifest["status"])
    except (OSError, TypeError, ValueError) as error:
        manifest["recording"] = recorder.summary()
        manifest["recording"]["finalization_error"] = f"{type(error).__name__}: {error}"
        manifest["status"] = "failed"
        manifest.setdefault("error", f"Recording finalization failed: {error}")
