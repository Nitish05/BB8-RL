"""Bounded live RGB buffer; immutable pixels retained for each admitted outcome."""

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path, payload):
    """Publish complete bytes exclusively; preserve and reject conflicting files."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing visual evidence differs: {path.name}")
        _sync_directory(path.parent)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".visual-evidence-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            if handle.write(payload) != len(payload):
                raise OSError("Incomplete visual evidence write")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard link publishes complete bytes without replacing any existing
            # artifact, including one concurrently created after the first check.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(
                    f"Existing visual evidence differs: {path.name}"
                ) from None
        # The callback may return only after both bytes and the published name
        # are durable. A failed directory flush withholds the learning receipt.
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class VisualEvidenceArchive:
    def __init__(self, run_dir, *, audit=False):
        self.directory = Path(run_dir) / "visual-evidence"
        self.directory.mkdir(exist_ok=True)
        self.audit = audit
        self.frames = OrderedDict()
        self.retained = set()

    def capture(self, rgb, *, step, timestamp):
        rgb = np.ascontiguousarray(rgb)
        raw_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError("Could not encode visual outcome evidence")
        payload = encoded.tobytes()
        record = {
            "rgb_path": f"visual-evidence/{raw_hash}.png",
            "rgb_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_rgb_sha256": raw_hash,
            "step": step,
            "timestamp": timestamp,
            "retained": self.audit,
        }
        self.frames[raw_hash] = (payload, record)
        self.frames.move_to_end(raw_hash)
        # More than the maximum 102 interaction samples plus before/after windows.
        while len(self.frames) > 128:
            self.frames.popitem(last=False)
        if self.audit:
            self._save(raw_hash)
        return record

    def _save(self, raw_hash):
        payload, record = self.frames[raw_hash]
        _write_immutable(self.directory / f"{raw_hash}.png", payload)
        record["retained"] = True
        return dict(record)

    def _verify_existing(self, path, evidence, hashes):
        """Exact receipt replay remains checkable after its live buffer expires."""
        payload = path.read_bytes()
        try:
            archive = json.loads(payload)
            if (
                not isinstance(archive, dict)
                or set(archive) != {"evidence", "images"}
                or json.dumps(archive["evidence"], sort_keys=True, allow_nan=False)
                != json.dumps(evidence, sort_keys=True, allow_nan=False)
                or not isinstance(archive["images"], dict)
                or set(archive["images"]) != hashes
                or payload != _json_bytes(archive)
            ):
                raise ValueError("Existing visual receipt differs")
            for raw_hash, record in archive["images"].items():
                if (
                    not isinstance(record, dict)
                    or set(record)
                    != {
                        "rgb_path",
                        "rgb_sha256",
                        "raw_rgb_sha256",
                        "step",
                        "timestamp",
                        "retained",
                    }
                    or record["rgb_path"] != f"visual-evidence/{raw_hash}.png"
                    or record["raw_rgb_sha256"] != raw_hash
                    or record["retained"] is not True
                ):
                    raise ValueError("Existing visual image record differs")
                pixels = (self.directory / f"{raw_hash}.png").read_bytes()
                if hashlib.sha256(pixels).hexdigest() != record["rgb_sha256"]:
                    raise ValueError("Existing visual image differs")
                if raw_hash in self.frames:
                    if pixels != self.frames[raw_hash][0]:
                        raise ValueError("Existing visual image differs")
                else:
                    rgb = cv2.imdecode(
                        np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR
                    )
                    if (
                        rgb is None
                        or hashlib.sha256(
                            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).tobytes()
                        ).hexdigest()
                        != raw_hash
                    ):
                        raise ValueError("Existing visual image pixels differ")
        except (TypeError, KeyError, UnicodeDecodeError, cv2.error) as error:
            raise ValueError("Invalid existing visual receipt") from error
        _sync_directory(self.directory)

    def retain_outcome(self, evidence):
        digest = evidence["evidence_sha256"]
        hashes = {
            frame["frame_sha256"]
            for field in ("before_frames", "during_frames", "after_frames")
            for frame in evidence[field]
        }
        path = self.directory / f"outcome-{digest}.json"
        if path.exists():
            self._verify_existing(path, evidence, hashes)
        else:
            if not hashes.issubset(self.frames):
                raise ValueError(
                    "Admitted outcome pixels missing from bounded evidence buffer"
                )
            artifacts = {key: self._save(key) for key in sorted(hashes)}
            _write_immutable(
                path, _json_bytes({"evidence": evidence, "images": artifacts})
            )
        self.retained.add(digest)
