"""Durable image provenance without unbounded idle-session frame logging."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from bb8_rl.visual_evidence import VisualEvidenceArchive


def frame(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


def receipt(*hashes):
    return {
        "evidence_sha256": "a" * 64,
        **{
            name: [{"frame_sha256": digest} for digest in hashes]
            for name in ("before_frames", "during_frames", "after_frames")
        },
    }


def test_live_buffer_does_not_write_idle_pixels_and_retains_outcome(tmp_path):
    archive = VisualEvidenceArchive(tmp_path)
    records = [archive.capture(frame(i), step=i, timestamp=i * 0.05) for i in range(3)]
    assert not list(archive.directory.iterdir())
    evidence = receipt(*(r["raw_rgb_sha256"] for r in records))
    archive.retain_outcome(evidence)
    archive.retain_outcome(evidence)
    assert len(list(archive.directory.iterdir())) == 4
    saved = json.loads(next(archive.directory.glob("outcome-*.json")).read_text())
    assert saved["evidence"] == evidence
    for record in saved["images"].values():
        path = tmp_path / record["rgb_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["rgb_sha256"]
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == record["raw_rgb_sha256"]


def test_buffer_bounded_and_missing_outcome_evidence_rejected(tmp_path):
    archive = VisualEvidenceArchive(tmp_path)
    first = archive.capture(frame(0), step=0, timestamp=0)
    for index in range(1, 140):
        archive.capture(frame(index), step=index, timestamp=index * 0.05)
    assert len(archive.frames) == 128
    with pytest.raises(ValueError, match="missing"):
        archive.retain_outcome(receipt(first["raw_rgb_sha256"]))
    assert not list(archive.directory.iterdir())


def test_audit_retains_every_distinct_image(tmp_path):
    archive = VisualEvidenceArchive(tmp_path, audit=True)
    for index in range(140):
        record = archive.capture(frame(index), step=index, timestamp=index * 0.05)
        assert record["retained"]
        assert (tmp_path / record["rgb_path"]).is_file()
    assert len(list(archive.directory.glob("*.png"))) == 140


@pytest.mark.parametrize("artifact", ["png", "receipt"])
@pytest.mark.parametrize("new_receipt", [False, True])
def test_partial_atomic_write_leaves_no_published_fragment_and_retry_is_valid(
    tmp_path, monkeypatch, artifact, new_receipt
):
    from bb8_rl import visual_evidence

    archive = VisualEvidenceArchive(tmp_path)
    record = archive.capture(frame(15), step=1, timestamp=0.05)
    evidence = receipt(record["raw_rgb_sha256"])
    original = visual_evidence.tempfile.NamedTemporaryFile
    failed = []

    class InterruptedFile:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, payload):
            selected = (
                payload.startswith(b"\x89PNG")
                if artifact == "png"
                else payload.startswith(b"{")
            )
            if selected and not failed:
                failed.append(True)
                self.handle.write(payload[:8])
                self.handle.flush()
                raise OSError("partial device write")
            return self.handle.write(payload)

    monkeypatch.setattr(
        visual_evidence.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: InterruptedFile(original(*args, **kwargs)),
    )
    with pytest.raises(OSError, match="partial device write"):
        archive.retain_outcome(evidence)
    assert not list(archive.directory.glob("outcome-*.json"))
    assert not list(archive.directory.glob(".visual-evidence-*.tmp"))
    if artifact == "png":
        assert not list(archive.directory.glob("*.png"))
    if new_receipt:
        evidence["evidence_sha256"] = "b" * 64
    archive.retain_outcome(evidence)
    saved = json.loads(next(archive.directory.glob("outcome-*.json")).read_text())
    assert saved["evidence"] == evidence
    image = saved["images"][record["raw_rgb_sha256"]]
    assert (
        hashlib.sha256((tmp_path / image["rgb_path"]).read_bytes()).hexdigest()
        == image["rgb_sha256"]
    )
    assert not list(archive.directory.glob(".visual-evidence-*.tmp"))


@pytest.mark.parametrize("new_receipt", [False, True])
def test_corrupt_existing_png_is_preserved_and_rejected_even_after_retention(
    tmp_path, new_receipt
):
    archive = VisualEvidenceArchive(tmp_path)
    record = archive.capture(frame(20), step=0, timestamp=0)
    evidence = receipt(record["raw_rgb_sha256"])
    archive.retain_outcome(evidence)
    target = tmp_path / record["rgb_path"]
    target.write_bytes(b"partial old PNG")
    if new_receipt:
        evidence["evidence_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs"):
        archive.retain_outcome(evidence)
    assert target.read_bytes() == b"partial old PNG"
    assert len(list(archive.directory.glob("outcome-*.json"))) == 1


@pytest.mark.parametrize(
    "corruption", ["partial", "conflicting", "equal_value_wrong_type"]
)
def test_corrupt_existing_receipt_is_preserved_and_rejected(tmp_path, corruption):
    archive = VisualEvidenceArchive(tmp_path)
    record = archive.capture(frame(20), step=0, timestamp=0)
    evidence = receipt(record["raw_rgb_sha256"])
    evidence["generation"] = 1
    archive.retain_outcome(evidence)
    path = next(archive.directory.glob("outcome-*.json"))
    if corruption == "partial":
        corrupted = b'{"evidence":'
    else:
        payload = json.loads(path.read_text())
        if corruption == "equal_value_wrong_type":
            payload["evidence"]["generation"] = True
        else:
            payload["evidence"]["unexpected"] = True
        corrupted = (json.dumps(payload, indent=2) + "\n").encode()
    path.write_bytes(corrupted)
    with pytest.raises(ValueError):
        archive.retain_outcome(evidence)
    assert path.read_bytes() == corrupted


def test_exact_retained_receipt_replay_survives_eviction_and_archive_restart(tmp_path):
    archive = VisualEvidenceArchive(tmp_path)
    record = archive.capture(frame(0), step=0, timestamp=0)
    evidence = receipt(record["raw_rgb_sha256"])
    archive.retain_outcome(evidence)
    original = {path.name: path.read_bytes() for path in archive.directory.iterdir()}
    for index in range(1, 140):
        archive.capture(frame(index), step=index, timestamp=index * 0.05)
    assert record["raw_rgb_sha256"] not in archive.frames
    archive.retain_outcome(evidence)
    restarted = VisualEvidenceArchive(tmp_path)
    restarted.retain_outcome(evidence)
    assert {
        path.name: path.read_bytes() for path in archive.directory.iterdir()
    } == original


def test_atomic_publication_never_replaces_a_concurrent_conflicting_artifact(
    tmp_path, monkeypatch
):
    from bb8_rl import visual_evidence

    archive = VisualEvidenceArchive(tmp_path)
    record = archive.capture(frame(5), step=0, timestamp=0)
    original = visual_evidence.os.link

    def conflict(source, destination):
        destination.write_bytes(b"concurrent original")
        return original(source, destination)

    monkeypatch.setattr(visual_evidence.os, "link", conflict)
    with pytest.raises(ValueError, match="differs"):
        archive.retain_outcome(receipt(record["raw_rgb_sha256"]))
    assert (tmp_path / record["rgb_path"]).read_bytes() == b"concurrent original"
    assert not list(archive.directory.glob("outcome-*.json"))
    assert not list(archive.directory.glob(".visual-evidence-*.tmp"))


@pytest.mark.parametrize("failed_publication", [1, 2])
def test_directory_sync_failure_withholds_retention_and_retries_complete_artifacts(
    tmp_path, monkeypatch, failed_publication
):
    import os
    import stat

    from bb8_rl import visual_evidence

    archive = VisualEvidenceArchive(tmp_path)
    record = archive.capture(frame(7), step=0, timestamp=0)
    evidence = receipt(record["raw_rgb_sha256"])
    original = os.fsync
    directory_calls = []

    def interrupted(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_calls.append(True)
            if len(directory_calls) == failed_publication:
                raise OSError("directory sync failed")
        return original(descriptor)

    monkeypatch.setattr(visual_evidence.os, "fsync", interrupted)
    with pytest.raises(OSError, match="directory sync failed"):
        archive.retain_outcome(evidence)
    assert not archive.retained
    assert not list(archive.directory.glob(".visual-evidence-*.tmp"))
    published = {path.name: path.read_bytes() for path in archive.directory.iterdir()}
    archive.retain_outcome(evidence)
    assert len(directory_calls) > failed_publication
    assert archive.retained == {evidence["evidence_sha256"]}
    assert all(
        (archive.directory / name).read_bytes() == data
        for name, data in published.items()
    )
