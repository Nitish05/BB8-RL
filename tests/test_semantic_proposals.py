"""Advisory model output cannot create motion targets or revive stale requests."""

import json
import sys
import time
from dataclasses import replace

import pytest

from bb8_rl.semantic_proposals import (
    MAX_RESPONSE_BYTES,
    SemanticCandidate,
    SemanticRequest,
    SubprocessCandidateRanker,
    build_prompt,
    validate_choice,
)


@pytest.fixture
def semantic_request():
    return SemanticRequest(
        "frame-12",
        3,
        100.0,
        120.0,
        {"localization": "visible", "evidence": "camera-A-frame-12"},
        (
            SemanticCandidate(
                "inspect-blue", "Attend to the blue obstacle", ("frame-12",)
            ),
            SemanticCandidate("revisit", "Revisit an existing candidate"),
        ),
    )


def checked(output, semantic_request, **kwargs):
    return validate_choice(
        output,
        semantic_request,
        now=kwargs.get("now", 101.0),
        current_epoch=kwargs.get("epoch", 3),
        current_snapshot_id=kwargs.get("snapshot", "frame-12"),
    )


def test_offered_id_only_and_wait(semantic_request):
    assert (
        checked('{"candidate_id":"inspect-blue"}', semantic_request).candidate_id
        == "inspect-blue"
    )
    assert checked('{"candidate_id":"wait"}', semantic_request).status == "wait"
    assert (
        checked('{"candidate_id":"invented-target"}', semantic_request).status
        == "unknown_candidate"
    )


@pytest.mark.parametrize(
    "output",
    [
        '{"candidate_id":"inspect-blue","x":1.0,"y":1.0}',
        '{"candidate_id":["inspect-blue"]}',
        '{"candidate_id":true}',
        '{"candidate_id":null}',
        '"inspect-blue"',
        '["inspect-blue"]',
        "{}",
        '```json\n{"candidate_id":"inspect-blue"}\n```',
        'Go! {"candidate_id":"inspect-blue"}',
        '{"candidate_id":"inspect-blue","candidate_id":"revisit"}',
        '{"candidate_id":"inspect-blue"} {"candidate_id":"revisit"}',
    ],
)
def test_no_salvaging_unsafe_or_ambiguous_outputs(output, semantic_request):
    assert checked(output, semantic_request).candidate_id is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epoch": 4},
        {"snapshot": "frame-13"},
        {"now": 120.0},
        {"now": 99.0},
        {"now": float("nan")},
    ],
)
def test_expired_or_interrupted_inference_cannot_be_accepted(kwargs, semantic_request):
    assert (
        checked('{"candidate_id":"inspect-blue"}', semantic_request, **kwargs).status
        == "stale"
    )


def test_byte_limits_and_prompt_path_privacy(semantic_request):
    assert (
        checked("💥" * MAX_RESPONSE_BYTES, semantic_request).status
        == "response_too_large"
    )
    with pytest.raises(ValueError, match="byte limit"):
        replace(semantic_request, observation={"unbounded": "x" * 33_000})
    prompt = build_prompt(
        replace(semantic_request, image_paths=("/private/room.png",)).to_payload()
    )
    assert "/private/room.png" not in prompt
    assert "candidate_id" in prompt


def test_duplicate_candidates_and_remote_images_rejected(semantic_request):
    with pytest.raises(ValueError):
        replace(semantic_request, candidates=(semantic_request.candidates[0],) * 2)
    with pytest.raises(ValueError):
        replace(semantic_request, image_paths=("https://remote.invalid/image.png",))
    with pytest.raises(ValueError):
        SemanticCandidate("wait", "reserved")
    with pytest.raises(ValueError):
        replace(semantic_request, observation={"bad": float("nan")})


def child(source, timeout=1.0):
    return SubprocessCandidateRanker(
        [sys.executable, "-c", source], timeout_seconds=timeout
    )


def test_real_subprocess_protocol(semantic_request):
    ranker = child(
        "import sys,json; p=json.load(sys.stdin); "
        'print(json.dumps({"candidate_id":p["candidates"][0]["candidate_id"]}))'
    )
    result = ranker.rank(semantic_request, current=lambda: (101.0, 3, "frame-12"))
    assert result.status == "proposed"
    assert result.candidate_id == "inspect-blue"


def test_epoch_checked_after_process_returns(semantic_request):
    states = iter([(101.0, 3, "frame-12"), (102.0, 4, "frame-12")])
    ranker = child('print(\'{"candidate_id":"inspect-blue"}\')')
    assert ranker.rank(semantic_request, current=lambda: next(states)).status == "stale"


def test_timeout_and_output_flood_are_abstentions(semantic_request):
    start = time.monotonic()
    hung = child("import time; time.sleep(10)", timeout=0.1)
    assert (
        hung.rank(semantic_request, current=lambda: (101.0, 3, "frame-12")).status
        == "timeout"
    )
    assert time.monotonic() - start < 2.0
    flood = child('import sys; sys.stdout.write("x"*1000000); sys.stdout.flush()')
    result = flood.rank(semantic_request, current=lambda: (101.0, 3, "frame-12"))
    assert result.status == "response_too_large"
    assert result.candidate_id is None


def test_hung_nonreading_worker_cannot_block_stdin(semantic_request):
    semantic_request = replace(semantic_request, observation={"large": "x" * 29_000})
    ranker = child("import time; time.sleep(10)", timeout=0.1)
    start = time.monotonic()
    assert (
        ranker.rank(semantic_request, current=lambda: (101.0, 3, "frame-12")).status
        == "timeout"
    )
    assert time.monotonic() - start < 2.0


def test_worker_failure_and_missing_runtime_abstain(semantic_request):
    assert (
        child("raise RuntimeError('failure')")
        .rank(semantic_request, current=lambda: (101.0, 3, "frame-12"))
        .status
        == "worker_failed"
    )
    missing = SubprocessCandidateRanker(["/nonexistent/bb8-semantic-python"])
    assert (
        missing.rank(semantic_request, current=lambda: (101.0, 3, "frame-12")).status
        == "unavailable"
    )


def test_no_candidates_skips_worker(semantic_request):
    semantic_request = replace(semantic_request, candidates=())
    ranker = SubprocessCandidateRanker(["/nonexistent/bb8-semantic-python"])
    assert (
        ranker.rank(semantic_request, current=lambda: (101.0, 3, "frame-12")).status
        == "wait"
    )


def test_request_can_be_serialized_without_extra_dependencies(semantic_request):
    assert json.loads(json.dumps(semantic_request.to_payload()))["epoch"] == 3
