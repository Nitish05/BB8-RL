#!/usr/bin/env python3
"""Offline RGB advisory trial; optional MLX imports stay in a bounded worker.

Run with scripts/python.sh. Use --python for the isolated MLX environment.
--worker implements the strict stdout protocol for SubprocessCandidateRanker.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bb8_rl.semantic_proposals import (
    MAX_REQUEST_BYTES,
    SemanticCandidate,
    SemanticRequest,
    build_prompt,
    validate_choice,
)


def request_from_payload(payload: dict) -> SemanticRequest:
    return SemanticRequest(
        snapshot_id=payload["snapshot_id"],
        epoch=payload["epoch"],
        observed_at=payload["observed_at"],
        expires_at=payload["expires_at"],
        observation=payload["observation"],
        candidates=tuple(
            SemanticCandidate(
                c["candidate_id"], c["description"], tuple(c.get("evidence_ids", []))
            )
            for c in payload["candidates"]
        ),
        image_paths=tuple(payload.get("image_paths", [])),
    )


def load_model(model_path: str):
    if not Path(model_path).is_dir():
        raise ValueError("model must be an explicitly downloaded local directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from mlx_vlm import load

    return load(model_path, trust_remote_code=False)


def infer(model, processor, request: SemanticRequest) -> dict:
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    images = []
    for name in request.image_paths:
        path = Path(name)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("image exceeds 16 MiB limit")
        with Image.open(path) as original:
            if original.width * original.height > 8_000_000:
                raise ValueError("image exceeds 8 megapixel limit")
            rgb = original.convert("RGB")
            rgb.thumbnail((512, 512))
            images.append(rgb)
    prompt = apply_chat_template(
        processor,
        model.config,
        build_prompt(request.to_payload()),
        num_images=len(images),
        enable_thinking=False,
    )
    mx.reset_peak_memory()
    start = time.perf_counter()
    output = generate(
        model,
        processor,
        prompt,
        image=images or None,
        max_tokens=96,
        temperature=0.0,
        verbose=False,
    )
    mx.synchronize()
    return {
        "output": output.text,
        "inference_seconds": time.perf_counter() - start,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "prompt_tokens": getattr(output, "prompt_tokens", None),
        "generation_tokens": getattr(output, "generation_tokens", None),
        "image_sizes": [list(image.size) for image in images],
    }


def worker(model_path: str, batch: bool) -> None:
    limit = MAX_REQUEST_BYTES * (24 if batch else 1)
    raw = sys.stdin.buffer.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("worker input exceeds byte limit")
    payload = json.loads(raw)
    with contextlib.redirect_stdout(sys.stderr):
        start = time.perf_counter()
        model, processor = load_model(model_path)
        load_seconds = time.perf_counter() - start
        if not batch:
            result = infer(model, processor, request_from_payload(payload))
        else:
            if not isinstance(payload, list) or not 1 <= len(payload) <= 24:
                raise ValueError("batch must contain 1–24 cases")
            rows = []
            for case in payload:
                request = request_from_payload(case["request"])
                row = infer(model, processor, request)
                # Replay uses recorded evidence, not a live localization clock.
                choice = validate_choice(
                    row["output"],
                    request,
                    now=request.observed_at,
                    current_epoch=request.epoch,
                    current_snapshot_id=request.snapshot_id,
                )
                row.update(
                    name=case["name"],
                    status=choice.status,
                    candidate_id=choice.candidate_id,
                    expected_ids=case.get("expected_ids", []),
                )
                expected = case.get("expected_ids")
                row["matches_expected"] = (
                    (
                        (choice.candidate_id or "wait") in expected
                        and choice.status in {"proposed", "wait"}
                    )
                    if expected
                    else None
                )
                rows.append(row)
            result = {
                "load_seconds": load_seconds,
                "cases": rows,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("mlx-vlm", "mlx", "transformers", "Pillow")
                },
            }
    print(json.dumps(result) if batch else result["output"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch-worker", action="store_true")
    args = parser.parse_args()
    if args.worker or args.batch_worker:
        worker(str(args.model.resolve()), args.batch_worker)
        return
    if not args.manifest or not args.output or not 0 < args.timeout <= 600:
        parser.error("manifest, output, and timeout in (0, 600] are required")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    if not 1 <= len(cases) <= 24:
        parser.error("manifest must contain 1–24 cases")
    for case in cases:
        request_from_payload(case["request"])
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--batch-worker",
        "--model",
        str(args.model.resolve()),
    ]
    start = time.perf_counter()
    report = {
        "scope": "offline recorded RGB smoke trial; no motion or personality learning",
        "model": manifest.get("model", {}),
        "dataset": manifest.get("dataset", {}),
        "completed": False,
    }
    try:
        run = subprocess.run(
            command,
            input=json.dumps(cases),
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        report["returncode"] = run.returncode
        report["diagnostics_tail"] = run.stderr[-4000:]
        if run.returncode == 0:
            report.update(json.loads(run.stdout))
            report["completed"] = True
            times = [c["inference_seconds"] for c in report["cases"]]
            report["latency_median_seconds"] = statistics.median(times)
            report["latency_max_seconds"] = max(times)
        else:
            report["error"] = "worker failed; no model result accepted"
    except subprocess.TimeoutExpired:
        report["error"] = "trial timeout; worker terminated; no model result accepted"
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
    report["total_wall_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("completed", "total_wall_seconds", "error")
                if key in report
            }
        )
    )


if __name__ == "__main__":
    main()
