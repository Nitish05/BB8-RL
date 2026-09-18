"""Finite native probes and machine evidence; no hardware imports on the default path."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from genesis_studio.services.atomic_io import atomic_write_text


def write_report(path: Path, report: dict):
    atomic_write_text(path, json.dumps(report, indent=2, allow_nan=False) + "\n")


def machine_report(config_path: Path) -> dict:
    from genesis_studio.project import REPOSITORY_ROOT as studio_root

    from .config import load_config

    config, project_path = load_config(config_path)

    def command(*args):
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    source_hash = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for source in sorted(source_root.rglob("*.py")):
        source_hash.update(str(source.relative_to(source_root)).encode())
        source_hash.update(source.read_bytes())
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string")
        if sys.platform == "darwin"
        else platform.processor(),
        "memory_bytes": command("sysctl", "-n", "hw.memsize")
        if sys.platform == "darwin"
        else "unavailable",
        "git_revision": command(
            "git", "-C", str(config_path.resolve().parent), "rev-parse", "HEAD"
        ),
        "git_worktree_changes": command(
            "git", "-C", str(config_path.resolve().parent), "status", "--short"
        ),
        "bb8_source_sha256": source_hash.hexdigest(),
        "studio_checkout": str(studio_root),
        "studio_git_revision": command(
            "git", "-C", str(studio_root), "rev-parse", "HEAD"
        ),
        "studio_worktree_changes": command(
            "git", "-C", str(studio_root), "status", "--short"
        ),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "project_sha256": hashlib.sha256(project_path.read_bytes()).hexdigest(),
        "provenance": config.provenance,
        "parameters": config.model_dump(),
        "packages": dict(sorted(packages.items())),
        "camera": {
            "status": "not_tested",
            "reason": "External camera check was not requested",
        },
        "ble": {
            "status": "not_tested",
            "reason": "No Sphero adapter or hardware command test in M1; hardware compatibility remains pending",
        },
    }


def learner_probe(device: str) -> dict:
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable; use native Mac access or explicitly select CPU"
        )
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x, target = torch.randn(64, 8, device=device), torch.randn(64, 2, device=device)
    initial = next(model.parameters()).detach().clone()
    started = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(
            not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError("Learner produced non-finite loss or gradients")
        optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    if torch.equal(initial, next(model.parameters())):
        raise RuntimeError("Optimizer failed to change the model weights")
    return {
        "status": "passed",
        "requested_device": device,
        "resolved_device": str(next(model.parameters()).device),
        "dtype": "float32",
        "loss": float(loss.detach().cpu()),
        "updates": 3,
        "seconds": time.perf_counter() - started,
    }


def simulation_probe(
    config_path: Path,
    backend: str,
    *,
    render: bool = True,
    image_path: Path | None = None,
) -> dict:
    from .world import NavigationWorld

    with NavigationWorld(config_path, backend=backend, render=render) as world:
        world.step(40)
        action_durations = []
        started = time.perf_counter()
        for _ in range(20):
            action_started = time.perf_counter()
            world.drive(1, 0)
            driven = world.step(world.config.action_steps)
            action_durations.append(time.perf_counter() - action_started)
        elapsed = time.perf_counter() - started
        if driven.position[0] <= 0.02 or abs(driven.position[1]) > 0.01:
            raise RuntimeError(f"Free-body positive X drive check failed: {driven}")
        world.stop()
        stopped = world.step(400)
        if math.hypot(*stopped.velocity) >= 0.03:
            raise RuntimeError(f"Bounded braking failed: {stopped}")
        image_shape = None
        if render:
            import numpy as np

            rgb = world.frames[-1].rgb
            if (
                rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[-1] != 3
                or np.ptp(rgb) == 0
            ):
                raise RuntimeError("RGB render is empty or has an unexpected format")
            image_shape = list(rgb.shape)
            if image_path is not None:
                from PIL import Image

                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(image_path)
        reset = world.reset()
        if (
            any(
                abs(a - b) > 1e-6
                for a, b in zip(reset.position, world.body_spec.position[:2])
            )
            or math.hypot(*reset.velocity) > 1e-6
        ):
            raise RuntimeError(
                "Reset failed to restore initial position and zero velocity"
            )
        # Expiry, rotated shell, Y response and obstacle contacts are covered by integration tests.
        return {
            "status": "passed",
            "requested_backend": backend,
            "resolved_backend": world.gs.backend.name,
            "tensor_device": str(world.gs.device),
            "dtype": "float32",
            "body_mass_kg": world.backend.mass,
            "drive": asdict(driven),
            "stop": asdict(stopped),
            "reset": asdict(reset),
            "physics_steps_per_second": 20 * world.config.action_steps / elapsed,
            "action_steps_per_second": 20 / elapsed,
            "action_duration_p95_seconds": sorted(action_durations)[18],
            "configured_action_period_seconds": world.config.action_steps * world.dt,
            "timed_action_samples": 20,
            "timing_includes": "actuation, state reads, head pose, contacts and camera cadence when enabled",
            "render": render,
            "rgb_shape": image_shape,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        }


def camera_probe(index: int) -> dict:
    import cv2

    camera = cv2.VideoCapture(index)
    arrivals, durations = [], []
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Camera {index} could not open; connect it and check macOS camera permissions"
            )
        for _ in range(60):
            start = time.monotonic()
            ok, frame = camera.read()
            arrived = time.monotonic()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            arrivals.append(arrived)
            durations.append(arrived - start)
        return {
            "status": "passed",
            "device_index": index,
            "frames": len(arrivals),
            "resolution": [frame.shape[1], frame.shape[0]],
            "delivered_fps": (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]),
            "read_duration_p95_seconds": sorted(durations)[
                int(0.95 * (len(durations) - 1))
            ],
            "timestamp_source": "host monotonic arrival; sensor exposure timestamps unavailable",
            "end_to_end_capture_latency": "unmeasured; read duration is not exposure-to-host latency",
        }
    finally:
        camera.release()


def doctor(
    config_path: Path,
    backend: str,
    device: str,
    output: Path,
    camera_index: int | None = None,
):
    report = machine_report(config_path)
    output.mkdir(parents=True, exist_ok=True)
    report["simulation"] = {}
    checks = [("cpu", ["--backend", "cpu"])]
    if backend != "cpu":
        checks.append((backend, ["--backend", backend]))
    checks.append(("learner", ["--device", device]))
    if camera_index is not None:
        checks.append(("camera", ["--camera-index", str(camera_index)]))
    for name, options in checks:
        result_path = (output / f"{name}.json").resolve()
        log_path = output / f"{name}.log"
        cmd = [
            sys.executable,
            "-m",
            "bb8_rl.cli",
            "_probe",
            "--config",
            str(config_path.resolve()),
            "--output",
            str(result_path),
            *options,
        ]
        result_path.unlink(missing_ok=True)
        try:
            with log_path.open("w") as log:
                result = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
                )
            data = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {
                    "status": "failed",
                    "reason": f"Probe exited {result.returncode}; see {log_path.name}",
                }
            )
            if result.returncode and data.get("status") == "passed":
                data = {
                    "status": "failed",
                    "reason": f"Probe teardown exited {result.returncode}",
                }
        except subprocess.TimeoutExpired:
            data = {"status": "failed", "reason": "Probe exceeded 180 seconds"}
        finally:
            # Retain bounded diagnostic output, including failures from native initialization.
            if log_path.exists() and log_path.stat().st_size > 65536:
                with log_path.open("rb") as stream:
                    stream.seek(-65536, 2)
                    tail = stream.read()
                log_path.write_bytes(tail)
        if name in ("learner", "camera"):
            report[name] = data
        else:
            report["simulation"][name] = data
        write_report(output / "machine-report.json", report)
    report["simulation_gate"] = (
        "passed"
        if all(r["status"] == "passed" for r in report["simulation"].values())
        and report["learner"]["status"] == "passed"
        else "failed"
    )
    report["hardware_gate"] = "pending"
    write_report(output / "machine-report.json", report)
    lock = "# Exact native diagnostic environment; not a cross-platform or hashed resolver lock.\n"
    lock += (
        "# Install the local Studio/navigation packages separately with --no-deps -e.\n"
    )
    lock += (
        "\n".join(
            f"{name}=={v}"
            for name, v in report["packages"].items()
            if not name.lower().startswith("genesis-studio")
            and name.lower() != "bb8-rl"
        )
        + "\n"
    )
    atomic_write_text(output / "requirements-macos-py312.lock.txt", lock)
    return report
