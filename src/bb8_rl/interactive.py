"""Loopback control room. Rendering/inference live in a separate native worker."""

import argparse
import json
import math
import mimetypes
import multiprocessing
import queue
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .demo_assets import checked_path, validate_assets

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).with_name("web")
ACTIONS = {"goal", "stop", "reset", "mode", "demo", "heartbeat"}


def validate_command(data):
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        raise ValueError("Unknown command")
    action = data["action"]
    allowed = {"action"} | (
        {"x", "y"} if action == "goal" else {"mode"} if action == "mode" else set()
    )
    if set(data) != allowed:
        raise ValueError("Unexpected or missing command fields")
    if action == "goal":
        for key in ("x", "y"):
            if type(data[key]) not in (int, float) or not math.isfinite(data[key]):
                raise ValueError("Destination needs two finite metric coordinates")
    if action == "mode" and (
        type(data["mode"]) is not int or data["mode"] not in (1, 2, 3)
    ):
        raise ValueError("Camera mode must be 1, 2 or 3")
    return dict(data)


def map_payload(asset_dir, demo):
    import numpy as np

    with np.load(
        checked_path(asset_dir, demo["memory"]) / "free-grid.npz", allow_pickle=False
    ) as data:
        cells = np.zeros_like(data["free_mask"], dtype=np.uint8)
        cells[data["free_mask"]] = 1
        cells[data["occupied_mask"]] = 2
        extent, resolution = float(data["extent"]), float(data["resolution"])
    return {
        "bounds": [-extent, -extent, extent, extent],
        "resolution": resolution,
        "width": cells.shape[1],
        "height": cells.shape[0],
        "cells": cells.tolist(),
        "demo_start": demo["start"],
        "demo_goal": demo["goal"],
        "provenance": "Estimated scan memory. Unknown cells remain blocked.",
    }


def goal_cell(map_data, x, y):
    left, bottom, right, top = map_data["bounds"]
    if not left <= x < right or not bottom <= y < top:
        raise ValueError("Destination is outside the scanned map")
    col = int((x - left) / map_data["resolution"])
    row = int((y - bottom) / map_data["resolution"])
    if map_data["cells"][row][col] != 1:
        raise ValueError(
            "Destination is occupied or unknown; choose observed clear space"
        )
    return row, col


class Supervisor:
    def __init__(
        self, asset_dir, run_dir, *, mode=1, worker_target=None, control_profile=None
    ):
        from .control_profiles import get_profile

        self.control_profile = get_profile(control_profile).name
        self.asset_dir, self.run_dir = (
            Path(asset_dir).resolve(),
            Path(run_dir).resolve(),
        )
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.context = multiprocessing.get_context("spawn")
        self.process = self.commands = self.events = self.worker_stop = None
        self.worker_target = worker_target
        self.generation = 0
        self.restart_requested = False
        self.pending_demo = False
        self.frames = {}
        self.map = None
        self.state = {
            "phase": "starting",
            "mode": mode,
            "sim_time": 0,
            "pose": None,
            "goal": None,
            "route": [],
            "position_radius": None,
            "localization_status": "uninitialized",
            "localization_valid": False,
            "controller_status": "initializing",
            "source_ids": [],
            "views": {},
            "frame_seq": 0,
            "processing_ms": None,
            "message": "Loading checked demo assets…",
            "asset_ready": False,
            "generation": 0,
        }
        try:
            self.demo = validate_assets(self.asset_dir)
            self.map = map_payload(self.asset_dir, self.demo)
            self.state["asset_ready"] = True
            self.restart_requested = True
        except (OSError, ValueError, KeyError) as error:
            self.state.update(
                phase="missing_assets",
                message=f"Demo assets unavailable: {error}. Run scripts/install-demo-assets.py.",
            )
        self.thread = threading.Thread(
            target=self._manage, daemon=True, name="bb8-supervisor"
        )
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return {
                **self.state,
                "token": self.token,
                "generation": self.generation,
                "control_profile": getattr(self, "control_profile", None),
                "scope": "Synthetic room · lockstep simulation · estimated map",
            }

    def _send(self, data):
        if self.commands is None:
            return False
        data = {**data, "generation": self.generation}
        try:
            self.commands.put_nowait(data)
            return True
        except queue.Full:
            # Saturation cannot leave an older destination executing indefinitely.
            if data["action"] == "heartbeat":
                return False
            self.worker_stop.set()
            self.state.update(
                phase="error",
                message="Command queue full; simulation stopped. Reset to resume.",
            )
            return False

    def command(self, data):
        data = validate_command(data)
        action = data["action"]
        with self.lock:
            if action == "heartbeat":
                self._send(data)
                return {"accepted": True}
            if action == "stop":
                self.generation += 1
                self.pending_demo = False
                sent = self._send(data)
                self.state.update(goal=None, route=[])
                if sent:
                    self.state.update(
                        phase="stopping", message="Stopping and cancelling the route…"
                    )
                elif self.state["phase"] != "error":
                    self.state.update(
                        phase="stopped", message="Stopped; no active command channel."
                    )
                return {"accepted": True, "generation": self.generation}
            if not self.state["asset_ready"]:
                raise ValueError(
                    "Install the checked demo assets before starting simulation"
                )
            if action in ("reset", "mode", "demo"):
                self.generation += 1
                self.pending_demo = action == "demo"
                if action == "mode":
                    self.state["mode"] = data["mode"]
                self.state.update(
                    phase="starting",
                    message="Resetting the simulation…",
                    pose=None,
                    goal=None,
                    route=[],
                    source_ids=[],
                    views={},
                    frame_seq=0,
                    localization_status="uninitialized",
                    localization_valid=False,
                    position_radius=None,
                    last_seen_pose=None,
                    last_seen_age_s=None,
                    last_seen_radius_m=None,
                    recovery_anchor_pose=None,
                    recovery_anchor_time=None,
                    recovery_anchor_source=None,
                    raw_position_radius=None,
                    raw_prediction_time=None,
                    raw_pose=None,
                    raw_velocity=None,
                    raw_velocity_radius=None,
                    reacquisition_samples=0,
                    braking_prediction=None,
                    requires_new_goal=False,
                )
                self.frames.clear()
                self.restart_requested = True
                if self.worker_stop is not None:
                    self.worker_stop.set()
                return {"accepted": True, "generation": self.generation}
            if self.state["phase"] in (
                "starting",
                "missing_assets",
                "error",
                "stopping",
            ):
                raise ValueError(
                    "Wait for the camera estimate before choosing a destination"
                )
            if self.state.get("localization_valid") is False or self.state.get(
                "localization_status"
            ) in ("lost", "reacquiring"):
                raise ValueError(
                    "Localization lost; wait for visual reacquisition or reset."
                )
            goal_cell(self.map, data["x"], data["y"])
            self.generation += 1
            if not self._send(data):
                raise ValueError("Command channel unavailable; reset the simulation")
            self.state.update(
                phase="braking",
                controller_status="checking_route",
                message="Checking the route and stopping envelope…",
                goal=[data["x"], data["y"]],
                route=[],
            )
            return {"accepted": True, "generation": self.generation}

    def _shutdown_worker(self):
        # Detach under the same lock used by HTTP commands, then perform the
        # potentially slow joins outside it. A sender can never use a channel
        # while it is being closed, and Reset/Stop stay responsive during join.
        with self.lock:
            process, commands, events, stop = (
                self.process,
                self.commands,
                self.events,
                self.worker_stop,
            )
            self.process = self.commands = self.events = self.worker_stop = None
        if process is not None:
            stop.set()
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            for channel in (commands, events):
                channel.cancel_join_thread()
                channel.close()

    def _start_worker(self):
        self._shutdown_worker()
        with self.lock:
            if self.closed.is_set():
                return
            # Coalesce any Reset/Mode requests received during shutdown. The
            # current generation/mode below already includes those requests;
            # leaving their flag set would launch this same generation twice.
            self.restart_requested = False
            if self.worker_target is None:
                from .interactive_runtime import worker_main

                target = worker_main
            else:
                target = self.worker_target
            self.commands = self.context.Queue(maxsize=32)
            self.events = self.context.Queue(maxsize=8)
            self.worker_stop = self.context.Event()
            run = self.run_dir / f"session-{self.generation:04d}"
            run.mkdir(parents=True, exist_ok=False)
            config = {
                "asset_dir": str(self.asset_dir),
                "run_dir": str(run),
                "mode": self.state["mode"],
                "generation": self.generation,
                "control_profile": getattr(self, "control_profile", None),
            }
            self.process = self.context.Process(
                target=target,
                args=(config, self.commands, self.events, self.worker_stop),
                name="BB8 native simulation",
            )
            self.process.start()
            self._send({"action": "heartbeat"})

    def _accept_event(self, event):
        """Consume only current-generation presentation state under the lock."""
        with self.lock:
            if self.restart_requested:
                return False
            if event.get("type") == "state":
                state = event["state"]
                if state.get("generation", -1) != self.generation:
                    return False
                self.state.update(state)
                if (
                    self.pending_demo
                    and state.get("phase") == "idle"
                    and state.get("pose") is not None
                ):
                    self.pending_demo = False
                    self._send({"action": "demo"})
            elif event.get("type") == "frames":
                if event.get("generation", -1) != self.generation:
                    return False
                self.frames.update(event["frames"])
            elif event.get("type") == "error":
                # Errors without a generation are terminal worker failures;
                # the queue itself belongs exclusively to the current worker.
                if event.get("generation", self.generation) != self.generation:
                    return False
                self.state.update(phase="error", message=event["message"])
            else:
                return False
            return True

    def _manage(self):
        try:
            while not self.closed.wait(0.025):
                with self.lock:
                    restart, self.restart_requested = self.restart_requested, False
                if restart:
                    self._start_worker()
                if self.events is None:
                    continue
                for _ in range(12):
                    try:
                        event = self.events.get_nowait()
                    except queue.Empty:
                        break
                    self._accept_event(event)
                if self.process is not None and not self.process.is_alive():
                    with self.lock:
                        if (
                            not self.restart_requested
                            and self.state["phase"] != "error"
                        ):
                            self.state.update(
                                phase="error",
                                message="Simulation worker exited. Reset to start a new session.",
                            )
        except Exception as error:  # noqa: BLE001 — process boundary must expose failure in the UI
            with self.lock:
                self.state.update(
                    phase="error", message=f"Simulation supervisor failed: {error}"
                )
        finally:
            self._shutdown_worker()

    def close(self):
        self.closed.set()
        with self.lock:
            if self.worker_stop is not None:
                self.worker_stop.set()
        self.thread.join(timeout=7)


def make_handler(supervisor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _local_request(self):
            host = self.headers.get("Host", "")
            expected = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            return host in expected

        def reply(self, status, payload, content_type="application/json"):
            body = (
                json.dumps(payload, allow_nan=False).encode()
                if content_type == "application/json"
                else payload
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._local_request():
                return self.reply(403, {"error": "Loopback host required"})
            path = urlsplit(self.path).path
            if path == "/api/state":
                return self.reply(200, supervisor.snapshot())
            if path == "/api/map":
                return (
                    self.reply(200, supervisor.map)
                    if supervisor.map
                    else self.reply(503, {"error": "Demo map unavailable"})
                )
            if path.startswith("/api/frame/"):
                camera_id = path.rsplit("/", 1)[-1]
                with supervisor.lock:
                    frame = (
                        supervisor.frames.get(camera_id)
                        if camera_id in "ABC" and len(camera_id) == 1
                        else None
                    )
                return (
                    self.reply(200, frame, "image/jpeg")
                    if frame
                    else self.reply(404, {"error": "Waiting for RGB"})
                )
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path not in static:
                return self.reply(404, {"error": "Not found"})
            file = WEB / static[path]
            try:
                body = file.read_bytes()
            except OSError:
                return self.reply(503, {"error": "Interface files unavailable"})
            return self.reply(
                200, body, mimetypes.guess_type(file.name)[0] or "text/plain"
            )

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self._local_request() or (
                origin and origin != f"http://{self.headers.get('Host')}"
            ):
                return self.reply(
                    403, {"error": "Same-origin loopback requests required"}
                )
            if urlsplit(self.path).path != "/api/command":
                return self.reply(404, {"error": "Not found"})
            if not secrets.compare_digest(
                self.headers.get("X-BB8-Token", ""), supervisor.token
            ):
                return self.reply(403, {"error": "Invalid session token"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < length <= 4096
                    or self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    raise ValueError("Expected a small JSON command")
                result = supervisor.command(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError) as error:
                return self.reply(400, {"error": str(error)})
            return self.reply(200, result)

    return Handler


def main(argv=None):
    from .control_profiles import PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--control-profile", choices=tuple(PROFILES))
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be 0–65535")
    run_dir = args.output or ROOT / "work/interactive" / time.strftime("%Y%m%d-%H%M%S")
    if run_dir.exists():
        parser.error("Choose a fresh output directory")
    supervisor = Supervisor(
        args.assets, run_dir, mode=args.mode, control_profile=args.control_profile
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(supervisor))
    except OSError:
        supervisor.close()
        raise
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"BB8-RL control room: {url}\nEvidence: {run_dir}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        supervisor.close()


if __name__ == "__main__":
    main()
