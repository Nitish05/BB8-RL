"""Control-room command and HTTP boundaries without a renderer or model."""

import http.client
import json
import queue
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from bb8_rl.interactive import Supervisor, goal_cell, make_handler, validate_command


@pytest.mark.parametrize(
    "command",
    [
        {"action": "goal", "x": float("nan"), "y": 0},
        {"action": "goal", "x": True, "y": 0},
        {"action": "goal", "x": 1},
        {"action": "mode", "mode": True},
        {"action": "mode", "mode": 4},
        {"action": "reset", "x": 0},
        {"action": "shell", "command": "anything"},
        None,
    ],
)
def test_reject_malformed_commands(command):
    with pytest.raises(ValueError):
        validate_command(command)


def sample_map():
    return {"bounds": [-1, -1, 1, 1], "resolution": 1, "cells": [[0, 1], [2, 1]]}


def test_goal_coordinates_use_metric_lower_left_origin():
    assert goal_cell(sample_map(), 0.5, -0.5) == (0, 1)
    assert goal_cell(sample_map(), 0.5, 0.5) == (1, 1)
    for x, y in [(-0.5, -0.5), (-0.5, 0.5), (1, 0), (0, 1), (-2, 0)]:
        with pytest.raises(ValueError):
            goal_cell(sample_map(), x, y)


def fake_supervisor():
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.lock = threading.RLock()
    supervisor.token = "test-session-token"
    supervisor.generation = 0
    supervisor.commands = queue.Queue(maxsize=32)
    supervisor.worker_stop = threading.Event()
    supervisor.pending_demo = False
    supervisor.restart_requested = False
    supervisor.frames = {}
    supervisor.map = sample_map()
    supervisor.state = {
        "asset_ready": True,
        "phase": "idle",
        "mode": 1,
        "goal": None,
        "route": [],
    }
    return supervisor


def test_stop_invalidates_old_goal_and_mode_restarts_without_route():
    supervisor = fake_supervisor()
    supervisor.command({"action": "goal", "x": 0.5, "y": -0.5})
    supervisor.command({"action": "stop"})
    goal, stop = supervisor.commands.get_nowait(), supervisor.commands.get_nowait()
    assert goal["generation"] < stop["generation"]
    assert supervisor.state["goal"] is None and supervisor.state["route"] == []
    supervisor.command({"action": "mode", "mode": 3})
    assert supervisor.worker_stop.is_set()
    assert supervisor.restart_requested and supervisor.state["mode"] == 3
    assert supervisor.state["pose"] is None


def test_rejected_click_does_not_cancel_or_replace_valid_goal():
    supervisor = fake_supervisor()
    supervisor.command({"action": "goal", "x": 0.5, "y": -0.5})
    with pytest.raises(ValueError):
        supervisor.command({"action": "goal", "x": -0.5, "y": -0.5})
    assert supervisor.commands.qsize() == 1
    assert supervisor.state["goal"] == [0.5, -0.5]


def test_saturated_command_channel_stops_worker():
    supervisor = fake_supervisor()
    for _ in range(32):
        supervisor.commands.put_nowait({"action": "heartbeat"})
    supervisor.command({"action": "stop"})
    assert supervisor.worker_stop.is_set()
    assert supervisor.state["phase"] == "error"


@pytest.fixture
def http_server():
    supervisor = fake_supervisor()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(supervisor))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, supervisor
    server.shutdown()
    server.server_close()
    thread.join()


def request(server, method, path, data=None, **headers):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    body = json.dumps(data) if data is not None else None
    connection.request(
        method, path, body, {"Content-Type": "application/json", **headers}
    )
    response = connection.getresponse()
    status, payload = response.status, response.read()
    connection.close()
    return status, payload


def test_http_requires_token_same_origin_and_local_host(http_server):
    server, supervisor = http_server
    assert request(server, "POST", "/api/command", {"action": "stop"})[0] == 403
    assert (
        request(
            server,
            "POST",
            "/api/command",
            {"action": "stop"},
            **{"X-BB8-Token": supervisor.token, "Origin": "https://example.com"},
        )[0]
        == 403
    )
    assert request(server, "GET", "/api/state", Host="example.com")[0] == 403
    status, _ = request(
        server,
        "POST",
        "/api/command",
        {"action": "stop"},
        **{"X-BB8-Token": supervisor.token},
    )
    assert status == 200 and supervisor.commands.get_nowait()["action"] == "stop"


def test_http_rejects_arbitrary_files_and_unknown_json_fields(http_server):
    server, supervisor = http_server
    assert request(server, "GET", "/../../pyproject.toml")[0] == 404
    assert (
        request(
            server,
            "POST",
            "/api/command",
            {"action": "reset", "path": "/tmp"},
            **{"X-BB8-Token": supervisor.token},
        )[0]
        == 400
    )
    assert supervisor.commands.empty()


def test_old_generation_state_and_frames_cannot_repopulate_new_session():
    supervisor = fake_supervisor()
    supervisor.command({"action": "mode", "mode": 2})
    supervisor.restart_requested = False
    assert not supervisor._accept_event(
        {
            "type": "state",
            "state": {"generation": 0, "phase": "arrived", "goal": [0.5, 0.5]},
        }
    )
    assert not supervisor._accept_event(
        {"type": "frames", "generation": 0, "frames": {"A": b"old"}}
    )
    assert supervisor.frames == {}
    assert supervisor.state["goal"] is None
    assert supervisor._accept_event(
        {"type": "frames", "generation": 1, "frames": {"A": b"new"}}
    )
    assert supervisor.frames == {"A": b"new"}


def test_demo_waits_for_current_ready_pose_and_stop_cancels_pending_demo():
    supervisor = fake_supervisor()
    supervisor.command({"action": "demo"})
    supervisor.restart_requested = False
    supervisor._accept_event(
        {
            "type": "state",
            "state": {"generation": 1, "phase": "localizing", "pose": None},
        }
    )
    assert supervisor.commands.empty()
    assert supervisor.pending_demo
    supervisor._accept_event(
        {"type": "state", "state": {"generation": 1, "phase": "idle", "pose": [0, 0]}}
    )
    assert supervisor.commands.get_nowait() == {"action": "demo", "generation": 1}
    assert not supervisor.pending_demo
    supervisor.command({"action": "demo"})
    supervisor.command({"action": "stop"})
    assert not supervisor.pending_demo


class ClosingQueue(queue.Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        self.closed = False

    def cancel_join_thread(self):
        pass

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, join_hook=None):
        self.join_hook = join_hook
        self.started = False

    def start(self):
        self.started = True

    def join(self, timeout):
        if self.join_hook is not None:
            hook, self.join_hook = self.join_hook, None
            hook()

    def is_alive(self):
        return self.started

    def terminate(self):
        self.started = False


def test_mode_change_during_worker_shutdown_is_coalesced_into_one_latest_start(
    tmp_path,
):
    supervisor = fake_supervisor()
    supervisor.closed = threading.Event()
    supervisor.asset_dir = tmp_path / "assets"
    supervisor.run_dir = tmp_path / "runs"
    supervisor.worker_target = lambda *args: None
    old_commands, old_events = ClosingQueue(), ClosingQueue()
    supervisor.commands, supervisor.events = old_commands, old_events
    spawned = []

    def process_factory(**kwargs):
        spawned.append(kwargs)
        return FakeProcess()

    supervisor.context = SimpleNamespace(
        Queue=ClosingQueue, Event=threading.Event, Process=process_factory
    )

    def during_shutdown():
        # HTTP requests can arrive while join waits. The old queue is already
        # detached, so these requests cannot race its subsequent close.
        assert supervisor.commands is None
        supervisor.command({"action": "mode", "mode": 3})
        supervisor.command({"action": "heartbeat"})

    supervisor.process = FakeProcess(join_hook=during_shutdown)
    supervisor.restart_requested = True
    supervisor._start_worker()
    assert old_commands.closed and old_events.closed
    assert len(spawned) == 1
    config = spawned[0]["args"][0]
    assert config["generation"] == 1 and config["mode"] == 3
    assert not supervisor.restart_requested
    assert (supervisor.run_dir / "session-0001").is_dir()
    assert supervisor.commands.get_nowait() == {"action": "heartbeat", "generation": 1}


def test_stop_without_channel_reports_stopped_and_does_not_schedule_demo():
    supervisor = fake_supervisor()
    supervisor.commands = None
    supervisor.pending_demo = True
    supervisor.command({"action": "stop"})
    assert supervisor.state["phase"] == "stopped"
    assert not supervisor.pending_demo
