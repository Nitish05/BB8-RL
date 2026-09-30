"""Control-room command and HTTP boundaries without a renderer or model."""

import http.client
import json
import queue
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
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
        {"action": "recheck_scene", "fault_epoch": 1},
        {"action": "recheck_scene", "generation": 1},
        {"action": "recheck_scene", "reference": "replacement"},
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
    supervisor.scene_invalidated_event = threading.Event()
    supervisor.scene_validity_enabled = False
    supervisor.scene_invalidated = False
    supervisor.scene_fault_epoch = 0
    supervisor.scene_recovered_epoch = 0
    supervisor.pending_scene_recheck = None
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


def faulted_supervisor(epoch=1):
    supervisor = fake_supervisor()
    supervisor.scene_validity_enabled = True
    supervisor.scene_invalidated_event.set()
    supervisor._accept_event(
        {
            "type": "state",
            "state": {
                "generation": 0,
                "phase": "scene_invalid",
                "scene_validity": {
                    "fault_epoch": epoch,
                    "ready": True,
                    "invalidated": True,
                    "navigation_allowed": False,
                },
                "agency": {"enabled": False, "intention": None},
            },
        }
    )
    return supervisor


def recovery_state(generation, epoch=1, status="succeeded"):
    return {
        "type": "state",
        "state": {
            "generation": generation,
            "phase": "stopped",
            "goal": None,
            "route": [],
            "scene_validity": {
                "ready": True,
                "invalidated": status != "succeeded",
                "navigation_allowed": status == "succeeded",
                "fault_epoch": epoch,
                "recovery": {
                    "generation": generation,
                    "fault_epoch": epoch,
                    "status": status,
                },
            },
        },
    }


def test_recheck_is_explicit_maintenance_with_server_owned_ticket():
    assert validate_command({"action": "recheck_scene"}) == {"action": "recheck_scene"}
    supervisor = faulted_supervisor()
    assert supervisor.snapshot()["scene_recheck_available"]
    accepted = supervisor.command({"action": "recheck_scene"})
    assert accepted == {"accepted": True, "generation": 1}
    assert supervisor.commands.get_nowait() == {
        "action": "recheck_scene",
        "generation": 1,
        "fault_epoch": 1,
    }
    assert supervisor.scene_invalidated_event.is_set()
    assert supervisor.snapshot()["scene_invalidated"]
    assert supervisor.snapshot()["scene_recheck_pending"]
    assert not supervisor.snapshot()["scene_recheck_available"]
    assert supervisor.state["goal"] is None and supervisor.state["route"] == []
    assert not supervisor.state["agency"]["enabled"]
    with pytest.raises(ValueError, match="stopped, locked"):
        supervisor.command({"action": "recheck_scene"})


@pytest.mark.parametrize(
    "change",
    [
        {"scene_validity_enabled": False},
        {"scene_fault_epoch": 0},
        {"commands": None},
        {"restart_requested": True},
    ],
)
def test_recheck_requires_current_fault_and_live_maintenance_channel(change):
    supervisor = faulted_supervisor()
    for name, value in change.items():
        setattr(supervisor, name, value)
    with pytest.raises(ValueError, match="stopped, locked"):
        supervisor.command({"action": "recheck_scene"})
    assert supervisor.generation == 0 and supervisor.scene_invalidated


@pytest.mark.parametrize(
    "state",
    [
        {"phase": "running"},
        {"phase": "stopping"},
        {"goal": [0, 0]},
        {"route": [[0, 0], [1, 1]]},
        {"agency": {"enabled": True}},
        {"scene_validity": {"ready": False, "invalidated": True, "fault_epoch": 1}},
    ],
)
def test_recheck_cannot_be_used_as_a_motion_command(state):
    supervisor = faulted_supervisor()
    supervisor.state.update(state)
    with pytest.raises(ValueError, match="stopped, locked"):
        supervisor.command({"action": "recheck_scene"})
    assert supervisor.commands.empty()


def test_recovery_requires_worker_event_clear_and_never_resumes_authority():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    event = recovery_state(1)
    assert supervisor._accept_event(event)
    assert supervisor.scene_invalidated  # A success presentation is not authority.
    supervisor.scene_invalidated_event.clear()  # Worker commit only.
    event["state"].update(goal=[0.5, 0.5], route=[[0, 0], [0.5, 0.5]])
    event["state"]["agency"] = {"enabled": True, "intention": {"goal": [0.5, 0.5]}}
    assert supervisor._accept_event(event)
    assert not supervisor.scene_invalidated and supervisor.scene_recovered_epoch == 1
    assert supervisor.pending_scene_recheck is None
    assert supervisor.state["phase"] == "stopped"
    assert supervisor.state["requires_new_goal"]
    assert supervisor.state["goal"] is None and supervisor.state["route"] == []
    assert not supervisor.state["agency"]["enabled"]
    assert supervisor.state["agency"]["intention"] is None
    assert supervisor.commands.qsize() == 1  # No hidden goal, Demo or enable.


@pytest.mark.parametrize("generation", [0, 2, True, None])
def test_stale_future_or_malformed_recovery_generation_never_unlocks(generation):
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    supervisor.scene_invalidated_event.clear()
    supervisor._accept_event(recovery_state(generation))
    assert supervisor.scene_invalidated and supervisor.scene_recovered_epoch == 0


@pytest.mark.parametrize("epoch", [0, 2, True, None])
def test_stale_future_or_malformed_recovery_epoch_never_unlocks(epoch):
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    supervisor.scene_invalidated_event.clear()
    supervisor._accept_event(recovery_state(1, epoch))
    assert supervisor.scene_invalidated and supervisor.scene_recovered_epoch == 0


def test_late_stop_cancels_success_ticket_and_allows_explicit_same_epoch_retry():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    supervisor.command({"action": "stop"})
    supervisor.scene_invalidated_event.clear()  # Recovery raced the late Stop.
    assert not supervisor._accept_event(recovery_state(1))
    assert supervisor.scene_invalidated and supervisor.pending_scene_recheck is None
    stopped = recovery_state(1)
    stopped["state"]["generation"] = 2
    assert supervisor._accept_event(stopped)
    assert (
        supervisor.scene_invalidated
        and supervisor.snapshot()["scene_recheck_available"]
    )
    supervisor.command({"action": "recheck_scene"})
    assert supervisor.pending_scene_recheck == {"generation": 3, "fault_epoch": 1}
    assert supervisor._accept_event(recovery_state(3))
    assert not supervisor.scene_invalidated


def test_rejected_recheck_keeps_lock_and_can_retry_without_new_reference():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    assert supervisor._accept_event(recovery_state(1, status="rejected"))
    assert supervisor.scene_invalidated and supervisor.pending_scene_recheck is None
    supervisor.command({"action": "recheck_scene"})
    assert supervisor.pending_scene_recheck == {"generation": 2, "fault_epoch": 1}


def test_acknowledged_epoch_discards_old_invalid_presentation_but_not_new_signal():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    supervisor.scene_invalidated_event.clear()
    supervisor._accept_event(recovery_state(1))
    old = recovery_state(1, status="checking")
    assert not supervisor._accept_event(old)
    assert not supervisor.scene_invalidated
    assert not supervisor.state["scene_validity"]["invalidated"]
    supervisor.scene_invalidated_event.set()  # New fault can precede its state.
    assert not supervisor._accept_event(old)
    assert supervisor.scene_invalidated
    new = recovery_state(1, epoch=2, status="rejected")
    assert supervisor._accept_event(new)
    assert supervisor.scene_fault_epoch == 2 and supervisor.scene_invalidated


def test_new_fault_supersedes_pending_ticket_without_unlocking():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    supervisor._accept_event(recovery_state(1, epoch=2, status="rejected"))
    assert supervisor.scene_fault_epoch == 2 and supervisor.scene_invalidated
    assert supervisor.pending_scene_recheck is None


def test_recheck_queue_failure_stops_worker_and_preserves_fault():
    supervisor = faulted_supervisor()
    for _ in range(32):
        supervisor.commands.put_nowait({"action": "heartbeat"})
    with pytest.raises(ValueError, match="lock retained"):
        supervisor.command({"action": "recheck_scene"})
    assert (
        supervisor.worker_stop.is_set() and supervisor.scene_invalidated_event.is_set()
    )
    assert supervisor.scene_invalidated and supervisor.pending_scene_recheck is None
    assert not supervisor.snapshot()["scene_recheck_available"]


def test_success_without_an_explicit_pending_recheck_never_unlocks():
    supervisor = faulted_supervisor()
    supervisor.scene_invalidated_event.clear()
    supervisor._accept_event(recovery_state(0))
    assert supervisor.scene_invalidated and supervisor.scene_recovered_epoch == 0


def test_inflight_worker_state_without_recovery_preserves_pending_ticket():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    event = recovery_state(1, status="checking")
    event["state"]["scene_validity"]["recovery"] = None
    assert supervisor._accept_event(event)
    assert supervisor.pending_scene_recheck == {"generation": 1, "fault_epoch": 1}
    assert supervisor.scene_invalidated


def test_recovery_acknowledgement_requires_both_ticket_and_state_epoch():
    supervisor = faulted_supervisor()
    supervisor.command({"action": "recheck_scene"})
    supervisor.scene_invalidated_event.clear()
    event = recovery_state(1)
    event["state"]["scene_validity"]["fault_epoch"] = 2
    supervisor._accept_event(event)
    assert supervisor.scene_invalidated and supervisor.pending_scene_recheck is not None


@pytest.mark.parametrize(
    "command",
    [
        {"action": "reset"},
        {"action": "mode", "mode": 2},
        {"action": "demo"},
        {"action": "autonomy", "enabled": True},
        {"action": "goal", "x": 0.5, "y": 0.5},
    ],
)
def test_scene_lock_cannot_be_cleared_by_reset_or_new_motion(command):
    supervisor = fake_supervisor()
    supervisor.scene_invalidated = True
    with pytest.raises(ValueError, match="Reset cannot clear"):
        supervisor.command(command)
    assert not supervisor.restart_requested and supervisor.commands.empty()
    assert supervisor.command({"action": "stop"})["accepted"]
    assert supervisor.command({"action": "heartbeat"})["accepted"]


def test_scene_fault_from_inflight_capture_survives_stop_generation():
    supervisor = fake_supervisor()
    supervisor.command({"action": "stop"})
    assert not supervisor._accept_event(
        {
            "type": "state",
            "state": {"generation": 0, "scene_validity": {"invalidated": True}},
        }
    )
    assert supervisor.scene_invalidated
    with pytest.raises(ValueError, match="Reset cannot clear"):
        supervisor.command({"action": "reset"})


def test_shared_scene_fault_survives_lost_presentation_and_reset_race():
    supervisor = fake_supervisor()
    supervisor.scene_invalidated_event = threading.Event()
    supervisor.closed = threading.Event()
    supervisor.command({"action": "reset"})
    # The worker can set this while shutdown is joining it, after Reset was
    # accepted and with no presentation event ever reaching the supervisor.
    supervisor._shutdown_worker = supervisor.scene_invalidated_event.set
    supervisor._start_worker()
    assert supervisor.scene_invalidated
    assert not supervisor.restart_requested
    assert supervisor.state["phase"] == "scene_invalid"
    assert not supervisor.state["localization_valid"]
    with pytest.raises(ValueError, match="Reset cannot clear"):
        supervisor.command({"action": "reset"})


def test_invalidation_event_is_latched_even_while_reset_is_pending():
    supervisor = fake_supervisor()
    supervisor.command({"action": "reset"})
    assert not supervisor._accept_event(
        {
            "type": "state",
            "state": {"generation": 0, "scene_validity": {"invalidated": True}},
        }
    )
    assert supervisor.scene_invalidated


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


@pytest.mark.parametrize("status", ["lost", "reacquiring", "uninitialized"])
def test_localization_loss_rejects_goals_but_keeps_reset_available(status):
    supervisor = fake_supervisor()
    supervisor.state.update(
        localization_status=status,
        localization_valid=False,
        last_seen_pose=[0.1, 0.2],
        raw_position_radius=7.19,
        raw_pose=[0.1, 0.2],
        raw_velocity=[0.0, 0.0],
        raw_velocity_radius=0.03,
        raw_prediction_time=240.0,
        reacquisition_samples=2,
    )
    with pytest.raises(ValueError, match="Localization lost"):
        supervisor.command({"action": "goal", "x": 0.5, "y": -0.5})
    assert supervisor.commands.empty()
    supervisor.command({"action": "reset"})
    assert supervisor.restart_requested
    assert supervisor.state["localization_status"] == "uninitialized"
    assert supervisor.state["last_seen_pose"] is None
    assert supervisor.state["raw_position_radius"] is None
    for name in (
        "raw_pose",
        "raw_velocity",
        "raw_velocity_radius",
        "raw_prediction_time",
    ):
        assert supervisor.state[name] is None
    assert supervisor.state["reacquisition_samples"] == 0


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


def test_shutdown_kills_worker_that_ignores_term_before_replacement():
    supervisor = fake_supervisor()
    supervisor.commands, supervisor.events = ClosingQueue(), ClosingQueue()
    calls = []

    class StubbornProcess(FakeProcess):
        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            self.started = False

    process = StubbornProcess()
    process.started = True
    supervisor.process = process
    supervisor._shutdown_worker()
    assert calls == ["terminate", "kill"]
    assert not process.is_alive()
    assert supervisor.process is None


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


def test_new_worker_has_no_acknowledged_epoch_from_previous_reference(tmp_path):
    supervisor = fake_supervisor()
    supervisor.closed = threading.Event()
    supervisor.asset_dir = tmp_path / "assets"
    supervisor.run_dir = tmp_path / "runs"
    supervisor.worker_target = lambda *args: None
    supervisor.process = None
    supervisor.events = None
    supervisor.scene_fault_epoch = supervisor.scene_recovered_epoch = 3
    supervisor.state["scene_validity"] = {
        "fault_epoch": 3,
        "invalidated": False,
        "navigation_allowed": True,
    }
    supervisor.context = SimpleNamespace(
        Queue=ClosingQueue,
        Event=threading.Event,
        Process=lambda **kwargs: FakeProcess(),
    )
    supervisor._start_worker()
    assert supervisor.scene_fault_epoch == supervisor.scene_recovered_epoch == 0
    assert supervisor.pending_scene_recheck is None
    assert "scene_validity" not in supervisor.state
    # First fault in the replacement worker must not look like an old receipt.
    event = recovery_state(0, epoch=1, status="rejected")
    assert supervisor._accept_event(event) and supervisor.scene_invalidated


def test_stop_without_channel_reports_stopped_and_does_not_schedule_demo():
    supervisor = fake_supervisor()
    supervisor.commands = None
    supervisor.pending_demo = True
    supervisor.command({"action": "stop"})
    assert supervisor.state["phase"] == "stopped"
    assert not supervisor.pending_demo


@pytest.mark.parametrize(
    ("mode", "directory"),
    [("purpose", "agency"), ("coverage", "agency"), ("visual", "visual-agency")],
)
def test_agency_mode_memory_isolation_and_worker_config(
    tmp_path, monkeypatch, mode, directory
):
    from bb8_rl import interactive

    spawned = []
    monkeypatch.setattr(interactive, "validate_assets", lambda path: {})
    monkeypatch.setattr(interactive, "map_payload", lambda *args: sample_map())
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)

    def process_factory(**kwargs):
        spawned.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        interactive.multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(
            Event=threading.Event,
            Queue=ClosingQueue,
            Process=process_factory,
        ),
    )
    supervisor = Supervisor(
        tmp_path / "assets",
        tmp_path / "runs",
        agency_mode=mode,
        worker_target=lambda *args: None,
    )
    expected = interactive.ROOT / "work" / directory / "bb8.sqlite3"
    assert supervisor.agency_memory == expected
    assert not supervisor.state["agency"]["enabled"]
    assert supervisor.snapshot()["agency_mode"] == mode
    if mode == "visual":
        assert (
            supervisor.state["agency"]["selection_policy"] == "learned_station_outcomes"
        )
        assert supervisor.state["agency"]["resource_source"] == "native_scene_rgb"
        assert supervisor.state["agency"]["resource"] is None
    supervisor._start_worker()
    config = spawned[0]["args"][0]
    assert config["agency_mode"] == mode
    assert config["agency_memory"] == str(expected)


def test_visual_memory_override_is_respected_without_opening_database(
    tmp_path, monkeypatch
):
    from bb8_rl import interactive

    memory = tmp_path / "existing.sqlite3"
    memory.write_bytes(b"preserved database bytes")
    monkeypatch.setattr(interactive, "validate_assets", lambda path: {})
    monkeypatch.setattr(interactive, "map_payload", lambda *args: sample_map())
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    supervisor = Supervisor(
        tmp_path / "assets",
        tmp_path / "runs",
        agency_mode="visual",
        agency_memory=memory,
    )
    assert supervisor.agency_memory == memory
    assert memory.read_bytes() == b"preserved database bytes"


@pytest.mark.parametrize("memory_argument", [None, "selected.sqlite3"])
def test_visual_cli_accepts_mode_and_defers_default_memory_selection(
    tmp_path, monkeypatch, memory_argument
):
    from bb8_rl import interactive

    calls = []
    monkeypatch.setattr(
        interactive,
        "Supervisor",
        lambda *args, **kwargs: (
            calls.append(kwargs) or SimpleNamespace(close=lambda: None)
        ),
    )

    def no_server(*args):
        raise OSError("contract test stops before serving")

    monkeypatch.setattr(interactive, "ThreadingHTTPServer", no_server)
    arguments = [
        "--agency-mode",
        "visual",
        "--no-browser",
        "--output",
        str(tmp_path / "run"),
    ]
    if memory_argument:
        arguments.extend(["--agency-memory", memory_argument])
    with pytest.raises(OSError, match="contract test"):
        interactive.main(arguments)
    assert calls[0]["agency_mode"] == "visual"
    assert calls[0]["agency_memory"] == (
        Path(memory_argument) if memory_argument else None
    )
