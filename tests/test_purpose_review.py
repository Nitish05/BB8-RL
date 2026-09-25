"""Independent regressions for evidence boundaries found during integration review."""

import importlib.util
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from bb8_rl.interaction_environment import DEFAULT_STATIONS, SOURCE
from bb8_rl.interactive import Supervisor
from bb8_rl.purpose import PurposeEngine, PurposeStore
from bb8_rl.purpose_runtime import PurposeCoordinator


class ArrivedSession:
    localization_lost = False

    def __init__(self):
        self.cancelled = False

    def state(self):
        return {
            "phase": "arrived",
            "localization_valid": True,
            "localization_status": "measured",
        }

    def cancel(self, *_args):
        self.cancelled = True


@pytest.fixture
def interacting(tmp_path):
    store = PurposeStore(tmp_path / "outcomes.sqlite", "review-map")
    engine = PurposeEngine(store)
    # Register declared stations without selecting motion or manufacturing evidence.
    engine.select(DEFAULT_STATIONS, resource=0.8, pose=(0, 0), now=0)
    coordinator = PurposeCoordinator(engine, DEFAULT_STATIONS)
    coordinator.enable(0)
    coordinator.status = "interacting"
    coordinator.interaction_since = 0.0
    coordinator.intention = {
        "event_id": "receipt-1",
        "candidate_id": "station-b",
        "label": "Station B",
    }
    yield coordinator, ArrivedSession()
    store.close()


def receipt(*, gain=0.45, timestamp=1.0):
    return {
        "request_id": "receipt-1",
        "station_id": "station-b",
        "before": 0.35,
        "after": 0.35 + gain,
        "timestamp": timestamp,
        "source": SOURCE,
    }


def deliver(coordinator, outcome, *, now=1.0):
    coordinator.observe(
        {
            "resource": outcome["after"],
            "timestamp": now,
            "source": SOURCE,
            "outcome": outcome,
        }
    )


def test_freshly_delivered_receipt_still_requires_dwell_at_effect_time(interacting):
    coordinator, session = interacting
    # Delivery itself is after one second; the effect occurred too early.
    deliver(coordinator, receipt(timestamp=0.95), now=1.0)
    assert coordinator.tick(session, 1.0)
    assert session.cancelled and not coordinator.enabled
    assert coordinator.snapshot()["episodes"] == 0


@pytest.mark.parametrize("extra", ["world_pose", "hidden_effect", "velocity"])
def test_nested_simulator_fields_cannot_cross_receipt_boundary(interacting, extra):
    coordinator, _ = interacting
    coordinator.observe({"resource": 0.35, "timestamp": 0.0, "source": SOURCE})
    previous = dict(coordinator.observation)
    with pytest.raises(ValueError, match="receipt fields"):
        deliver(coordinator, {**receipt(), extra: [1, 2]})
    assert coordinator.observation == previous
    assert coordinator.snapshot()["episodes"] == 0


def test_observation_receipt_is_detached_from_provider_owned_dictionary(interacting):
    coordinator, session = interacting
    incoming = receipt()
    deliver(coordinator, incoming)
    incoming.update(after=0.35, request_id="changed", world_pose=[1, 2])
    assert not coordinator.tick(session, 1.0)
    assert coordinator.snapshot()["episodes"] == 1
    assert coordinator.recent_experience["after"] == pytest.approx(0.8)
    assert coordinator.recent_experience["request_id"] == "receipt-1"
    assert "world_pose" not in coordinator.recent_experience


@pytest.mark.parametrize(
    "gain,label,responses",
    [
        (0.0, "no_effect", 0),
        (0.02, "no_meaningful_gain", 0),
        (0.0399, "no_meaningful_gain", 0),
        (0.04, "restored", 1),
        (0.05, "restored", 1),
        (-0.02, "no_meaningful_gain", 0),
    ],
)
def test_visible_outcome_classification_matches_learned_response(
    interacting, gain, label, responses
):
    coordinator, session = interacting
    deliver(coordinator, receipt(gain=gain))
    assert not coordinator.tick(session, 1.0)
    assert coordinator.recent_experience["outcome"] == label
    station = next(
        item
        for item in coordinator.snapshot()["preferences"]
        if item["candidate_id"] == "station-b"
    )
    assert station["responses"] == responses
    assert station["outcomes"] == 1


@pytest.mark.parametrize("mode", ["purpose", "coverage"])
def test_supervisor_forwards_requested_agency_mode_to_worker(tmp_path, mode):
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.lock = threading.RLock()
    supervisor.process = supervisor.commands = supervisor.events = None
    supervisor.worker_stop = None
    supervisor.closed = threading.Event()
    supervisor.worker_target = lambda *_args: None
    supervisor.run_dir = tmp_path / "run"
    supervisor.asset_dir = tmp_path / "assets"
    supervisor.agency_memory = tmp_path / "memory.sqlite"
    supervisor.agency_mode = mode
    supervisor.generation = 7
    supervisor.state = {"mode": 3}
    supervisor.restart_requested = True
    spawned = []

    def spawn(**kwargs):
        spawned.append(kwargs)
        return SimpleNamespace(start=lambda: None)

    supervisor.context = SimpleNamespace(
        Queue=queue.Queue, Event=threading.Event, Process=spawn
    )
    supervisor._start_worker()
    assert len(spawned) == 1
    config = spawned[0]["args"][0]
    assert config["agency_mode"] == mode
    assert config["agency_memory"] == str(supervisor.agency_memory)
    assert config["generation"] == 7 and config["mode"] == 3
    assert supervisor.commands.get_nowait() == {"action": "heartbeat", "generation": 7}


def test_native_effect_audit_rejects_speed_excursion_between_action_endpoints():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/benchmark-purpose-navigation.py"
    )
    spec = importlib.util.spec_from_file_location("purpose_review_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    station = DEFAULT_STATIONS[1]
    samples = [
        {"time": index * 0.005, "position": list(station.xy), "velocity": [0.0, 0.0]}
        for index in range(1, 201)
    ]
    rows = [
        {
            "time": 0.0,
            "agency": {
                "interaction_request": {
                    "request_id": "receipt-1",
                    "station_id": station.id,
                }
            },
            "after_step": {
                "time": 1.0,
                "physics_trace_scoring_only": samples,
                "interaction_telemetry": {
                    "resource": 0.8,
                    "timestamp": 1.0,
                    "source": SOURCE,
                    "outcome": receipt(),
                },
            },
        }
    ]
    assert audit.score_effects(rows)["passed"]
    # Endpoints and sampled 50 ms boundaries remain stationary. The full physics
    # trace must still invalidate this receipt for its short intervening motion.
    samples[96]["velocity"] = [0.031, 0.0]
    score = audit.score_effects(rows)
    assert not score["passed"]
    assert not score["effects"][0]["physics_dwell"]
    assert "invalid_effect:receipt-1" in score["errors"]
