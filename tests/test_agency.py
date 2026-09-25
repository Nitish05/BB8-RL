import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from bb8_rl.agency import AgencyEngine, AgencyStore, Candidate

CANDIDATES = [
    Candidate("a", (-1.0, 0.0), "West activity"),
    Candidate("b", (1.0, 0.0), "East activity"),
]


def registered(tmp_path, *, synthetic=False, strategy="learned"):
    store = AgencyStore(
        tmp_path / "memory.sqlite",
        map_version="estimated-map-1",
        allow_synthetic=synthetic,
    )
    engine = AgencyEngine(store, strategy=strategy)
    engine.select(CANDIDATES, now=0, pose=(0, 0))
    return store, engine


def test_navigation_evidence_and_restart_are_durable(tmp_path):
    store, engine = registered(tmp_path)
    assert engine.record_outcome(
        event_id="trip-1", candidate_id="a", outcome="arrived", now=1
    )
    assert engine.record_outcome(
        event_id="trip-2", candidate_id="b", outcome="rejected", now=2
    )
    before = engine.snapshot()
    assert before["preferences"][0]["candidate_id"] == "a"
    assert before["preferences"][0]["competence"] == 1
    assert before["evidence"] == "navigation outcomes"
    store.close()
    reopened = AgencyStore(tmp_path / "memory.sqlite", map_version="estimated-map-1")
    assert AgencyEngine(reopened).snapshot() == before
    reopened.close()


def test_replayed_event_is_idempotent_but_conflicting_payload_rejected(tmp_path):
    store, engine = registered(tmp_path)
    payload = {
        "event_id": "trip-1",
        "candidate_id": "a",
        "outcome": "arrived",
        "now": 1.0,
    }
    assert engine.record_outcome(**payload)
    before = engine.snapshot()
    for _ in range(50):
        assert not engine.record_outcome(**payload)
    assert engine.snapshot() == before
    for field, value in [("candidate_id", "b"), ("outcome", "rejected"), ("now", 2)]:
        with pytest.raises(ValueError, match="conflicting payload"):
            engine.record_outcome(**{**payload, field: value})
        assert engine.snapshot() == before
    store.close()


def test_concurrent_delivery_applies_one_event_once(tmp_path):
    store, engine = registered(tmp_path)

    def deliver(_):
        return engine.record_outcome(
            event_id="same-event", candidate_id="a", outcome="arrived", now=1
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        accepted = list(executor.map(deliver, range(64)))
    assert sum(accepted) == 1
    assert engine.snapshot()["episodes"] == 1
    assert engine.snapshot()["preferences"][0]["visits"] == 1
    store.close()


def test_proposals_and_interruptions_are_not_successes_or_dislikes(tmp_path):
    store, engine = registered(tmp_path)
    for now in range(1, 10):
        engine.select(CANDIDATES, now=now, pose=(0, 0))
    assert engine.snapshot()["episodes"] == 0
    for i, reason in enumerate(("cancelled", "localization_lost")):
        engine.record_outcome(
            event_id=str(i), candidate_id="a", outcome=reason, now=10 + i
        )
    snapshot = engine.snapshot()
    assert snapshot["episodes"] == 2
    assert all(
        p["value"] == 0 and p["visits"] == 0 and p["competence"] is None
        for p in snapshot["preferences"]
    )
    store.close()


def test_map_and_agent_identity_isolate_memory(tmp_path):
    store, engine = registered(tmp_path)
    engine.record_outcome(event_id="1", candidate_id="a", outcome="arrived", now=1)
    other_map = AgencyStore(tmp_path / "memory.sqlite", map_version="estimated-map-2")
    other_agent = AgencyStore(
        tmp_path / "memory.sqlite", agent_id="bb9", map_version="estimated-map-1"
    )
    assert AgencyEngine(other_map).snapshot()["preferences"] == []
    assert AgencyEngine(other_agent).snapshot()["episodes"] == 0
    with pytest.raises(ValueError, match="not registered"):
        AgencyEngine(other_map).record_outcome(
            event_id="1", candidate_id="a", outcome="arrived", now=1
        )
    for memory in (store, other_map, other_agent):
        memory.close()


def test_live_store_cannot_receive_synthetic_rewards_or_truth(tmp_path):
    store, engine = registered(tmp_path)
    with pytest.raises(ValueError, match="explicitly synthetic"):
        engine.record_experience(event_id="fake", candidate_id="a", utility=1, now=1)
    with pytest.raises(TypeError):
        engine.record_outcome(
            event_id="fake", candidate_id="a", outcome="arrived", now=1, truth=True
        )
    with pytest.raises(ValueError, match="cannot share"):
        AgencyStore(
            tmp_path / "memory.sqlite",
            map_version="estimated-map-1",
            allow_synthetic=True,
        )
    assert engine.snapshot()["episodes"] == 0
    store.close()


def test_synthetic_store_cannot_be_reopened_live(tmp_path):
    store, _ = registered(tmp_path, synthetic=True)
    store.close()
    with pytest.raises(ValueError, match="cannot share"):
        AgencyStore(tmp_path / "memory.sqlite", map_version="estimated-map-1")


def test_history_changes_choices_and_reversed_outcomes_change_them_again(tmp_path):
    store, engine = registered(tmp_path, synthetic=True)
    for i in range(25):
        for candidate, utility in [("a", 0.8), ("b", -0.8)]:
            engine.record_experience(
                event_id=f"first-{candidate}-{i}",
                candidate_id=candidate,
                utility=utility,
                now=i,
            )
    first = [
        engine.select(CANDIDATES, now=30 + i, pose=(0, 0))["candidate_id"]
        for i in range(100)
    ]
    assert first.count("a") >= 85
    for i in range(25):
        for candidate, utility in [("a", -0.8), ("b", 0.8)]:
            engine.record_experience(
                event_id=f"reversed-{candidate}-{i}",
                candidate_id=candidate,
                utility=utility,
                now=200 + i,
            )
    second = [
        engine.select(CANDIDATES, now=250 + i, pose=(0, 0))["candidate_id"]
        for i in range(100)
    ]
    assert second.count("b") >= 85
    store.close()


def test_restart_preserves_future_exploration_sequence(tmp_path):
    store, engine = registered(tmp_path, synthetic=True)
    for i in range(8):
        engine.record_experience(event_id=str(i), candidate_id="a", utility=0.5, now=i)
    # Back up at the exact same counter to compare uninterrupted vs reopened state.
    clone = AgencyStore(
        tmp_path / "clone.sqlite", map_version="estimated-map-1", allow_synthetic=True
    )
    store.db.backup(clone.db)
    store.close()
    reopened = AgencyStore(
        tmp_path / "memory.sqlite", map_version="estimated-map-1", allow_synthetic=True
    )
    restored = AgencyEngine(reopened)
    uninterrupted = AgencyEngine(clone)
    for i in range(30):
        assert restored.select(
            CANDIDATES, now=50 + i, pose=(0, 0)
        ) == uninterrupted.select(CANDIDATES, now=50 + i, pose=(0, 0))
    reopened.close()
    clone.close()


def test_memory_baseline_uses_visits_without_outcome_valence(tmp_path):
    store, engine = registered(tmp_path, synthetic=True, strategy="memory")
    for i in range(5):
        engine.record_experience(event_id=str(i), candidate_id="a", utility=1, now=i)
    assert engine.select(CANDIDATES, now=6, pose=(0, 0))["candidate_id"] == "b"
    store.close()


def test_fixed_baseline_ignores_experience(tmp_path):
    store, engine = registered(tmp_path, synthetic=True, strategy="fixed")
    initial = engine.select(CANDIDATES, now=1, pose=(0, 0))["candidate_id"]
    for i in range(30):
        engine.record_experience(
            event_id=str(i), candidate_id=initial, utility=-1, now=2 + i
        )
    assert engine.select(CANDIDATES, now=32, pose=(0, 0))["candidate_id"] == initial
    store.close()


def test_changed_anchor_requires_new_map_and_registration_rolls_back(tmp_path):
    store, engine = registered(tmp_path)
    with pytest.raises(ValueError, match="coordinates changed"):
        engine.select(
            [Candidate("new", (2, 2)), Candidate("a", (3, 0))], now=1, pose=(0, 0)
        )
    assert {p["candidate_id"] for p in engine.snapshot()["preferences"]} == {"a", "b"}
    store.close()


def test_no_candidates_or_already_at_only_candidate_produces_no_goal(tmp_path):
    store, engine = registered(tmp_path)
    before = engine.snapshot()["decisions"]
    assert engine.select([], now=1, pose=(0, 0)) is None
    assert engine.select([CANDIDATES[0]], now=2, pose=(-1, 0)) is None
    assert engine.snapshot()["decisions"] == before
    store.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "1"])
def test_invalid_events_cannot_poison_memory(tmp_path, bad):
    store, engine = registered(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        engine.record_outcome(
            event_id="bad", candidate_id="a", outcome="arrived", now=bad
        )
    assert engine.snapshot()["episodes"] == 0
    json.dumps(engine.snapshot(), allow_nan=False)
    store.close()


def test_candidates_validate_coordinates_and_ids():
    for kwargs in [
        {"id": "", "xy": (0, 0)},
        {"id": "x", "xy": (0, math.inf)},
        {"id": "x", "xy": (1,)},
    ]:
        with pytest.raises(ValueError):
            Candidate(**kwargs)
