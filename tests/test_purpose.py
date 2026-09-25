"""Behavioral checks for learned simulated interaction outcomes, not personality."""

import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from bb8_rl.agency import AgencyEngine, AgencyStore, Candidate
from bb8_rl.purpose import (
    EVIDENCE_WINDOW,
    INITIAL_PROBES,
    MAX_INFORMATION_VALUE,
    PurposeEngine,
    PurposeStore,
    Station,
)

STATIONS = [
    Station("a", (0.5, 1.5), "Station A"),
    Station("b", (-0.5, 1.5), "Station B"),
]


def registered(tmp_path, name="memory", **kwargs):
    store = PurposeStore(tmp_path / f"{name}.sqlite3", "estimated-map", **kwargs)
    engine = PurposeEngine(store)
    engine.select(STATIONS, resource=0.8, pose=(0, 1.5), now=0)
    return store, engine


def observe(engine, station, event, gain=0.45, source="simulated_station_telemetry"):
    return engine.record_outcome(
        str(event), station, before=0.35, after=0.35 + gain, now=1, source=source
    )


def select(engine, **kwargs):
    return engine.select(
        STATIONS, **{"resource": 0.35, "pose": (0, 1.5), "now": 2, **kwargs}
    )


def preferences(engine):
    return {row["candidate_id"]: row for row in engine.snapshot()["preferences"]}


def test_registration_proposals_and_arrival_only_give_no_learning(tmp_path):
    store, engine = registered(tmp_path)
    baseline = preferences(engine)
    for _ in range(20):
        assert select(engine)
    snapshot = engine.snapshot()
    assert snapshot["episodes"] == 0
    assert snapshot["decisions"] == 20
    for row in snapshot["preferences"]:
        assert row["outcomes"] == 0
        assert (
            row["response_probability"]
            == baseline[row["candidate_id"]]["response_probability"]
        )
        assert row["value"] == baseline[row["candidate_id"]]["value"]
    with pytest.raises(TypeError):
        engine.record_outcome("arrival", "a", outcome="arrived", now=2)
    store.close()


@pytest.mark.parametrize("resource", [0.8, 0.9, 1.0])
def test_need_satisfaction_selects_idle_even_with_uncertain_alternatives(
    tmp_path, resource
):
    store, engine = registered(tmp_path)
    observe(engine, "a", "useful")
    before = engine.snapshot()
    assert select(engine, resource=resource) is None
    assert engine.snapshot() == before
    store.close()


def test_same_pose_and_need_opposite_histories_produce_different_choices(tmp_path):
    stores = []
    for useful in ("a", "b"):
        store, engine = registered(tmp_path, useful)
        stores.append(store)
        for trial in range(6):
            observe(engine, useful, f"helpful-{trial}")
            observe(engine, "b" if useful == "a" else "a", f"ineffective-{trial}", 0)
        choice = select(engine)
        assert choice["candidate_id"] == useful
        assert choice["score_terms"]["outcomes"] == 6
        assert choice["score_terms"]["response_probability"] > 0.8
        assert "simulated resource" in choice["question"]
    for store in stores:
        store.close()


def test_gain_magnitude_affects_choice_at_equal_success_probability(tmp_path):
    store, engine = registered(tmp_path)
    for trial in range(6):
        observe(engine, "a", f"a-{trial}", 0.08)
        observe(engine, "b", f"b-{trial}", 0.45)
    assert select(engine)["candidate_id"] == "b"
    assert (
        preferences(engine)["a"]["response_probability"]
        == preferences(engine)["b"]["response_probability"]
    )
    store.close()


def test_cost_can_make_idle_better_than_small_remaining_need(tmp_path):
    store, engine = registered(tmp_path)
    for trial in range(5):
        observe(engine, "a", f"a-{trial}")
    assert select(engine, resource=0.795) is None
    assert select(engine, resource=0.35)["candidate_id"] == "a"
    store.close()


@pytest.mark.parametrize("resource", [0.75, 0.7575, 0.77])
def test_useful_local_station_can_top_up_resource_after_travel(tmp_path, resource):
    store, engine = registered(tmp_path)
    observe(engine, "a", "ineffective", 0)
    observe(engine, "b", "useful", 0.45)
    choice = select(engine, resource=resource, pose=STATIONS[1].xy)
    assert choice["candidate_id"] == "b"
    store.close()


def test_useless_and_small_noisy_telemetry_have_finite_probe_allowance(tmp_path):
    store, engine = registered(tmp_path)
    choices = []
    while (choice := select(engine)) is not None:
        station = choice["candidate_id"]
        choices.append(station)
        # Jitter can change sign but never restore a meaningful amount of resource.
        gain = 0.02 if len(choices) % 2 else -0.02
        observe(engine, station, len(choices), gain)
        assert len(choices) <= INITIAL_PROBES * len(STATIONS)
    assert len(choices) == INITIAL_PROBES * len(STATIONS)
    snapshot = engine.snapshot()
    assert all(row["suppressed"] for row in snapshot["preferences"])
    assert all(row["responses"] == 0 for row in snapshot["preferences"])
    assert snapshot["change_epoch"] == 0
    for now in (1e4, 1e9):
        assert select(engine, now=now) is None
    store.close()


def test_unreliable_large_responses_do_not_trigger_change_reprobe_chasing(tmp_path):
    store, engine = registered(tmp_path)
    for i in range(4):
        observe(engine, "b", f"b-{i}", 0)
    # Alternating actual resource restoration is sometimes useful but never stable.
    for i in range(20):
        observe(engine, "a", f"a-{i}", 0.45 if i % 2 == 0 else 0)
    for i in range(4):
        observe(engine, "a", f"end-{i}", 0)
    assert engine.snapshot()["change_epoch"] == 0
    assert select(engine) is None
    store.close()


def test_stable_outcome_reversal_reopens_suppressed_alternative_once(tmp_path):
    store, engine = registered(tmp_path)
    for i in range(4):
        observe(engine, "b", f"old-b-{i}", 0)
    for i in range(8):
        observe(engine, "a", f"old-a-{i}")
    assert select(engine)["candidate_id"] == "a"
    assert preferences(engine)["b"]["suppressed"]
    trials = []
    for step in range(8):
        choice = select(engine)
        assert choice is not None
        station = choice["candidate_id"]
        trials.append(station)
        observe(engine, station, f"reversal-{step}", 0.45 if station == "b" else 0)
    assert "b" in trials[:5]
    assert trials[-4:] == ["b"] * 4
    assert engine.snapshot()["change_epoch"] == 1
    assert preferences(engine)["b"]["last_change_epoch"] == 1
    store.close()


def test_failed_reopened_probe_stops_instead_of_starting_endless_laps(tmp_path):
    store, engine = registered(tmp_path)
    for i in range(4):
        observe(engine, "b", f"b-{i}", 0)
    for i in range(6):
        observe(engine, "a", f"a-{i}")
    after_change = []
    while (choice := select(engine)) is not None:
        after_change.append(choice["candidate_id"])
        observe(engine, choice["candidate_id"], f"changed-{len(after_change)}", 0)
        assert len(after_change) <= 5
    assert after_change.count("a") == 4
    assert after_change.count("b") == 1
    assert engine.snapshot()["change_epoch"] == 1
    store.close()


def test_recent_evidence_forgets_stale_response_probability_and_gain(tmp_path):
    store, engine = registered(tmp_path)
    for i in range(40):
        observe(engine, "a", f"old-{i}", 0.45)
    for i in range(EVIDENCE_WINDOW):
        observe(engine, "a", f"new-{i}", 0.08)
    row = preferences(engine)["a"]
    assert row["observations"] == 40 + EVIDENCE_WINDOW
    assert row["effective_observations"] == EVIDENCE_WINDOW
    assert row["conditional_gain"] == pytest.approx(0.08)
    for i in range(EVIDENCE_WINDOW):
        observe(engine, "a", f"failed-{i}", 0)
    row = preferences(engine)["a"]
    assert row["response_probability"] == pytest.approx(1 / (EVIDENCE_WINDOW + 2))
    store.close()


def test_information_value_is_bounded_and_expected_gain_is_need_limited(tmp_path):
    store, engine = registered(tmp_path)
    choice = select(engine)
    terms = choice["score_terms"]
    assert 0 <= terms["information_value"] <= MAX_INFORMATION_VALUE
    assert 0 <= terms["predicted_deficit_reduction"] <= terms["deficit"]
    assert terms["total"] == pytest.approx(
        terms["predicted_deficit_reduction"]
        + terms["information_value"]
        - terms["interaction_cost"]
        - terms["distance_cost"]
    )
    assert terms["idle_value"] == 0
    store.close()


def test_durable_model_event_replay_and_conflicting_payload(tmp_path):
    store, engine = registered(tmp_path)
    assert observe(engine, "b", "event")
    before = engine.snapshot()
    assert not observe(engine, "b", "event")
    assert engine.snapshot() == before
    with pytest.raises(ValueError, match="conflicting payload"):
        observe(engine, "a", "event")
    with pytest.raises(ValueError, match="conflicting payload"):
        observe(engine, "b", "event", 0)
    assert engine.snapshot() == before
    store.close()
    reopened = PurposeStore(tmp_path / "memory.sqlite3", "estimated-map")
    assert PurposeEngine(reopened).snapshot() == before
    assert select(PurposeEngine(reopened))["candidate_id"] == "b"
    reopened.close()


def test_event_and_model_transaction_roll_back_together(tmp_path):
    store, engine = registered(tmp_path)
    before = engine.snapshot()
    store.db.execute("""
        CREATE TRIGGER fail_purpose_model BEFORE UPDATE ON purpose_stations
        BEGIN SELECT RAISE(ABORT, 'injected failure'); END
    """)
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        observe(engine, "a", "failure")
    assert engine.snapshot() == before
    assert store.db.execute("SELECT COUNT(*) FROM purpose_events").fetchone()[0] == 0
    store.db.execute("DROP TRIGGER fail_purpose_model")
    assert observe(engine, "a", "failure")
    store.close()


def test_concurrent_replays_are_applied_once(tmp_path):
    store, engine = registered(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: observe(engine, "a", "same"), range(32)))
    assert sum(results) == 1
    assert engine.snapshot()["episodes"] == 1
    store.close()


def test_separately_opened_connections_serialize_duplicate_delivery(tmp_path):
    store, engine = registered(tmp_path)
    another = PurposeStore(tmp_path / "memory.sqlite3", "estimated-map")
    other_engine = PurposeEngine(another)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda e: observe(e, "a", "same"), [engine, other_engine])
        )
    assert sum(results) == 1
    assert engine.snapshot() == other_engine.snapshot()
    assert engine.snapshot()["episodes"] == 1
    store.close()
    another.close()


def test_map_agent_and_old_arrival_learner_are_isolated(tmp_path):
    path = tmp_path / "shared.sqlite3"
    old_store = AgencyStore(path, map_version="estimated-map")
    old_engine = AgencyEngine(old_store)
    old_engine.select([Candidate("a", (0.5, 1.5), "Old point")], now=0, pose=(0, 0))
    old_engine.record_outcome(
        event_id="old-arrival", candidate_id="a", outcome="arrived", now=1
    )
    old_snapshot = old_engine.snapshot()
    store = PurposeStore(path, "estimated-map")
    engine = PurposeEngine(store)
    engine.select(STATIONS, resource=0.8, pose=(0, 0), now=0)
    assert engine.snapshot()["episodes"] == 0
    observe(engine, "a", "new-outcome")
    assert old_engine.snapshot() == old_snapshot
    for map_version, agent in (
        ("another-map", "bb8"),
        ("estimated-map", "other-agent"),
    ):
        other = PurposeStore(path, map_version, agent)
        assert PurposeEngine(other).snapshot()["episodes"] == 0
        assert PurposeEngine(other).snapshot()["preferences"] == []
        other.close()
    store.close()
    old_store.close()


def test_simulated_telemetry_and_benchmark_evidence_cannot_mix(tmp_path):
    store, engine = registered(tmp_path)
    observe(engine, "a", "native")
    before = engine.snapshot()
    with pytest.raises(ValueError, match="cannot share"):
        observe(engine, "a", "synthetic", source="synthetic_outcome_benchmark")
    assert engine.snapshot() == before
    for source in ("arrival", "ground_truth", "rgb", "", None):
        with pytest.raises(ValueError, match="approved explicit"):
            observe(engine, "a", "invalid", source=source)
    store.close()


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, True, "0.5"])
def test_invalid_resource_evidence_does_not_change_memory(tmp_path, value):
    store, engine = registered(tmp_path)
    before = engine.snapshot()
    for field in ("before", "after"):
        with pytest.raises((TypeError, ValueError)):
            engine.record_outcome(
                "bad", "a", now=1, **{"before": 0.35, "after": 0.8, field: value}
            )
    with pytest.raises((TypeError, ValueError)):
        select(engine, resource=value)
    assert engine.snapshot() == before
    store.close()


def test_changed_station_coordinates_and_unknown_outcome_are_rejected(tmp_path):
    store, engine = registered(tmp_path)
    before = engine.snapshot()
    with pytest.raises(ValueError, match="coordinates changed"):
        engine.select([Station("a", (0, 0), "A")], resource=0.35, pose=(0, 0), now=0)
    with pytest.raises(ValueError, match="not registered"):
        observe(engine, "unknown", "bad")
    with pytest.raises(ValueError, match="Duplicate"):
        engine.select([STATIONS[0], STATIONS[0]], resource=0.35, pose=(0, 0), now=0)
    assert engine.snapshot() == before
    store.close()


def test_terminal_no_response_can_be_observed_without_moving_again(tmp_path):
    store, engine = registered(tmp_path)
    # A station interaction is meaningful even when already at its location.
    assert select(engine, pose=STATIONS[0].xy)["candidate_id"] == "a"
    observe(engine, "a", "ineffective", 0)
    assert engine.snapshot()["episodes"] == 1
    assert preferences(engine)["a"]["responses"] == 0
    store.close()
