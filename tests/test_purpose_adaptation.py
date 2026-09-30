"""Bounded synthetic effect changes, including histories shorter than stability.

These tests exercise declared stations and observed resource deltas, not RGB,
navigation, authority, or physical station behavior.
"""

import json
import random

import pytest

from bb8_rl.purpose import INITIAL_PROBES, PurposeEngine, PurposeStore, Station

STATIONS = [Station("a", (0.0, 1.0), "A"), Station("b", (-0.5, 1.0), "B")]


def open_engine(path):
    store = PurposeStore(path, "bounded-change-test")
    engine = PurposeEngine(store)
    engine.select(STATIONS, resource=0.8, pose=(0, 1), now=0)
    return store, engine


def observe(engine, station, event, gain):
    return engine.record_outcome(
        event,
        station,
        before=0.35,
        after=0.35 + gain,
        now=1,
        source="synthetic_outcome_benchmark",
    )


def choose(engine, *, now=2, pose=(0, 1)):
    return engine.select(STATIONS, resource=0.35, pose=pose, now=now)


def establish(engine, useful, successes, rng):
    other = "b" if useful == "a" else "a"
    for index in range(INITIAL_PROBES):
        observe(engine, other, f"initial-failure-{index}", rng.uniform(-0.019, 0.019))
    for index in range(successes):
        observe(engine, useful, f"initial-success-{index}", rng.uniform(0.42, 0.49))
    return other


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("useful", ["a", "b"])
@pytest.mark.parametrize("successes", [2, 3, 5, 6, 8, 24])
def test_early_and_late_reversals_recover_with_seeded_telemetry(
    tmp_path, seed, useful, successes
):
    rng = random.Random(seed)
    store, engine = open_engine(tmp_path / "outcomes.sqlite")
    other = establish(engine, useful, successes, rng)
    pose = (rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.2))
    assert choose(engine, pose=pose)["candidate_id"] == useful
    choices = []
    for index in range(9):
        choice = choose(engine, pose=pose)
        assert choice is not None
        station = choice["candidate_id"]
        choices.append(station)
        gain = (
            rng.uniform(0.42, 0.49) if station == other else rng.uniform(-0.019, 0.019)
        )
        event = f"reversal-{index}"
        assert observe(engine, station, event, gain)
        before_replay = engine.snapshot()
        assert not observe(engine, station, event, gain)
        assert engine.snapshot() == before_replay
    assert other in choices[:5]
    assert choices[-4:] == [other] * 4
    assert engine.snapshot()["change_epoch"] == 1
    assert engine.snapshot()["episodes"] == INITIAL_PROBES + successes + len(choices)
    store.close()


@pytest.mark.parametrize("restart_after_failures", [0, 1, 2, 3, 4])
def test_restart_across_early_change_and_replay_retains_single_reopening(
    tmp_path, restart_after_failures
):
    path = tmp_path / "restart.sqlite"
    store, engine = open_engine(path)
    establish(engine, "b", 2, random.Random(10))
    for index in range(restart_after_failures):
        observe(engine, "b", f"failed-{index}", 0)
    before = engine.snapshot()
    store.close()
    store, engine = open_engine(path)
    assert engine.snapshot() == before
    for index in range(restart_after_failures):
        assert not observe(engine, "b", f"failed-{index}", 0)
    assert engine.snapshot() == before
    for index in range(restart_after_failures, 4):
        observe(engine, "b", f"failed-{index}", 0)
    assert engine.snapshot()["change_epoch"] == 1
    assert choose(engine)["candidate_id"] == "a"
    observe(engine, "a", "failed-reopened-alternative", 0)
    assert choose(engine) is None
    settled = engine.snapshot()
    store.close()
    store, engine = open_engine(path)
    assert engine.snapshot() == settled
    for now in (10, 1000, 1e9):
        assert choose(engine, now=now) is None
    assert engine.snapshot() == settled
    store.close()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("successes", [2, 5, 6, 24])
def test_noisy_burst_then_ineffective_world_settles_with_exact_finite_budget(
    tmp_path, seed, successes
):
    rng = random.Random(seed)
    store, engine = open_engine(tmp_path / "burst.sqlite")
    # Whether these initial readings reflect actual utility or measurement errors
    # is unknowable from telemetry. Once meaningful readings stop, curiosity stops.
    establish(engine, "b", successes, rng)
    counts = {"a": 0, "b": 0}
    for index in range(20):
        choice = choose(engine)
        if choice is None:
            break
        station = choice["candidate_id"]
        counts[station] += 1
        gain = rng.uniform(-0.019, 0.019)
        assert observe(engine, station, f"noisy-{index}", gain)
        assert not observe(engine, station, f"noisy-{index}", gain)
    assert counts == {"a": 1, "b": INITIAL_PROBES}
    assert engine.snapshot()["change_epoch"] == 1
    # Additional explicitly observed ineffective interactions shift the retained
    # evidence window. They cannot recycle the same old successful burst.
    for index in range(30):
        observe(
            engine, "b", f"later-explicit-failure-{index}", rng.uniform(-0.019, 0.019)
        )
        assert choose(engine, now=100 + index) is None
    assert engine.snapshot()["change_epoch"] == 1
    store.close()


@pytest.mark.parametrize("seed", range(10))
def test_static_ineffective_noise_uses_only_initial_budget(tmp_path, seed):
    rng = random.Random(seed)
    store, engine = open_engine(tmp_path / "noise.sqlite")
    counts = {"a": 0, "b": 0}
    for index in range(20):
        choice = choose(engine)
        if choice is None:
            break
        station = choice["candidate_id"]
        counts[station] += 1
        observe(engine, station, f"noise-{index}", rng.uniform(-0.039, 0.039))
    assert counts == {"a": INITIAL_PROBES, "b": INITIAL_PROBES}
    assert choose(engine, now=1e9) is None
    assert engine.snapshot()["change_epoch"] == 0
    store.close()


def test_one_isolated_meaningful_reading_does_not_reopen_alternatives(tmp_path):
    store, engine = open_engine(tmp_path / "isolated.sqlite")
    establish(engine, "b", 1, random.Random(0))
    for index in range(4):
        assert choose(engine)["candidate_id"] == "b"
        observe(engine, "b", f"failed-{index}", 0)
    assert choose(engine) is None
    assert engine.snapshot()["change_epoch"] == 0
    store.close()


def test_new_corroborating_evidence_can_support_more_than_one_reversal(tmp_path):
    store, engine = open_engine(tmp_path / "repeated.sqlite")
    useful = "a"
    establish(engine, useful, 2, random.Random(0))
    for phase in range(4):
        useful = "b" if useful == "a" else "a"
        choices = []
        for index in range(6):
            choice = choose(engine)
            assert choice is not None
            station = choice["candidate_id"]
            choices.append(station)
            observe(
                engine,
                station,
                f"phase-{phase}-{index}",
                0.45 if station == useful else 0,
            )
        assert choices[-2:] == [useful, useful]
        assert engine.snapshot()["change_epoch"] == phase + 1
    store.close()


def test_stable_history_survives_intermittent_failures_before_change(tmp_path):
    store, engine = open_engine(tmp_path / "intermittent.sqlite")
    establish(engine, "a", 6, random.Random(0))
    for index in range(20):
        observe(engine, "a", f"intermittent-{index}", 0.45 if index % 2 else 0)
    assert engine.snapshot()["change_epoch"] == 0
    for index in range(4):
        observe(engine, "a", f"changed-{index}", 0)
    assert choose(engine)["candidate_id"] == "b"
    assert engine.snapshot()["change_epoch"] == 1
    store.close()


def legacy_stuck_memory(path, *, epoch=0):
    store, engine = open_engine(path)
    establish(engine, "b", 2, random.Random(0))
    for index in range(4):
        observe(engine, "b", f"legacy-failed-{index}", 0)
    # Reproduce the old detector's metadata with the same complete receipt log:
    # it never reopened A because B had fewer than six consecutive successes.
    with store.db:
        store.db.execute("UPDATE purpose_scopes SET change_epoch=?", (epoch,))
        store.db.execute(
            "UPDATE purpose_stations SET probes_remaining=0,last_change_epoch=0 "
            "WHERE id='a'"
        )
    return store, engine


def test_legacy_stuck_history_recovers_only_on_unsatisfied_explicit_selection(tmp_path):
    path = tmp_path / "legacy.sqlite"
    store, engine = legacy_stuck_memory(path)
    before = engine.snapshot()
    receipts = [tuple(row) for row in store.db.execute("SELECT * FROM purpose_events")]
    store.close()
    store, engine = open_engine(path)
    assert engine.snapshot() == before
    assert not engine.reconsider_outcomes(resource=0.8)
    assert not engine.reconsider_outcomes(resource=1.0)
    assert engine.snapshot() == before
    choice = choose(engine)
    assert choice["candidate_id"] == "a"
    after = engine.snapshot()
    assert after["change_epoch"] == 1
    assert after["episodes"] == before["episodes"] == 10
    assert [
        tuple(row) for row in store.db.execute("SELECT * FROM purpose_events")
    ] == receipts
    for row in store.db.execute("SELECT id,payload FROM purpose_events").fetchall():
        assert not engine.record_outcome(row["id"], **json.loads(row["payload"]))
    assert engine.snapshot() == after
    assert not engine.reconsider_outcomes(resource=0.35)
    assert engine.snapshot() == after
    # The single renewed probe fails. Re-enabling selection or reopening memory
    # cannot regenerate a probe from those same historical successful receipts.
    observe(engine, "a", "legacy-reopened-failed", 0)
    settled = engine.snapshot()
    for _ in range(3):
        store.close()
        store, engine = open_engine(path)
        assert engine.snapshot() == settled
        assert not engine.reconsider_outcomes(resource=0.35)
        assert choose(engine, now=1e9) is None
        assert engine.snapshot() == settled
    store.close()


def test_legacy_catch_up_does_not_guess_about_previously_consumed_epochs(tmp_path):
    store, engine = legacy_stuck_memory(tmp_path / "ambiguous.sqlite", epoch=1)
    before = engine.snapshot()
    assert not engine.reconsider_outcomes(resource=0.35)
    assert choose(engine) is None
    assert engine.snapshot() == before
    store.close()
