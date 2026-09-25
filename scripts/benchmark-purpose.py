#!/usr/bin/env python3
"""Synthetic outcome-learning comparison; no physics, RGB or personality evaluation.

Run with scripts/python.sh. Each strategy sees the same declared resource target,
station locations, interaction budget and observed before/after values. Only the
learned strategy uses previous outcomes to choose the next interaction. Hidden
station efficacy is used by this evaluator only, after a choice has been made.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from bb8_rl.purpose import (
    DISTANCE_COST,
    EVIDENCE_WINDOW,
    INITIAL_PROBES,
    INTERACTION_COST,
    MAX_INFORMATION_VALUE,
    MEANINGFUL_GAIN,
    PRIOR_GAIN,
    RESOURCE_TARGET,
    PurposeEngine,
    PurposeStore,
    Station,
)

STATIONS = (
    Station("station-a", (0.5, 1.5), "Station A"),
    Station("station-b", (-0.5, 1.5), "Station B"),
)
STRATEGIES = ("learned_outcomes", "random", "no_memory_nearest")
SOURCE = "synthetic_outcome_benchmark"


def run_case(
    path: Path, strategy: str, seed: int, initially_useful: str, episodes: int
) -> dict:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite outcome evidence: {path}")
    rng = random.Random(seed)
    pose = (rng.uniform(-0.08, 0.08), 1.5)
    store = PurposeStore(path, "synthetic-purpose-benchmark-v1")
    engine = PurposeEngine(store)
    engine.select(STATIONS, resource=RESOURCE_TARGET, pose=pose, now=0)
    events = 0
    phases = {}
    try:
        for phase in ("acquisition", "reversal"):
            useful = initially_useful
            if phase == "reversal":
                useful = "station-b" if initially_useful == "station-a" else "station-a"
            phase_episodes = []
            for episode in range(episodes):
                resource = 0.35
                choices = []
                gains = []
                # The same finite interaction budget applies to all strategies.
                for step in range(8):
                    if strategy == "learned_outcomes":
                        choice = engine.select(
                            STATIONS, resource=resource, pose=pose, now=events
                        )
                        station_id = choice["candidate_id"] if choice else None
                    elif resource >= RESOURCE_TARGET - 1e-9:
                        station_id = None
                    elif strategy == "random":
                        station_id = rng.choice(STATIONS).id
                    else:
                        station_id = min(
                            STATIONS, key=lambda s: math.dist(s.xy, pose)
                        ).id
                    if station_id is None:
                        break
                    before = resource
                    # This outcome function is never included in a policy request.
                    resource = min(
                        1.0, resource + (0.45 if station_id == useful else 0.0)
                    )
                    event_id = f"{phase}-{episode}-{step}"
                    accepted = engine.record_outcome(
                        event_id,
                        station_id,
                        before=before,
                        after=resource,
                        now=events,
                        source=SOURCE,
                    )
                    assert accepted
                    assert not engine.record_outcome(
                        event_id,
                        station_id,
                        before=before,
                        after=resource,
                        now=events,
                        source=SOURCE,
                    )
                    choices.append(station_id)
                    gains.append(resource - before)
                    events += 1
                phase_episodes.append(
                    {
                        "episode": episode,
                        "useful_station": useful,
                        "choices": choices,
                        "observed_gains": gains,
                        "resource": resource,
                        "restored": resource >= RESOURCE_TARGET - 1e-9,
                    }
                )
            last_half = phase_episodes[episodes // 2 :]
            phases[phase] = {
                "episodes": phase_episodes,
                "restoration_rate": statistics.mean(
                    row["restored"] for row in phase_episodes
                ),
                "mean_interactions": statistics.mean(
                    len(row["choices"]) for row in phase_episodes
                ),
                "late_first_choice_useful_rate": statistics.mean(
                    bool(row["choices"]) and row["choices"][0] == useful
                    for row in last_half
                ),
            }
            # Snapshot equality is checked before any new action on the reopened engine.
            before_restart = engine.snapshot()
            store.close()
            store = PurposeStore(path, "synthetic-purpose-benchmark-v1")
            engine = PurposeEngine(store)
            phases[phase]["restart_exact"] = engine.snapshot() == before_restart
        snapshot = engine.snapshot()
        return {
            "strategy": strategy,
            "seed": seed,
            "initially_useful": initially_useful,
            "pose": pose,
            "phases": phases,
            "unique_observations": events,
            "snapshot": snapshot,
            "dedup_exact": snapshot["episodes"] == events,
        }
    finally:
        store.close()


def noise_case(path: Path, seed: int) -> dict:
    rng = random.Random(seed)
    store = PurposeStore(path, "synthetic-noise-v1")
    engine = PurposeEngine(store)
    choices = []
    try:
        for step in range(100):
            choice = engine.select(STATIONS, resource=0.35, pose=(0, 1.5), now=step)
            if choice is None:
                break
            choices.append(choice["candidate_id"])
            # Unpredictable sensor variation has no useful resource consequence.
            engine.record_outcome(
                f"noise-{step}",
                choice["candidate_id"],
                before=0.35,
                after=0.35 + rng.uniform(-0.019, 0.019),
                now=step,
                source=SOURCE,
            )
        idle_later = (
            engine.select(STATIONS, resource=0.35, pose=(0, 1.5), now=1e9) is None
        )
        return {
            "seed": seed,
            "interactions": len(choices),
            "finite": len(choices) <= INITIAL_PROBES * len(STATIONS),
            "remains_idle": idle_later,
            "snapshot": engine.snapshot(),
        }
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=24)
    args = parser.parse_args()
    if args.seeds < 1 or args.episodes < 12:
        parser.error("--seeds must be positive and --episodes at least 12")
    output = args.output or Path("work/purpose") / datetime.now(timezone.utc).strftime(
        "outcomes-%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case(
            output / f"{strategy}-{seed}-{useful}.sqlite3",
            strategy,
            seed,
            useful,
            args.episodes,
        )
        for strategy in STRATEGIES
        for seed in range(100, 100 + args.seeds)
        for useful in ("station-a", "station-b")
    ]
    noise = [
        noise_case(output / f"noise-{seed}.sqlite3", seed)
        for seed in range(100, 100 + args.seeds)
    ]
    summary = {}
    for strategy in STRATEGIES:
        group = [case for case in cases if case["strategy"] == strategy]
        summary[strategy] = {
            phase: {
                metric: statistics.mean(case["phases"][phase][metric] for case in group)
                for metric in (
                    "restoration_rate",
                    "mean_interactions",
                    "late_first_choice_useful_rate",
                )
            }
            for phase in ("acquisition", "reversal")
        }
    learned = summary["learned_outcomes"]
    checks = {
        "all_outcomes_deduplicated": all(case["dedup_exact"] for case in cases),
        "all_restarts_exact": all(
            case["phases"][phase]["restart_exact"]
            for case in cases
            for phase in ("acquisition", "reversal")
        ),
        "learned_restores_in_all_synthetic_episodes": all(
            learned[phase]["restoration_rate"] == 1
            for phase in ("acquisition", "reversal")
        ),
        "learned_late_choices_follow_reversed_outcomes": all(
            learned[phase]["late_first_choice_useful_rate"] == 1
            for phase in ("acquisition", "reversal")
        ),
        "learned_uses_fewer_interactions_than_random": all(
            learned[phase]["mean_interactions"]
            < summary["random"][phase]["mean_interactions"]
            for phase in ("acquisition", "reversal")
        ),
        "learned_beats_no_memory_late_choice": all(
            learned[phase]["late_first_choice_useful_rate"]
            > summary["no_memory_nearest"][phase]["late_first_choice_useful_rate"]
            for phase in ("acquisition", "reversal")
        ),
        "sensor_noise_exhausts_finite_probe_budget": all(
            row["finite"] and row["remains_idle"] for row in noise
        ),
    }
    result = {
        "evidence": "SYNTHETIC action-contingent resource outcomes; no Genesis, RGB, people, or personality evaluated",
        "limitations": [
            "The resource objective, priors, costs, station affordances and change-detection thresholds are designed.",
            "The learned model predicts consequences; this benchmark does not establish personality or social cognition.",
            "No navigation distance, occlusion, telemetry delay, or perception errors are modeled here.",
            "All baselines know when resource is satisfied; only learned_outcomes uses past outcomes in its choices.",
        ],
        "seed_start": 100,
        "seeds": args.seeds,
        "episodes_per_phase": args.episodes,
        "designed_parameters": {
            "resource_target": RESOURCE_TARGET,
            "meaningful_gain": MEANINGFUL_GAIN,
            "evidence_window": EVIDENCE_WINDOW,
            "initial_probes": INITIAL_PROBES,
            "prior_conditional_gain": PRIOR_GAIN,
            "interaction_cost": INTERACTION_COST,
            "distance_cost_per_metre": DISTANCE_COST,
            "maximum_information_value": MAX_INFORMATION_VALUE,
        },
        "cases": cases,
        "noise_cases": noise,
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Purposeful interaction: synthetic outcome benchmark",
        "",
        result["evidence"],
        "",
        "| Choice rule | Acquisition: interactions per need | Reversal: interactions per need | Late useful choice, acquisition / reversal |",
        "|---|---:|---:|---:|",
    ]
    for strategy, row in summary.items():
        a, r = row["acquisition"], row["reversal"]
        lines.append(
            f"| {strategy} | {a['mean_interactions']:.3f} | {r['mean_interactions']:.3f} | "
            f"{a['late_first_choice_useful_rate']:.1%} / {r['late_first_choice_useful_rate']:.1%} |"
        )
    lines += ["", *result["limitations"], "", "Checks:", ""]
    lines += [
        f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks.items()
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": summary,
                "checks": checks,
                "passed": result["passed"],
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
