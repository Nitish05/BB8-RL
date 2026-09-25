#!/usr/bin/env python3
"""Synthetic controlled activity-choice comparison; does not start Genesis."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from bb8_rl.agency import STRATEGIES, AgencyEngine, AgencyStore, Candidate

CANDIDATES = [
    Candidate("west", (-1, 0), "West activity"),
    Candidate("east", (1, 0), "East activity"),
    Candidate("noise", (0, 1), "Unpredictable activity"),
]


def run_case(path: Path, strategy: str, seed: int, preferred: str) -> dict:
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    store = AgencyStore(
        path, agent_id="bb8", map_version="synthetic-choice-v1", allow_synthetic=True
    )
    engine = AgencyEngine(store, strategy=strategy, seed=seed)
    repeated = 0
    selections: list[str] = []

    def train(phase: str, target: str, steps: int = 120) -> None:
        nonlocal repeated
        for tick in range(steps):
            now = len(selections)
            choice = engine.select(CANDIDATES, now=now, pose=(0, 0))
            candidate = choice["candidate_id"]
            selections.append(candidate)
            # Hidden outcome schedule exists only in this synthetic evaluator.
            # Neither engine.select nor its candidate definitions receive it.
            if candidate == "noise":
                utility = rng.choice((-0.7, 0.7))
            else:
                utility = 0.8 if candidate == target else -0.5
                if rng.random() < 0.05:
                    utility *= -1
            event = {
                "event_id": f"{phase}-{tick}",
                "candidate_id": candidate,
                "utility": utility,
                "now": now,
            }
            assert engine.record_experience(**event)
            # Duplicate messages are frequent in this deliberate stress case.
            for _ in range(3):
                assert not engine.record_experience(**event)
                repeated += 1

    def evaluate(target: str, base: int) -> dict:
        picked = [
            engine.select(CANDIDATES, now=base + i, pose=(0, 0))["candidate_id"]
            for i in range(60)
        ]
        # Frozen evaluation: proposals advance exploration, utility is unchanged.
        return {
            "preferred_rate": picked.count(target) / len(picked),
            "noise_rate": picked.count("noise") / len(picked),
        }

    train("acquisition", preferred)
    acquired = evaluate(preferred, 1000)
    before = engine.snapshot()
    store.close()
    store = AgencyStore(
        path, agent_id="bb8", map_version="synthetic-choice-v1", allow_synthetic=True
    )
    engine = AgencyEngine(store, strategy=strategy, seed=seed)
    restart_exact = before == engine.snapshot()
    retained = evaluate(preferred, 2000)
    reversed_target = "east" if preferred == "west" else "west"
    train("reversal", reversed_target)
    reversed_result = evaluate(reversed_target, 3000)
    final = engine.snapshot()
    store.close()
    return {
        "strategy": strategy,
        "seed": seed,
        "history_preferred": preferred,
        "acquisition": acquired,
        "after_restart": retained,
        "after_reversal": reversed_result,
        "restart_exact": restart_exact,
        "unique_events": final["episodes"],
        "ignored_repeats": repeated,
        "snapshot": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/agency-audit"))
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    # Seed 0 is the development/unit-test seed. 100+ is the held-out audit set;
    # algorithm constants remain fixed for every baseline, history and seed.
    for strategy in STRATEGIES:
        for seed in range(100, 100 + args.seeds):
            for preferred in ("west", "east"):
                path = args.output / f"{strategy}-{seed}-{preferred}.sqlite"
                cases.append(run_case(path, strategy, seed, preferred))
    summary = {}
    for strategy in STRATEGIES:
        group = [case for case in cases if case["strategy"] == strategy]
        summary[strategy] = {
            phase: {
                metric: statistics.mean(case[phase][metric] for case in group)
                for metric in ("preferred_rate", "noise_rate")
            }
            for phase in ("acquisition", "after_restart", "after_reversal")
        }
        summary[strategy]["cases"] = len(group)
        summary[strategy]["restart_exact"] = all(
            case["restart_exact"] for case in group
        )
    learned = summary["learned"]
    checks = {
        "all_restarts_exact": all(case["restart_exact"] for case in cases),
        "duplicates_do_not_add_episodes": all(
            case["unique_events"] == 240 and case["ignored_repeats"] == 720
            for case in cases
        ),
        "learned_acquisition_at_least_85pct": learned["acquisition"]["preferred_rate"]
        >= 0.85,
        "learned_reversal_at_least_85pct": learned["after_reversal"]["preferred_rate"]
        >= 0.85,
        "learned_beats_memory_by_30_points": learned["acquisition"]["preferred_rate"]
        - summary["memory"]["acquisition"]["preferred_rate"]
        >= 0.30,
        "noise_distractor_below_15pct": learned["after_reversal"]["noise_rate"] <= 0.15,
    }
    result = {
        "evidence": "SYNTHETIC utility schedule; no LLM, Genesis, humans or physical robot evaluated",
        "seed_start": 100,
        "seeds": args.seeds,
        "cases": cases,
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Persistent agency: synthetic activity-choice audit",
        "",
        result["evidence"],
        "",
        f"Held-out seeds: 100–{99 + args.seeds}; two opposite experience histories per seed.",
        "",
        "| Strategy | After acquisition | After restart | After reversal | Noisy activity after reversal |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, row in summary.items():
        lines.append(
            f"| {strategy} | {row['acquisition']['preferred_rate']:.1%} | {row['after_restart']['preferred_rate']:.1%} | {row['after_reversal']['preferred_rate']:.1%} | {row['after_reversal']['noise_rate']:.1%} |"
        )
    lines += [
        "",
        "Rates measure choosing the activity with the evaluator's positive synthetic consequences.",
        "The fixed baseline is a seeded fixed ranking, not a prompted language model.",
        "The memory baseline explores less-visited activities without using outcome valence.",
        "Restart checks compare the complete stored snapshot exactly; evaluation retains exploratory choices.",
        "No human preference, personality authenticity or real-world autonomy claim follows from this audit.",
        "",
        "Checks:",
        "",
    ]
    lines += [
        f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in checks.items()
    ]
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
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
