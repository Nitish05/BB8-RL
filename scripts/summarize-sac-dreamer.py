"""Summarize the fixed M5 pilot without claiming a general algorithm ranking."""

import argparse
import json
from pathlib import Path

import numpy as np

from bb8_rl.diagnostics import write_report
from bb8_rl.evaluate import plot_paths

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
run, output = args.run.resolve(), args.output.resolve()
experiment = json.loads((run / "experiment.json").read_text())
if experiment["status"] != "complete":
    raise SystemExit("Only summarize a completed comparison")
output.mkdir(parents=True, exist_ok=True)
train = {
    name: json.loads((run / name / "manifest.json").read_text())
    for name in ("sac", "dreamer")
}
assert train["sac"]["contract"] == train["dreamer"]["contract"]
assert train["sac"]["bb8_source_sha256"] == train["dreamer"]["bb8_source_sha256"]
for package in ("numpy", "genesis-world", "torch", "opencv-python"):
    assert train["sac"]["packages"][package] == train["dreamer"]["packages"][package]
records, scores, eval_manifests = [], {}, {}
for folder in ("eval-sac", "eval-dreamer", "references"):
    manifest = json.loads((run / folder / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    eval_manifests[folder] = manifest
    scores.update(json.loads((run / folder / "summary.json").read_text()))
    records.extend(
        json.loads(file.read_text())
        for file in sorted((run / folder / "episodes").glob("*.json"))
    )
for case in eval_manifests["eval-sac"]["cases"]:
    rows = [r for r in records if r["id"] == case["id"]]
    assert len(rows) == 3
    for row in rows[1:]:
        for key in ("start", "goal", "obstacles", "obstacle_sizes"):
            np.testing.assert_array_equal(rows[0][key], row[key])
plot_paths(output, records)
result = {
    "schema_version": 1,
    "status": "complete",
    "seed": experiment["seed"],
    "paired_geometry_verified": True,
    "final_test_used": False,
    "scores": scores,
    "training": {
        name: {
            key: value[key]
            for key in (
                "steps",
                "updates",
                "elapsed_seconds",
                "gradient_seconds",
                "bb8_source_sha256",
                "final_checkpoint",
                "episode_boundary_overshoot",
            )
        }
        for name, value in train.items()
    },
    "scope": "One seed, one authored room, oracle local observations; not a vision or hardware result",
}
write_report(output / "summary.json", result)
lines = [
    "# BB8-RL: SAC + HER versus DreamerV3",
    "",
    "Completed local M5 pilot, 16 September 2026. Both learners trained from scratch",
    "in the same synthetic 4 m room with seven obstacles, the corrected failure",
    "penalty, identical 180 state/map features, and native CPU Genesis physics.",
    "",
    "| Controller | Successful arrivals | Contact episodes | Timeouts | Mean final error |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for name in ("sac-her", "dreamerv3", "baseline"):
    s = scores[name]
    lines.append(
        f"| {name} | {s['successes']}/{s['episodes']} | {s['collision_episodes']} | {s['timeouts']} | {s['mean_final_error_m']:.3f} m |"
    )
lines += [
    "",
    "Arrival requires ≤10 cm position error and ≤3 cm/s speed held for 0.5 s.",
    "Contacts exclude the floor. All 12 starts, goals and obstacle geometries",
    "were checked to match exactly across controllers.",
    "",
    "| Learner | Actual actions | Updates | Training wall time | Optimization time |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for name in ("sac", "dreamer"):
    t = train[name]
    lines.append(
        f"| {name} | {t['steps']:,} | {t['updates']:,} | {t['elapsed_seconds']:.1f} s | {t['gradient_seconds']:.1f} s |"
    )
lines += [
    "",
    "Each run requested 30,000 actions and finished its last episode. Reset rows",
    "are excluded from the action budget. Training runs were sequential. Wall time",
    "includes startup, compilation and checkpoints; Dreamer's optimization time",
    "includes its first training compilation. Update counts differ in meaning:",
    "SAC uses transition minibatches/HER; Dreamer uses sequences and imagined rollouts.",
    "",
    "## Interpretation",
    "",
    "This is a single-seed, limited-budget pilot. It cannot establish a reliable",
    "algorithm ranking. Both learners see oracle local state/map features, not",
    "camera frames. Dreamer has recurrent memory and uses the authors' size1m preset",
    "with a CPU-friendly replay ratio of 16, below the upstream state-control",
    "benchmark recipe. Both have discount 0.98 and use the unchanged penalty-v2 task.",
    "",
    "SAC evaluation uses deterministic actions. Dreamer retains the authors' sampled",
    "evaluation actions, with its recurrent state reset and action RNG counter set",
    "from each case's seed. This operational distinction is recorded, not hidden.",
    "",
    "A* has a full map and therefore serves as a full-information reference. One",
    "authored layout does not test new-room generalization. The final 200-case test",
    "suite was not used. Neither simulated success nor a saved checkpoint establishes",
    "physical BB-8 compatibility.",
    "",
    "## Trajectories",
    "",
    "![Paired policy trajectories](paths.png)",
    "",
    "## Provenance",
    "",
    f"BB8-RL source SHA-256: `{train['sac']['bb8_source_sha256']}`.",
    "",
    f"Dreamer authors' revision: `{train['dreamer']['upstream_revision']}`.",
    "",
    "Upstream Dreamer and Genesis Studio source trees remain unchanged. The first",
    "SAC launch was cancelled after a dependency preflight found inherited image",
    "packages requiring newer NumPy; both accepted runs start fresh after resolving",
    "those conflicts. Failed setup attempts and the cancelled run are retained",
    "separately under project work storage and excluded from this result table.",
    "",
    "Dreamer checkpoints have verified checksums and support inference reload; exact",
    "Dreamer training resume is not implemented. SAC retains its existing resume",
    "support. Per-case JSON retains all failures and complete trajectories.",
    "",
    "Source: [authors' DreamerV3 implementation](https://github.com/danijar/dreamerv3).",
    "",
]
(output / "report.md").write_text("\n".join(lines))
print(json.dumps(result, indent=2))
