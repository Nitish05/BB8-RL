"""Summarize all declared seeds without selecting the best run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def summarize(root, output):
    experiment = read(root / "experiment.json")
    if experiment["status"] != "complete":
        raise ValueError("The declared experiment is not complete")
    output.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for seed in experiment["seeds"]:
        stages = [root / f"train-{seed}"]
        if seed == 17:
            stages.append(root / "resume-17")
        training = [read(stage / "manifest.json") for stage in stages]
        row = {
            "controller": f"SAC + HER seed {seed}",
            "seed": seed,
            **read(root / f"eval-{seed}-final/summary.json")["sac-her"],
            "training_steps": training[-1]["steps"],
            "updates": training[-1]["updates"],
            "training_wall_seconds": sum(m["elapsed_seconds"] for m in training),
            "gradient_seconds": sum(m["gradient_seconds"] for m in training),
            "checkpoint": training[-1]["final_checkpoint"],
            "resumed": len(stages) > 1,
        }
        rows.append(row)
        curve = []
        for label in ("initial", "mid", "final"):
            folder = root / f"eval-{seed}-{label}"
            m = read(folder / "manifest.json")
            cases = [
                read(folder / "episodes" / f"validation-{i:03d}-sac-her.json")
                for i in range(4)
            ]
            curve.append(
                {
                    "stage": label,
                    "steps": m["checkpoint_state"]["steps"],
                    "success_rate_fixed_four": sum(r["success"] for r in cases) / 4,
                    "mean_error_fixed_four_m": float(
                        np.mean([r["final_error_m"] for r in cases])
                    ),
                }
            )
        curves[seed] = curve
    references = read(root / "references/summary.json")
    comparison = rows + [
        {"controller": name, **metrics} for name, metrics in references.items()
    ]
    summary = {
        "scope": "single authored synthetic room, oracle local observations; full-map A* reference",
        "seeds": rows,
        "references": references,
        "curves_fixed_four": curves,
        "mean_seed_success_rate": float(np.mean([r["success_rate"] for r in rows])),
        "std_seed_success_rate": float(
            np.std([r["success_rate"] for r in rows], ddof=1)
        ),
        "all_seed_minimum_budget_met": all(r["training_steps"] >= 12000 for r in rows),
        "final_test_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "controller",
        "successes",
        "episodes",
        "collision_episodes",
        "timeouts",
        "mean_final_error_m",
        "training_steps",
        "updates",
        "training_wall_seconds",
    ]
    with (output / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, (seed, curve) in zip(axes, curves.items()):
        ax.plot(
            [p["steps"] for p in curve],
            [p["success_rate_fixed_four"] for p in curve],
            "o-",
            color="#175e91",
        )
        ax.set(
            title=f"Seed {seed}",
            xlabel="Training transitions",
            ylim=(-0.05, 1.05),
            yticks=[0, 0.25, 0.5, 0.75, 1],
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Arrival fraction (same four cases)")
    fig.suptitle("SAC + HER pilot — fixed validation subset at each checkpoint")
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in experiment["seeds"]:
        stages = [root / f"train-{seed}"] + ([root / "resume-17"] if seed == 17 else [])
        episodes = [
            json.loads(line)
            for stage in stages
            for line in (stage / "episodes.jsonl").read_text().splitlines()
        ]
        steps = [r["step"] for r in episodes]
        returns = [
            np.mean([r["return"] for r in episodes[max(0, i - 4) : i + 1]])
            for i in range(len(episodes))
        ]
        axes[0].plot(steps, returns, label=f"Seed {seed}")
        losses = []
        for stage in stages:
            with (stage / "logs/progress.csv").open() as f:
                for row in csv.DictReader(f):
                    if row.get("train/critic_loss") and row.get("train/n_updates"):
                        losses.append(
                            (
                                float(row["train/n_updates"]),
                                float(row["train/critic_loss"]),
                            )
                        )
        if losses:
            axes[1].plot(*np.asarray(losses).T, label=f"Seed {seed}")
    axes[0].set(
        xlabel="Training transitions",
        ylabel="Episode return (rolling five)",
        title="Exploratory training episodes",
    )
    axes[1].set(
        xlabel="Gradient updates",
        ylabel="Critic loss",
        title="Optimization diagnostics",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training-diagnostics.png", dpi=160)
    plt.close(fig)
    lines = [
        "# SAC + HER pilot results",
        "",
        "Every declared seed is reported. These are synthetic oracle results in one room.",
        "",
        "| Controller | Success | 95% Wilson interval | Contact episodes | Timeouts | Mean final error |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in comparison:
        lo, hi = r["success_95ci"]
        lines.append(
            f"| {r['controller']} | {r['successes']}/{r['episodes']} | {lo:.1%}–{hi:.1%} | {r['collision_episodes']} | {r['timeouts']} | {r['mean_final_error_m']:.3f} m |"
        )
    lines += [
        "",
        "| Seed | Actual transitions | Gradient updates | Training wall time | Restarted from midpoint |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['training_steps']} | {r['updates']} | {r['training_wall_seconds']:.1f} s | {'Yes' if r['resumed'] else 'No'} |"
        )
    lines += [
        "",
        "Episode-boundary completion explains transition counts above 12,000. Timing includes initialization/checkpoints and excludes evaluation.",
        "",
        "![Learning curves](learning-curves.png)",
        "",
        "![Training diagnostics](training-diagnostics.png)",
        "",
        "Curves use the same four cases at all checkpoints; the table uses all twelve final validation cases. Training returns use varied exploratory episodes and are not validation success.",
        "",
        "The policy uses a pooled local map; A* has the full map. Seeds vary learning and training endpoints, not authored-room geometry. These small correlated same-room samples do not establish generalization, vision performance or physical transfer. The final test suite was not used.",
        "",
    ]
    if all(r["successes"] == 0 for r in rows):
        lines += [
            "The pipeline optimized and resumed successfully, but none of the final policies solved a validation episode. This pilot does not establish useful navigation. A new experiment should first test goal-reaching in the empty room and then add obstacle difficulty, retaining the present reward/evaluation results as the reference.",
            "",
        ]
    else:
        lines += [
            "Some arrivals were observed. The small single-room pilot is not a validated navigation policy; inspect every seed and failure before planning a larger experiment.",
            "",
        ]
    (output / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "mean_seed_success_rate": summary["mean_seed_success_rate"],
                "report": str(output / "report.md"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
