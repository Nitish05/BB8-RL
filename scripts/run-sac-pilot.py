"""Run the declared three-seed pilot sequentially, retaining every stage log."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from bb8_rl.diagnostics import write_report
from bb8_rl.training import load_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/sac-her.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, task = load_training(args.config)
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Choose a fresh experiment directory")
    root.mkdir(parents=True)
    record = {
        "status": "running",
        "seeds": [17, 29, 43],
        "minimum_steps_per_seed": 12000,
        "task": str(task),
        "validation_cases": {"initial": 4, "mid": 4, "final": 12},
        "stages": [],
    }
    write_report(root / "experiment.json", record)

    def run(label, *arguments):
        print(json.dumps({"stage": label, "status": "starting"}), flush=True)
        with (root / f"{label}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "bb8_rl.cli", *map(str, arguments)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record["stages"].append(label)
        write_report(root / "experiment.json", record)
        print(json.dumps({"stage": label, "status": "complete"}), flush=True)

    try:
        for seed in record["seeds"]:
            settings = cfg.model_dump()
            settings.update(
                seed=seed, task=str(task), total_steps=6000 if seed == 17 else 12000
            )
            config_path = root / f"seed-{seed}.yaml"
            config_path.write_text(yaml.safe_dump(settings))
            run(
                f"train-{seed}",
                "train",
                "--config",
                config_path,
                "--output",
                root / f"train-{seed}",
            )
            checkpoint_paths = sorted(
                (root / f"train-{seed}/checkpoints").glob("step-*")
            )
            initial, middle = checkpoint_paths[0], checkpoint_paths[1]
            if seed == 17:
                settings["total_steps"] = 12000
                continued = root / "seed-17-resume.yaml"
                continued.write_text(yaml.safe_dump(settings))
                run(
                    "resume-17",
                    "train",
                    "--config",
                    continued,
                    "--output",
                    root / "resume-17",
                    "--resume",
                    middle,
                )
                final = Path(
                    json.loads((root / "resume-17/latest.json").read_text())[
                        "checkpoint"
                    ]
                )
            else:
                final = Path(
                    json.loads((root / f"train-{seed}/latest.json").read_text())[
                        "checkpoint"
                    ]
                )
            for label, checkpoint, episodes in (
                ("initial", initial, 4),
                ("mid", middle, 4),
                ("final", final, 12),
            ):
                run(
                    f"eval-{seed}-{label}",
                    "evaluate",
                    "--task",
                    task,
                    "--suite",
                    "validation",
                    "--episodes",
                    episodes,
                    "--controllers",
                    "sac-her",
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    root / f"eval-{seed}-{label}",
                )
        run(
            "references",
            "evaluate",
            "--task",
            task,
            "--suite",
            "validation",
            "--episodes",
            12,
            "--controllers",
            "baseline",
            "random",
            "--output",
            root / "references",
        )
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_report(root / "experiment.json", record)


if __name__ == "__main__":
    main()
