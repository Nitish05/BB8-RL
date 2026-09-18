"""Sequential, bounded M5 comparison with a shared Python/Genesis runtime."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

from bb8_rl.diagnostics import write_report

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = args.output.resolve()
if output.exists():
    raise SystemExit("Choose a fresh output directory to preserve existing evidence")
output.mkdir(parents=True)
experiment = {
    "status": "running",
    "protocol": "docs/M5_PLAN.md",
    "seed": 17,
    "requested_steps_per_learner": 30000,
    "validation_cases": 12,
    "python": sys.executable,
    "stages": [],
    "final_test_used": False,
    "selected_runtime_versions": {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "jax",
            "jaxlib",
            "torch",
            "genesis-world",
            "stable-baselines3",
            "opencv-python",
            "tifffile",
        )
    },
}


def stage(name, arguments):
    command = [sys.executable, "-u", "-m", "bb8_rl.cli", *arguments]
    record = {"name": name, "command": command, "status": "running"}
    experiment["stages"].append(record)
    write_report(output / "experiment.json", experiment)
    print(f"Starting {name}", flush=True)
    started = time.perf_counter()
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    record.update(
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_report(output / "experiment.json", experiment)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
    print(f"Completed {name}", flush=True)


try:
    for name, command, config in (
        ("sac", "train", "configs/training/sac-m5.yaml"),
        ("dreamer", "train-dreamer", "configs/training/dreamer-m5.yaml"),
    ):
        stage(
            f"train-{name}",
            [command, "--config", config, "--output", str(output / name)],
        )
    task = "projects/bb8/synthetic-room/task-penalty-v2.yaml"
    for name, controller in (("sac", "sac-her"), ("dreamer", "dreamerv3")):
        manifest = json.loads((output / name / "manifest.json").read_text())
        stage(
            f"eval-{name}",
            [
                "evaluate",
                "--task",
                task,
                "--episodes",
                "12",
                "--controllers",
                controller,
                "--checkpoint",
                manifest["final_checkpoint"],
                "--output",
                str(output / f"eval-{name}"),
            ],
        )
    stage(
        "references",
        [
            "evaluate",
            "--task",
            task,
            "--episodes",
            "12",
            "--controllers",
            "baseline",
            "--output",
            str(output / "references"),
        ],
    )
    experiment["status"] = "complete"
except BaseException as error:
    experiment.update(status="failed", error=repr(error))
    raise
finally:
    write_report(output / "experiment.json", experiment)
