"""BB8-RL simulation, navigation checks and reproducible evaluation commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .diagnostics import doctor, learner_probe, simulation_probe, write_report
from .world import NavigationWorld, validate_world


def run(argv=None):
    parser = argparse.ArgumentParser(
        description="BB8-RL simulation and navigation tools using Genesis Studio"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "simulate", "fixture-check", "_probe"):
        item = sub.add_parser(name)
        item.add_argument(
            "--config", type=Path, default=Path("configs/navigation/bb8-state.yaml")
        )
        item.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
        item.add_argument("--output", type=Path)
        if name in ("doctor", "_probe"):
            item.add_argument("--device", choices=("cpu", "mps"), default=None)
            item.add_argument("--camera-index", type=int)
        if name == "simulate":
            item.add_argument("--viewer", action="store_true")
            item.add_argument("--render", action="store_true")
            item.add_argument("--seconds", type=float, default=4.0)
            item.add_argument(
                "--drive", type=float, nargs=2, default=(0.5, 0.0), metavar=("X", "Y")
            )
    world = sub.add_parser("world").add_subparsers(dest="world_command", required=True)
    validate = world.add_parser("validate")
    validate.add_argument(
        "--config", type=Path, default=Path("configs/navigation/bb8-state.yaml")
    )
    for name in ("env-check", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument(
            "--task", type=Path, default=Path("configs/navigation/bb8-task.yaml")
        )
        command.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
        command.add_argument("--output", type=Path)
        if name == "evaluate":
            command.add_argument(
                "--suites", type=Path, default=Path("configs/navigation/suites.json")
            )
            command.add_argument(
                "--suite", choices=("train", "validation", "test"), default="validation"
            )
            command.add_argument("--episodes", type=int)
            command.add_argument(
                "--controllers",
                nargs="+",
                choices=("baseline", "random", "sac-her", "dreamerv3"),
                default=["baseline", "random"],
            )
            command.add_argument("--viewer", action="store_true")
            command.add_argument("--checkpoint", type=Path)
            command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    trainer = sub.add_parser("train")
    trainer.add_argument(
        "--config", type=Path, default=Path("configs/training/sac-her.yaml")
    )
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--resume", type=Path)
    dreamer = sub.add_parser("train-dreamer")
    dreamer.add_argument(
        "--config", type=Path, default=Path("configs/training/dreamer-m5.yaml")
    )
    dreamer.add_argument("--output", type=Path, required=True)
    room = sub.add_parser("room").add_subparsers(dest="room_command", required=True)
    generate = room.add_parser("generate")
    generate.add_argument(
        "--base-config", type=Path, default=Path("configs/navigation/bb8-state.yaml")
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--side", type=float, default=4.0)
    generate.add_argument("--obstacles", type=int, default=7)
    preview = room.add_parser("preview")
    preview.add_argument(
        "--config", type=Path, default=Path("projects/bb8/synthetic-room/room.yaml")
    )
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "train-dreamer":
            from .dreamer import train

            result = train(args.config, args.output)
            print(json.dumps(result))
            return 0 if result["status"] == "complete" else 130
        elif args.command == "train":
            from .training import train

            result = train(args.config, args.output, resume=args.resume)
            print(json.dumps(result))
            return 0 if result["status"] == "complete" else 130
        elif args.command == "room":
            from .room import generate_room, preview_room

            result = (
                generate_room(
                    args.base_config,
                    args.output,
                    seed=args.seed,
                    side=args.side,
                    obstacles=args.obstacles,
                )
                if args.room_command == "generate"
                else preview_room(args.config, args.output, backend=args.backend)
            )
            print(json.dumps(result))
            return 0 if result["status"] == "passed" else 1
        elif args.command == "env-check":
            from gymnasium.utils.env_checker import check_env

            from .env import NavigationEnv

            with NavigationEnv(args.task, backend=args.backend) as env:
                check_env(env, skip_render_check=True)
            result = {
                "status": "passed",
                "check": "Gymnasium scalar environment",
                "observations": "oracle",
            }
        elif args.command == "evaluate":
            from .evaluate import evaluate

            summary = evaluate(
                args.task,
                args.suites,
                split=args.suite,
                count=args.episodes,
                controllers=tuple(args.controllers),
                backend=args.backend,
                output=args.output or Path("work/evaluation"),
                viewer=args.viewer,
                checkpoint=args.checkpoint,
                device=args.device,
            )
            print(json.dumps(summary))
            return 0
        elif args.command == "world":
            config, project, body, _head, camera = validate_world(args.config)
            result = {
                "status": "passed",
                "project_schema": project.schema_version,
                "provenance": config.provenance,
                "body_radius_m": body.radius,
                "camera": camera.name,
                "notes": config.notes,
            }
        elif args.command == "doctor":
            output = args.output or Path("work/bb8-doctor")
            result = doctor(
                args.config,
                args.backend,
                args.device or "cpu",
                output,
                args.camera_index,
            )
            print(
                json.dumps(
                    {
                        "simulation_gate": result["simulation_gate"],
                        "hardware_gate": result["hardware_gate"],
                        "camera": result["camera"]["status"],
                        "report": str(output / "machine-report.json"),
                    }
                )
            )
            passed = result["simulation_gate"] == "passed" and (
                args.camera_index is None or result["camera"]["status"] == "passed"
            )
            return 0 if passed else 1
        elif args.command == "_probe" and args.device:
            result = learner_probe(args.device)
        elif args.command == "_probe" and args.camera_index is not None:
            from .diagnostics import camera_probe

            result = camera_probe(args.camera_index)
        elif args.command in ("_probe", "fixture-check"):
            result = simulation_probe(
                args.config,
                args.backend,
                image_path=args.output.with_suffix(".png") if args.output else None,
            )
        else:
            import math

            if not math.isfinite(args.seconds) or not 0.5 <= args.seconds <= 60:
                raise ValueError("Choose a finite duration between 0.5 and 60 seconds")
            # Reject invalid commands before graphics startup.
            from bb8_rl.control.contract import PlanarDriveCommand

            PlanarDriveCommand(*args.drive, 0, 0.1)
            started = time.perf_counter()
            with NavigationWorld(
                args.config,
                backend=args.backend,
                viewer=args.viewer,
                render=args.render,
            ) as fixture:
                # Last second is a bounded brake; long drives stop at the arena boundary.
                while fixture.time < args.seconds:
                    if args.viewer and not fixture.scene.viewer.is_alive():
                        break
                    if (
                        fixture.time < args.seconds - min(1.0, args.seconds / 2)
                        and not fixture.boundary_failure
                    ):
                        fixture.drive(*args.drive)
                    else:
                        fixture.stop()
                    fixture.step(fixture.config.action_steps)
                state = fixture.backend.read_planar_state(now=fixture.time)
                result = {
                    "status": "passed"
                    if not fixture.boundary_failure
                    else "boundary_failure",
                    "final": asdict(state),
                    "physics_steps": fixture.steps,
                    "non_floor_contact_samples": fixture.contacts,
                    "wall_seconds_including_startup": time.perf_counter() - started,
                }
        if getattr(args, "output", None):
            write_report(args.output, result)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["status"] == "passed" else 1
    except KeyboardInterrupt:
        print("Simulation cancelled; native resources closed.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 — CLI records unexpected engine failures too.
        result = {"status": "failed", "reason": str(error)}
        if getattr(args, "output", None) and args.command not in (
            "doctor",
            "evaluate",
            "room",
            "train",
            "train-dreamer",
        ):
            write_report(args.output, result)
        print(json.dumps(result), file=sys.stderr)
        return 1


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
