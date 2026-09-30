"""Run a frozen camera-navigation case/profile and independently score its log."""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import queue
import shutil
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from bb8_rl.control_profiles import PROFILES, get_profile
from bb8_rl.demo_assets import validate_assets
from bb8_rl.interactive_runtime import worker_main

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    protocol = json.loads(args.protocol.read_text())
    case = protocol["cases"][args.case]
    profile = get_profile(args.profile)
    expected_inputs = {
        "bundle_manifest_sha256": "bundle.json",
        "map_manifest_sha256": "memory/manifest.json",
        "room_memory_sha256": "memory/room-memory.json",
    }
    for key, name in expected_inputs.items():
        if digest(args.assets / name) != protocol[key]:
            raise ValueError(f"Assets differ from frozen protocol: {name}")
    minimum_end = next(
        (entry["minimum_end_sim_time"] for entry in protocol["failure_controls"]
         if entry["id"] == args.case), 0.0
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark-source.py").write_bytes(Path(__file__).read_bytes())
    (output / "protocol.json").write_bytes(args.protocol.read_bytes())
    assets = output / "assets"
    shutil.copytree(args.assets, assets)
    demo_path = assets / "demo.json"
    demo = json.loads(demo_path.read_text())
    demo.update(start=case["start"], goal=case["goal"])
    demo_path.write_text(json.dumps(demo, indent=2) + "\n")
    bundle_path = assets / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["sha256"]["demo.json"] = digest(demo_path)
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    validate_assets(assets)
    config = {
        "asset_dir": str(assets),
        "recording_mode": "audit",
        "run_dir": str(output / "worker"),
        "mode": case["mode"],
        "max_steps": round(case["duration_s"] / 0.05),
        "control_profile": profile.name,
        "generation": 0,
        "save_frames": args.save_frames,
        "scoring_labels": True,
        **{
            key: case[key]
            for key in ("camera_dropouts", "invalidate_memory_at")
            if key in case
        },
    }
    (output / "case.json").write_text(json.dumps({
        "case_id": args.case, "case": case, "config": config,
        "profile": profile.record(), "protocol_sha256": digest(args.protocol),
        "harness_sha256": digest(__file__),
    }, indent=2) + "\n")
    context = multiprocessing.get_context("spawn")
    commands, events, stop = context.Queue(32), context.Queue(16), context.Event()
    process = context.Process(target=worker_main, args=(config, commands, events, stop))
    commands.put({"action": "goal", "generation": 1,
                  "x": demo["goal"][0], "y": demo["goal"][1]})
    commands.put({"action": "heartbeat", "generation": 1})
    process.start()
    started, last_heartbeat, sent_stop = time.monotonic(), 0.0, False
    errors, last_state = [], None
    try:
        while process.is_alive():
            now = time.monotonic()
            if now - started > 900:
                errors.append("worker_wall_timeout")
                break
            if now - last_heartbeat >= 0.3:
                commands.put({"action": "heartbeat", "generation": 1})
                last_heartbeat = now
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("type") == "error":
                errors.append(event)
            if event.get("type") != "state":
                continue
            last_state = event["state"]
            if (last_state.get("phase") == "arrived"
                    and case.get("intent", "arrival") == "arrival"
                    and last_state.get("sim_time", 0) >= minimum_end):
                stop.set()
            if ("stop_at" in case and not sent_stop
                    and last_state.get("sim_time", 0) >= case["stop_at"]):
                commands.put({"action": "stop", "generation": 2})
                commands.put({"action": "goal", "generation": 1,
                              "x": demo["goal"][0], "y": demo["goal"][1]})
                sent_stop = True
    finally:
        stop.set()
        process.join(30)
        if process.is_alive():
            errors.append("worker_cleanup_timeout")
            process.terminate()
            process.join(10)
        for channel in (commands, events):
            channel.close()
        (output / "harness.json").write_text(json.dumps({
            "exitcode": process.exitcode, "errors": errors, "last_state": last_state,
            "wall_seconds": time.monotonic() - started,
        }, indent=2) + "\n")
    if errors or process.exitcode != 0:
        raise RuntimeError(f"Native worker failed; inspect {output}")
    spec = importlib.util.spec_from_file_location("independent_audit", ROOT / "scripts/audit-interactive.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.main(SimpleNamespace(run=output / "worker", output=output / "audit"))
    report = json.loads((output / "audit/report.json").read_text())
    rows = [json.loads(line) for line in (output / "worker/rows.jsonl").read_text().splitlines()]
    statuses = [r["controller_status"] for r in rows]
    guards = {"stopping_envelope_not_certified", "proactive_envelope_not_certified", "route_not_certified"}
    episodes = sum(s in guards and (i == 0 or statuses[i - 1] not in guards) for i, s in enumerate(statuses))
    arrivals = report["arrival_events"]
    summary = {
        "case": args.case, "profile": profile.name, "intent": case.get("intent", "arrival"),
        "audit_pass": report["audit_pass"],
        "valid_arrival": report["valid_arrivals"] > 0,
        "arrival_time_s": arrivals[0]["time"] if arrivals else None,
        "recorded_simulation_s": rows[-1]["after_step"]["time"],
        "guarded_stop_episodes": episodes, "status_counts": dict(Counter(statuses)),
        "collision": report["collision"], "boundary": report["boundary"],
        "premature_arrivals": report["premature_or_unobservable_arrivals"],
        "report_sha256": digest(output / "audit/report.json"),
        "harness_sha256": digest(__file__),
        "images_saved": args.save_frames,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/interactive-assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-frames", action="store_true")
    run(parser.parse_args())
