"""Native RGB rig observation/handover probe; prescribed motion, not visual control."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bb8_rl.camera_rig import CameraRig, CameraView, calibration_from_live_camera
from bb8_rl.diagnostics import write_report
from bb8_rl.env import NavigationEnv
from bb8_rl.guided import TASKS
from bb8_rl.training import file_hash
from bb8_rl.vision import LearnedObserver


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    records = []
    with NavigationEnv(
        TASKS["authored"],
        render_mode="rgb_array",
        camera_positions=[(2.8, -3.4, 5.2), (-2.8, 3.4, 5.2)],
    ) as env:
        raw, _ = env.reset(
            seed=751001,
            options={
                "layout_seed": 1001,
                "start": [-1.75, -1.75],
                "goal": [-1.0, -1.75],
            },
        )
        env.world.camera_period = env.world._next_frame = 1e9
        views = []
        for name, camera in env.world.cameras.items():
            calibration = calibration_from_live_camera(camera, 2)
            views.append(
                CameraView(
                    name,
                    "synthetic-v1",
                    calibration,
                    LearnedObserver(
                        calibration, args.checkpoint, floor_refinement="guided"
                    ),
                )
            )
        rig = CameraRig(views)
        for step in range(24):
            started = time.perf_counter()
            frames = env.world.capture_rig()
            capture_seconds = time.perf_counter() - started
            omitted = (
                ("A",) if 6 <= step <= 11 else ("A", "B") if 16 <= step <= 18 else ()
            )
            started = time.perf_counter()
            result = rig.observe(
                [f for f in frames if f.camera_id not in omitted], now=env.world.time
            )
            observation_seconds = time.perf_counter() - started
            records.append(
                {
                    "step": step,
                    "time": env.world.time,
                    "omitted_camera_ids": omitted,
                    "status": result.status,
                    "source_ids": result.source_ids,
                    "handover": result.handover,
                    "position": result.xy.tolist() if result.xy is not None else None,
                    "error_m": float(
                        np.linalg.norm(result.xy - raw["achieved_goal"][:2])
                    )
                    if result.xy is not None
                    else None,
                    "truth_scoring_only": raw["achieved_goal"].tolist(),
                    "view_status": {k: v.status for k, v in result.views.items()},
                    "capture_wall_seconds": capture_seconds,
                    "observation_wall_seconds": observation_seconds,
                }
            )
            # The fixed command only generates a short observation sequence.
            # Rig output never influences it; this is not an occlusion controller.
            raw, _, terminal, truncated, _ = env.step([0.2, 0] if step < 15 else [0, 0])
            if terminal or truncated:
                raise RuntimeError("Observation probe ended unexpectedly")
        assert all(
            r["position"] is None for r in records if len(r["omitted_camera_ids"]) == 2
        )
        assert env.world.contacts == 0
    measured = [r["error_m"] for r in records if r["error_m"] is not None]
    report = {
        "status": "complete",
        "scope": "native two-camera observation and injected stream loss; prescribed motion, no visual control or geometric-occlusion claim",
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_sha256": file_hash(Path(__file__)),
        "camera_count": 2,
        "measured_frames": len(measured),
        "frames": len(records),
        "localization_error_p95_m": float(np.quantile(measured, 0.95))
        if measured
        else None,
        "handover_events": sum(r["handover"] for r in records),
        "contacts": 0,
        "all_missing_frames_have_no_position": True,
        "records": records,
    }
    write_report(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    main(parser.parse_args())
