"""Causal CV/CA replay, followed by separate scoring on recorded simulator truth."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from bb8_rl.occlusion_belief import OcclusionBelief, observation_input


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay(observations, commands, stream, model):
    """The only dynamics inputs are sanitized observations; commands are context.

    Commands at the capture instant have not yet been issued by the recording
    loop. Only strictly earlier command records are exposed at that instant.
    CV/CA use neither future commands nor a calibrated command-response model.
    """
    belief = OcclusionBelief(model)
    rows, cursor = [], 0
    for record in observations:
        inputs = observation_input(record, stream)
        issued = []
        while (
            cursor < len(commands)
            and commands[cursor]["issue_time"] < inputs["timestamp"] - 1e-10
        ):
            issued.append(
                {
                    "issue_time": commands[cursor]["issue_time"],
                    "action": commands[cursor]["action"],
                }
            )
            cursor += 1
        row = belief.update(**inputs)
        row["new_past_issued_commands"] = issued
        rows.append(row)
    return {"assumptions": belief.assumptions, "estimates": rows}


def score(rows, records, stream):
    scored = []
    for estimate, record in zip(rows, records):
        if stream == "fused":
            hidden = all(
                view["head_pixels_scoring_only"] == 0
                for view in record["views"].values()
            )
        else:
            hidden = record["views"][stream]["head_pixels_scoring_only"] == 0
        position_error = speed_error = covered = None
        if estimate["position"] is not None:
            position_error = float(
                np.linalg.norm(
                    np.asarray(estimate["position"])
                    - record["truth_position_xy_scoring_only"]
                )
            )
            speed_error = abs(
                estimate["speed_m_s"]
                - float(np.linalg.norm(record["truth_velocity_xy_scoring_only"]))
            )
            covered = bool(position_error <= estimate["assumed_position_radius_m"])
        scored.append(
            {
                "time": record["time"],
                "fully_hidden_scoring_only": hidden,
                "mode": estimate["mode"],
                "position_error_m": position_error,
                "speed_error_m_s": speed_error,
                "inside_assumed_radius": covered,
            }
        )
    hidden = [row for row in scored if row["fully_hidden_scoring_only"]]
    available = [row for row in hidden if row["position_error_m"] is not None]
    residuals = [row for row in rows if row["reacquisition_residual_m"] is not None]
    return {
        "requested_frames": len(records),
        "fully_hidden_frames": len(hidden),
        "hidden_estimates_available": len(available),
        "hidden_position_error_p95_m": float(
            np.quantile([r["position_error_m"] for r in available], 0.95)
        )
        if available
        else None,
        "hidden_position_error_max_m": max(r["position_error_m"] for r in available)
        if available
        else None,
        "hidden_speed_error_p95_m_s": float(
            np.quantile([r["speed_error_m_s"] for r in available], 0.95)
        )
        if available
        else None,
        "hidden_pointwise_assumed_radius_coverage": sum(
            r["inside_assumed_radius"] for r in available
        )
        / len(available)
        if available
        else None,
        "hidden_entire_recorded_path_inside_assumed_radius": all(
            r["inside_assumed_radius"] for r in hidden
        )
        if hidden
        else None,
        "max_assumed_position_radius_m": max(
            (
                r["assumed_position_radius_m"]
                for r in rows
                if r["assumed_position_radius_m"] is not None
            ),
            default=None,
        ),
        "reacquisition_attempts": [
            {
                key: row[key]
                for key in (
                    "time",
                    "measurement_accepted",
                    "reacquisition_residual_m",
                    "filter_status",
                )
            }
            for row in residuals
        ],
        "scored_frames": scored,
    }


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh replay output")
    raw = json.loads(args.report.read_text())
    if raw["status"] != "complete" or not raw["checkpoint_unchanged"]:
        raise ValueError("Require complete frozen-checkpoint capture")
    observations, commands = raw["observations"], raw["commands"]
    if [row["step"] for row in observations] != raw["requested_observation_steps"]:
        raise ValueError("Capture omitted requested observations")
    if any(
        b["issue_time"] <= a["issue_time"] for a, b in pairwise(commands)
    ):
        raise ValueError("Commands must have increasing issue timestamps")
    args.output.mkdir(parents=True)
    results = {}
    # Produce and persist every estimate before accessing scoring truth below.
    for stream in ("A", "B", "fused"):
        results[stream] = {
            model: replay(observations, commands, stream, model)
            for model in ("cv", "ca")
        }
    write(args.output / "causal-estimates.json", results)
    scores = {
        stream: {
            model: score(result["estimates"], observations, stream)
            for model, result in models.items()
        }
        for stream, models in results.items()
    }
    report = {
        "status": "complete",
        "scope": "Single prescribed-motion geometric-occlusion trajectory replay; no closed-loop control or motion authorization",
        "source_capture_sha256": digest(args.report),
        "source_checkpoint_sha256": raw["checkpoint_sha256"],
        "script_sha256": digest(Path(__file__)),
        "belief_source_sha256": digest(
            Path(__file__).resolve().parents[1] / "src/bb8_rl/occlusion_belief.py"
        ),
        "causal_estimates_sha256": digest(args.output / "causal-estimates.json"),
        "parameters_selected_from_hidden_truth": False,
        "commands_role": "Past issued commands retained as causal context; CV/CA dynamics do not use them. No command-response model is claimed.",
        "truth_role": "Scored only after all estimates were produced and persisted; never initialization, correction or noise tuning",
        "uncertainty_status": "Assumed process/measurement covariance and three-sigma radius; empirical coverage on one correlated trajectory is not calibrated coverage",
        "map_queries": "unavailable",
        "motion_authorized": False,
        "scores": scores,
    }
    write(args.output / "report.json", report)
    print(
        json.dumps(
            {
                stream: {
                    model: {
                        key: value
                        for key, value in score.items()
                        if key != "scored_frames"
                    }
                    for model, score in models.items()
                }
                for stream, models in scores.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
