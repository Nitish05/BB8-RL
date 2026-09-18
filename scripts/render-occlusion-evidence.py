"""Render scoring-only native trajectory evidence; never a controller input."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main(args):
    rows = [
        json.loads(line) for line in (args.case / "rows.jsonl").read_text().splitlines()
    ]
    protocol = json.loads((args.case.parent / "protocol.json").read_text())
    case = next(c for c in protocol["cases"] if c["id"] == args.case.name)
    truth = np.array([r["truth_before_scoring_only"]["position"][:2] for r in rows])
    belief = np.array(
        [
            (r["controller"].get("xy") or [np.nan, np.nan])
            if r["controller"].get("state_is_current")
            else [np.nan, np.nan]
            for r in rows
        ]
    )
    times = np.array([r["time"] for r in rows])
    hidden = np.array([r["head_pixels_scoring_only"] == 0 for r in rows])
    dropout = np.array([r["dropout_injected"] for r in rows])
    radii = np.array(
        [
            (r["controller"].get("position_radius_m") or np.nan)
            if r["controller"].get("state_is_current")
            else np.nan
            for r in rows
        ]
    )
    error = np.linalg.norm(truth - belief, axis=1)
    speed = np.array(
        [np.linalg.norm(r["truth_before_scoring_only"]["velocity"][:2]) for r in rows]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((args.case.parent / "calibration.json").read_text())
    world_points = np.c_[
        np.vstack([truth, case["goal"]]), np.full(len(truth) + 1, 0.085)
    ]
    transform = np.asarray(calibration["estimated_world_to_camera"])
    projected = (world_points @ transform[:3, :3].T + transform[:3, 3]) @ np.asarray(
        calibration["intrinsics"]
    ).T
    pixels = projected[:, :2] / projected[:, 2:]
    crop_center = (pixels.min(axis=0) + pixels.max(axis=0)) / 2
    crop_width = max(320, int(np.ceil(np.ptp(pixels, axis=0).max() + 160)))
    crop_height = int(np.ceil(crop_width * 0.75))
    image_width, image_height = calibration["resolution"]
    crop_width, crop_height = (
        min(crop_width, image_width),
        min(crop_height, image_height),
    )
    left = int(np.clip(crop_center[0] - crop_width / 2, 0, image_width - crop_width))
    top = int(np.clip(crop_center[1] - crop_height / 2, 0, image_height - crop_height))
    crop_box = (left, top, left + crop_width, top + crop_height)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    native = json.loads((args.case / "native-static-obstacles.json").read_text())
    for position, size in zip(native["positions"], native["sizes"], strict=True):
        axes[0, 0].add_patch(
            Rectangle(
                np.array(position[:2]) - np.array(size[:2]) / 2,
                *size[:2],
                color="#bec3ce",
            )
        )
    axes[0, 0].plot(*truth.T, label="Native physics (scoring only)", color="#293d5a")
    axes[0, 0].plot(*belief.T, "--", label="RGB / command estimate", color="#227fac")
    axes[0, 0].scatter(
        *truth[hidden].T, s=12, color="#c5592e", label="Head hidden at capture"
    )
    axes[0, 0].scatter(
        *case["goal"], marker="*", s=140, color="#459158", label="Requested goal"
    )
    route_points = np.vstack([truth, case["goal"], case["start"]])
    center = (route_points.min(axis=0) + route_points.max(axis=0)) / 2
    half_span = max(0.5, float(np.ptp(route_points, axis=0).max()) / 2 + 0.3)
    axes[0, 0].set(
        xlim=(center[0] - half_span, center[0] + half_span),
        ylim=(center[1] - half_span, center[1] + half_span),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="Route detail in the scanned room",
    )
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 1].plot(times, error * 100, label="Actual localization error")
    axes[0, 1].plot(times, radii * 100, "--", label="Assumed position radius")
    axes[0, 1].set(
        xlabel="Simulation time (s)",
        ylabel="cm",
        title="Prediction error and declared uncertainty",
    )
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(times, speed, label="Native speed")
    axes[1, 0].plot(
        times,
        [
            r["controller"].get("speed_m_s", np.nan)
            if r["controller"].get("state_is_current")
            else np.nan
            for r in rows
        ],
        "--",
        label="Estimated speed",
    )
    axes[1, 0].axhline(0.03, linestyle=":", color="#666", label="Arrival speed limit")
    axes[1, 0].set(
        xlabel="Simulation time (s)", ylabel="m/s", title="Motion and braking"
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].step(
        times,
        [r["head_pixels_scoring_only"] for r in rows],
        where="post",
        label="Rendered head pixels",
    )
    axes[1, 1].set(
        xlabel="Simulation time (s)",
        ylabel="pixels",
        title="Geometric visibility; omissions are separate",
    )
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.fill_between(
            times,
            0,
            1,
            where=hidden,
            transform=axis.get_xaxis_transform(),
            color="#c5592e",
            alpha=0.15,
            step="post",
            label="Head hidden",
        )
        axis.fill_between(
            times,
            0,
            1,
            where=dropout,
            transform=axis.get_xaxis_transform(),
            color="#8253b5",
            alpha=0.15,
            step="post",
            label="Injected omission",
        )
        axis.grid(alpha=0.15)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        f"{case['id']} — one fixed camera after RGB scan\nNative lockstep synthetic simulation; geometry and truth used only for scoring",
        fontsize=13,
    )
    fig.savefig(args.output / "trajectory.png", dpi=160)
    plt.close(fig)
    # Include the first measured-hidden interval if present; otherwise show the run.
    selected = np.flatnonzero(hidden)
    center = times[selected[0]] if len(selected) else times[len(times) // 2]
    indices = np.flatnonzero((times >= center - 1.5) & (times <= center + 3))
    frames = []
    clip_indices = list(indices[::2])
    if len(indices) and clip_indices[-1] != indices[-1]:
        clip_indices.append(indices[-1])
    for index in clip_indices:
        row = rows[index]
        source = Image.open(args.case / row["rgb_path"]).convert("RGB")
        source = source.crop(crop_box).resize((768, 576), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (768, 636), "#142033")
        canvas.paste(source, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (14, 588),
            f"{case['id']} | t={row['time']:.2f}s | {row['controller_status']}",
            fill="white",
        )
        draw.text(
            (14, 610),
            f"Native RGB crop | scoring: head pixels={row['head_pixels_scoring_only']} | injected omission={row['dropout_injected']}",
            fill="#a8c7e0",
        )
        frames.append(canvas)
    if frames:
        frames[0].save(
            args.output / "native-clip.gif",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
