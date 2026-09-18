"""Score a finished reconstruction against a synthetic scene, without fitting it.

Truth is used only here, after the RGB backend has written its immutable artifact.
Surface precision and sampled obstacle completeness are different measurements.
Neither turns sparse or learned surfaces into certified navigable free volume.
"""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_surface_distance(points, center, size):
    """Unsigned distance to four side rectangles and the top; never the bottom."""
    low, high = (
        np.asarray(center) - np.asarray(size) / 2,
        np.asarray(center) + np.asarray(size) / 2,
    )
    distance = np.full(len(points), np.inf)
    for axis in range(3):
        for face in (low[axis], high[axis]):
            if axis == 2 and face == low[axis]:
                continue
            closest = np.clip(points, low, high)
            closest[:, axis] = face
            distance = np.minimum(distance, np.linalg.norm(points - closest, axis=1))
    return distance


def validate_boxes(boxes, room_extent=2.0):
    """Require ground-supported axis-aligned boxes without buried room faces.

    Wall intersections outside the scored primary room are allowed. Their
    individual faces are not an exact union-surface metric outside that room.
    """
    bounds = []
    for box in boxes:
        center, size = (
            np.asarray(box["position"], float),
            np.asarray(box["size"], float),
        )
        euler = np.asarray(box.get("euler", [0, 0, 0]), float)
        if (
            center.shape != (3,)
            or size.shape != (3,)
            or euler.shape != (3,)
            or not np.isfinite(np.r_[center, size, euler]).all()
            or np.any(size <= 0)
            or not np.allclose(euler, 0, rtol=0, atol=1e-10)
            or not np.isclose(center[2] - size[2] / 2, 0, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                "Scoring requires finite axis-aligned boxes resting on z=0"
            )
        bounds.append((center - size / 2, center + size / 2))
    for i, (low, high) in enumerate(bounds):
        for j in range(i):
            other_low, other_high = bounds[j]
            overlap_low, overlap_high = (
                np.maximum(low, other_low),
                np.minimum(high, other_high),
            )
            widths = overlap_high - overlap_low
            # Shared faces are buried too; edge/point contacts have zero area.
            if np.any(widths < -1e-10) or np.count_nonzero(widths > 1e-10) < 2:
                continue
            interior = np.all(overlap_high[:2] > -room_extent + 1e-10) and np.all(
                overlap_low[:2] < room_extent - 1e-10
            )
            walls = all(
                str(boxes[k].get("name", "")).startswith("wall_") for k in (i, j)
            )
            if interior or not walls:
                raise ValueError(
                    "Boxes overlap or share a face; only wall intersections outside primary room are supported"
                )


def exposed_floor_distance(points, boxes):
    """Exact distance to z=0 outside the union of axis-aligned box footprints."""
    if not boxes:
        return abs(points[:, 2])
    lows = np.array(
        [np.asarray(b["position"][:2]) - np.asarray(b["size"][:2]) / 2 for b in boxes]
    )
    highs = np.array(
        [np.asarray(b["position"][:2]) + np.asarray(b["size"][:2]) / 2 for b in boxes]
    )
    inside = np.zeros(len(points), bool)
    for low, high in zip(lows, highs):
        inside |= np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)
    horizontal = np.zeros(len(points))
    if not inside.any():
        return abs(points[:, 2])
    xs, ys = [np.unique(np.r_[lows[:, axis], highs[:, axis]]) for axis in range(2)]
    xx, yy = np.meshgrid((xs[1:] + xs[:-1]) / 2, (ys[1:] + ys[:-1]) / 2, indexing="ij")
    occupied = np.zeros(xx.shape, bool)
    for low, high in zip(lows, highs):
        occupied |= (xx > low[0]) & (xx < high[0]) & (yy > low[1]) & (yy < high[1])
    selected = points[inside, :2]
    distance = np.full(len(selected), np.inf)
    for i, j in np.argwhere(occupied):
        for axis, delta in ((0, -1), (0, 1), (1, -1), (1, 1)):
            neighbor = [i, j]
            neighbor[axis] += delta
            if (
                0 <= neighbor[0] < occupied.shape[0]
                and 0 <= neighbor[1] < occupied.shape[1]
                and occupied[tuple(neighbor)]
            ):
                continue
            low, high = np.array([xs[i], ys[j]]), np.array([xs[i + 1], ys[j + 1]])
            edge = (low if delta < 0 else high)[axis]
            closest = np.clip(selected, low, high)
            closest[:, axis] = edge
            distance = np.minimum(distance, np.linalg.norm(selected - closest, axis=1))
    horizontal[inside] = distance
    return np.hypot(points[:, 2], horizontal)


def nearest_surface(points, boxes):
    validate_boxes(boxes)
    distances = [exposed_floor_distance(points, boxes)]
    for box in boxes:
        distances.append(
            box_surface_distance(
                points, np.array(box["position"]), np.array(box["size"])
            )
        )
    values = np.array(distances)
    return values.min(0), values.argmin(0)


def sample_exterior(box, spacing=0.05):
    low = np.array(box["position"]) - np.array(box["size"]) / 2
    high = low + box["size"]
    samples = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        grids = [
            np.linspace(
                low[i], high[i], max(2, int(np.ceil((high[i] - low[i]) / spacing)) + 1)
            )
            for i in others
        ]
        for sign in (0, 1):
            if axis == 2 and sign == 0:
                continue  # underside sits on ground and is not observable
            for a, b in product(*grids):
                point = np.empty(3)
                point[axis] = (low, high)[sign][axis]
                point[others] = a, b
                samples.append(point)
    return np.unique(samples, axis=0)


def summarize(errors):
    if not len(errors):
        return {"points": 0}
    return {
        "points": len(errors),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.quantile(errors, 0.95)),
        "max_m": float(max(errors)),
        "within_2cm_fraction": float(np.mean(errors <= 0.02)),
        "within_5cm_fraction": float(np.mean(errors <= 0.05)),
    }


def plot(points, colors, boxes, output, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    inside = (
        (abs(points[:, :2]).max(1) < 2.2)
        & (points[:, 2] >= -0.25)
        & (points[:, 2] < 1.5)
    )
    selected, rgb = points[inside], colors[inside]
    stride = max(1, len(selected) // 50000)
    selected, rgb = selected[::stride], rgb[::stride]
    fig = plt.figure(figsize=(13, 6.7), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(*selected.T, c=rgb / 255, s=1, rasterized=True)
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
        xlim=(-2.2, 2.2),
        ylim=(-2.2, 2.2),
        zlim=(-0.15, 0.8),
    )
    ax.set_box_aspect((4.4, 4.4, 1.7))
    ax.view_init(elev=31, azim=-64)
    ax2 = fig.add_subplot(122)
    elevated = selected[:, 2] > 0.05
    dots = ax2.scatter(
        selected[elevated, 0],
        selected[elevated, 1],
        c=selected[elevated, 2],
        cmap="viridis",
        s=2,
        vmin=0,
        vmax=0.5,
    )
    for box in boxes:
        center, size = np.array(box["position"]), np.array(box["size"])
        lo, hi = center - size / 2, center + size / 2
        corners = np.array(list(product(*zip(lo, hi))))
        for i, a in enumerate(corners):
            for b in corners[i + 1 :]:
                if np.count_nonzero(a != b) == 1:
                    ax.plot(
                        *np.stack([a, b]).T, color="#555555", alpha=0.4, linewidth=0.6
                    )
        if box["name"].startswith("room_obstacle"):
            ax2.add_patch(
                Rectangle(
                    lo[:2],
                    size[0],
                    size[1],
                    fill=False,
                    edgecolor="#555555",
                    linewidth=1,
                )
            )
    fig.colorbar(dots, ax=ax2, label="estimated surface height (m)", shrink=0.7)
    ax2.set(
        xlim=(-2.2, 2.2),
        ylim=(-2.2, 2.2),
        xlabel="x (m)",
        ylabel="y (m)",
        title="Elevated estimates; gray outlines are scoring truth",
        aspect="equal",
    )
    fig.suptitle(
        title
        + "\nRGB + known synthetic scan calibration; no depth or obstacle labels used for reconstruction",
        fontsize=13,
    )
    fig.savefig(output / "geometry.png", dpi=170)
    plt.close(fig)


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh evaluation output")
    args.output.mkdir(parents=True)
    arrays = np.load(args.reconstruction)
    if args.field == "points":
        points, colors = arrays["points"], arrays["colors"]
    else:
        keep = arrays["valid_mask"]
        points, colors = arrays[args.field][keep], arrays["rgb_uint8"][keep]
    finite = np.isfinite(points).all(1)
    points, colors = points[finite], colors[finite]
    scene = json.loads(args.truth_scene.read_text())
    boxes = [o for o in scene["objects"] if o["format"] == "box" and o.get("fixed")]
    errors, nearest = nearest_surface(points, boxes)
    inside = abs(points[:, :2]).max(1) < 2
    elevated = inside & (points[:, 2] > 0.05)
    tree = cKDTree(points) if len(points) else None
    coverage = []
    for box in boxes:
        if not box["name"].startswith("room_obstacle"):
            continue
        samples = sample_exterior(box)
        distances = (
            tree.query(samples)[0]
            if tree is not None
            else np.repeat(float("inf"), len(samples))
        )
        coverage.append(
            {
                "name": box["name"],
                "sample_count": len(samples),
                "fraction_with_reconstructed_point_within_2cm": float(
                    np.mean(distances <= 0.02)
                ),
                "fraction_with_reconstructed_point_within_5cm": float(
                    np.mean(distances <= 0.05)
                ),
            }
        )
    report = {
        "scope": "scoring only; no alignment, scale fitting, outlier fitting or reconstruction refinement using truth",
        "reconstruction_sha256": digest(args.reconstruction),
        "truth_scene_sha256": digest(args.truth_scene),
        "field": args.field,
        "surface_metric": "Exposed floor outside footprint union plus four box sides and tops; box undersides excluded. Axis-aligned boxes rest on z=0 and neither overlap nor share faces inside primary room xy (-2,2). Permitted exterior wall intersections use individual wall faces, so all-point distances outside the primary room are not exact union-surface distances.",
        "surface_distance_all": summarize(errors),
        "surface_distance_in_room_xy": summarize(errors[inside]),
        "surface_distance_elevated_in_room_xy": summarize(errors[elevated]),
        "closest_surface_is_ground_fraction": float(np.mean(nearest == 0))
        if len(nearest)
        else None,
        "obstacle_exterior_completeness": coverage,
        "coverage_note": "Samples every <=5cm on four sides and top, including surfaces potentially unobserved in this subset; excludes underside.",
        "robot_in_scan": "Parked BB8 is not scored as a static room surface; its samples may be outliers here.",
        "free_space_certified": False,
        "nonfinite_points_rejected": int((~finite).sum()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot(points, colors, boxes, args.output, args.title)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--field", default="points")
    parser.add_argument(
        "--truth-scene",
        type=Path,
        default=Path("projects/bb8/synthetic-room/room.genesis.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Calibrated sparse scan reconstruction")
    main(parser.parse_args())
