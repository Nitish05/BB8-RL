"""Serialize and draw analytic room-memory fixtures; never reconstruct RGB or drive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bb8_rl.mapping import (
    Bounds3D,
    EvidenceSource,
    ObjectBounds,
    Provenance,
    RoomMemory,
    SpaceState,
)

COLORS = {
    SpaceState.FREE: "#218b72",
    SpaceState.OCCUPIED: "#d78223",
    SpaceState.UNKNOWN: "#a2aab8",
}
CORRIDOR = Bounds3D((-0.95, -0.3, 0), (0.95, 0.3, 0.3))
CORRIDOR_QUERY = Bounds3D((-0.75, -0.05, 0), (0.75, 0.05, 0.15))
OBJECT = Bounds3D((0.2, 0.35, 0), (0.5, 0.7, 0.25))
OBJECT_QUERY = Bounds3D((0.3, 0.45, 0.02), (0.4, 0.55, 0.1))


def source(name, observed_at=1.0):
    return EvidenceSource(
        source_id=name,
        view_ids=(f"{name}-view-001", f"{name}-view-002"),
        observed_at=observed_at,
        provenance=Provenance.ANALYTIC_SYNTHETIC,
    )


def new_room():
    return RoomMemory(
        Bounds3D((-1, -1, 0), (1, 1, 0.5)),
        0.05,
        scene_version="analytic-static-v1",
        calibration_version="analytic-registration-v1",
    )


def add_memory(memory, name="scan"):
    memory.observe_volume(CORRIDOR, SpaceState.FREE, source(name))
    memory.remember_object(
        ObjectBounds("remembered-box", OBJECT, source(f"{name}-box"))
    )


def fixtures():
    visible = new_room()
    add_memory(visible, "current-view")
    hidden = new_room()
    add_memory(hidden, "prior-scan")
    never_observed = new_room()
    never_observed.remember_object(
        ObjectBounds("observed-box", OBJECT, source("box-only"))
    )
    fork = new_room()
    fork.observe_volume(
        Bounds3D((-0.95, -0.8, 0), (0.95, -0.2, 0.3)),
        SpaceState.FREE,
        source("south-branch"),
    )
    fork.observe_volume(
        Bounds3D((0.5, 0.2, 0), (0.95, 0.8, 0.3)),
        SpaceState.FREE,
        source("north-endpoint"),
    )
    moved_object = new_room()
    add_memory(moved_object)
    moved_object.invalidate_if_changed(
        scene_version="analytic-object-moved-v2",
        calibration_version="analytic-registration-v1",
        at_time=3.0,
    )
    moved_camera = new_room()
    add_memory(moved_camera)
    moved_camera.invalidate_if_changed(
        scene_version="analytic-static-v1",
        calibration_version="analytic-registration-moved-v2",
        at_time=3.0,
    )
    return [
        (
            "01-visible",
            "Visible corridor",
            "Explicit current-view volume evidence",
            visible,
            SpaceState.FREE,
        ),
        (
            "02-remembered-hidden",
            "Remembered, currently hidden",
            "Prior scan evidence; no new observation",
            hidden,
            SpaceState.FREE,
        ),
        (
            "03-never-observed",
            "Never observed corridor",
            "Known object size does not certify empty space",
            never_observed,
            SpaceState.UNKNOWN,
        ),
        (
            "04-ambiguous-fork",
            "Ambiguous fork",
            "South branch FREE; north connection UNKNOWN",
            fork,
            SpaceState.UNKNOWN,
        ),
        (
            "05-changed-obstacle",
            "Object moved: memory invalid",
            "Historical evidence retained; queries UNKNOWN",
            moved_object,
            SpaceState.UNKNOWN,
        ),
        (
            "06-changed-camera",
            "Camera moved: registration invalid",
            "Re-registration required; queries UNKNOWN",
            moved_camera,
            SpaceState.UNKNOWN,
        ),
    ]


def query_record(name, memory, bounds, expected):
    actual = memory.query(bounds)
    return {
        "name": name,
        "bounds_m": bounds.to_dict(),
        "expected": expected.value,
        "actual": actual.value,
        "passed": actual is expected,
    }


def prism(axis, bounds, *, color, alpha, linestyle="solid"):
    lo, hi = bounds.minimum, bounds.maximum
    vertices = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    faces = [
        vertices[index]
        for index in (
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        )
    ]
    axis.add_collection3d(
        Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=0.7,
            linestyle=linestyle,
        )
    )


def draw_scene(axis, name, title, subtitle, memory):
    # Display the actual query state of small complete body-height volumes.
    # No private occupancy buffers or hand-coded free-space labels are used.
    locations = {state: [] for state in SpaceState}
    for x in np.linspace(-0.9, 0.9, 19):
        for y in np.linspace(-0.9, 0.9, 19):
            state = memory.query(
                Bounds3D((x - 0.01, y - 0.01, 0), (x + 0.01, y + 0.01, 0.15))
            )
            locations[state].append([x, y, 0.003])
    for state, points in locations.items():
        if points:
            points = np.asarray(points)
            axis.scatter(
                *points.T, color=COLORS[state], s=9, alpha=0.8, depthshade=False
            )
    for volume in memory.evidence:
        prism(
            axis,
            volume.bounds,
            color=COLORS[volume.state] if memory.valid else COLORS[SpaceState.UNKNOWN],
            alpha=0.07,
            linestyle="solid" if memory.valid else "dashed",
        )
    for obj in memory.objects:
        prism(
            axis,
            obj.bounds,
            color=COLORS[SpaceState.OCCUPIED] if memory.valid else "#b6bbc5",
            alpha=0.40 if memory.valid else 0.10,
        )
    if name == "02-remembered-hidden":
        axis.text(
            0.15,
            0.72,
            0.28,
            "Stored box\n0.30 × 0.35 × 0.25 m",
            fontsize=8,
            color="#945411",
        )
    ys = [-0.5, 0.5] if name == "04-ambiguous-fork" else [0]
    for y in ys:
        allowed = memory.segment_free((-0.7, y), (0.7, y), radius_m=0.05, height_m=0.15)
        axis.plot(
            [-0.7, 0.7],
            [y, y],
            [0.075, 0.075],
            color="#087960" if allowed else "#c45157",
            linewidth=3,
            linestyle="-" if allowed else "--",
        )
    axis.set_title(title, fontsize=12, fontweight="bold", color="#1c2a41", pad=7)
    axis.text2D(
        0.5,
        0.97,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color="#53627a",
    )
    axis.set(
        xlim=(-1, 1),
        ylim=(-1, 1),
        zlim=(0, 0.5),
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    axis.set_xticks([-1, 0, 1])
    axis.set_yticks([-1, 0, 1])
    axis.set_zticks([0, 0.25, 0.5])
    axis.tick_params(labelsize=8)
    axis.set_box_aspect((1, 1, 0.35))
    axis.view_init(elev=30, azim=-62)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor("#f5f7fa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    scenes = fixtures()
    figure = plt.figure(figsize=(18, 11), facecolor="white")
    figure.suptitle(
        "3D room memory: what remains known when the camera cannot see",
        fontsize=21,
        fontweight="bold",
        x=0.5,
        y=0.975,
        color="#1c2a41",
    )
    figure.text(
        0.5,
        0.935,
        "Analytic synthetic contract fixtures — not reconstructed from RGB",
        ha="center",
        fontsize=13,
        color="#9a5b23",
    )
    for index, (name, title, subtitle, memory, expected) in enumerate(scenes):
        path = args.output / f"{name}.json"
        memory.save(path)
        # All visualizations and checks use the round-tripped memory.
        loaded = RoomMemory.load(path)
        checks = [query_record("corridor-volume", loaded, CORRIDOR_QUERY, expected)]
        if name == "04-ambiguous-fork":
            checks.extend(
                [
                    query_record(
                        "south-branch",
                        loaded,
                        Bounds3D((-0.75, -0.55, 0), (0.75, -0.45, 0.15)),
                        SpaceState.FREE,
                    ),
                    query_record(
                        "north-branch",
                        loaded,
                        Bounds3D((-0.75, 0.45, 0), (0.75, 0.55, 0.15)),
                        SpaceState.UNKNOWN,
                    ),
                ]
            )
        else:
            checks.append(
                query_record(
                    "object-volume",
                    loaded,
                    OBJECT_QUERY,
                    SpaceState.OCCUPIED if loaded.valid else SpaceState.UNKNOWN,
                )
            )
        expected_segment = expected is SpaceState.FREE
        actual_segment = loaded.segment_free(
            (-0.7, 0), (0.7, 0), radius_m=0.05, height_m=0.15
        )
        checks.append(
            {
                "name": "swept-body-corridor",
                "expected": expected_segment,
                "actual": actual_segment,
                "passed": actual_segment == expected_segment,
            }
        )
        checks.append(
            {
                "name": "serialization-round-trip",
                "expected": True,
                "actual": loaded.to_dict() == memory.to_dict(),
                "passed": loaded.to_dict() == memory.to_dict(),
            }
        )
        rows.append(
            {
                "fixture": name,
                "description": subtitle,
                "memory_file": path.name,
                "valid": loaded.valid,
                "map_version": loaded.map_version,
                "scene_version": loaded.scene_version,
                "calibration_version": loaded.calibration_version,
                "queries": checks,
                "objects": [
                    {
                        "object_id": obj.object_id,
                        "size_m": list(obj.bounds.size),
                        "source": obj.source.to_dict(),
                    }
                    for obj in loaded.objects
                ],
            }
        )
        draw_scene(
            figure.add_subplot(2, 3, index + 1, projection="3d"),
            name,
            title,
            subtitle,
            loaded,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=COLORS[state],
            markersize=8,
            label=state.value.title(),
        )
        for state in SpaceState
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#087960",
                linewidth=3,
                label="Swept-volume query passes",
            ),
            Line2D(
                [0],
                [0],
                color="#c45157",
                linewidth=3,
                linestyle="--",
                label="Swept-volume query rejects",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=11,
    )
    figure.text(
        0.5,
        0.018,
        "Floor dots show queried body-height volumes. Transparent prisms retain evidence. Invalidated maps return UNKNOWN; they do not regain permission from old geometry.",
        ha="center",
        fontsize=10,
        color="#53627a",
    )
    figure.subplots_adjust(
        top=0.88, bottom=0.12, left=0.03, right=0.97, hspace=0.20, wspace=0.03
    )
    figure.savefig(
        args.output / "room-memory-contracts.png", dpi=150, facecolor="white"
    )
    plt.close(figure)
    checks = [check for row in rows for check in row["queries"]]
    report = {
        "scope": "analytic synthetic contract fixtures — not reconstructed from RGB",
        "native_simulation": False,
        "actuation": False,
        "resolution_m": 0.05,
        "query_body_height_m": 0.15,
        "query_envelope_radius_m": 0.05,
        "fixtures": rows,
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "all_passed": all(check["passed"] for check in checks),
        "visualization": "room-memory-contracts.png",
        "limitations": [
            "Geometry and provenance are supplied analytically, not inferred from images.",
            "Camera/scene changes are explicitly signaled, not automatically detected.",
            "Free volumes need an upstream evidence producer; unknown space never becomes free by absence of objects.",
            "Object boxes and swept bounding prisms are conservative approximations, not full reconstructed meshes or motion plans.",
            "Prior scan evidence persists only under the static-scene and valid-registration assumptions.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "checks_passed": report["checks_passed"],
                "checks_total": len(checks),
                "all_passed": report["all_passed"],
            }
        )
    )
    if not report["all_passed"]:
        raise SystemExit("One or more analytic contract checks failed")


if __name__ == "__main__":
    main()
