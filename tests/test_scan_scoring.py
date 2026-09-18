import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scan_scoring", ROOT / "scripts/evaluate-scan-geometry.py"
)
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


def box(position=(0, 0, 0.5), size=(1, 1, 1), name="room_obstacle_test"):
    return {"name": name, "position": list(position), "size": list(size)}


def test_underside_and_hallucinated_floor_inside_solid_receive_no_surface_credit():
    points = np.array([[0, 0, 0], [0, 0, 0.01], [0, 0, 0.25], [0, 0, -0.01]])
    distances, _ = scoring.nearest_surface(points, [box()])
    np.testing.assert_allclose(distances, [0.5, 0.5, 0.5, np.hypot(0.5, 0.01)])


def test_true_sides_top_and_exposed_floor_still_receive_zero_error():
    points = np.array([[0.5, 0, 0.2], [0, 0, 1], [0.8, 0, 0], [0.5, 0, 0]])
    distances, _ = scoring.nearest_surface(points, [box()])
    np.testing.assert_allclose(distances, 0)


def test_distance_to_finite_side_and_top_rectangles_is_euclidean():
    points = np.array([[0.8, 0.9, 0.5], [0, 0, 1.3], [0.8, 0, -0.4]])
    np.testing.assert_allclose(
        scoring.box_surface_distance(points, np.array([0, 0, 0.5]), np.ones(3)),
        [0.5, 0.3, 0.5],
    )


def test_exposed_floor_uses_union_boundary_even_for_overlapping_footprints():
    boxes = [box((-0.25, 0, 0.5)), box((0.25, 0, 0.5))]
    points = np.array([[0, 0, 0], [0, 0, 0.3], [1, 0, -0.2]])
    # Distance to either individual edge is .25, but neither is exposed floor.
    np.testing.assert_allclose(
        scoring.exposed_floor_distance(points, boxes), [0.5, np.hypot(0.5, 0.3), 0.2]
    )


def test_exposed_floor_does_not_treat_shared_edge_as_visible():
    boxes = [box((-0.5, 0, 0.5)), box((0.5, 0, 0.5))]
    np.testing.assert_allclose(
        scoring.exposed_floor_distance(np.array([[0, 0, 0]]), boxes), [0.5]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"euler": [0, 0, 0.1]},
        {"position": [0, 0, 0.6]},
        {"size": [1, -1, 1]},
        {"position": [0, 0, float("nan")]},
    ],
)
def test_unsupported_box_geometry_is_rejected(change):
    candidate = box()
    candidate.update(change)
    with pytest.raises(ValueError, match="axis-aligned"):
        scoring.nearest_surface(np.zeros((1, 3)), [candidate])


def test_overlapping_room_interiors_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        scoring.validate_boxes([box(), box((0.25, 0, 0.5))])


def test_shared_interior_box_face_is_rejected_as_unexposed():
    with pytest.raises(ValueError, match="share a face"):
        scoring.validate_boxes([box((-0.5, 0, 0.5)), box((0.5, 0, 0.5))])


def test_only_wall_intersections_outside_primary_room_are_allowed():
    walls = [
        box((2.1, 0, 0.5), (0.2, 4.4, 1), "wall_x"),
        box((0, 2.1, 0.5), (4.4, 0.2, 1), "wall_y"),
    ]
    scoring.validate_boxes(walls)
    walls[0]["position"][0] = 1.95
    walls[1]["position"][1] = 1.95
    with pytest.raises(ValueError, match="overlap"):
        scoring.validate_boxes(walls)


def test_current_authored_scene_meets_geometry_contract():
    scene = json.loads(
        (ROOT / "projects/bb8/synthetic-room/room.genesis.json").read_text()
    )
    scoring.validate_boxes(
        [b for b in scene["objects"] if b["format"] == "box" and b.get("fixed")]
    )


def test_empty_cloud_and_floor_only_cloud_are_supported():
    distances, nearest = scoring.nearest_surface(np.empty((0, 3)), [box()])
    assert distances.size == nearest.size == 0
    distances, nearest = scoring.nearest_surface(np.array([[0, 0, -0.3]]), [])
    np.testing.assert_allclose(distances, [0.3])
    assert nearest.tolist() == [0]
