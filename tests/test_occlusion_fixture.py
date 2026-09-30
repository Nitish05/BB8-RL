import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "geometric_probe",
    Path(__file__).resolve().parents[1] / "scripts/probe-geometric-occlusion.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "x,hidden",
    [
        (-1.6, (False, False)),
        (-0.7, (True, False)),
        (0, (True, True)),
        (0.7, (False, True)),
        (1.6, (False, False)),
    ],
)
def test_box_shadow_covers_entire_head_then_releases_it(x, hidden):
    assert (
        tuple(
            probe.head_cube_hidden(camera, (x, 0)) for camera in probe.CAMERAS.values()
        )
        == hidden
    )
    if hidden == (False, False):
        # These endpoints also have unobstructed center rays; "not fully
        # hidden" alone would not demonstrate reappearance.
        assert not any(
            probe.ray_hits_box(camera, (x, 0, 0.083), box)
            for camera in probe.CAMERAS.values()
            for box in probe.BOXES
        )


def test_prescribed_lane_is_clear_of_boxes_and_walls():
    for box in probe.BOXES:
        assert abs(box["position"][1]) - box["size"][1] / 2 == pytest.approx(0.4)
    assert 2 - max(abs(probe.START[0]), abs(probe.GOAL[0])) > 0.147


def test_command_schedule_is_time_only_with_bounded_braking_phase():
    for time, expected in (
        (0, [0, 0]),
        (0.5, [0.8, 0]),
        (12.49, [0.8, 0]),
        (12.5, [0, 0]),
        (13.5, [0, 0]),
    ):
        np.testing.assert_array_equal(probe.prescribed_command(time), expected)
    with pytest.raises(ValueError):
        probe.prescribed_command(float("nan"))


def test_weak_head_is_not_reported_as_fully_hidden():
    assert [probe.visibility_class(value) for value in (0, 1, 2, 3)] == [
        "fully_hidden",
        "weak",
        "weak",
        "visible",
    ]


def test_summary_counts_hidden_false_accepts_and_reappearance_denominators():
    rows = []
    for index, (pixels, detected) in enumerate(
        ((12, True), (0, True), (0, False), (2, False), (10, False), (11, True))
    ):
        view = {
            "head_pixels_scoring_only": pixels,
            "truth_visibility_scoring_only": probe.visibility_class(pixels),
            "status": "visible" if detected else "missing",
            "error_m_scoring_only": 0.01 if detected else None,
        }
        rows.append(
            {
                "time": index * 0.1,
                "views": {"A": dict(view), "B": dict(view)},
                "rig": {"position": [0, 0] if detected else None, "handover": False},
            }
        )
    summary = probe.summarize(rows)
    assert summary["views"]["A"]["fully_hidden_frames"] == 2
    assert summary["views"]["A"]["fully_hidden_false_accepts"] == 1
    assert summary["views"]["A"]["visible_head_misses"] == 1
    assert summary["views"]["A"]["weak_1_to_2_pixel_frames"] == 1
    assert summary["both_hidden_rig_false_accepts"] == 1
    assert summary["views"]["A"]["reappearances"][0]["delay_seconds"] == pytest.approx(
        0.1
    )


@pytest.mark.studio
def test_fixture_writes_valid_separate_world_without_native_initialization(tmp_path):
    import json

    import yaml

    probe.create_fixture(tmp_path / "fixture")
    project = json.loads((tmp_path / "fixture/room.genesis.json").read_text())
    boxes = [
        obj for obj in project["objects"] if obj["name"].startswith("room_obstacle")
    ]
    assert len(boxes) == 2
    assert all(obj["size"][2] == 1.2 and obj["fixed"] for obj in boxes)
    config = yaml.safe_load((tmp_path / "fixture/room.yaml").read_text())
    assert config["arena_half_extent"] == 2
    assert config["obstacle_names"] == [obj["name"] for obj in boxes]
