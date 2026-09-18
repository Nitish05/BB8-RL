import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "rig_route_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit-rig-routes.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ([-2, 0], [2, 0], 0),
        ([0, 0], [0.1, 0.1], 0),
        ([-1, 0.7], [1, 0.7], 0.2),
        ([0.6, 0.6], [1, 1], np.sqrt(0.02)),
        ([0.6, 0.6], [0.6, 0.6], np.sqrt(0.02)),
        ([-1, 0.5], [1, 0.5], 0),
        ([0.7, -2], [0.7, 2], 0.2),
        ([-2, 0.7], [0.7, -2], np.sqrt(0.045)),
    ],
)
def test_continuous_segment_rectangle_clearance(start, end, expected):
    # The final diagonal passes near the lower-left corner; endpoint-only
    # checks would miss its much smaller interior-segment clearance.
    assert audit.segment_box_distance(start, end, [0, 0, 1, 1]) == pytest.approx(
        expected
    )
    assert audit.segment_box_distance(end, start, [0, 0, 1, 1]) == pytest.approx(
        expected
    )


def test_corner_clearance_is_euclidean_not_square_inflation():
    distance = audit.segment_box_distance([0.58, 0.58], [1, 1], [0, 0, 1, 1])
    assert distance > 0.10  # Both axis gaps are only8cm, but radial gap is11.3cm.


def test_physical_envelope_includes_inner_walls_and_rejects_outside():
    assert audit.segment_clearance([1.91, 0], [1.8, 1], []) == pytest.approx(0.09)
    assert audit.segment_clearance([2.1, 0], [2.2, 1], []) < 0


def test_physical_envelope_does_not_add_body_radius_twice():
    boxes = [[0, 0, 1, 1]]
    assert audit.segment_clearance([0.61, -1], [0.61, 1], boxes) > audit.CLEARANCE
    assert audit.segment_clearance([0.59, -1], [0.59, 1], boxes) < audit.CLEARANCE


@pytest.mark.parametrize("bad", [[0, 0, -1, 1], [0, 0, float("nan"), 1]])
def test_invalid_scoring_geometry_fails(bad):
    with pytest.raises(ValueError, match="positive-size box"):
        audit.segment_box_distance([0, 0], [1, 1], bad)
