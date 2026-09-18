import cv2
import numpy as np
import pytest

from bb8_rl.mapping.planar_completion import (
    CompletionParameters,
    complete_depth,
    validation_tiles,
)


def scene():
    y, x = np.indices((64, 64))
    depth = (1 / (0.5 + 0.001 * x + 0.0005 * y)).astype(np.float32)
    rgb = np.full((64, 64, 3), 100, np.uint8)
    anchors = (x >= 8) & (x <= 55) & (y >= 8) & (y <= 55)
    anchors &= (x % 3 == 0) & (y % 3 == 0)
    return rgb, depth, anchors


def test_plane_recovery_preserves_anchors_and_stays_in_fit_hull():
    rgb, depth, anchors = scene()
    untrusted = depth.copy()
    untrusted[~anchors] = 999  # This must never become geometric support.
    result, completed, report = complete_depth(rgb, untrusted, anchors)
    assert completed.sum() > 1000
    np.testing.assert_array_equal(result[anchors], depth[anchors])
    np.testing.assert_allclose(result[completed], depth[completed], atol=1e-6)
    assert not np.any(completed & anchors)
    assert np.all(np.isnan(result[~(anchors | completed)]))
    record = next(item for item in report["regions"] if item["status"] == "accepted")
    hull = np.array(record["fit_hull_xy"], np.float32)
    assert all(
        cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0
        for y, x in zip(*np.nonzero(completed), strict=True)
    )
    assert not completed[:8].any()
    assert report["free_space_certified"] is False


def test_color_hole_and_disconnected_region_never_get_completed():
    rgb, depth, anchors = scene()
    rgb[25:35, 25:35] = 240
    anchors[25:35, 25:35] = False
    rgb[:, 42:44] = 240
    anchors[:, 44:] = False
    _, completed, _ = complete_depth(rgb, depth, anchors)
    assert completed.any()
    assert not completed[25:35, 25:35].any()
    assert not completed[:, 42:].any()


def test_heldout_nonplanarity_rejects_even_perfect_fit_tiles():
    rgb, depth, anchors = scene()
    y, x = np.nonzero(anchors)
    heldout = validation_tiles(np.column_stack((x, y)))
    depth[y[heldout], x[heldout]] += 0.2
    result, completed, report = complete_depth(rgb, depth, anchors)
    assert not completed.any()
    assert report["regions"][0]["status"] == "validation_failed"
    np.testing.assert_array_equal(result[anchors], depth[anchors])


def test_collinear_support_is_rejected():
    rgb, depth, _ = scene()
    anchors = np.zeros(depth.shape, bool)
    anchors[24, :] = True
    _, completed, report = complete_depth(rgb, depth, anchors)
    assert not completed.any()
    assert report["regions"][0]["status"] == "degenerate_fit"


def test_insufficient_split_support_is_rejected():
    rgb, depth, anchors = scene()
    y, x = np.nonzero(anchors)
    heldout = validation_tiles(np.column_stack((x, y)))
    anchors[y[heldout][7:], x[heldout][7:]] = False
    _, completed, report = complete_depth(rgb, depth, anchors)
    assert not completed.any()
    assert report["regions"][0]["status"] == "insufficient_split_support"


def test_spatial_tile_split_has_no_pixel_leakage():
    y, x = np.indices((64, 64))
    xy = np.column_stack((x.ravel(), y.ravel()))
    selected = validation_tiles(xy)
    tile = xy // 8
    fit_tiles = set(map(tuple, tile[~selected]))
    validation = set(map(tuple, tile[selected]))
    assert not fit_tiles.intersection(validation)
    assert len(fit_tiles) == 48
    assert len(validation) == 16


def test_invalid_anchor_and_relaxed_gates_are_rejected():
    rgb, depth, anchors = scene()
    depth[anchors] = np.nan
    with pytest.raises(ValueError, match="finite positive"):
        complete_depth(rgb, depth, anchors)
    with pytest.raises(ValueError, match="stricter"):
        CompletionParameters(maximum_p95_m=0.06)
