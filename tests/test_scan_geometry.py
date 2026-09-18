import numpy as np
import pytest

from bb8_rl.mapping.scan_geometry import (
    distinct_landmark_matches,
    fundamental,
    project,
    triangulate_track,
)


def cameras():
    k = np.array([[800.0, 0, 640], [0, 800, 480], [0, 0, 1]])
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[:, 2, 3] = 3
    transforms[1, 0, 3], transforms[2, 1, 3] = -0.7, -0.5
    return np.repeat(k[None], 3, axis=0), transforms


def test_metric_triangulation_and_epipolar_constraint():
    ks, ts = cameras()
    point = np.array([0.25, -0.15, 0.4])
    pixels = np.array([project(point, k, t)[0][0] for k, t in zip(ks, ts)])
    solved = triangulate_track(pixels, ks, ts)
    np.testing.assert_allclose(solved[0], point, atol=1e-12)
    assert max(solved[1]) < 1e-10
    assert solved[2] > 5
    f = fundamental(ks[0], ts[0], ks[1], ts[1])
    assert abs(np.r_[pixels[1], 1] @ f @ np.r_[pixels[0], 1]) < 1e-10


def test_wrong_correspondence_and_weak_baseline_do_not_create_geometry():
    ks, ts = cameras()
    pixels = np.array([project([0.2, 0.1, 0.3], k, t)[0][0] for k, t in zip(ks, ts)])
    pixels[2] += [15, 12]
    assert triangulate_track(pixels, ks, ts) is None
    ts[:] = ts[0]
    pixels[:] = pixels[0]
    assert triangulate_track(pixels, ks, ts) is None


def test_negative_depth_and_nonfinite_input_rejected():
    ks, ts = cameras()
    pixels = np.array([project([0.2, 0.1, -4], k, t)[0][0] for k, t in zip(ks, ts)])
    assert triangulate_track(pixels, ks, ts) is None
    pixels[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        triangulate_track(pixels, ks, ts)


def test_almost_opposite_collinear_rays_are_not_strong_parallax():
    ks, ts = cameras()
    ts[1] = np.diag([1.0, -1.0, -1.0, 1.0])
    ts[1, 2, 3] = 3
    pixels = np.array(
        [project([0.001, 0, 0], k, t)[0][0] for k, t in zip(ks[:2], ts[:2])]
    )
    assert triangulate_track(pixels, ks[:2], ts[:2]) is None


def test_ratio_search_reaches_a_distinct_landmark_beyond_first_eight_rows():
    query = np.zeros((1, 128), np.float32)
    descriptors = np.repeat(query, 11, axis=0)
    descriptors[-1] = 10
    ids = np.array([0] * 10 + [1])
    assert distinct_landmark_matches(query, descriptors, ids) == [(0, 0, 0)]
    assert not distinct_landmark_matches(query, descriptors[:10], ids[:10])
