import numpy as np
import pytest

from bb8_rl.mapping.depth_calibration import (
    apply_depth_transform,
    choose_depth_transform,
    deduplicate_rgb_tracks,
    fit_depth_transform,
    sample_bilinear,
    triangulated_rgb_anchors,
)


def test_bilinear_masks_holes_instead_of_crossing_them():
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = np.ones((2, 2), bool)
    assert sample_bilinear(z, m, [[0.5, 0.5]])[0] == pytest.approx(2.5)
    m[1, 1] = False
    assert np.isnan(sample_bilinear(z, m, [[0.5, 0.5]])[0])


def test_affine_depth_fit_recovers_known_relation_with_outlier():
    x = np.linspace(2, 9, 80)
    y = 1.07 * x - 0.12
    y[15] += 10
    t = fit_depth_transform(x, y, kind="affine")
    assert t["scale"] == pytest.approx(1.07, abs=1e-5)
    assert t["offset"] == pytest.approx(-0.12, abs=1e-5)


def test_inverse_depth_fit_and_application():
    x = np.linspace(2, 9, 40)
    y = 1 / (0.9 / x + 0.01)
    t = fit_depth_transform(x, y, kind="inverse_affine")
    np.testing.assert_allclose(apply_depth_transform(x, t), y, atol=1e-7)


def test_validation_prevents_bad_training_fit_from_being_selected():
    x = np.linspace(2, 8, 30)
    y = x.copy()
    y[:20] *= 1.3
    validation = np.arange(30) >= 20
    t, _ = choose_depth_transform(x, y, validation)
    assert t["kind"] == "identity"


def test_queries_rejected_and_insufficient_validation_stays_identity():
    with pytest.raises(ValueError, match="Frozen query"):
        triangulated_rgb_anchors([], [14])
    t, r = choose_depth_transform(np.ones(8), np.ones(8), np.zeros(8, bool))
    assert t["kind"] == "identity"
    assert r["status"] == "insufficient_independent_validation"


def test_spatial_duplicate_landmark_observations_share_one_fold():
    tracks = [
        {"point": [0.0, 0.0, 1.0], "view_ids": [0, 1, 2], "reprojection_max_px": 0.1},
        {
            "point": [0.001, 0.0, 1.0],
            "view_ids": [0, 1, 2, 3],
            "reprojection_max_px": 0.2,
        },
        {"point": [1.0, 0.0, 1.0], "view_ids": [0, 1, 2], "reprojection_max_px": 0.1},
    ]
    kept = deduplicate_rgb_tracks(tracks)
    assert len(kept) == 2
    assert len(kept[0]["view_ids"]) == 4
    repeated = deduplicate_rgb_tracks(list(reversed(tracks)))
    assert [(t["point"], t["validation"]) for t in kept] == [
        (t["point"], t["validation"]) for t in repeated
    ]
