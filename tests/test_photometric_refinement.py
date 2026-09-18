import cv2
import numpy as np

from bb8_rl.mapping.photometric_refinement import (
    refine,
    resample_prior,
    resized_intrinsics,
    select_depth,
    warped_correlation,
)


def test_resized_intrinsics_preserve_pixel_center_coordinates():
    k = np.array([[865.94, 0, 640], [0, 865.94, 480], [0, 0, 1]])
    new_k = resized_intrinsics(k, (1280, 960), (640, 480))
    np.testing.assert_allclose(new_k[:2, 2], [319.75, 239.75])
    points = np.array([[0.2, -0.5, 2], [-1, 0.6, 3]])
    native = points @ k.T
    resized = points @ new_k.T
    np.testing.assert_allclose(
        resized[:, :2] / resized[:, 2:],
        (native[:, :2] / native[:, 2:] + 0.5) * 0.5 - 0.5,
    )


def test_unique_minimum_interpolates_depth_but_ambiguous_surface_is_rejected():
    offsets = np.arange(-5, 6) * 0.1
    costs = (offsets[:, None, None] - 0.125) ** 2
    depth, valid, _, _ = select_depth(
        costs, np.array([[2.0]]), offsets, maximum_cost=0.02, minimum_gap=0.03
    )
    assert valid.all()
    np.testing.assert_allclose(depth, 2.125, atol=1e-6)
    ambiguous = np.minimum(costs, (offsets[:, None, None] + 0.375) ** 2)
    assert not select_depth(
        ambiguous, np.array([[2.0]]), offsets, maximum_cost=0.02, minimum_gap=0.03
    )[1].any()


def test_correlation_rejects_untextured_and_invalid_patches():
    flat = np.ones((21, 21), np.float32) * 0.3
    valid = np.ones_like(flat, bool)
    assert np.isinf(warped_correlation(flat, flat, valid)).all()
    rng = np.random.default_rng(77)
    image = rng.random((21, 21), dtype=np.float32)
    assert (warped_correlation(image, image, valid)[3:-3, 3:-3] < 1e-5).all()
    assert np.isinf(warped_correlation(image, image, valid)[:2]).all()
    assert np.isinf(warped_correlation(image, image, valid)[:, -2:]).all()
    valid[10, 10] = False
    assert np.isinf(warped_correlation(image, image, valid)[8:13, 8:13]).all()


def test_resampling_rejects_masked_neighbors_and_out_of_image_support():
    depth = np.full((4, 4), 2.0, np.float32)
    valid = np.ones((4, 4), bool)
    valid[2, 2] = False
    depth[2, 2] = np.nan
    u = np.array([[1.5, 1.0, -0.25, 3.25, np.nan]])
    v = np.array([[1.5, 1.0, 1, 1, 1]])
    sampled, support = resample_prior(depth, valid, u, v)
    assert support.tolist() == [[False, True, False, False, False]]
    assert sampled[0, 1] == 2.0
    assert np.isfinite(sampled).all()


def test_multiview_rgb_recovers_plane_depth_from_biased_prior():
    rng = np.random.default_rng(77)
    rgb = rng.integers(30, 230, (48, 64, 3), dtype=np.uint8)
    k = np.array([[120.0, 0, 32], [0, 120.0, 24], [0, 0, 1]])
    neighbors = []
    for translation in ([0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0]):
        transform = np.eye(4)
        transform[:3, 3] = translation
        homography = (
            k @ (np.eye(3) + np.outer(translation, [0, 0, 0.5])) @ np.linalg.inv(k)
        )
        neighbors.append(
            {
                "rgb": cv2.warpPerspective(rgb, homography, (64, 48)),
                "intrinsics": k,
                "world_to_camera": transform,
            }
        )
    result = refine(
        rgb,
        k,
        np.eye(4),
        np.ones((48, 64), np.float32) * 1.8,
        np.ones((48, 64), bool),
        neighbors,
        half_range_m=0.4,
        step_m=0.05,
    )
    assert result["valid"].sum() > 400
    assert np.quantile(abs(result["depth"][result["valid"]] - 2.0), 0.95) < 0.025
