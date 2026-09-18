import numpy as np
import pytest

from bb8_rl.mapping.scan_free_space import ScanRGBView, build_scan_free_memory
from bb8_rl.mapping.scan_revision import retain_accepted_free


def views():
    result = []
    for name, (x, y) in zip(
        "abcd", [(-1.6, 0), (0, 0), (1.6, 0), (0, 1.6)], strict=True
    ):
        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[:3, 3] = [-x, y, 3.0]
        result.append(
            ScanRGBView(
                name,
                np.zeros((80, 80, 3), np.uint8),
                np.array([[40.0, 0.0, 40.0], [0.0, 40.0, 40.0], [0.0, 0.0, 1.0]]),
                transform,
            )
        )
    return result


def build(cameras, masks, count, pixels):
    return build_scan_free_memory(
        cameras,
        extent=0.2,
        resolution=0.1,
        body_height_m=0.1,
        projection_margin_m=0.05,
        pixel_guard=pixels,
        minimum_views=count,
        minimum_baseline_m=0.5,
        floor_masks=masks,
        projection_shape="convex_hull",
    )


@pytest.mark.parametrize("baseline_observed", [True, False])
def test_strict_additions_union_only_explicit_evidence(baseline_observed):
    cameras = views()
    masks = [np.ones((80, 80), bool) for _ in cameras]
    old_masks = [mask.copy() for mask in masks[:3]]
    if not baseline_observed:
        old_masks[0][:] = False
    baseline = build(cameras[:3], old_masks, 3, 1)
    if baseline_observed:
        masks[3][:] = False  # Fourth witness absent, no new strict evidence.
    candidate = build(cameras, masks, 4, 2)
    assert bool(candidate.free_mask.any()) is not baseline_observed
    result = retain_accepted_free(candidate, baseline)
    np.testing.assert_array_equal(
        result.free_mask, candidate.free_mask | baseline.free_mask
    )
    np.testing.assert_array_equal(result.occupied_mask, baseline.occupied_mask)
    assert result.metadata["minimum_views"] == 3
    assert result.metadata["supplemental_addition_minimum_views"] == 4
    assert result.metadata["supplemental_addition_pixel_guard"] == 2
    added = result.free_mask & ~baseline.free_mask
    counts = np.array([int(v).bit_count() for v in candidate.support_bits[added]])
    assert np.all(counts >= 4)
    # Cells unknown in both evidence sets cannot become FREE through merging.
    assert not result.free_mask[~candidate.free_mask & ~baseline.free_mask].any()


def test_strict_additions_reject_weaker_candidate_or_changed_registration():
    cameras = views()
    masks = [np.ones((80, 80), bool) for _ in cameras]
    baseline = build(cameras[:3], masks[:3], 3, 1)
    weaker = build(cameras, masks, 3, 1)
    with pytest.raises(ValueError, match="stricter full-prism"):
        retain_accepted_free(weaker, baseline)
    candidate = build(cameras, masks, 4, 2)
    candidate.memory.scene_version = "changed"
    with pytest.raises(ValueError, match="compatible"):
        retain_accepted_free(candidate, baseline)
