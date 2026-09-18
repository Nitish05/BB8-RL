import numpy as np
import pytest

from bb8_rl.mapping import Bounds3D, EvidenceSource, Provenance, RoomMemory, SpaceState
from bb8_rl.mapping.scan_free_space import (
    ScanRGBView,
    build_scan_free_memory,
    prism_floor_support,
)


def camera(name, x):
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[:3, 3] = [-x, 0.0, 3.0]
    return ScanRGBView(
        name,
        np.zeros((80, 80, 3), np.uint8),
        np.array([[60.0, 0.0, 40.0], [0.0, 60.0, 40.0], [0.0, 0.0, 1.0]]),
        transform,
    )


def build(views, masks, **options):
    settings = {
        "extent": 0.2,
        "resolution": 0.1,
        "body_height_m": 0.1,
        "projection_margin_m": 0.05,
        "pixel_guard": 1,
        "minimum_views": 3,
        "minimum_baseline_m": 0.5,
        "floor_masks": masks,
        "projection_shape": "convex_hull",
        "evidence_mode": "grounded_column",
        "assume_floor_connected_columns": True,
    }
    settings.update(options)
    return build_scan_free_memory(views, **settings)


def test_grounded_column_separates_floor_footprint_from_background_in_prism():
    view = camera("a", 0.8)
    mask = np.ones((80, 80), bool)
    low, high = np.array([[-0.1, -0.1]]), np.array([[0.1, 0.1]])
    # The elevated projection shifts left toward this nonfloor background pixel;
    # the expanded pixel square does not touch the actual floor footprint.
    mask[40, 18] = False
    args = {
        "floor_z": 0.0,
        "top_z": 0.6,
        "pixel_guard": 1,
        "floor_mask": mask,
        "projection_shape": "convex_hull",
    }
    assert not prism_floor_support(view, low, high, **args).item()
    assert prism_floor_support(
        view, low, high, **args, evidence_mode="grounded_column"
    ).item()
    mask[40, 24] = False  # An interior floor-footprint pixel still rejects.
    assert not prism_floor_support(
        view, low, high, **args, evidence_mode="grounded_column"
    ).item()


def test_grounded_column_requires_explicit_strong_assumption_and_unchanged_guards():
    views = [camera("a", -1.6), camera("b", 0.0), camera("c", 1.6)]
    masks = [np.ones((80, 80), bool) for _ in views]
    for options in (
        {"assume_floor_connected_columns": False},
        {"minimum_views": 2},
        {"projection_margin_m": 0.049},
        {"pixel_guard": 0},
        {"floor_masks": None},
    ):
        with pytest.raises(ValueError, match="Grounded-column evidence"):
            build(views, masks, **options)


def test_grounded_column_keeps_three_separated_views_and_missing_unknown():
    views = [camera("a", -1.6), camera("b", 0.0), camera("c", 1.6)]
    masks = [np.ones((80, 80), bool) for _ in views]
    result = build(views, masks)
    assert result.free_mask[1, 1]
    assert result.metadata["free_evidence_mode"] == "grounded_column"
    assert result.metadata["floor_connected_columns_assumed"] is True
    assert result.metadata["body_volume_observed_directly"] is False
    assert any("entire vertical" in text for text in result.metadata["assumptions"])
    assert not build(views[:2], masks[:2]).free_mask.any()
    masks[2][:] = False
    assert not build(views, masks).free_mask.any()
    near_views = [camera("a", -0.1), camera("b", 0.0), camera("c", 0.1)]
    assert not build(near_views, [np.ones((80, 80), bool)] * 3).free_mask.any()


def test_grounded_column_retains_occupied_columns_and_rejects_query_ancestry():
    views = [camera("a", -1.6), camera("b", 0.0), camera("c", 1.6)]
    masks = [np.ones((80, 80), bool) for _ in views]
    memory = RoomMemory(
        Bounds3D((-0.2, -0.2, 0.0), (0.2, 0.2, 0.3)),
        0.1,
        scene_version="s",
        calibration_version="c",
    )
    memory.observe_volume(
        Bounds3D((-0.19, -0.19, 0.01), (-0.11, -0.11, 0.09)),
        SpaceState.OCCUPIED,
        EvidenceSource("conflict", ("a",), 0.0, Provenance.RGB_RECONSTRUCTION),
    )
    result = build(views, masks, occupied_memory=memory)
    assert result.candidate_free_mask[0, 0]
    assert result.occupied_mask[0, 0] and not result.free_mask[0, 0]
    assert result.memory.evidence[0] == memory.evidence[0]
    views[0] = camera("scan-014", -1.6)
    with pytest.raises(ValueError, match="Frozen query"):
        build(views, masks)


def test_full_prism_default_still_equals_explicit_mode():
    views = [camera("a", -0.8), camera("b", 0.8)]
    masks = [np.ones((80, 80), bool) for _ in views]
    options = {
        "extent": 0.2,
        "resolution": 0.1,
        "body_height_m": 0.1,
        "floor_masks": masks,
    }
    a = build_scan_free_memory(views, **options)
    b = build_scan_free_memory(views, **options, evidence_mode="full_prism")
    np.testing.assert_array_equal(a.free_mask, b.free_mask)
    np.testing.assert_array_equal(a.support_bits, b.support_bits)
    assert a.metadata["free_evidence_mode"] == "full_prism"
