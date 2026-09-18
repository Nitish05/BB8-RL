import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from bb8_rl.mapping import Bounds3D, RoomMemory
from bb8_rl.mapping.scan_free_space import (
    ScanFreeMemory,
    ScanRGBView,
    build_scan_free_memory,
    floor_homography_masks,
)

SPEC = importlib.util.spec_from_file_location(
    "dense_scan_builder",
    Path(__file__).parents[1] / "scripts/build-dense-scan-memory.py",
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def memory(count, bits):
    room = RoomMemory(
        Bounds3D((-0.2, -0.2, 0.0), (0.2, 0.2, 0.2)),
        0.2,
        scene_version="s",
        calibration_version="c",
    )
    zero = np.zeros((2, 2), bool)
    values = np.full((2, 2), bits, np.uint64)
    return ScanFreeMemory(
        room,
        zero,
        zero,
        zero,
        values,
        [f"v{i}" for i in range(count)],
        {
            "extent_m": 0.2,
            "resolution_m": 0.2,
            "floor_z_m": 0.0,
            "certified_height_m": 0.2,
        },
    )


@pytest.mark.parametrize(
    "count,indices,dtype",
    [(32, [31], np.uint32), (48, [32, 47], np.uint64), (64, [63], np.uint64)],
)
def test_support_high_bits_roundtrip_without_changing_legacy_dtype(
    tmp_path, count, indices, dtype
):
    bits = sum(1 << i for i in indices)
    source = memory(count, bits)
    source.save(tmp_path / "memory")
    loaded = ScanFreeMemory.load(tmp_path / "memory")
    assert loaded.support_bits.dtype == dtype
    assert int(loaded.support_bits[0, 0]) == bits
    assert loaded.supporting_views((0, 0)) == tuple(f"v{i}" for i in indices)
    with np.load(tmp_path / "memory/free-grid.npz", allow_pickle=False) as data:
        assert data["support_bits"].dtype == dtype
        np.testing.assert_array_equal(data["support_bits"], source.support_bits)


def test_undeclared_support_and_more_than64_views_fail_closed():
    with pytest.raises(ValueError, match="declared scan views"):
        memory(32, 1 << 32)
    with pytest.raises(ValueError, match="declared scan views"):
        memory(65, 1)


def camera(name, x=0.0):
    hsv = np.full((80, 80, 3), [105, 150, 70], np.uint8)
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[:3, 3] = [-x, 0.0, 3.0]
    return ScanRGBView(
        name,
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
        np.array([[40.0, 0.0, 40.0], [0.0, 40.0, 40.0], [0.0, 0.0, 1.0]]),
        transform,
    )


def test_builder_sets_bit47_but_one_witness_does_not_create_free():
    cameras = [camera(f"v{i}") for i in range(48)]
    masks = [np.zeros((80, 80), bool) for _ in cameras]
    masks[-1][:] = True
    result = build_scan_free_memory(
        cameras, extent=0.2, resolution=0.2, body_height_m=0.2, floor_masks=masks
    )
    assert result.support_bits.dtype == np.uint64
    assert int(result.support_bits[0, 0]) == 1 << 47
    assert not result.free_mask.any()


def test_selected_homography_sources_equal_corresponding_all_source_outputs():
    cameras = [camera("a", -0.8), camera("b", 0.8), camera("c", 1.6)]
    options = {"patch_size": None, "minimum_matches": 1, "minimum_angle_degrees": 10}
    all_masks, all_counts = floor_homography_masks(cameras, **options)
    masks, counts = floor_homography_masks(cameras, source_indices=[2, 0], **options)
    for at, index in enumerate([2, 0]):
        np.testing.assert_array_equal(masks[at], all_masks[index])
        np.testing.assert_array_equal(counts[at], all_counts[index])
    with pytest.raises(ValueError, match="Source indices"):
        floor_homography_masks(cameras, source_indices=[0, 0], **options)


def test_dense_builder_frozen_ancestry_keeps_queries_out():
    assert len(BUILDER.ALL_IDS) == 48
    assert len(set(BUILDER.ALL_IDS)) == 48
    assert not {"scan-014", "scan-017", "scan-020", "scan-023"} & set(BUILDER.ALL_IDS)
    assert BUILDER.NEW_IDS == tuple(f"scan-{i:03}" for i in range(36, 52))
