import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "map_improvement_audit", Path(__file__).parents[1] / "scripts/audit-map-improvement.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sphere_intersection_catches_overlap_without_cell_center_inside():
    low, high = np.array([[.9, -.1, -.1], [1., -.1, -.1], [1.1, -.1, -.1]]), np.array([[1.3, .1, .1], [1.2, .1, .1], [1.2, .1, .1]])
    overlap, contact = MODULE.sphere_intersections(low, high, [0, 0, 0], 1.)
    assert overlap.tolist() == [True, False, False]
    assert contact.tolist() == [False, True, False]


def test_sphere_corner_and_invalid_shapes():
    overlap, _ = MODULE.sphere_intersections([[.6, .6, 0]], [[.8, .8, .2]], [0, 0, 0], 1.)
    assert overlap[0]
    with pytest.raises(ValueError):
        MODULE.sphere_intersections([[0, 0, 0]], [[0, 1, 1]], [0, 0, 0], 1.)


def record(case, passed):
    return {"case_id": case, "radius_m": .14, "connected_and_certified": passed}


def test_gap_success_cannot_hide_regression_on_previous_route():
    protocol = {"cases": [{"id": "reported-gap"}, {"id": "previous"}], "radii_m": [.14], "required_reported_gap_radius_m": .14}
    base = [record("reported-gap", False), record("previous", True)]
    candidate = [record("reported-gap", True), record("previous", False)]
    result = MODULE.coverage_gate(base, candidate, protocol)
    assert not result["pass"]
    assert result["regressions"] == [{"case_id": "previous", "radius_m": .14}]
    candidate[-1]["connected_and_certified"] = True
    assert MODULE.coverage_gate(base, candidate, protocol)["pass"]


def test_missing_failed_case_is_not_removed_from_denominator():
    protocol = {"cases": [{"id": "reported-gap"}, {"id": "failed"}], "radii_m": [.14], "required_reported_gap_radius_m": .14}
    base = [record("reported-gap", False), record("failed", False)]
    assert not MODULE.coverage_gate(base, [record("reported-gap", True)], protocol)["pass"]
    duplicates = [record("reported-gap", True), record("reported-gap", True)]
    assert not MODULE.coverage_gate(base, duplicates, protocol)["pass"]


def test_legacy_full_volume_gate_catches_overlap_not_center_sampling():
    box = {"position": [0, 0, .05], "size": [.10, .10, .10], "euler": [0, 0, 0]}
    overlaps, _ = MODULE.LEGACY.solid_intersections([[.04, -.01, 0]], [[.08, .01, .12]], [box])
    assert overlaps[0]  # Cell center x=.06 is outside, but its volume intersects.


def test_supplemental_membership_exception_cannot_hide_geometry_or_provenance_failure():
    result = {
        "free_prisms": 1, "false_free_solid_intersection_voxels": 0,
        "free_voxels_outside_room_or_below_floor": 0, "free_solid_intersection_prisms": 0,
        "free_without_two_view_support": 0, "support_bits_outside_view_ids": False,
        "heldout_query_in_map": [], "free_sources_rgb_only": True,
        "source_rgb_hashes_checked": True, "input_provenance_checks": {"source": True},
    }
    ids = [str(i) for i in range(32)]
    assert MODULE.extended_fullvolume_gate(result, ids, ids, True)
    result["false_free_solid_intersection_voxels"] = 1
    assert not MODULE.extended_fullvolume_gate(result, ids, ids, True)
    result["false_free_solid_intersection_voxels"] = 0
    result["input_provenance_checks"]["source"] = False
    assert not MODULE.extended_fullvolume_gate(result, ids, ids, True)


def test_supplemental_membership_requires_all_distinct_allowed_views():
    result = {"free_prisms": 1}
    assert not MODULE.extended_fullvolume_gate(result, ["a", "a"], ["a", "b"], True)


def strict_fixture():
    baseline = np.array([[True, False, False]])
    occupied = np.array([[False, False, True]])
    candidate = np.array([[False, True, True]])
    strict_bits = np.array([[0, 15, 15]], dtype=np.uint32)
    final = np.array([[True, True, False]])
    final_bits = np.array([[7, 15, 15]], dtype=np.uint32)
    arrays = {"added_mask": final & ~baseline, "strict_candidate_mask": candidate,
              "strict_support_bits": strict_bits, "retained_baseline_free_mask": baseline.copy()}
    return final, occupied, final_bits, baseline, arrays


def test_strict_addition_preserves_old_three_view_evidence_and_occupied_conflicts():
    args = strict_fixture()
    assert all(MODULE.strict_addition_array_checks(*args, 4, 32).values())
    args[0][0, 2] = True
    assert not MODULE.strict_addition_array_checks(*args, 4, 32)["final_free_exact_evidence_union"]


def test_inherited_three_view_support_cannot_authorize_an_added_cell():
    args = strict_fixture()
    args[4]["strict_support_bits"][0, 1] = 7
    checks = MODULE.strict_addition_array_checks(*args, 4, 32)
    assert not checks["every_addition_has_strict_support"]
    assert not checks["addition_support_matches_final"]


def test_revision_cannot_relabel_addition_as_inherited_or_fabricate_view_bit():
    args = strict_fixture()
    args[4]["retained_baseline_free_mask"][0, 1] = True
    args[4]["strict_support_bits"][0, 1] = 31
    checks = MODULE.strict_addition_array_checks(*args, 4, 4)
    assert not checks["retained_baseline_mask_exact"]
    assert not checks["strict_bits_within_allowed_views"]


def test_high_view_bits_cannot_be_truncated_during_load():
    raw = np.array([[1 | (1 << 32) | (1 << 47)]], dtype=np.uint64)
    assert all(MODULE.support_storage_checks(raw, raw.copy(), 48).values())
    narrowed = raw.astype(np.uint32)
    checks = MODULE.support_storage_checks(raw, narrowed, 48)
    assert not checks["all_support_bits_roundtrip_exact"]
    assert not checks["wide_storage_when_required"]
    assert not MODULE.support_storage_checks(raw, raw, 47)["no_undeclared_view_bits"]


def test_strict_addition_high_bits_count_as_distinct_views_without_lower_bit_substitution():
    args = strict_fixture()
    args[4]["strict_support_bits"] = np.array([[0, (15 << 44), (15 << 44)]], dtype=np.uint64)
    bits = np.array([[7, (15 << 44), (15 << 44)]], dtype=np.uint64)
    assert all(MODULE.strict_addition_array_checks(args[0], args[1], bits, args[3], args[4], 4, 48).values())
    bits[0, 1] = 15
    checks = MODULE.strict_addition_array_checks(args[0], args[1], bits, args[3], args[4], 4, 48)
    assert not checks["addition_support_matches_final"]
