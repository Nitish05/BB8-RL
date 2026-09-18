"""Independent frozen-map improvement gate; authored geometry is scoring only.

This script never changes or produces a controller map. A passing result only
permits bounded native testing; static connectivity is not navigation success.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from bb8_rl.mapping.scan_free_space import ScanFreeMemory

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "scripts/audit-occluded-control.py"
SPEC = importlib.util.spec_from_file_location("frozen_fullvolume_audit", LEGACY_PATH)
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(relative):
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "<BB8-RL>")
    return value


def sphere_intersections(low, high, center, radius):
    """Exact positive-volume AABB/sphere overlap, including corner-only overlap."""
    low, high, center = (np.asarray(v, dtype=float) for v in (low, high, center))
    if (low.shape != high.shape or low.ndim != 2 or low.shape[1] != 3
            or center.shape != (3,) or not np.isfinite(np.r_[low.ravel(), high.ravel(), center, radius]).all()
            or radius <= 0 or np.any(high <= low)):
        raise ValueError("Need finite positive boxes and a positive sphere")
    distance_squared = np.sum((np.clip(center, low, high) - center) ** 2, axis=1)
    interior = distance_squared < radius**2 - 1e-12
    contact = (distance_squared <= radius**2 + 1e-12) & ~interior
    return interior, contact


def scan_robot_checks(memory, robot):
    xyz = np.argwhere(memory.memory._cells == memory.memory._FREE)
    low = np.asarray(memory.memory.bounds.minimum) + xyz * memory.memory.resolution_m
    high = low + memory.memory.resolution_m
    yx = np.argwhere(memory.free_mask)
    prism_low = np.c_[-memory.extent + yx[:, 1] * memory.resolution,
                      -memory.extent + yx[:, 0] * memory.resolution,
                      np.full(len(yx), memory.floor_z)]
    prism_high = prism_low + [memory.resolution, memory.resolution, memory.body_height_m]
    result = {"scope": "Supplementary scan-time movable-object assumption check; not a static obstacle inserted into online control."}
    for part in ("body", "head"):
        center = [*robot["position_xy"], robot[f"{part}_center_z_m"]]
        radius = robot[f"{part}_radius_m"]
        overlap, contact = sphere_intersections(low, high, center, radius)
        prism_overlap, prism_contact = sphere_intersections(prism_low, prism_high, center, radius)
        result[part] = {
            "center_scoring_only": center, "radius_m": radius,
            "free_voxel_solid_intersections": int(overlap.sum()),
            "free_voxel_face_contacts": int(contact.sum()),
            "free_prism_solid_intersections": int(prism_overlap.sum()),
            "free_prism_face_contacts": int(prism_contact.sum()),
            "collision_object": part == "body",
        }
    result["pass"] = all(result[p]["free_voxel_solid_intersections"] == result[p]["free_prism_solid_intersections"] == 0 for p in ("body", "head"))
    return result


def coverage(memory, protocol):
    grids = {r: memory.planning_grid(r) for r in protocol["radii_m"]}
    cache, output = {}, []
    for case in protocol["cases"]:
        for radius, grid in grids.items():
            key = (tuple(case["start"]), tuple(case["goal"]), radius)
            if key not in cache:
                result = {
                    "radius_m": radius, "start_free": grid.free(grid.cell(case["start"])),
                    "goal_free": grid.free(grid.cell(case["goal"])), "connected_and_certified": False,
                }
                try:
                    route = grid.route(case["start"], case["goal"])
                    certificates = [memory.segment_free(a, b, radius) for a, b in pairwise(route)]
                    result.update(
                        connected_and_certified=all(certificates), route=route.tolist(),
                        exact_segment_certificates=certificates,
                        route_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
                    )
                except ValueError as error:
                    result["failure_reason"] = str(error)
                cache[key] = result
            output.append({"case_id": case["id"], "family": case["family"], **cache[key]})
    return output


def coverage_gate(baseline, candidate, protocol):
    by_key = {(c["case_id"], c["radius_m"]): c for c in candidate}
    regressions = [
        {"case_id": c["case_id"], "radius_m": c["radius_m"]}
        for c in baseline if c["connected_and_certified"]
        and not by_key.get((c["case_id"], c["radius_m"]), {}).get("connected_and_certified", False)
    ]
    gap = by_key.get(("reported-gap", protocol["required_reported_gap_radius_m"]), {})
    expected = {(c["id"], r) for c in protocol["cases"] for r in protocol["radii_m"]}
    baseline_keys = [(c["case_id"], c["radius_m"]) for c in baseline]
    candidate_keys = [(c["case_id"], c["radius_m"]) for c in candidate]
    complete = (len(candidate_keys) == len(baseline_keys) == len(expected)
                and set(candidate_keys) == set(baseline_keys) == expected)
    return {
        "complete_case_radius_denominator": complete,
        "reported_gap_at_required_radius": gap.get("connected_and_certified", False),
        "regressions": regressions, "pass": complete and gap.get("connected_and_certified", False) and not regressions,
    }


def evidence_checks(memory, baseline, protocol, expected_hashes=None):
    m = memory.metadata
    expected_hashes = expected_hashes or protocol["frozen_mapping_input_sha256"]
    counts = np.array([int(x).bit_count() for x in memory.support_bits.ravel()]).reshape(memory.free_mask.shape)
    expected_ids = set(expected_hashes)
    raw = memory.memory.to_dict()
    old = baseline.memory.to_dict()
    canonical = lambda record: json.dumps(record, sort_keys=True, separators=(",", ":"))
    old_occupied = Counter(canonical(v) for v in old["volumes"] if v["state"] == "occupied")
    new_occupied = Counter(canonical(v) for v in raw["volumes"] if v["state"] == "occupied")
    compatible = memory.free_mask.shape == baseline.free_mask.shape and np.isclose(memory.resolution, baseline.resolution)
    checks = {
        "all_original_input_hashes_unchanged": all(m.get("input_sha256", {}).get(v) == h for v, h in protocol["frozen_mapping_input_sha256"].items()),
        "all_allowed_input_hashes_match_binding": m.get("input_sha256") == expected_hashes,
        "input_hash_ids_exactly_allowed": set(m.get("input_sha256", {})) == expected_ids,
        "mapping_ancestry_exactly_allowed": set(m.get("mapping_input_view_ids", [])) == expected_ids,
        "support_ids_exactly_allowed": set(memory.view_ids) == expected_ids,
        "source_paths_present_for_rehash": bool(m.get("scan_dir")) and bool(m.get("occupied_source")),
        "body_height_preserved": np.isclose(memory.body_height_m, protocol["required_body_height_m"]),
        "resolution_preserved": np.isclose(memory.resolution, protocol["required_resolution_m"]),
        "extent_preserved": np.isclose(memory.extent, protocol["required_extent_m"]),
        "declared_minimum_views_preserved": m.get("minimum_views", 0) >= protocol["required_minimum_views"],
        "every_free_cell_has_declared_support_count": bool(np.all(counts[memory.free_mask] >= m.get("minimum_views", 0))),
        "projection_xy_guard_not_shrunk": m.get("projection_margin_m", 0) >= .05,
        "pairwise_baseline_gate_not_shrunk": m.get("minimum_baseline_m", 0) >= .5,
        "pairwise_angle_gate_not_shrunk": m.get("minimum_angle_degrees", 0) >= 15,
        "all_baseline_occupied_records_retained": not bool(old_occupied - new_occupied),
        "all_baseline_occupied_cells_retained": compatible and bool(np.all(memory.occupied_mask[baseline.occupied_mask])),
        "retains_unknown_cells": bool(np.any(~memory.free_mask & ~memory.occupied_mask)),
        "missing_geometry_never_declared_free": m.get("missing_geometry_creates_free") is False,
    }
    return {"checks": {k: bool(v) for k, v in checks.items()}, "pass": bool(all(checks.values())),
            "limit": "Support bits and declared producer assumptions are checked, not interpreted as independently calibrated confidence. Producer source and RGB causality need separate review."}


def capture_binding_checks(path, protocol_path, protocol, memory, baseline_dir, candidate_dir):
    binding = json.loads(path.read_text())
    plan_path = resolve(binding["capture_plan_path"])
    capture_path = resolve(binding["capture_manifest_path"])
    plan, capture = json.loads(plan_path.read_text()), json.loads(capture_path.read_text())
    expected = binding["input_sha256"]
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text())
    learned = memory.metadata.get("learned_mask_intersection", {})
    checks = {
        "supplemental_protocol_hash": binding["supplemental_protocol_sha256"] == digest(protocol_path),
        "capture_plan_hash": digest(plan_path) == binding["capture_plan_sha256"] == capture["plan_sha256"],
        "capture_manifest_hash": digest(capture_path) == binding["capture_manifest_sha256"],
        "complete_capture": capture["status"] == "complete",
        "capture_input_hashes_match_binding": capture["input_sha256"] == expected,
        "exact_allowed_ids": set(expected) == set(protocol["allowed_mapping_view_ids"]),
        "exact_supplemental_ids": capture["additional_view_ids"] == protocol["supplemental_view_ids"] == [v["id"] for v in plan["views"]],
        "task_hash": capture["task_sha256"] == binding["task_sha256"],
        "capture_source_snapshot": digest(capture_path.parent / "source/capture-overhead-scan.py") == capture["source_sha256"] == binding["capture_source_sha256"],
        "capture_plan_snapshot": digest(capture_path.parent / "capture-plan.json") == binding["capture_plan_sha256"],
        "no_depth_or_segmentation": capture["depth_used"] is False and capture["segmentation_used"] is False and plan["depth_used"] is False and plan["segmentation_used"] is False,
        "original_robot_pose": plan["parked_robot_xy"] == protocol["scan_time_robot_scoring_only"]["position_xy"],
        "full_prism_evidence": memory.metadata.get("free_evidence_mode") == "full_prism",
        "candidate_capture_manifest_hash": memory.metadata.get("supplemental_capture_manifest_sha256") == binding["capture_manifest_sha256"],
        "candidate_capture_plan_hash": memory.metadata.get("capture_plan_sha256") == binding["capture_plan_sha256"],
        "frozen_vision_checkpoint": learned.get("checkpoint_sha256") == baseline_manifest["learned_mask_intersection"]["checkpoint_sha256"],
        "frozen_floor_threshold": learned.get("floor_threshold") == .5,
        "frozen_floor_refinement": learned.get("floor_refinement") == "guided-radius4-epsilon1e-4",
        "no_new_training_or_threshold_selection": learned.get("training_or_threshold_selection") is False,
    }
    for view, hashes in expected.items():
        for kind, extension in (("rgb", ".png"), ("calibration", ".json")):
            checks[f"captured:{view}:{kind}"] = digest(capture_path.parent / (view + extension)) == hashes[kind]
    for view in plan["views"]:
        meta = json.loads((capture_path.parent / (view["id"] + ".json")).read_text())
        checks[f"pose_matches_plan:{view['id']}"] = meta["position"] == view["position"] and meta["lookat"] == view["lookat"] and meta["depth_used"] is False and meta["segmentation_used"] is False
    old_masks = np.load(baseline_dir / "floor-masks.npz")
    new_masks = np.load(candidate_dir / "floor-masks.npz")
    new_ids = list(new_masks["view_ids"])
    old_values, new_values = old_masks["masks"], new_masks["masks"]
    checks["mask_ids_exactly_allowed"] = set(new_ids) == set(expected) and len(new_ids) == len(expected)
    for i, view in enumerate(old_masks["view_ids"]):
        checks[f"original_mask_unchanged:{view}"] = view in new_ids and np.array_equal(old_values[i], new_values[new_ids.index(view)])
    if protocol.get("prior_mask_directory"):
        prior_mask_path = resolve(protocol["prior_mask_directory"]) / "floor-masks.npz"
        checks["previous_mask_artifact_hash"] = digest(prior_mask_path) == protocol["prior_floor_masks_sha256"]
        with np.load(prior_mask_path, allow_pickle=False) as prior_masks:
            prior_ids, prior_values = list(prior_masks["view_ids"]), prior_masks["masks"]
            checks["previous_mask_ids_exact"] = prior_ids == protocol["prior_floor_mask_view_ids"]
            for i, view in enumerate(prior_ids):
                checks[f"previous_mask_unchanged:{view}"] = view in new_ids and np.array_equal(prior_values[i], new_values[new_ids.index(view)])
        prior_binding_path = resolve(protocol["prior_capture_binding_path"])
        prior_binding = json.loads(prior_binding_path.read_text())
        checks["prior_capture_binding_hash"] = digest(prior_binding_path) == protocol["prior_capture_binding_sha256"] == binding["prior_capture_binding_sha256"]
        checks["prior_capture_manifest_hash"] = plan["prior_capture_manifest_sha256"] == prior_binding["capture_manifest_sha256"]
        checks["prior_capture_view_ids"] = plan["retained_view_ids"] == protocol["retained_mapping_view_ids"]
    old_masks.close()
    new_masks.close()
    return expected, {"binding_sha256": digest(path), "checks": checks, "pass": all(checks.values())}


def extended_fullvolume_gate(result, actual_ids, allowed_ids, memory_valid):
    """Supersede only legacy20 membership; retain every full-volume condition."""
    return bool(
        memory_valid and len(actual_ids) == len(set(actual_ids)) == len(allowed_ids)
        and set(actual_ids) == set(allowed_ids)
        and result["free_prisms"] > 0
        and result["false_free_solid_intersection_voxels"] == 0
        and result["free_voxels_outside_room_or_below_floor"] == 0
        and result["free_solid_intersection_prisms"] == 0
        and result["free_without_two_view_support"] == 0
        and not result["support_bits_outside_view_ids"]
        and not result["heldout_query_in_map"]
        and result["free_sources_rgb_only"]
        and result["source_rgb_hashes_checked"]
        and result["input_provenance_checks"]
        and all(result["input_provenance_checks"].values())
    )


def strict_addition_array_checks(final_free, occupied, final_bits, baseline_free, arrays, minimum_views, view_count):
    """Separate inherited evidence from additions; inherited support cannot fill a new cell."""
    expected_names = {"added_mask", "strict_candidate_mask", "strict_support_bits", "retained_baseline_free_mask"}
    if set(arrays) != expected_names:
        return {"exact_array_schema": False}
    if any(np.asarray(v).shape != final_free.shape for v in arrays.values()):
        return {"compatible_array_shapes": False}
    bits = arrays["strict_support_bits"]
    if (any(arrays[k].dtype != bool for k in expected_names - {"strict_support_bits"})
            or not np.issubdtype(bits.dtype, np.unsignedinteger)):
        return {"valid_array_dtypes": False}
    additions = final_free & ~baseline_free
    count = np.array([int(v).bit_count() for v in bits.ravel()]).reshape(bits.shape)
    candidate = arrays["strict_candidate_mask"]
    return {
        "retained_baseline_mask_exact": np.array_equal(arrays["retained_baseline_free_mask"], baseline_free),
        "added_mask_exact": np.array_equal(arrays["added_mask"], additions),
        "final_free_exact_evidence_union": np.array_equal(final_free, baseline_free | (candidate & ~occupied)),
        "every_addition_has_strict_support": bool(np.all(count[additions] >= minimum_views)),
        "all_strict_candidates_have_strict_support": bool(np.all(count[candidate] >= minimum_views)),
        "strict_bits_within_allowed_views": bool(np.all(bits.astype(np.uint64) >> view_count == 0)),
        "addition_support_matches_final": np.array_equal(final_bits[additions], bits[additions]),
    }


def support_storage_checks(raw_bits, loaded_bits, view_count):
    """Catch silent narrowing of valid high camera bits at the persistence boundary."""
    raw, loaded = np.asarray(raw_bits), np.asarray(loaded_bits)
    unsigned = np.issubdtype(raw.dtype, np.unsignedinteger) and np.issubdtype(loaded.dtype, np.unsignedinteger)
    return {
        "unsigned_storage": unsigned,
        "all_support_bits_roundtrip_exact": raw.shape == loaded.shape and np.array_equal(raw, loaded),
        "wide_storage_when_required": view_count <= 32 or (raw.dtype == np.uint64 and loaded.dtype == np.uint64),
        "no_undeclared_view_bits": unsigned and bool(np.all(raw.astype(np.uint64) >> view_count == 0)),
    }


def revision_binding_checks(plan_path, memory, baseline, candidate_dir):
    if not memory.metadata.get("revision_plan_sha256"):
        if plan_path is not None:
            raise ValueError("Revision plan supplied for a candidate without revision provenance")
        return {"required": False, "pass": True}
    if plan_path is None:
        raise ValueError("A stricter revision requires its independently frozen --revision-plan")
    plan = json.loads(plan_path.read_text())
    metadata = memory.metadata
    settings = plan["new_evidence"]
    with np.load(candidate_dir / "supplemental-additions.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    checks = strict_addition_array_checks(
        memory.free_mask, memory.occupied_mask, memory.support_bits, baseline.free_mask,
        arrays, settings["minimum_pairwise_separated_views"], len(memory.view_ids),
    )
    checks.update(
        revision_plan_hash=digest(plan_path) == metadata["revision_plan_sha256"] == digest(candidate_dir / "revision-plan.json"),
        declared_strict_view_count=metadata.get("supplemental_addition_minimum_views") == settings["minimum_pairwise_separated_views"] == 4,
        declared_strict_pixel_guard=metadata.get("supplemental_addition_pixel_guard") == settings["pixel_guard"] == 2,
        declared_metric_guard=metadata["projection_margin_m"] == settings["metric_projection_margin_m"] == .05,
        no_truth_locations=plan["truth_geometry_or_failed_cell_locations_used"] is False,
        addition_artifact_hash=digest(candidate_dir / "supplemental-additions.npz") == metadata["artifact_sha256"]["supplemental-additions.npz"],
    )
    if plan["revision"] == "full-prism-extended-v2":
        pinned = plan["inputs_sha256"]
        prior_dir = ROOT / "work/m712-map-improvement/full-prism-extended-v1"
        prior_prefix = str(prior_dir.relative_to(ROOT)) + "/"
        checks.update(
            complete_mask_file_reused=digest(candidate_dir / "floor-masks.npz") == pinned[prior_prefix + "floor-masks.npz"] == digest(prior_dir / "floor-masks.npz"),
            reused_prior_manifest=metadata.get("reused_floor_evidence_manifest_sha256") == pinned[prior_prefix + "manifest.json"] == digest(prior_dir / "manifest.json"),
            retained_baseline_manifest=metadata.get("source_memory_manifest_sha256") == pinned["work/interactive-assets/memory/manifest.json"],
        )
    elif plan["revision"] == "full-prism-extended-v3":
        prior_dir = ROOT / "work/m712-map-improvement/full-prism-extended-v2"
        prior = ScanFreeMemory.load(prior_dir)
        checks.update(
            reused_prior_manifest=metadata.get("reused_floor_evidence_manifest_sha256") == plan["retained_manifest_sha256"] == digest(prior_dir / "manifest.json"),
            retained_mask_artifact=digest(prior_dir / "floor-masks.npz") == plan["retained_floor_masks_sha256"],
            retained_baseline_manifest=metadata.get("source_memory_manifest_sha256") == plan["baseline_manifest_sha256"],
            previous_revision_free_cells_retained=bool(np.all(memory.free_mask[prior.free_mask])),
            capture_plan_frozen=metadata["capture_plan_sha256"] == plan["capture_plan_sha256"],
            capture_manifest_frozen=metadata["supplemental_capture_manifest_sha256"] == plan["capture_manifest_sha256"],
            checkpoint_frozen=metadata["learned_mask_intersection"]["checkpoint_sha256"] == plan["checkpoint_sha256"],
            view_order_exact=list(memory.view_ids) == plan["retained_view_ids"] + plan["new_view_ids"],
            strict_support_uses_uint64=arrays["strict_support_bits"].dtype == np.uint64,
        )
    else:
        raise ValueError("Unsupported revision plan; preserve this attempt and declare its audit policy")
    return {"required": True, "revision_plan_sha256": digest(plan_path), "checks": checks,
            "new_free_cells": int((memory.free_mask & ~baseline.free_mask).sum()), "pass": all(checks.values())}


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh output; preserve every failed candidate")
    protocol = json.loads(args.protocol.read_text())
    scene_path = resolve(protocol["scene"])
    if digest(scene_path) != protocol["scene_sha256"]:
        raise ValueError("Frozen scoring scene changed")
    if digest(LEGACY_PATH) != protocol["source_sha256"]["legacy_fullvolume_scorer"]:
        raise ValueError("Frozen full-volume scorer changed")
    baseline_dir = resolve(protocol["baseline_directory"])
    if digest(baseline_dir / "manifest.json") != protocol["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline changed")
    baseline, candidate = ScanFreeMemory.load(baseline_dir), ScanFreeMemory.load(args.memory)
    scene = json.loads(scene_path.read_text())
    full_volume = LEGACY.audit_memory(args.memory, scene)
    binding_checks = {"required": False, "pass": True}
    expected_hashes = protocol["frozen_mapping_input_sha256"]
    if protocol.get("supplemental_view_ids"):
        if not args.capture_binding:
            raise ValueError("Supplemental views require a frozen capture-input binding")
        expected_hashes, binding_checks = capture_binding_checks(args.capture_binding, args.protocol, protocol, candidate, baseline_dir, args.memory)
        full_volume["legacy_twenty_view_status"] = full_volume["status"]
        full_volume["explicit_membership_exception"] = protocol["membership_exception"]
        full_volume["status"] = "pass" if extended_fullvolume_gate(full_volume, candidate.view_ids, protocol["allowed_mapping_view_ids"], candidate.memory.valid) else "fail"
    evidence = evidence_checks(candidate, baseline, protocol, expected_hashes)
    revision_checks = revision_binding_checks(args.revision_plan, candidate, baseline, args.memory)
    with np.load(args.memory / "free-grid.npz", allow_pickle=False) as raw_grid:
        storage_checks = support_storage_checks(raw_grid["support_bits"], candidate.support_bits, len(candidate.view_ids))
    high_counts = {v: int(np.sum(candidate.free_mask & ((candidate.support_bits.astype(np.uint64) >> i) & 1).astype(bool)))
                   for i, v in enumerate(candidate.view_ids) if i >= 32}
    support_storage = {"checks": storage_checks, "high_view_free_support_counts": high_counts, "pass": all(storage_checks.values())}
    robot = scan_robot_checks(candidate, protocol["scan_time_robot_scoring_only"])
    baseline_coverage, candidate_coverage = coverage(baseline, protocol), coverage(candidate, protocol)
    route_gate = coverage_gate(baseline_coverage, candidate_coverage, protocol)
    gate = full_volume["status"] == "pass" and evidence["pass"] and robot["pass"] and route_gate["pass"] and binding_checks["pass"] and revision_checks["pass"] and support_storage["pass"]
    result = {
        "schema": "bb8.map-improvement-audit.v1", "scope": "Offline whole-volume safety and static connectivity; truth never repairs or enters a candidate map.",
        "native_trial_gate_pass": gate, "runtime_promotion_authorized": False,
        "protocol_sha256": digest(args.protocol), "scorer_sha256": digest(__file__),
        "legacy_fullvolume_scorer_sha256": digest(LEGACY_PATH),
        "baseline_manifest_sha256": digest(baseline_dir / "manifest.json"),
        "candidate_manifest_sha256": digest(args.memory / "manifest.json"),
        "candidate_artifact_sha256": candidate.metadata["artifact_sha256"],
        "declared_candidate_semantics": {k: candidate.metadata.get(k) for k in ("assumptions", "free_evidence_mode", "free_evidence_semantics", "uncertainty_status")},
        "full_volume": full_volume, "evidence": evidence, "scan_time_robot": robot,
        "capture_binding": binding_checks,
        "strict_revision": revision_checks,
        "support_storage": support_storage,
        "baseline_scan_time_robot": scan_robot_checks(baseline, protocol["scan_time_robot_scoring_only"]),
        "coverage_gate": route_gate, "baseline_coverage": baseline_coverage, "candidate_coverage": candidate_coverage,
        "coverage_counts": {str(r): {
            "case_denominator": len(protocol["cases"]),
            "unique_endpoint_denominator": protocol["unique_endpoint_pair_denominator"],
            "baseline_connected": sum(c["connected_and_certified"] for c in baseline_coverage if c["radius_m"] == r),
            "candidate_connected": sum(c["connected_and_certified"] for c in candidate_coverage if c["radius_m"] == r),
        } for r in protocol["radii_m"]},
        "limitations": [
            "All geometry is authored synthetic scorer data; no unseen-room or physical safety claim.",
            "Six original failed case IDs include four identical endpoint pairs. Map connectivity cannot establish their original occlusion/control intent.",
            "The reported route and regression endpoints are development cases; all candidate attempts and failed gates must remain visible.",
            "A passing static gate permits fresh bounded native trials only; original arrival, visibility, actuation and failure-control gates remain mandatory.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "scorer-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
    (args.output / "report.json").write_text(json.dumps(sanitize(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"native_trial_gate_pass": gate, "full_volume": full_volume["status"],
                      "evidence": evidence["pass"], "scan_robot": robot["pass"],
                      "coverage": result["coverage_counts"], "output": sanitize(str(args.output))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--capture-binding", type=Path)
    parser.add_argument("--revision-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
