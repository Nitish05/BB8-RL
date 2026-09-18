"""Preserve accepted map evidence while requiring stricter positive additions."""

import numpy as np

from .contracts import SpaceState
from .room_memory import RoomMemory
from .scan_free_space import ScanFreeMemory, _memory_planar_states


def retain_accepted_free(candidate, baseline):
    """Union accepted FREE records with stricter evidence, never clear occupancy.

    A producer must separately freeze and validate each evidence layer. The
    candidate's support bits remain the strict-addition proof; retained baseline
    support is added only on baseline FREE cells. No unobserved-cell fill occurs.
    """
    if (
        not candidate.memory.valid
        or not baseline.memory.valid
        or candidate.extent != baseline.extent
        or candidate.resolution != baseline.resolution
        or candidate.body_height_m != baseline.body_height_m
        or candidate.floor_z != baseline.floor_z
        or candidate.memory.scene_version != baseline.memory.scene_version
        or candidate.memory.calibration_version != baseline.memory.calibration_version
        or candidate.memory.world_frame != baseline.memory.world_frame
        or candidate.memory.bounds != baseline.memory.bounds
        or candidate.view_ids[: len(baseline.view_ids)] != baseline.view_ids
        or not np.array_equal(candidate.occupied_mask, baseline.occupied_mask)
        or candidate.metadata["minimum_views"] < 4
        or candidate.metadata["pixel_guard"] < 2
        or candidate.metadata["free_evidence_mode"] != "full_prism"
    ):
        raise ValueError(
            "Need compatible accepted baseline and stricter full-prism additions"
        )
    memory = RoomMemory.from_dict(candidate.memory.to_dict())
    for evidence in baseline.memory.evidence:
        if evidence.state is SpaceState.FREE:
            memory.observe_volume(evidence.bounds, evidence.state, evidence.source)
    free, occupied = _memory_planar_states(
        memory,
        candidate.extent,
        candidate.resolution,
        candidate.floor_z + candidate.body_height_m,
    )
    if not np.array_equal(free, candidate.free_mask | baseline.free_mask):
        raise ValueError("Combined evidence differs from the exact accepted FREE union")
    if not np.array_equal(occupied, baseline.occupied_mask):
        raise ValueError("Combined evidence changed an occupied cell")
    bits = candidate.support_bits.copy()
    bits[baseline.free_mask] |= baseline.support_bits[baseline.free_mask]
    positive = candidate.candidate_free_mask | baseline.free_mask
    metadata = dict(candidate.metadata)
    metadata.update(
        minimum_views=baseline.metadata["minimum_views"],
        pixel_guard=baseline.metadata["pixel_guard"],
        supplemental_addition_minimum_views=candidate.metadata["minimum_views"],
        supplemental_addition_pixel_guard=candidate.metadata["pixel_guard"],
        retained_original_free_evidence=True,
        free_cells=int(free.sum()),
        candidate_free_cells=int(positive.sum()),
        occupied_cells=int(occupied.sum()),
        unknown_cells=int((~free & ~occupied).sum()),
        free_candidates_blocked_by_occupied=int((positive & occupied).sum()),
        free_evidence_semantics=(
            "Retained accepted baseline full-prism FREE evidence, plus new full-prism "
            "evidence requiring four mutually separated cameras and a two-pixel "
            "projection guard. Whole-map minimum fields describe retained evidence; "
            "supplemental_addition fields describe every newly FREE cell."
        ),
    )
    return ScanFreeMemory(
        memory, free, positive, occupied, bits, candidate.view_ids, metadata
    )
