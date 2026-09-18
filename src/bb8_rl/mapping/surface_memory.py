"""Store estimated scan surfaces conservatively, retaining view provenance.

This importer does not reconstruct free volume, identify static objects, or
validate reconstruction accuracy. Unseen and unsampled cells remain unknown.
"""

import math

import numpy as np

from .contracts import Bounds3D, EvidenceSource, Provenance, SpaceState
from .room_memory import RoomMemory


def import_surfaces(
    points_by_view,
    valid_masks,
    view_ids,
    *,
    bounds,
    resolution_m,
    uncertainty_m,
    scene_version,
    calibration_version,
    minimum_height_m=0.04,
):
    points = np.asarray(points_by_view)
    masks = np.asarray(valid_masks)
    if (
        points.ndim != 4
        or points.shape[-1] != 3
        or masks.shape != points.shape[:-1]
        or masks.dtype != bool
        or len(view_ids) != points.shape[0]
        or len(set(view_ids)) != len(view_ids)
        or not all(isinstance(v, str) and v for v in view_ids)
        or not math.isfinite(uncertainty_m)
        or uncertainty_m < 0
        or not math.isfinite(minimum_height_m)
    ):
        raise ValueError("Invalid surface arrays, source views or assumed margin")
    memory = RoomMemory(
        bounds,
        resolution_m,
        scene_version=scene_version,
        calibration_version=calibration_version,
    )
    voxels = {}
    low, high = np.array(bounds.minimum), np.array(bounds.maximum)
    sample_count = 0
    for view, cloud, mask in zip(view_ids, points, masks):
        keep = mask & np.isfinite(cloud).all(axis=-1)
        keep &= (cloud >= low).all(axis=-1) & (cloud < high).all(axis=-1)
        keep &= cloud[..., 2] >= minimum_height_m
        sample_count += int(keep.sum())
        indices = np.floor((cloud[keep] - low) / resolution_m).astype(int)
        for cell in np.unique(indices, axis=0):
            voxels.setdefault(tuple(cell), set()).add(view)
    for number, (index, views) in enumerate(sorted(voxels.items())):
        origin = low + np.array(index) * resolution_m
        source = EvidenceSource(
            f"surface-voxel-{number}",
            tuple(sorted(views)),
            0.0,
            Provenance.RGB_RECONSTRUCTION,
            uncertainty_m,
        )
        memory.observe_volume(
            Bounds3D(tuple(origin), tuple(np.minimum(origin + resolution_m, high))),
            SpaceState.OCCUPIED,
            source,
        )
    return memory, {
        "retained_surface_samples": sample_count,
        "occupied_source_voxels": len(voxels),
        "free_space_certified": False,
        "minimum_height_m": minimum_height_m,
        "height_filter_semantics": "Known world support-plane prior; lower samples retained in source cloud, not obstacle memory.",
        "uncertainty_m": uncertainty_m,
        "uncertainty_status": "assumed import margin, not calibrated coverage",
        "surface_semantics": "Unclassified scan-time surfaces, potentially including the parked robot; persistent static object identity is not established.",
    }
