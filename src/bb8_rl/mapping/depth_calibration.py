"""Offline depth calibration against triangulated RGB tracks, never scene truth."""

import hashlib
from itertools import combinations

import numpy as np

from bb8_rl.mapping.scan_geometry import match_calibrated, triangulate_track

FROZEN_QUERY_IDS = frozenset({14, 17, 20, 23})


def sample_bilinear(depth, valid, pixels):
    """Reject boundaries and masked neighbors instead of sampling across holes."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    x, y = pixels.T
    u, v = np.floor(x).astype(int), np.floor(y).astype(int)
    inside = (u >= 0) & (v >= 0) & (u + 1 < depth.shape[1]) & (v + 1 < depth.shape[0])
    result = np.full(len(pixels), np.nan)
    indices = np.flatnonzero(inside)
    uu, vv = u[indices], v[indices]
    good = valid[vv, uu] & valid[vv + 1, uu] & valid[vv, uu + 1] & valid[vv + 1, uu + 1]
    indices, uu, vv = indices[good], uu[good], vv[good]
    dx, dy = x[indices] - uu, y[indices] - vv
    result[indices] = (
        (1 - dx) * (1 - dy) * depth[vv, uu]
        + dx * (1 - dy) * depth[vv, uu + 1]
        + (1 - dx) * dy * depth[vv + 1, uu]
        + dx * dy * depth[vv + 1, uu + 1]
    )
    return result


def triangulated_rgb_anchors(features, view_ids, *, min_views=3, max_error_px=1.0):
    """Three-view RGB tracks with known-pose epipolar and reprojection gates.

    No operating-region, floor-height, object or learned-depth filter is applied.
    """
    if set(view_ids) & FROZEN_QUERY_IDS:
        raise ValueError("Frozen query cannot contribute calibration tracks")
    parent, observations, pair_counts = {}, {}, []

    def find(node):
        parent.setdefault(node, node)
        observations.setdefault(node, {node[0]: node[1]})
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    for a, b in combinations(range(len(features)), 2):
        if view_ids[b] - view_ids[a] > 3:
            continue
        matches = match_calibrated(
            features[a], features[b], ratio=0.75, max_epipolar_px=1.0
        )
        pair_counts.append([view_ids[a], view_ids[b], len(matches)])
        for i, j, _ in matches:
            x, y = find((a, i)), find((b, j))
            if x == y or any(
                k in observations[y] and observations[y][k] != v
                for k, v in observations[x].items()
            ):
                continue
            parent[y] = x
            observations[x].update(observations[y])
    tracks = []
    for node in parent:
        if find(node) != node or len(observations[node]) < min_views:
            continue
        track = observations[node]
        views = sorted(track)
        solved = triangulate_track(
            [features[v]["pixels"][track[v]] for v in views],
            [features[v]["intrinsics"] for v in views],
            [features[v]["world_to_camera"] for v in views],
            max_error_px=max_error_px,
            min_angle_degrees=4.0,
        )
        if solved is None:
            continue
        point, errors, angle = solved
        tracks.append(
            {
                "point": point.tolist(),
                "view_ids": [view_ids[v] for v in views],
                "pixels": [features[v]["pixels"][track[v]].tolist() for v in views],
                "reprojection_max_px": float(max(errors)),
                "parallax_degrees": angle,
            }
        )
    kept = deduplicate_rgb_tracks(tracks)
    return kept, {
        "raw_tracks": len(tracks),
        "deduplicated_tracks": len(kept),
        "pair_match_counts": pair_counts,
    }


def deduplicate_rgb_tracks(tracks):
    """Spatial duplicates cannot leak between calibration and validation folds."""
    kept = []
    for track in sorted(
        tracks, key=lambda t: (-len(t["view_ids"]), t["reprojection_max_px"])
    ):
        if any(
            np.linalg.norm(np.asarray(track["point"]) - old["point"]) < 0.02
            for old in kept
        ):
            continue
        identity = tuple(np.rint(np.asarray(track["point"]) * 50).astype(int))
        digest = hashlib.sha256(repr(identity).encode()).digest()
        track["validation"] = int.from_bytes(digest[:4], "little") % 5 == 0
        kept.append(track)
    return kept


def fit_depth_transform(predicted, target, *, kind="affine"):
    """Huber IRLS fit; samples/weights contain RGB triangulation only."""
    x, y = np.asarray(predicted, float), np.asarray(target, float)
    if (
        x.shape != y.shape
        or x.ndim != 1
        or len(x) < 6
        or not np.isfinite(np.r_[x, y]).all()
        or min(x.min(), y.min()) <= 0
    ):
        raise ValueError("Need at least six finite positive paired depth observations")
    if kind == "scale":
        scale, offset = float(np.median(y / x)), 0.0
    else:
        if kind == "inverse_affine":
            x, y = 1 / x, 1 / y
        elif kind != "affine":
            raise ValueError("Unknown depth transform")
        design = np.c_[x, np.ones(len(x))]
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError("Depth range cannot determine affine transform")
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        for _ in range(30):
            residual = design @ coefficients - y
            scale_mad = max(
                1e-8, 1.4826 * float(np.median(abs(residual - np.median(residual))))
            )
            weights = np.minimum(
                1.0, 1.345 * scale_mad / np.maximum(abs(residual), 1e-12)
            )
            weighted = np.sqrt(weights)
            updated = np.linalg.lstsq(
                design * weighted[:, None], y * weighted, rcond=None
            )[0]
            if np.max(abs(updated - coefficients)) < 1e-9:
                coefficients = updated
                break
            coefficients = updated
        scale, offset = map(float, coefficients)
    if not 0.5 <= scale <= 2.0:
        raise ValueError("Calibration scale outside fixed admissible range [0.5,2]")
    return {"kind": kind, "scale": scale, "offset": offset}


def apply_depth_transform(depth, transform):
    depth = np.asarray(depth)
    if transform["kind"] == "identity":
        return depth.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if transform["kind"] == "inverse_affine":
            result = 1 / (transform["scale"] / depth + transform["offset"])
        else:
            result = transform["scale"] * depth + transform["offset"]
    return result


def error_summary(predicted, target):
    errors = abs(np.asarray(predicted) - np.asarray(target))
    return {
        "count": len(errors),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(max(errors)),
    }


def choose_depth_transform(predicted, target, validation):
    """Choose only using held-out RGB landmarks; truth scoring happens elsewhere.

    Require both validation median and p95 to improve; identity stays available.
    """
    x, y, validation = (
        np.asarray(predicted),
        np.asarray(target),
        np.asarray(validation, bool),
    )
    identity = {"kind": "identity", "scale": 1.0, "offset": 0.0}
    if validation.sum() < 4 or (~validation).sum() < 8:
        return identity, {
            "status": "insufficient_independent_validation",
            "count": len(x),
        }
    baseline = error_summary(x[validation], y[validation])
    candidates = [{"transform": identity, "validation": baseline}]
    for kind in ["scale", "affine", "inverse_affine"]:
        try:
            transform = fit_depth_transform(x[~validation], y[~validation], kind=kind)
            corrected = apply_depth_transform(x[validation], transform)
            if not np.isfinite(corrected).all() or np.any(corrected <= 0):
                continue
            candidates.append(
                {
                    "transform": transform,
                    "validation": error_summary(corrected, y[validation]),
                }
            )
        except ValueError:
            continue
    accepted = [
        c
        for c in candidates
        if c["validation"]["median_m"] <= baseline["median_m"]
        and c["validation"]["p95_m"] <= baseline["p95_m"]
    ]
    best = min(accepted, key=lambda c: c["validation"]["median_m"])
    return best["transform"], {
        "status": "selected_on_rgb_anchor_validation_only",
        "candidates": candidates,
        "selected": best,
        "training_observations": int((~validation).sum()),
        "validation_observations": int(validation.sum()),
    }
