"""Learned RGB correspondence registration with independently calibrated map views.

Optional LightGlue dependencies load only on explicit construction. No renderer
depth, query pose, object geometry or navigation action enters this module.
"""

from collections import defaultdict

import cv2
import numpy as np

from .scan_geometry import fundamental, project, triangulate_track


def epipolar_matches(first, second, matches, *, threshold_px=1.5):
    """Filter learned map-to-map matches using known metric scan cameras."""
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    if not len(matches):
        return np.zeros(0, bool)
    f = fundamental(
        first["intrinsics"],
        first["world_to_camera"],
        second["intrinsics"],
        second["world_to_camera"],
    )
    a = np.c_[first["keypoints"][matches[:, 0]], np.ones(len(matches))]
    b = np.c_[second["keypoints"][matches[:, 1]], np.ones(len(matches))]
    la, lb = b @ f, a @ f.T
    numerator = abs(np.sum(a * la, axis=1))
    error = np.maximum(
        numerator / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12),
        numerator / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12),
    )
    return np.isfinite(error) & (error <= threshold_px)


def build_landmarks(views, pairs, *, min_views=3, reprojection_px=2.0):
    """Merge geometry-checked learned matches and triangulate fixed-camera tracks.

    Each track contains at most one point per image. A track is accepted only
    when all its image observations pass depth, parallax and reprojection tests.
    The query views must be excluded by the caller, before pair generation.
    """
    parent, observations = {}, {}

    def find(node):
        parent.setdefault(node, node)
        observations.setdefault(node, {node[0]: node[1]})
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    edges, statistics = [], []
    for pair in pairs:
        first, second = pair["first"], pair["second"]
        matches, scores = np.asarray(pair["matches"]), np.asarray(pair["scores"])
        keep = epipolar_matches(views[first], views[second], matches)
        statistics.append(
            {
                "first": first,
                "second": second,
                "matches": len(matches),
                "epipolar_accepted": int(keep.sum()),
            }
        )
        edges.extend(
            (float(score), (first, int(a)), (second, int(b)))
            for (a, b), score in zip(matches[keep], scores[keep])
        )
    for _, first, second in sorted(edges, reverse=True):
        x, y = find(first), find(second)
        if x == y or any(
            view in observations[y] and observations[y][view] != point
            for view, point in observations[x].items()
        ):
            continue
        parent[y] = x
        observations[x].update(observations[y])
    tracks = [
        observations[node]
        for node in parent
        if find(node) == node and len(observations[node]) >= min_views
    ]
    points, metadata, lookup = [], [], {}
    for track in tracks:
        names = sorted(track)
        pixels = [views[name]["keypoints"][track[name]] for name in names]
        solved = triangulate_track(
            pixels,
            [views[name]["intrinsics"] for name in names],
            [views[name]["world_to_camera"] for name in names],
            max_error_px=reprojection_px,
            min_angle_degrees=2.0,
        )
        if solved is None:
            continue
        point, errors, angle = solved
        if np.max(abs(point[:2])) > 2.5 or not -0.2 <= point[2] <= 2.0:
            continue
        identifier = len(points)
        points.append(point)
        nodes = [(name, int(track[name])) for name in names]
        for node in nodes:
            lookup[node] = identifier
        metadata.append(
            {
                "observations": nodes,
                "reprojection_errors_px": errors.tolist(),
                "maximum_parallax_degrees": angle,
            }
        )
    return {
        "points": np.asarray(points, dtype=float).reshape(-1, 3),
        "tracks": metadata,
        "lookup": lookup,
        "pair_statistics": statistics,
        "candidate_tracks": len(tracks),
    }


def collect_query_correspondences(pairs, lookup, *, minimum_views=2):
    """Require the same query keypoint/landmark association in multiple map views."""
    votes = defaultdict(dict)
    for pair in pairs:
        name = pair["map_view"]
        for (query_index, map_index), score in zip(pair["matches"], pair["scores"]):
            landmark = lookup.get((name, int(map_index)))
            if landmark is not None:
                key = (int(query_index), landmark)
                votes[key][name] = max(float(score), votes[key].get(name, 0))
    ranked = sorted(
        (
            (len(support), sum(support.values()), query, landmark, sorted(support))
            for (query, landmark), support in votes.items()
            if len(support) >= minimum_views
        ),
        reverse=True,
    )
    used_query, used_landmark, result = set(), set(), []
    for count, score, query, landmark, support in ranked:
        if query in used_query or landmark in used_landmark:
            continue
        used_query.add(query)
        used_landmark.add(landmark)
        result.append(
            {
                "query_index": query,
                "landmark": landmark,
                "support_count": count,
                "score_sum": score,
                "support_views": support,
            }
        )
    return result


def solve_registration(
    query_pixels, intrinsics, points, correspondences, *, minimum_inliers=12
):
    """PnP candidate from measured pixels and metric RGB landmarks, no pose prior."""
    if len(correspondences) < minimum_inliers:
        return {
            "status": "rejected",
            "reason": "insufficient_supported_correspondences",
            "matches": len(correspondences),
            "world_to_camera": None,
        }
    xyz = np.asarray([points[row["landmark"]] for row in correspondences], dtype=float)
    pixels = np.asarray(
        [query_pixels[row["query_index"]] for row in correspondences], dtype=float
    )
    if not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
        raise ValueError("Nonfinite query correspondences")
    cv2.setRNGSeed(77)
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        xyz,
        pixels,
        np.asarray(intrinsics, dtype=float),
        None,
        iterationsCount=3000,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < minimum_inliers:
        return {
            "status": "rejected",
            "reason": "pnp_inliers",
            "matches": len(xyz),
            "inliers": 0 if inliers is None else len(inliers),
            "world_to_camera": None,
        }
    indices = inliers.ravel()
    rotation, translation = cv2.solvePnPRefineLM(
        xyz[indices],
        pixels[indices],
        np.asarray(intrinsics, dtype=float),
        None,
        rotation,
        translation,
    )
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.ravel()
    projected, depth = project(xyz, intrinsics, transform)
    errors = np.linalg.norm(projected - pixels, axis=1)
    if np.any(depth[indices] <= 0) or np.quantile(errors[indices], 0.95) > 2:
        return {
            "status": "rejected",
            "reason": "reprojection_or_cheirality",
            "world_to_camera": None,
        }
    spread = np.linalg.svd(xyz[indices] - xyz[indices].mean(0), compute_uv=False)
    if spread[1] < 0.05:
        return {
            "status": "rejected",
            "reason": "collinear_landmarks",
            "world_to_camera": None,
        }
    return {
        "status": "candidate",
        "world_to_camera": transform.tolist(),
        "matches": len(xyz),
        "inliers": len(indices),
        "inlier_indices": indices.tolist(),
        "reprojection_p50_p95_max_px": np.quantile(
            errors[indices], [0.5, 0.95, 1]
        ).tolist(),
        "all_correspondence_reprojection_errors_px": errors.tolist(),
        "landmark_spread_singular_values_m": spread.tolist(),
    }
