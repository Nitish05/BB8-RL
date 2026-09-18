"""Calibrated RGB feature triangulation and held-out camera registration.

Known metric scan poses establish scale. This is not unknown-pose monocular SLAM.
Sparse points describe observed surfaces, never certified empty volume.
"""

from itertools import combinations

import cv2
import numpy as np


def project(points, intrinsics, world_to_camera):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    camera = np.c_[points, np.ones(len(points))] @ np.asarray(world_to_camera)[:3].T
    pixels = camera @ np.asarray(intrinsics).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


def triangulate_track(
    pixels, intrinsics, transforms, *, max_error_px=2.0, min_angle_degrees=2.0
):
    """DLT using all views; reject negative depth, weak baseline and inconsistency."""
    pixels = np.asarray(pixels, dtype=float)
    intrinsics, transforms = (
        np.asarray(intrinsics, dtype=float),
        np.asarray(transforms, dtype=float),
    )
    count = len(pixels)
    if (
        count < 2
        or pixels.shape != (count, 2)
        or intrinsics.shape != (count, 3, 3)
        or transforms.shape != (count, 4, 4)
        or not all(np.isfinite(v).all() for v in (pixels, intrinsics, transforms))
    ):
        raise ValueError("Need finite corresponding pixels and calibrated cameras")
    rows = []
    for xy, k, t in zip(pixels, intrinsics, transforms):
        p = k @ t[:3]
        rows.extend((xy[0] * p[2] - p[0], xy[1] * p[2] - p[1]))
    _, _, vh = np.linalg.svd(np.array(rows))
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-10:
        return None
    xyz = homogeneous[:3] / homogeneous[3]
    errors, rays = [], []
    for xy, k, t in zip(pixels, intrinsics, transforms):
        uv, depth = project(xyz, k, t)
        if depth[0] <= 0 or not np.isfinite(uv).all():
            return None
        errors.append(float(np.linalg.norm(uv[0] - xy)))
        center = -t[:3, :3].T @ t[:3, 3]
        ray = xyz - center
        rays.append(ray / np.linalg.norm(ray))
    angle = max(
        np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))
        for a, b in combinations(rays, 2)
    )
    if max(errors) > max_error_px or angle < min_angle_degrees:
        return None
    return xyz, np.asarray(errors), float(angle)


def fundamental(k1, t1, k2, t2):
    relative = t2 @ np.linalg.inv(t1)
    x, y, z = relative[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.linalg.inv(k2).T @ skew @ relative[:3, :3] @ np.linalg.inv(k1)


def match_calibrated(first, second, *, ratio=0.75, max_epipolar_px=1.5):
    """Mutual descriptor ratios plus known-geometry symmetric epipolar distance."""
    if first["descriptors"] is None or second["descriptors"] is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(first["descriptors"], second["descriptors"], k=2)
    reverse = matcher.knnMatch(second["descriptors"], first["descriptors"], k=2)
    backward = {
        m.queryIdx: m.trainIdx
        for pair in reverse
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    }
    f = fundamental(
        first["intrinsics"],
        first["world_to_camera"],
        second["intrinsics"],
        second["world_to_camera"],
    )
    result = []
    for pair in forward:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance >= ratio * n.distance or backward.get(m.trainIdx) != m.queryIdx:
            continue
        a = np.r_[first["pixels"][m.queryIdx], 1]
        b = np.r_[second["pixels"][m.trainIdx], 1]
        la, lb = f.T @ b, f @ a
        error = max(
            abs(a @ la) / max(np.linalg.norm(la[:2]), 1e-10),
            abs(b @ lb) / max(np.linalg.norm(lb[:2]), 1e-10),
        )
        if error <= max_epipolar_px:
            result.append((m.queryIdx, m.trainIdx, m.distance))
    return sorted(result, key=lambda row: row[2])


def extract(rgb, intrinsics, world_to_camera, *, nfeatures=5000):
    detector = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    keypoints, descriptors = detector.detectAndCompute(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None
    )
    return {
        "pixels": np.array([k.pt for k in keypoints], dtype=float).reshape(-1, 2),
        "descriptors": descriptors,
        "intrinsics": np.asarray(intrinsics, dtype=float),
        "world_to_camera": np.asarray(world_to_camera, dtype=float),
        "rgb": rgb,
    }


def reconstruct(features, *, min_views=3, neighbors=3, max_error_px=2.0):
    """Merge consistent matches into tracks, then triangulate metric landmarks."""
    parent, observations = {}, {}

    def find(node):
        parent.setdefault(node, node)
        observations.setdefault(node, {node[0]: node[1]})
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    pairs = []
    for a, b in combinations(range(len(features)), 2):
        if b - a > neighbors:
            continue
        matches = match_calibrated(features[a], features[b])
        pairs.append({"first": a, "second": b, "matches": len(matches)})
        for i, j, _ in matches:
            x, y = find((a, i)), find((b, j))
            if x == y:
                continue
            # One feature per camera: conflicting track merges stay separate.
            if any(
                k in observations[y] and observations[y][k] != v
                for k, v in observations[x].items()
            ):
                continue
            parent[y] = x
            observations[x].update(observations[y])
    tracks = [
        observations[node]
        for node in parent
        if find(node) == node and len(observations[node]) >= min_views
    ]
    landmarks, descriptor_rows, descriptor_landmarks, colors, metadata = (
        [],
        [],
        [],
        [],
        [],
    )
    for track in tracks:
        views = sorted(track)
        pixels = [features[v]["pixels"][track[v]] for v in views]
        solved = triangulate_track(
            pixels,
            [features[v]["intrinsics"] for v in views],
            [features[v]["world_to_camera"] for v in views],
            max_error_px=max_error_px,
        )
        if solved is None:
            continue
        xyz, errors, angle = solved
        # Configured operating region, not obstacle labels or map completion.
        if np.max(abs(xyz[:2])) > 2.5 or not -0.2 <= xyz[2] <= 2:
            continue
        index = len(landmarks)
        landmarks.append(xyz)
        metadata.append(
            {
                "views": views,
                "keypoint_indices": [track[v] for v in views],
                "reprojection_errors_px": errors.tolist(),
                "maximum_parallax_degrees": angle,
            }
        )
        rgb = features[views[0]]["rgb"]
        u, v = np.rint(pixels[0]).astype(int)
        colors.append(
            rgb[np.clip(v, 0, rgb.shape[0] - 1), np.clip(u, 0, rgb.shape[1] - 1)]
        )
        for view in views:
            descriptor_rows.append(features[view]["descriptors"][track[view]])
            descriptor_landmarks.append(index)
    return {
        "points": np.asarray(landmarks, dtype=float).reshape(-1, 3),
        "colors": np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        "descriptors": np.asarray(descriptor_rows, dtype=np.float32).reshape(-1, 128),
        "descriptor_landmarks": np.asarray(descriptor_landmarks, dtype=np.int64),
        "tracks": metadata,
        "pairs": pairs,
        "candidate_tracks": len(tracks),
    }


def distinct_landmark_matches(
    query_descriptors, descriptors, landmark_ids, *, ratio=0.75
):
    """Ratio test against different landmarks, despite many descriptors per point."""
    if len(descriptors) < 2 or len(np.unique(landmark_ids)) < 2:
        return []
    matcher = cv2.BFMatcher()
    initial_k = min(8, len(descriptors))
    matches = matcher.knnMatch(query_descriptors, descriptors, k=initial_k)
    candidates = []
    for query_index, neighbors in enumerate(matches):
        k = initial_k
        while True:
            distinct = []
            for match in neighbors:
                landmark = int(landmark_ids[match.trainIdx])
                if landmark not in [row[0] for row in distinct]:
                    distinct.append((landmark, match))
                if len(distinct) == 2:
                    break
            if len(distinct) == 2 or k == len(descriptors):
                break
            k = min(k * 2, len(descriptors))
            neighbors = matcher.knnMatch(
                query_descriptors[query_index : query_index + 1], descriptors, k=k
            )[0]
        if (
            len(distinct) == 2
            and distinct[0][1].distance < ratio * distinct[1][1].distance
        ):
            candidates.append((query_index, distinct[0][0], distinct[0][1].distance))
    # One query per point, retaining its closest descriptor correspondence.
    used, selected = set(), []
    for candidate in sorted(candidates, key=lambda row: row[2]):
        if candidate[1] not in used:
            selected.append(candidate)
            used.add(candidate[1])
    return selected


def register_camera(rgb, intrinsics, cloud, *, ratio=0.75, min_inliers=12):
    """Estimate a fixed camera pose from query RGB + K + reconstructed landmarks.

    No query ground-truth pose, floor dimensions or object labels enter PnP.
    A result is a candidate registration; physical accuracy is independently scored.
    """
    query = extract(rgb, intrinsics, np.eye(4))
    if query["descriptors"] is None or len(cloud["descriptors"]) < 2:
        return {"status": "insufficient_features"}
    matches = distinct_landmark_matches(
        query["descriptors"],
        cloud["descriptors"],
        cloud["descriptor_landmarks"],
        ratio=ratio,
    )
    points = [cloud["points"][landmark] for _, landmark, _ in matches]
    pixels = [query["pixels"][q] for q, _, _ in matches]
    if len(points) < min_inliers:
        return {"status": "insufficient_matches", "matches": len(points)}
    points, pixels = np.asarray(points), np.asarray(pixels)
    cv2.setRNGSeed(76)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points,
        pixels,
        np.asarray(intrinsics),
        None,
        iterationsCount=3000,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return {
            "status": "registration_rejected",
            "matches": len(points),
            "inliers": 0 if inliers is None else len(inliers),
        }
    indices = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(
        points[indices], pixels[indices], np.asarray(intrinsics), None, rvec, tvec
    )
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    projection, depth = project(points[indices], intrinsics, transform)
    errors = np.linalg.norm(projection - pixels[indices], axis=1)
    if np.any(depth <= 0) or np.quantile(errors, 0.95) > 2:
        return {
            "status": "registration_rejected",
            "reason": "reprojection_or_cheirality",
        }
    return {
        "status": "candidate",
        "world_to_camera": transform.tolist(),
        "matches": len(points),
        "inliers": len(indices),
        "reprojection_p95_px": float(np.quantile(errors, 0.95)),
        "landmark_extent_z_m": float(np.ptp(points[indices, 2])),
        "landmark_spread_singular_values_m": np.linalg.svd(
            points[indices] - points[indices].mean(0), compute_uv=False
        ).tolist(),
    }


def save_ply(path, points, colors):
    with open(path, "w") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        stream.writelines(
            " ".join([*(f"{v:.7f}" for v in point), *(str(int(v)) for v in color)])
            + "\n"
            for point, color in zip(points, colors)
        )
