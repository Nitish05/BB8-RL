import numpy as np

from bb8_rl.mapping.registration import (
    build_landmarks,
    collect_query_correspondences,
    epipolar_matches,
    solve_registration,
)
from bb8_rl.mapping.scan_geometry import project


def scene():
    rng = np.random.default_rng(77)
    points = rng.uniform([-0.8, -0.8, 0.05], [0.8, 0.8, 0.8], (30, 3))
    k = np.array([[800.0, 0, 640], [0, 800.0, 480], [0, 0, 1]])
    views = {}
    for i, offset in enumerate(([0, 0], [-0.7, 0], [0, -0.6])):
        transform = np.eye(4)
        transform[:2, 3] = offset
        transform[2, 3] = 3
        views[f"map-{i}"] = {
            "intrinsics": k,
            "world_to_camera": transform,
            "keypoints": project(points, k, transform)[0],
        }
    return points, views


def test_known_geometry_rejects_wrong_feature_coordinate():
    _, views = scene()
    first, second = views["map-0"], views["map-1"]
    matches = np.stack([np.arange(30), np.arange(30)], axis=1)
    assert epipolar_matches(first, second, matches).all()
    second["keypoints"][0, 1] += 15
    accepted = epipolar_matches(first, second, matches)
    assert not accepted[0] and accepted[1:].all()


def test_metric_tracks_require_multiple_map_views_and_recover_points():
    expected, views = scene()
    matches = np.stack([np.arange(30), np.arange(30)], axis=1)
    pairs = [
        {"first": first, "second": second, "matches": matches, "scores": np.ones(30)}
        for first, second in [("map-0", "map-1"), ("map-1", "map-2")]
    ]
    assert len(build_landmarks(views, pairs[:1])["points"]) == 0
    cloud = build_landmarks(views, pairs)
    assert len(cloud["points"]) == 30
    for index in range(30):
        landmark = cloud["lookup"][("map-0", index)]
        np.testing.assert_allclose(
            cloud["points"][landmark], expected[index], atol=1e-12
        )
        assert cloud["lookup"][("map-2", index)] == landmark


def test_repeated_matches_from_one_view_cannot_fake_query_support():
    lookup = {("A", 1): 4, ("B", 2): 4}
    pair = {
        "map_view": "A",
        "matches": np.array([[3, 1], [3, 1]]),
        "scores": np.array([0.7, 0.9]),
    }
    assert not collect_query_correspondences([pair, pair], lookup)
    rows = collect_query_correspondences(
        [
            pair,
            {"map_view": "B", "matches": np.array([[3, 2]]), "scores": np.array([0.8])},
        ],
        lookup,
    )
    assert len(rows) == 1 and rows[0]["support_count"] == 2 and rows[0]["landmark"] == 4


def test_query_associations_are_one_to_one_despite_competing_votes():
    lookup = {("A", 1): 4, ("B", 2): 4, ("A", 3): 5, ("B", 4): 5}
    pairs = [
        {
            "map_view": "A",
            "matches": np.array([[0, 1], [1, 1], [0, 3]]),
            "scores": [0.9, 0.6, 0.5],
        },
        {
            "map_view": "B",
            "matches": np.array([[0, 2], [1, 2], [0, 4]]),
            "scores": [0.9, 0.6, 0.5],
        },
    ]
    result = collect_query_correspondences(pairs, lookup)
    assert (
        len(result) == 1
        and result[0]["query_index"] == 0
        and result[0]["landmark"] == 4
    )


def test_pnp_recovers_query_pose_with_outliers_without_pose_prior():
    points, views = scene()
    k = views["map-0"]["intrinsics"]
    truth = np.eye(4)
    truth[:3, 3] = [0.2, -0.3, 3.5]
    pixels = project(points, k, truth)[0]
    pixels[-8:] += [100, -75]
    matches = [{"query_index": i, "landmark": i} for i in range(len(points))]
    solved = solve_registration(pixels, k, points, matches)
    assert solved["status"] == "candidate" and solved["inliers"] == 22
    np.testing.assert_allclose(solved["world_to_camera"], truth, atol=1e-6)
    assert solve_registration(pixels, k, points, matches[:5])["status"] == "rejected"
