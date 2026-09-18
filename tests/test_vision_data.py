from copy import deepcopy

import pytest

from bb8_rl.vision_data import MULTIVIEW_FORMAT, validate_scene_splits


def dataset(*, multiview=True):
    manifest = {"status": "complete", "sequences": []}
    if multiview:
        manifest["format"] = MULTIVIEW_FORMAT
    for group, split in enumerate(("train", "validation", "development")):
        for camera in "ABC" if multiview else "A":
            identifier = len(manifest["sequences"])
            entry = {
                "id": identifier,
                "split": split,
                "seed": 740000 + group,
                "path": f"sequence-{identifier:03}.npz",
                "sha256": "a" * 64,
                "frames": 2,
            }
            if multiview:
                entry.update(scene_group=group, camera_id=camera)
                entry["native_frames"] = [
                    {
                        **{
                            kind: f"scenes/{identifier:03}/{frame}-{kind}.png"
                            for kind in ("rgb", "floor", "self")
                        },
                        "record_index": frame,
                        "sha256": {kind: "b" * 64 for kind in ("rgb", "floor", "self")},
                    }
                    for frame in range(2)
                ]
            manifest["sequences"].append(entry)
    return manifest


@pytest.mark.parametrize("multiview", [True, False])
def test_accepts_complete_grouped_or_legacy_manifest_without_mutation(multiview):
    manifest = dataset(multiview=multiview)
    original = deepcopy(manifest)
    assert validate_scene_splits(manifest) is None
    assert manifest == original


def test_prevents_room_leakage_even_when_camera_and_sequence_differ():
    manifest = dataset()
    manifest["sequences"][3]["scene_group"] = 0
    with pytest.raises(ValueError, match="scene_group.*overlaps across splits"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize("multiview", [True, False])
def test_prevents_seed_leakage_despite_distinct_scene_names(multiview):
    manifest = dataset(multiview=multiview)
    index = 3 if multiview else 1
    manifest["sequences"][index]["seed"] = manifest["sequences"][0]["seed"]
    with pytest.raises(ValueError, match="seed.*overlaps across splits"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize("field", ["id", "path"])
def test_rejects_sequence_aliases(field):
    manifest = dataset()
    manifest["sequences"][1][field] = manifest["sequences"][0][field]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_scene_splits(manifest)


def test_requires_all_three_views_per_room():
    manifest = dataset()
    del manifest["sequences"][1]
    with pytest.raises(ValueError, match="exactly cameras A/B/C"):
        validate_scene_splits(manifest)


def test_rejects_duplicate_camera_with_distinct_archive():
    manifest = dataset()
    manifest["sequences"][1]["camera_id"] = "A"
    with pytest.raises(ValueError, match="Duplicate camera"):
        validate_scene_splits(manifest)


def test_group_cannot_combine_different_geometry_seeds():
    manifest = dataset()
    manifest["sequences"][1]["seed"] = 900000
    with pytest.raises(ValueError, match="share one seed"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize("field", ["scene_group", "camera_id", "native_frames"])
def test_multiview_schema_requires_group_camera_and_native_records(field):
    manifest = dataset()
    del manifest["sequences"][0][field]
    with pytest.raises(ValueError, match="Multiview"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize("index", [0, 2, -1, True])
def test_rejects_duplicate_out_of_range_or_invalid_frame_indices(index):
    manifest = dataset()
    manifest["sequences"][0]["native_frames"][1]["record_index"] = index
    with pytest.raises(ValueError, match="record_index"):
        validate_scene_splits(manifest)


def test_rejects_missing_native_frame():
    manifest = dataset()
    manifest["sequences"][0]["native_frames"].pop()
    with pytest.raises(ValueError, match="count must match"):
        validate_scene_splits(manifest)


def test_rejects_reused_native_file_across_views():
    manifest = dataset()
    first, second = manifest["sequences"][:2]
    second["native_frames"][0]["floor"] = first["native_frames"][0]["floor"]
    with pytest.raises(ValueError, match="Duplicate dataset path"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize("digest", [None, "a" * 63, "z" * 64])
def test_rejects_incomplete_or_malformed_native_hash(digest):
    manifest = dataset()
    manifest["sequences"][0]["native_frames"][0]["sha256"]["self"] = digest
    with pytest.raises(ValueError, match="SHA256"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize(
    "path", ["/outside.png", "../other.png", "a/../b.png", "a//b.png", "C:\\a.png"]
)
def test_rejects_absolute_traversing_or_aliased_native_paths(path):
    manifest = dataset()
    manifest["sequences"][0]["native_frames"][0]["rgb"] = path
    with pytest.raises(ValueError, match="canonical relative path"):
        validate_scene_splits(manifest)


def test_complete_status_does_not_hide_missing_evaluation_split():
    manifest = dataset()
    manifest["sequences"] = manifest["sequences"][:6]
    with pytest.raises(ValueError, match="missing.*development"):
        validate_scene_splits(manifest)


@pytest.mark.parametrize("manifest", [{}, {"status": "running"}, None])
def test_rejects_incomplete_manifest(manifest):
    with pytest.raises(ValueError, match="complete"):
        validate_scene_splits(manifest)


def test_rejects_unknown_format_instead_of_skipping_multiview_checks():
    manifest = dataset()
    manifest["format"] += "-unknown"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_scene_splits(manifest)
