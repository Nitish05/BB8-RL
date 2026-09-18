"""Filesystem-independent checks for room-disjoint vision dataset manifests."""

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

MULTIVIEW_FORMAT = "bb8-multiview-supervision-v1"
_SPLITS = frozenset(("train", "validation", "development"))
_CAMERAS = frozenset(("A", "B", "C"))
_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _relative_path(value, label):
    # Canonical paths also prevent aliases such as a/../b or a//b from
    # defeating the duplicate-file checks. Dataset manifests use POSIX paths.
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    return value


def _digest(value, label):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA256 hexadecimal digest")


def validate_scene_splits(manifest):
    """Reject leakage and inconsistent manifest records, without reading files.

    All three splits must be present and nonempty. Legacy single-view manifests
    may omit ``format``, ``scene_group`` and ``native_frames``; their sequence
    IDs identify scene groups. The multiview format requires one A/B/C sequence
    per group, a common group seed/split, and complete native frame references.

    Native ``record_index`` values must cover every sequence frame exactly once.
    Paths must be unique across sequence archives and native RGB/floor/self
    files. Hash syntax is checked here; consumers must verify file bytes and
    labels.json record counts when loading them. This function never accesses
    renderer labels or the filesystem, never mutates its input, and returns None.
    """
    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise ValueError("Require a complete vision dataset manifest")
    data_format = manifest.get("format")
    if data_format not in (None, MULTIVIEW_FORMAT):
        raise ValueError(f"Unsupported vision dataset format: {data_format!r}")
    multiview = data_format == MULTIVIEW_FORMAT
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("Dataset sequences must be a nonempty list")

    ids, paths, present_splits = set(), set(), set()
    group_splits, seed_splits, groups = {}, {}, {}

    def reserve_path(value, label):
        path = _relative_path(value, label)
        if path in paths:
            raise ValueError(f"Duplicate dataset path: {path}")
        paths.add(path)

    for entry in sequences:
        if not isinstance(entry, Mapping):
            raise ValueError("Every sequence must be a mapping")  # noqa: TRY004 - manifest validation contract
        identifier = _integer(entry.get("id"), "Sequence id")
        if identifier in ids:
            raise ValueError(f"Duplicate sequence id: {identifier}")
        ids.add(identifier)
        split = entry.get("split")
        if not isinstance(split, str) or split not in _SPLITS:
            raise ValueError(f"Invalid split for sequence {identifier}")
        present_splits.add(split)
        reserve_path(entry.get("path"), f"Sequence {identifier} path")
        if "sha256" in entry:
            _digest(entry["sha256"], f"Sequence {identifier} sha256")
        seed = _integer(entry.get("seed"), f"Sequence {identifier} seed")
        frames = _integer(entry.get("frames"), "Sequence frames", minimum=1)
        if multiview and "scene_group" not in entry:
            raise ValueError("Multiview sequences require scene_group")
        group = entry.get("scene_group", identifier)
        if (
            isinstance(group, bool)
            or not isinstance(group, (int, str))
            or (isinstance(group, str) and not group)
            or (isinstance(group, int) and group < 0)
        ):
            raise ValueError(
                "scene_group must be a nonnegative integer or nonempty string"
            )
        for value, seen, name in (
            (group, group_splits, "scene_group"),
            (seed, seed_splits, "seed"),
        ):
            if value in seen and seen[value] != split:
                raise ValueError(f"{name} {value!r} overlaps across splits")
            seen[value] = split

        if multiview:
            camera = entry.get("camera_id")
            if not isinstance(camera, str) or camera not in _CAMERAS:
                raise ValueError("Multiview camera_id must be A, B or C")
            previous_seed, cameras = groups.setdefault(group, (seed, set()))
            if previous_seed != seed:
                raise ValueError(f"Scene group {group!r} must share one seed")
            if camera in cameras:
                raise ValueError(f"Duplicate camera {camera} in scene group {group!r}")
            cameras.add(camera)
            if "native_frames" not in entry:
                raise ValueError("Multiview sequences require native_frames")

        if "native_frames" not in entry:
            continue
        native = entry["native_frames"]
        if not isinstance(native, list) or len(native) != frames:
            raise ValueError("native_frames count must match sequence frames")
        indices = set()
        for record in native:
            if not isinstance(record, Mapping):
                raise ValueError("Every native frame must be a mapping")  # noqa: TRY004 - manifest validation contract
            index = _integer(record.get("record_index"), "Native record_index")
            if index >= frames or index in indices:
                raise ValueError(
                    "Native record_index must cover every frame exactly once"
                )
            indices.add(index)
            hashes = record.get("sha256")
            if not isinstance(hashes, Mapping):
                raise ValueError("Native frame requires RGB/floor/self sha256 hashes")  # noqa: TRY004 - manifest validation contract
            for kind in ("rgb", "floor", "self"):
                reserve_path(record.get(kind), f"Native {kind} path")
                _digest(hashes.get(kind), f"Native {kind} sha256")

    missing_splits = _SPLITS - present_splits
    if missing_splits:
        raise ValueError(
            f"Need nonempty train/validation/development splits: missing {sorted(missing_splits)}"
        )
    for group, (_, cameras) in groups.items():
        if cameras != _CAMERAS:
            raise ValueError(f"Scene group {group!r} requires exactly cameras A/B/C")
