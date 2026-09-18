import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "extended_scan_builder",
    Path(__file__).parents[1] / "scripts/build-extended-scan-memory.py",
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def capture_record(plan):
    names = [*BUILDER.ORIGINAL_IDS, *BUILDER.SUPPLEMENTAL_IDS]
    return {
        "status": "complete",
        "plan_sha256": BUILDER.digest(plan),
        "depth_used": False,
        "segmentation_used": False,
        "additional_view_ids": list(BUILDER.SUPPLEMENTAL_IDS),
        "mapping_input_view_ids": names,
        "input_sha256": {
            name: {"rgb": "rgb", "calibration": "calibration"} for name in names
        },
    }


@pytest.mark.parametrize("mutation", ["query", "incomplete", "depth", "different_plan"])
def test_capture_manifest_rejects_unfrozen_ancestry_or_non_rgb_inputs(
    tmp_path, mutation
):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    record = capture_record(plan)
    if mutation == "query":
        record["mapping_input_view_ids"][-1] = "scan-023"
    elif mutation == "incomplete":
        record["status"] = "running"
    elif mutation == "depth":
        record["depth_used"] = True
    else:
        record["plan_sha256"] = "changed"
    manifest = tmp_path / "capture.json"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen RGB-only plan"):
        BUILDER.capture_input_hashes(manifest, plan)


def test_capture_manifest_pins_all_32_exact_allowed_ids(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    record = capture_record(plan)
    manifest = tmp_path / "capture.json"
    manifest.write_text(json.dumps(record))
    assert BUILDER.capture_input_hashes(manifest, plan) == record["input_sha256"]


def test_scan_reader_hashchecks_rgb_calibration_and_never_reads_queries(tmp_path):
    rgb_path, meta_path = tmp_path / "scan-024.png", tmp_path / "scan-024.json"
    Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(rgb_path)
    meta = {
        "calibration": "synthetic_exact",
        "depth_used": False,
        "segmentation_used": False,
        "intrinsics": [[10.0, 0.0, 6.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]],
        "world_to_camera": np.eye(4).tolist(),
    }
    meta_path.write_text(json.dumps(meta))
    expected = {
        "rgb": BUILDER.digest(rgb_path),
        "calibration": BUILDER.digest(meta_path),
    }
    view, hashes, _ = BUILDER.read_view(tmp_path, "scan-024", expected)
    assert view.rgb.shape == (10, 12, 3) and hashes == expected
    meta["intrinsics"][0][0] = 11.0
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hash mismatch"):
        BUILDER.read_view(tmp_path, "scan-024", expected)
    with pytest.raises(ValueError, match="Only frozen"):
        BUILDER.read_view(tmp_path, "scan-023")
