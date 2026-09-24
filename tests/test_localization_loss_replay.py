import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "localization_loss_replay", ROOT / "scripts/replay-localization-loss.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def write_prefix(path, *, broken_timeline=False):
    with path.open("w") as stream:
        for index in range(40):
            timestamp = index * 0.05
            timeline = (
                []
                if index == 0
                else REPLAY.zero_timeline((index - 1) * 0.05, timestamp)
            )
            if broken_timeline:
                timeline = []
            row = {
                "measurement": {
                    "timestamp": timestamp,
                    "status": "visible",
                    "xy": [0.0, 0.0],
                    "covariance": [[0.000004, 0.0], [0.0, 0.000004]],
                },
                "prior_applied_command": [0.0, 0.0],
                "prior_acknowledged_command_intervals": timeline,
                "truth_before_scoring_only": "UNUSABLE_TRUTH_MUST_NOT_BE_AN_INPUT",
                "head_pixels_scoring_only": {"A": -999},
            }
            stream.write(json.dumps(row) + "\n")


def test_replay_strips_truth_expires_pose_and_rejects_goal(tmp_path):
    source, output = tmp_path / "native-shaped-unit-fixture.jsonl", tmp_path / "out"
    write_prefix(source)
    report = REPLAY.replay(source, output, duration_seconds=3.0)
    assert report["passed"] and all(report["checks"].values())
    assert not report["full_240_second_bookkeeping"]
    assert report["first_lost_offset_seconds"] == pytest.approx(1.05)
    assert report["goal_attempt"]["record"]["outcome"] == "rejected"
    assert report["synthetic_frames"] == 60
    assert report["prefix_rows"] < 40
    assert report["checks"]["recovery_anchor_valid_and_fixed_after_loss"]
    assert report["checks"]["existing_prediction_limits_respected"]
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["truth_to_replay"] is False
    assert "counterfactual" in protocol["scope"]
    for line in (output / "prefix-inputs.jsonl").read_text().splitlines():
        assert set(json.loads(line)) == set(REPLAY.INPUT_KEYS)
        assert "UNUSABLE_TRUTH" not in line
        assert "head_pixels" not in line
    final = json.loads((output / "replay.jsonl").read_text().splitlines()[-1])
    assert final["kind"] == "synthetic_missing_rgb_zero_ack_extension"
    assert final["state"]["pose"] is None
    assert final["state"]["position_radius"] is None
    assert (
        final["state"]["raw_position_radius"]
        > report["samples"][0]["state"]["raw_position_radius"]
    )
    assert final["state"]["braking_prediction"]["valid"]
    lost_rows = [
        json.loads(line)
        for line in (output / "replay.jsonl").read_text().splitlines()
        if json.loads(line)["state"]["localization_status"] == "lost"
    ]
    assert len(lost_rows) > 1
    for row in lost_rows:
        assert {
            key: row["state"][key] for key in REPLAY.RECOVERY_ANCHOR_KEYS
        } == report["recovery_anchor"]
    with pytest.raises(ValueError, match="fresh"):
        REPLAY.replay(source, output, duration_seconds=3.0)


def test_incomplete_native_chronology_cannot_be_selected_as_stable_prefix(tmp_path):
    source = tmp_path / "bad-prefix.jsonl"
    write_prefix(source, broken_timeline=True)
    with pytest.raises(ValueError, match="not yet stably localized"):
        REPLAY.replay(source, tmp_path / "out", duration_seconds=3.0)
    assert not (tmp_path / "out").exists()


def test_replay_detects_a_recovery_anchor_that_follows_expired_prediction(
    tmp_path, monkeypatch
):
    original = REPLAY.ControlSession

    class MovingAnchorSession(original):
        def state(self):
            value = super().state()
            if value["localization_status"] == "lost":
                value["recovery_anchor_pose"][0] += self.capture_time
            return value

    monkeypatch.setattr(REPLAY, "ControlSession", MovingAnchorSession)
    source = tmp_path / "prefix.jsonl"
    write_prefix(source)
    report = REPLAY.replay(source, tmp_path / "out", duration_seconds=3.0)
    assert not report["passed"]
    assert not report["checks"]["recovery_anchor_valid_and_fixed_after_loss"]
