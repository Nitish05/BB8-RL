import json

import numpy as np
import pytest

from bb8_rl.baseline import RouteController
from bb8_rl.env import NavigationEnv
from bb8_rl.guided import TASKS, teacher_action
from bb8_rl.guided_evaluate import evaluate


def test_teacher_labels_match_existing_feedback_controller():
    with NavigationEnv(TASKS["authored"]) as env:
        reference = RouteController(env.task, env.config.drive.parameters())
        rng = np.random.default_rng(123)
        for _ in range(200):
            delta = rng.uniform(-2, 2, 2)
            velocity = rng.uniform(-0.3, 0.3, 2)
            reference.route = np.array([[0, 0], delta])
            reference.index = 1
            obs = {"achieved_goal": np.zeros(3), "observation": np.r_[0, 0, velocity]}
            actual = teacher_action(np.r_[delta, velocity / 0.35, 1])
            np.testing.assert_allclose(actual, reference.action(obs), atol=1e-6)
    assert np.array_equal(teacher_action([0.02, 0, 0, 0, 1]), [0, 0])


def test_final_test_rejects_unvalidated_or_changed_candidates(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="Final test requires"):
        evaluate("sac", checkpoint, tmp_path / "result", split="test", count=100)
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": "different",
                "algorithm": "sac",
                "status": "complete",
                "split": "validation",
            }
        )
    )
    (validation / "summary.json").write_text(
        json.dumps({"sac": {"episodes": 100, "success_rate": 1.0}})
    )
    with pytest.raises(ValueError, match="has not passed"):
        evaluate(
            "sac",
            checkpoint,
            tmp_path / "result",
            split="test",
            count=100,
            validation=validation,
        )


def test_teacher_batch_and_action_limits():
    rng = np.random.default_rng(5)
    obs = rng.normal(size=(1000, 5)).astype(np.float32)
    batch = teacher_action(obs)
    assert batch.shape == (1000, 2) and np.isfinite(batch).all()
    assert np.max(np.linalg.norm(batch, axis=1)) < 0.65
    np.testing.assert_allclose(
        batch, np.stack([teacher_action(row) for row in obs]), atol=1e-7
    )
