from copy import deepcopy

import numpy as np
import pytest

from bb8_rl.occlusion_belief import OcclusionBelief, observation_input


@pytest.mark.parametrize("model", ["cv", "ca"])
def test_missing_observation_never_becomes_new_visual_evidence(model):
    belief = OcclusionBelief(model)
    unseen = belief.update(0)
    assert unseen["mode"] == "uninitialized" and unseen["position"] is None
    observed = belief.update(
        0.1, xy=[1, 2], covariance=np.eye(2) * 0.0001, status="visible"
    )
    previous_radius = observed["assumed_position_radius_m"]
    for time in (0.2, 0.5, 1.0, 2.0):
        prediction = belief.update(time)
        assert prediction["mode"] == "predicted"
        assert prediction["last_visual_time"] == 0.1
        assert prediction["measured_xy"] is None
        assert not prediction["measurement_accepted"]
        assert prediction["assumed_position_radius_m"] >= previous_radius
        assert prediction["motion_authorized"] is False
        previous_radius = prediction["assumed_position_radius_m"]
    assert prediction["time_since_visual_seconds"] == pytest.approx(1.9)


@pytest.mark.parametrize("model", ["cv", "ca"])
def test_reacquisition_residual_uses_prediction_before_correction(model):
    belief = OcclusionBelief(model)
    belief.update(0, xy=[0, 0], covariance=np.eye(2) * 0.0001, status="visible")
    prediction = belief.update(1)
    reacquired = belief.update(
        2, xy=[1, 0], covariance=np.eye(2) * 0.0001, status="visible"
    )
    assert prediction["position"] == [0, 0]
    assert reacquired["reacquisition_residual"] == [1, 0]
    assert reacquired["measurement_accepted"]
    assert reacquired["last_visual_time"] == 2
    assert reacquired["mode"] == "measured"


@pytest.mark.parametrize("model", ["cv", "ca"])
def test_outlier_does_not_reset_age_or_shrink_reported_envelope(model):
    belief = OcclusionBelief(model)
    first = belief.update(0, xy=[0, 0], covariance=np.eye(2) * 0.0001, status="visible")
    outlier = belief.update(
        0.1, xy=[100, 0], covariance=np.eye(2) * 0.0001, status="visible"
    )
    assert outlier["mode"] == "predicted"
    assert outlier["filter_status"] == "outlier"
    assert outlier["last_visual_time"] == 0
    assert outlier["assumed_position_radius_m"] >= first["assumed_position_radius_m"]


def test_truth_fields_cannot_change_sanitized_observations():
    record = {
        "time": 1.0,
        "views": {
            "A": {
                "position": [1, 2],
                "covariance": [[0.01, 0], [0, 0.01]],
                "status": "visible",
                "head_pixels_scoring_only": 10,
            }
        },
        "truth_position_xy_scoring_only": [0, 0],
    }
    changed = deepcopy(record)
    changed["truth_position_xy_scoring_only"] = [999, 999]
    changed["views"]["A"]["head_pixels_scoring_only"] = 0
    assert observation_input(record, "A") == observation_input(changed, "A")
    for model in ("cv", "ca"):
        assert OcclusionBelief(model).update(
            **observation_input(record, "A")
        ) == OcclusionBelief(model).update(**observation_input(changed, "A"))


@pytest.mark.parametrize("model", ["cv", "ca"])
def test_future_samples_cannot_change_persisted_prefix(model):
    belief = OcclusionBelief(model)
    first = belief.update(0, xy=[0, 0], covariance=np.eye(2) * 0.01, status="visible")
    prefix = deepcopy(first)
    belief.update(0.1, xy=[0.02, 0], covariance=np.eye(2) * 0.01, status="visible")
    belief.update(0.2)
    assert first == prefix


def test_ca_estimates_acceleration_from_visible_positions_without_velocity_labels():
    belief = OcclusionBelief("ca")
    for time in np.linspace(0, 2, 21):
        result = belief.update(
            float(time),
            xy=[0.1 * time**2, 0],
            covariance=np.eye(2) * 0.00001,
            status="visible",
        )
    assert result["velocity"][0] == pytest.approx(0.4, abs=0.04)
    assert result["acceleration"][0] == pytest.approx(0.2, abs=0.04)


@pytest.mark.parametrize("model", ["cv", "ca"])
def test_invalid_or_noncausal_inputs_fail(model):
    belief = OcclusionBelief(model)
    with pytest.raises(ValueError, match="visible"):
        belief.update(0, xy=[0, 0], covariance=np.eye(2), status="predicted")
    with pytest.raises(ValueError, match="positive covariance"):
        belief.update(0, xy=[0, 0], covariance=-np.eye(2), status="visible")
    belief.update(0)
    with pytest.raises(ValueError, match="increase strictly"):
        belief.update(0)
