from dataclasses import asdict

import pytest

from bb8_rl.control_profiles import PROFILES, get_profile
from bb8_rl.occluded_control import OcclusionParameters


def test_named_profiles_preserve_uncertainty_arrival_and_geometry_contracts():
    baseline = asdict(OcclusionParameters())
    tunable = {"clearance_margin", "speed_cap", "proactive_horizon_seconds"}
    for name, profile in PROFILES.items():
        values = asdict(profile.parameters())
        assert {k: v for k, v in values.items() if k not in tunable} == {
            k: v for k, v in baseline.items() if k not in tunable
        }
        assert get_profile(name) is profile
        assert profile.record()["parameters"] == values
        assert profile.parameters() is not profile.parameters()


@pytest.mark.parametrize("name", ["unsafe", {}, 7])
def test_unknown_profile_rejected(name):
    with pytest.raises(ValueError, match="Unknown control profile"):
        get_profile(name)
