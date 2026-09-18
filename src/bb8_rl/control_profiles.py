"""Named, reproducible clearance/speed experiments; arrival gates stay fixed."""

from dataclasses import asdict, dataclass

from .occluded_control import OcclusionParameters


@dataclass(frozen=True)
class ControlProfile:
    name: str
    margin_m: float
    speed_m_s: float
    lookahead_s: float = 0.0
    planning_reserve_m: float = 0.0

    def parameters(self):
        return OcclusionParameters(
            clearance_margin=self.margin_m,
            speed_cap=self.speed_m_s,
            proactive_horizon_seconds=self.lookahead_s,
        )

    def record(self):
        return dict(asdict(self), parameters=asdict(self.parameters()))


PROFILES = {
    p.name: p
    for p in (
        ControlProfile("baseline", 0.04, 0.15),
        ControlProfile("margin-3cm", 0.03, 0.15),
        ControlProfile("margin-2cm", 0.02, 0.15),
        ControlProfile("slow-3cm", 0.03, 0.10),
        ControlProfile("lookahead-100ms", 0.03, 0.15, 0.10),
        ControlProfile("lookahead-200ms", 0.03, 0.15, 0.20),
        ControlProfile("reserve-3cm", 0.03, 0.15, planning_reserve_m=0.04),
    )
}
DEFAULT_PROFILE = "reserve-3cm"


def get_profile(name=None):
    try:
        return PROFILES[DEFAULT_PROFILE if name is None else name]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown control profile") from error
