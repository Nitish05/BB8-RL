"""Explicitly simulated station telemetry for the bounded purpose experiment.

The caller supplies world-side planar mechanics at every simulation step. These
inputs determine movement cost and whether a requested station interaction has
completed; they are never returned to the decision policy. Stations are virtual
task zones, not RGB detections or physical charging equipment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from .purpose import Station

DEFAULT_STATIONS = (
    Station("station-a", (0.0, 1.0), "Station A"),
    Station("station-b", (-0.5, 1.0), "Station B"),
)

SOURCE = "simulated_station_telemetry"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value, name):
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must have two finite coordinates")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must have two finite coordinates") from error
    if len(result) != 2:
        raise ValueError(f"{name} must have two finite coordinates")
    return tuple(_number(item, name) for item in result)


def _identifier(value, name):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        raise ValueError(f"Invalid {name}")
    return value


class InteractionEnvironment:
    """World-side resource accounting and exactly-once station effects.

    Resource decreases by 0.025 per metre of observed planar travel, with no
    idle drain. A station request completes after one continuous sampled second
    within 0.14 m and at no more than 0.03 m/s. Call ``advance`` at every world
    step: the environment cannot account for movement between omitted samples.

    Unspecified stations have zero effect when an explicit effects mapping is
    provided. Completed request IDs are retained for this environment's entire
    lifetime, so a retry can never apply a second effect. Restarting a simulated
    episode constructs a new environment and starts a new request namespace.
    """

    interaction_radius = 0.14
    maximum_speed = 0.03
    dwell_seconds = 1.0
    travel_cost_per_metre = 0.025

    def __init__(self, stations=DEFAULT_STATIONS, initial_resource=0.35, effects=None):
        try:
            stations = tuple(stations)
        except TypeError as error:
            raise ValueError("Expected validated Station objects") from error
        if not all(isinstance(station, Station) for station in stations):
            raise ValueError("Expected validated Station objects")
        if len({station.id for station in stations}) != len(stations):
            raise ValueError("Station IDs must be unique")
        self.stations = stations
        self._stations = {station.id: station for station in stations}
        resource = _number(initial_resource, "initial_resource")
        if not 0 <= resource <= 1:
            raise ValueError("initial_resource must be in [0,1]")
        if effects is None:
            effects = {
                station.id: 0.45 if station.id == "station-b" else 0.0
                for station in stations
            }
        if not isinstance(effects, Mapping):
            raise TypeError("effects must map station IDs to resource changes")
        if set(effects) - set(self._stations):
            raise ValueError("Effect refers to an unknown station")
        self._effects = {}
        for station in stations:
            value = _number(effects.get(station.id, 0.0), "station effect")
            if not -1 <= value <= 1:
                raise ValueError("Station effects must be in [-1,1]")
            self._effects[station.id] = value
        self.resource = resource
        self._timestamp = None
        self._position = None
        self._velocity = None
        self._active_request = None
        self._dwell_start = None
        self._request_stations = {}
        self._completed = {}

    def advance(self, *, now, position, velocity, request=None):
        """Return telemetry without exposing mechanics or hidden station effects.

        Invalid input is rejected before any resource, dwell or request ledger
        mutation. The same timestamp can be retried with unchanged mechanics.
        Completed requests return the original immutable outcome, even after
        leaving the station, alongside current top-level resource telemetry.
        """
        now = _number(now, "now")
        position = _vector(position, "position")
        velocity = _vector(velocity, "velocity")
        if now < 0 or self._timestamp is not None and now < self._timestamp:
            raise ValueError("Simulation time must be nonnegative and nondecreasing")
        if now == self._timestamp and (
            position != self._position or velocity != self._velocity
        ):
            raise ValueError("Repeated timestamp has conflicting mechanics")
        request_id = station_id = None
        if request is not None:
            if not isinstance(request, dict) or set(request) != {
                "request_id",
                "station_id",
            }:
                raise ValueError("Request must contain only request_id and station_id")
            request_id = _identifier(request["request_id"], "request_id")
            station_id = _identifier(request["station_id"], "station_id")
            if station_id not in self._stations:
                raise ValueError("Unknown requested station")
            previous_station = self._request_stations.get(request_id)
            if previous_station is not None and previous_station != station_id:
                raise ValueError("Request ID reused for a different station")
        distance = (
            0.0 if self._position is None else math.dist(position, self._position)
        )
        if not math.isfinite(distance) or not math.isfinite(math.hypot(*velocity)):
            raise ValueError("Mechanics exceed the supported numeric range")
        elapsed = None if self._timestamp is None else now - self._timestamp
        if elapsed is not None and not math.isfinite(elapsed):
            raise ValueError(
                "Elapsed simulation time exceeds the supported numeric range"
            )

        self.resource = max(0.0, self.resource - distance * self.travel_cost_per_metre)
        self._timestamp, self._position, self._velocity = now, position, velocity
        result = {"resource": self.resource, "source": SOURCE, "timestamp": now}
        if request_id is None:
            self._active_request = self._dwell_start = None
            return result
        self._request_stations[request_id] = station_id
        if request_id in self._completed:
            self._active_request = self._dwell_start = None
            result["outcome"] = dict(self._completed[request_id])
            return result

        station = self._stations[station_id]
        inside = math.dist(position, station.xy) <= self.interaction_radius
        slow = math.hypot(*velocity) <= self.maximum_speed
        continuous = (
            elapsed is None or elapsed == 0 or distance <= self.maximum_speed * elapsed
        )
        if not inside or not slow:
            self._active_request = self._dwell_start = None
            return result
        if self._active_request != request_id or not continuous:
            self._active_request, self._dwell_start = request_id, now
        if now - self._dwell_start < self.dwell_seconds:
            return result

        before = self.resource
        self.resource = min(1.0, max(0.0, before + self._effects[station_id]))
        outcome = {
            "request_id": request_id,
            "station_id": station_id,
            "before": before,
            "after": self.resource,
            "timestamp": now,
            "source": SOURCE,
        }
        self._completed[request_id] = outcome
        self._active_request = self._dwell_start = None
        return {
            "resource": self.resource,
            "source": SOURCE,
            "timestamp": now,
            "outcome": dict(outcome),
        }
