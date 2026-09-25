"""Learn action consequences for an explicitly simulated operational need.

This is a small, inspectable outcome model, not a personality or recurrent RL
policy. The designer supplies a resource target and interaction costs. Evidence
changes the model's predictions and choices; merely reaching a point does not.
Navigation, telemetry freshness, interaction completion and authority belong to
the coordinator. No simulator state, cameras, or motor API is imported here.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RESOURCE_TARGET = 0.8
TARGET_RESOURCE = RESOURCE_TARGET
MEANINGFUL_GAIN = 0.04
EVIDENCE_WINDOW = 12
INITIAL_PROBES = 4
PRIOR_GAIN = 0.35
INTERACTION_COST = 0.005
DISTANCE_COST = 0.015
MAX_INFORMATION_VALUE = 0.03
APPROVED_SOURCES = frozenset(
    {"simulated_station_telemetry", "synthetic_outcome_benchmark"}
)


def _number(value: float, name: str, *, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (unit and not 0 <= result <= 1):
        raise ValueError(f"Invalid {name}")
    return result


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"Invalid {name}")
    return value


def _xy(value: tuple[float, float], name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must have two coordinates")
    return tuple(_number(x, name) for x in value)


@dataclass(frozen=True)
class Station:
    """A stable, declared interaction location on the estimated map."""

    id: str
    xy: tuple[float, float]
    label: str

    def __post_init__(self) -> None:
        _identifier(self.id, "station id")
        object.__setattr__(self, "xy", _xy(self.xy, "station position"))
        _identifier(self.label, "station label")


class PurposeStore:
    """Durable, independently namespaced action/outcome memory.

    A scope is permanently bound to its first outcome source. Synthetic benchmark
    evidence cannot enter a scope containing native simulated telemetry (or vice
    versa). This does not authenticate telemetry: only the coordinator is allowed
    to attest that a complete, fresh before/after observation was received.
    """

    def __init__(
        self, path: str | Path, map_version: str, agent_id: str = "bb8"
    ) -> None:
        self.agent_id = _identifier(agent_id, "agent id")
        self.map_version = _identifier(map_version, "map version")
        self.lock = threading.RLock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS purpose_scopes (
                agent TEXT NOT NULL, map TEXT NOT NULL,
                schema_version INTEGER NOT NULL, source TEXT,
                decisions INTEGER NOT NULL DEFAULT 0,
                change_epoch INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(agent, map));
            CREATE TABLE IF NOT EXISTS purpose_stations (
                agent TEXT NOT NULL, map TEXT NOT NULL, id TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, label TEXT NOT NULL,
                proposals INTEGER NOT NULL DEFAULT 0,
                outcomes INTEGER NOT NULL DEFAULT 0,
                responses INTEGER NOT NULL DEFAULT 0,
                alpha REAL NOT NULL DEFAULT 1, beta REAL NOT NULL DEFAULT 1,
                gain REAL NOT NULL DEFAULT 0.35,
                probes_remaining INTEGER NOT NULL DEFAULT 4,
                success_streak INTEGER NOT NULL DEFAULT 0,
                failure_streak INTEGER NOT NULL DEFAULT 0,
                stable INTEGER NOT NULL DEFAULT 0,
                last_change_epoch INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(agent, map, id),
                FOREIGN KEY(agent, map) REFERENCES purpose_scopes(agent, map));
            CREATE TABLE IF NOT EXISTS purpose_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL, map TEXT NOT NULL, id TEXT NOT NULL,
                station TEXT NOT NULL, payload TEXT NOT NULL,
                gain REAL NOT NULL, response INTEGER NOT NULL,
                UNIQUE(agent, map, id),
                FOREIGN KEY(agent, map, station)
                    REFERENCES purpose_stations(agent, map, id));
            CREATE INDEX IF NOT EXISTS purpose_events_station
                ON purpose_events(agent, map, station, sequence);
        """)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO purpose_scopes(agent,map,schema_version) "
                "VALUES(?,?,?)",
                (*self.scope, SCHEMA_VERSION),
            )
        version = self.db.execute(
            "SELECT schema_version FROM purpose_scopes WHERE agent=? AND map=?",
            self.scope,
        ).fetchone()[0]
        if version != SCHEMA_VERSION:
            self.db.close()
            raise ValueError(f"Unsupported purpose schema version {version}")

    @property
    def scope(self) -> tuple[str, str]:
        return self.agent_id, self.map_version

    def close(self) -> None:
        with self.lock:
            self.db.close()


def _serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.store.lock:
            return method(self, *args, **kwargs)

    return wrapped


class PurposeEngine:
    """Choose a useful interaction or idle using learned outcome predictions.

    Response probability has a Beta(1, 1) prior and only the last twelve observed
    outcomes update its effective evidence. Conditional gain is the mean positive
    gain in that same window, with an explicit 0.35 cold-start prior. A one-step
    value-of-information calculation asks whether one more outcome could change
    the preferred interaction at the current resource deficit. Its contribution
    is capped; raw surprise and merely collecting observations earn nothing.

    Four consecutive ineffective completed trials suppress a station. Six
    consecutive useful outcomes establish stable efficacy; three subsequent
    failures then reopen one trial of suppressed alternatives. A fresh successful
    outcome restores the ordinary probe allowance. No wall-clock timer renews
    curiosity, so a static useless/noisy room eventually produces idle.

    The current stations have zero or positive designed effects. Negative resource
    deltas count as ineffective; this model does not estimate harm magnitude and
    must be extended before offering activities with potentially harmful effects.
    """

    def __init__(self, store: PurposeStore):
        self.store = store

    def _rows(self) -> list[sqlite3.Row]:
        return self.store.db.execute(
            "SELECT * FROM purpose_stations WHERE agent=? AND map=? ORDER BY id",
            self.store.scope,
        ).fetchall()

    def _register(self, stations: list[Station]) -> None:
        if len(stations) > 32:
            raise ValueError("At most 32 stations can be offered")
        if not all(isinstance(station, Station) for station in stations):
            raise TypeError("Expected Station objects")
        ids = [station.id for station in stations]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate station id")
        with self.store.db:
            for station in stations:
                old = self.store.db.execute(
                    "SELECT x,y FROM purpose_stations WHERE agent=? AND map=? AND id=?",
                    (*self.store.scope, station.id),
                ).fetchone()
                if old and math.dist((old[0], old[1]), station.xy) > 1e-6:
                    raise ValueError(
                        "Station coordinates changed; use a new map version or id"
                    )
                self.store.db.execute(
                    "INSERT INTO purpose_stations(agent,map,id,x,y,label) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(agent,map,id) "
                    "DO UPDATE SET label=excluded.label",
                    (*self.store.scope, station.id, *station.xy, station.label),
                )

    @_serialized
    def select(
        self,
        stations: list[Station],
        *,
        resource: float,
        pose: tuple[float, float],
        now: float,
    ) -> dict[str, Any] | None:
        resource = _number(resource, "resource", unit=True)
        pose = _xy(pose, "pose")
        if _number(now, "now") < 0:
            raise ValueError("now must be nonnegative")
        stations = list(stations)
        self._register(stations)
        deficit = max(0.0, RESOURCE_TARGET - resource)
        if deficit <= 1e-9:
            return None
        rows = {row["id"]: row for row in self._rows()}
        ranked = []
        for station in stations:
            row = rows[station.id]
            if row["probes_remaining"] <= 0:
                continue
            a, b = row["alpha"], row["beta"]
            probability = a / (a + b)
            gain = min(deficit, row["gain"])
            distance = math.dist(pose, station.xy)
            cost = INTERACTION_COST + DISTANCE_COST * distance
            expected = probability * gain
            terms = {
                "resource": resource,
                "resource_target": RESOURCE_TARGET,
                "deficit": deficit,
                "response_probability": probability,
                "conditional_gain": row["gain"],
                "predicted_deficit_reduction": expected,
                "probability_uncertainty": math.sqrt(
                    a * b / ((a + b) ** 2 * (a + b + 1))
                ),
                "interaction_cost": INTERACTION_COST,
                "distance_cost": DISTANCE_COST * distance,
                "probes_remaining": row["probes_remaining"],
                "outcomes": row["outcomes"],
                "observations": row["outcomes"],
                "base_value": expected - cost,
            }
            ranked.append((station, row, terms))
        for station, row, terms in ranked:
            alternatives = [
                other[2]["base_value"] for other in ranked if other[0].id != station.id
            ]
            best_other = max([0.0, *alternatives])
            n = row["alpha"] + row["beta"]
            gain = min(deficit, row["gain"])
            cost = terms["interaction_cost"] + terms["distance_cost"]
            response_value = (row["alpha"] + 1) / (n + 1) * gain - cost
            no_response_value = row["alpha"] / (n + 1) * gain - cost
            probability = terms["response_probability"]
            expected_best_after = probability * max(best_other, response_value) + (
                1 - probability
            ) * max(best_other, no_response_value)
            value_before = max(best_other, terms["base_value"])
            information = min(
                MAX_INFORMATION_VALUE, max(0.0, expected_best_after - value_before)
            )
            terms["information_value"] = information
            terms["idle_value"] = 0.0
            terms["total"] = terms["base_value"] + information
        if not ranked:
            return None
        station, row, terms = max(
            sorted(ranked, key=lambda item: item[0].id),
            key=lambda item: item[2]["total"],
        )
        if terms["total"] <= 1e-9:
            return None
        with self.store.db:
            self.store.db.execute(
                "UPDATE purpose_scopes SET decisions=decisions+1 WHERE agent=? AND map=?",
                self.store.scope,
            )
            self.store.db.execute(
                "UPDATE purpose_stations SET proposals=proposals+1 "
                "WHERE agent=? AND map=? AND id=?",
                (*self.store.scope, station.id),
            )
        probability = terms["response_probability"]
        explanation = (
            f"Simulated resource is {resource:.0%}; target is {RESOURCE_TARGET:.0%}. "
            f"{station.label} has a {probability:.0%} estimated chance of restoring "
            f"resource, based on {row['outcomes']} observed interactions. "
            f"Expected useful gain {terms['predicted_deficit_reduction']:.0%} "
            f"outweighs the travel and interaction cost."
        )
        if row["outcomes"] == 0:
            explanation += " This first trial tests an explicitly uncertain prior."
        return {
            "candidate_id": station.id,
            "label": station.label,
            "goal": list(station.xy),
            "question": f"Will interacting with {station.label} restore the simulated resource?",
            "explanation": explanation,
            "score_terms": terms,
            "evidence": "learned simulated interaction outcomes; explicit resource objective",
        }

    @_serialized
    def record_outcome(
        self,
        event_id: str,
        station_id: str,
        *,
        before: float,
        after: float,
        now: float,
        source: str = "simulated_station_telemetry",
    ) -> bool:
        """Atomically save one completed before/after observation and update belief.

        Returns False for an exact replay, rejects conflicting event IDs. Arrival,
        missing evidence, cancellation and stale telemetry are not outcomes and
        must not call this method. Resource decreases count as no useful response.
        """
        event_id = _identifier(event_id, "event id")
        station_id = _identifier(station_id, "station id")
        before = _number(before, "before", unit=True)
        after = _number(after, "after", unit=True)
        now = _number(now, "now")
        if now < 0:
            raise ValueError("now must be nonnegative")
        if source not in APPROVED_SOURCES:
            raise ValueError("An approved explicit observation source is required")
        payload = json.dumps(
            {
                "station_id": station_id,
                "before": before,
                "after": after,
                "now": now,
                "source": source,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        db, scope = self.store.db, self.store.scope
        # BEGIN IMMEDIATE also serializes writers using separately opened stores.
        # An exception rolls back the evidence, model, and change detector together.
        with db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT payload FROM purpose_events WHERE agent=? AND map=? AND id=?",
                (*scope, event_id),
            ).fetchone()
            if old:
                if old[0] != payload:
                    raise ValueError("Event id already exists with conflicting payload")
                return False
            row = db.execute(
                "SELECT * FROM purpose_stations WHERE agent=? AND map=? AND id=?",
                (*scope, station_id),
            ).fetchone()
            if row is None:
                raise ValueError("Station is not registered")
            existing_source = db.execute(
                "SELECT source FROM purpose_scopes WHERE agent=? AND map=?", scope
            ).fetchone()[0]
            if existing_source not in (None, source):
                raise ValueError(
                    "Different evidence sources cannot share a memory scope"
                )
            db.execute(
                "UPDATE purpose_scopes SET source=? WHERE agent=? AND map=?",
                (source, *scope),
            )
            gain = after - before
            response = int(gain >= MEANINGFUL_GAIN - 1e-9)
            db.execute(
                "INSERT INTO purpose_events(agent,map,id,station,payload,gain,response) "
                "VALUES(?,?,?,?,?,?,?)",
                (*scope, event_id, station_id, payload, gain, response),
            )
            recent = db.execute(
                "SELECT gain,response FROM purpose_events "
                "WHERE agent=? AND map=? AND station=? "
                "ORDER BY sequence DESC LIMIT ?",
                (*scope, station_id, EVIDENCE_WINDOW),
            ).fetchall()
            successes = sum(outcome["response"] for outcome in recent)
            positive = [outcome["gain"] for outcome in recent if outcome["response"]]
            mean_gain = sum(positive) / len(positive) if positive else PRIOR_GAIN
            success_streak = row["success_streak"] + 1 if response else 0
            failure_streak = 0 if response else row["failure_streak"] + 1
            probes = INITIAL_PROBES if response else max(0, row["probes_remaining"] - 1)
            stable = bool(row["stable"]) or success_streak >= 6
            changed = stable and failure_streak >= 3
            if changed:
                stable = False
                db.execute(
                    "UPDATE purpose_scopes SET change_epoch=change_epoch+1 "
                    "WHERE agent=? AND map=?",
                    scope,
                )
                epoch = db.execute(
                    "SELECT change_epoch FROM purpose_scopes WHERE agent=? AND map=?",
                    scope,
                ).fetchone()[0]
                db.execute(
                    "UPDATE purpose_stations SET probes_remaining=1,last_change_epoch=? "
                    "WHERE agent=? AND map=? AND id<>? AND probes_remaining=0",
                    (epoch, *scope, station_id),
                )
            db.execute(
                "UPDATE purpose_stations SET outcomes=outcomes+1,responses=responses+?,"
                "alpha=?,beta=?,gain=?,probes_remaining=?,success_streak=?,"
                "failure_streak=?,stable=? WHERE agent=? AND map=? AND id=?",
                (
                    response,
                    1 + successes,
                    1 + len(recent) - successes,
                    mean_gain,
                    probes,
                    success_streak,
                    failure_streak,
                    int(stable),
                    *scope,
                    station_id,
                ),
            )
        return True

    @_serialized
    def snapshot(self) -> dict[str, Any]:
        scope = self.store.db.execute(
            "SELECT * FROM purpose_scopes WHERE agent=? AND map=?", self.store.scope
        ).fetchone()
        preferences = []
        for row in self._rows():
            a, b = row["alpha"], row["beta"]
            preferences.append(
                {
                    "candidate_id": row["id"],
                    "label": row["label"],
                    "goal": [row["x"], row["y"]],
                    "proposals": row["proposals"],
                    "visits": row["outcomes"],
                    "outcomes": row["outcomes"],
                    "observations": row["outcomes"],
                    "responses": row["responses"],
                    "response_probability": a / (a + b),
                    "probability_uncertainty": math.sqrt(
                        a * b / ((a + b) ** 2 * (a + b + 1))
                    ),
                    "conditional_gain": row["gain"],
                    "mean_gain": row["gain"],
                    "expected_gain": a / (a + b) * row["gain"],
                    "value": a / (a + b) * row["gain"],
                    "effective_observations": int(a + b - 2),
                    "probes_remaining": row["probes_remaining"],
                    "suppressed": row["probes_remaining"] == 0,
                    "success_streak": row["success_streak"],
                    "failure_streak": row["failure_streak"],
                    "last_change_epoch": row["last_change_epoch"],
                }
            )
        source = scope["source"]
        return {
            "agent_id": self.store.agent_id,
            "map_version": self.store.map_version,
            "strategy": "learned_interaction_outcomes",
            "model": "windowed Beta response model and conditional resource gain",
            "resource_target": RESOURCE_TARGET,
            "evidence_window": EVIDENCE_WINDOW,
            "meaningful_gain": MEANINGFUL_GAIN,
            "evidence": source or "no interaction outcomes observed",
            "source": source,
            "episodes": sum(row["outcomes"] for row in preferences),
            "decisions": scope["decisions"],
            "change_epoch": scope["change_epoch"],
            "preferences": preferences,
        }
