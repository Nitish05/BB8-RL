"""Durable, bounded activity choice above navigation; no motor commands or LLM.

The learned strategy is an exponentially weighted contextual-free value learner
(a nonstationary bandit), not recurrent RL or evidence of felt personality.
Only the runtime can execute a selected destination and report its outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import threading
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STRATEGIES = ("fixed", "memory", "learned")
OUTCOMES = ("arrived", "rejected", "cancelled", "localization_lost")


def _number(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class Candidate:
    """Stable activity anchor supplied from the runtime's estimated map."""

    id: str
    xy: tuple[float, float]
    label: str = ""

    def __post_init__(self) -> None:
        _identifier(self.id, "candidate id")
        if not isinstance(self.xy, (tuple, list)) or len(self.xy) != 2:
            raise ValueError("xy must have two coordinates")
        object.__setattr__(self, "xy", tuple(_number(x, "coordinate") for x in self.xy))
        if not isinstance(self.label, str) or len(self.label) > 256:
            raise ValueError("Invalid candidate label")


class AgencyStore:
    """SQLite memory partitioned by stable agent identity and map version.

    Synthetic stores are deliberately incompatible with live navigation stores.
    SQLite transactions make event insertion and its value update atomic.
    """

    def __init__(
        self,
        path: str | Path,
        agent_id: str = "bb8",
        map_version: str = "v1",
        *,
        allow_synthetic: bool = False,
    ):
        self.agent_id = _identifier(agent_id, "agent id")
        self.map_version = _identifier(map_version, "map version")
        self.lock = threading.RLock()
        self.allow_synthetic = bool(allow_synthetic)
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            self.db.close()
            raise ValueError(f"Unsupported agency schema version {version}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS scopes (
                agent TEXT NOT NULL, map TEXT NOT NULL, evidence TEXT NOT NULL,
                decisions INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(agent, map));
            CREATE TABLE IF NOT EXISTS candidates (
                agent TEXT NOT NULL, map TEXT NOT NULL, id TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, label TEXT NOT NULL,
                proposals INTEGER NOT NULL DEFAULT 0,
                visits INTEGER NOT NULL DEFAULT 0, learned INTEGER NOT NULL DEFAULT 0,
                arrivals INTEGER NOT NULL DEFAULT 0, rejections INTEGER NOT NULL DEFAULT 0,
                value REAL NOT NULL DEFAULT 0, learning_progress REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(agent, map, id),
                FOREIGN KEY(agent, map) REFERENCES scopes(agent, map));
            CREATE TABLE IF NOT EXISTS events (
                agent TEXT NOT NULL, map TEXT NOT NULL, id TEXT NOT NULL,
                candidate TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(agent, map, id),
                FOREIGN KEY(agent, map, candidate) REFERENCES candidates(agent, map, id));
        """)
        self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        evidence = "synthetic" if self.allow_synthetic else "navigation"
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO scopes(agent,map,evidence) VALUES(?,?,?)",
                (*self.scope, evidence),
            )
        found = self.db.execute(
            "SELECT evidence FROM scopes WHERE agent=? AND map=?", self.scope
        ).fetchone()[0]
        if found != evidence:
            self.db.close()
            raise ValueError(
                "Synthetic and live navigation memory cannot share a scope"
            )

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


class AgencyEngine:
    """Select estimated-map destinations; learn from grounded terminal outcomes.

    ``fixed`` uses a seed-stable arbitrary activity ranking. ``memory`` seeks
    less-visited activities. ``learned`` adds adaptive utility and learning
    progress. All baselines receive the same events and retain the same memory;
    only their choice rules differ. None runs a language model.
    """

    def __init__(self, store: AgencyStore, strategy: str = "learned", seed: int = 0):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown agency strategy: {strategy}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.store, self.strategy, self.seed = store, strategy, seed

    def _rows(self) -> list[sqlite3.Row]:
        return self.store.db.execute(
            "SELECT * FROM candidates WHERE agent=? AND map=? ORDER BY id",
            self.store.scope,
        ).fetchall()

    def _register(self, candidates: list[Candidate]) -> None:
        ids = [c.id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate id in selection")
        with self.store.db:
            for c in candidates:
                old = self.store.db.execute(
                    "SELECT x,y FROM candidates WHERE agent=? AND map=? AND id=?",
                    (*self.store.scope, c.id),
                ).fetchone()
                if old and math.dist((old[0], old[1]), c.xy) > 1e-6:
                    raise ValueError(
                        "Candidate coordinates changed; use a new map version or id"
                    )
                self.store.db.execute(
                    "INSERT INTO candidates(agent,map,id,x,y,label) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(agent,map,id) DO UPDATE SET label=excluded.label",
                    (*self.store.scope, c.id, *c.xy, c.label or c.id),
                )

    @_serialized
    def select(
        self, candidates: list[Candidate], *, now: float, pose: tuple[float, float]
    ) -> dict[str, Any] | None:
        _number(now, "now", minimum=0)
        if not isinstance(pose, (tuple, list)) or len(pose) != 2:
            raise ValueError("pose must have two coordinates")
        pose = tuple(_number(x, "pose coordinate") for x in pose)
        candidates = list(candidates)
        if not all(isinstance(c, Candidate) for c in candidates):
            raise ValueError("Expected Candidate objects")
        if not candidates:
            return None
        self._register(candidates)
        # A stationary re-selection is not a new activity; avoid repeated arrivals.
        eligible = [c for c in candidates if math.dist(c.xy, pose) >= 0.15]
        if not eligible:
            return None
        rows = {row["id"]: row for row in self._rows()}
        decisions = self.store.db.execute(
            "SELECT decisions FROM scopes WHERE agent=? AND map=?", self.store.scope
        ).fetchone()[0]
        # The counter lives in SQLite so closing/reopening does not reset exploration.
        rng = random.Random(
            f"{self.seed}:{decisions}:{self.store.agent_id}:{self.store.map_version}"
        )
        ranked = []
        for c in sorted(eligible, key=lambda c: c.id):
            row = rows[c.id]
            trait = (
                int.from_bytes(
                    hashlib.sha256(f"{self.seed}:{c.id}".encode()).digest()[:8], "big"
                )
                / 2**64
            )
            novelty = 1 / math.sqrt(1 + row["visits"])
            distance_cost = min(0.04, math.dist(c.xy, pose) * 0.01)
            if self.strategy == "fixed":
                total = trait
            elif self.strategy == "memory":
                total = novelty + 0.001 * trait - distance_cost
            else:
                total = (
                    row["value"]
                    + 0.12 * novelty
                    + 0.05 * row["learning_progress"]
                    - distance_cost
                )
            terms = {
                "utility": row["value"],
                "familiarity_bonus": novelty,
                "learning_progress": row["learning_progress"],
                "distance_cost": distance_cost,
                "fixed_rank": trait,
                "total": total,
            }
            ranked.append((c, terms))
        exploratory = self.strategy == "learned" and rng.random() < 0.12
        chosen, terms = (
            rng.choice(ranked)
            if exploratory
            else max(ranked, key=lambda item: item[1]["total"])
        )
        with self.store.db:
            self.store.db.execute(
                "UPDATE scopes SET decisions=decisions+1 WHERE agent=? AND map=?",
                self.store.scope,
            )
            self.store.db.execute(
                "UPDATE candidates SET proposals=proposals+1 WHERE agent=? AND map=? AND id=?",
                (*self.store.scope, chosen.id),
            )
        explanation = {
            "fixed": "Seeded fixed activity ranking; outcomes do not change this choice rule.",
            "memory": "Favoring a less-visited activity using persistent visit memory.",
            "learned": (
                "Exploring an alternative to keep checking whether outcomes changed."
                if exploratory
                else "Using learned activity utility, familiarity and recent learning progress."
            ),
        }[self.strategy]
        return {
            "candidate_id": chosen.id,
            "goal": list(chosen.xy),
            "label": chosen.label or chosen.id,
            "explanation": explanation,
            "score_terms": {**terms, "exploratory": exploratory},
            "evidence": "synthetic"
            if self.store.allow_synthetic
            else "navigation outcomes",
        }

    def record_outcome(
        self,
        *,
        event_id: str,
        candidate_id: str,
        outcome: str,
        now: float,
        duration_s: float = 0.0,
        progress: float = 0.0,
    ) -> bool:
        if outcome not in OUTCOMES:
            raise ValueError(f"Unknown navigation outcome: {outcome}")
        duration_s = _number(duration_s, "duration", minimum=0)
        progress = _number(progress, "progress", minimum=0)
        if progress > 1:
            raise ValueError("progress must be in [0,1]")
        # This is designed navigation utility, not human affection or emotion.
        reward = {
            "arrived": 1.0,
            "rejected": -0.6,
            "cancelled": None,
            "localization_lost": None,
        }[outcome]
        return self._record(
            event_id,
            candidate_id,
            now,
            {
                "source": "navigation",
                "outcome": outcome,
                "duration_s": duration_s,
                "progress": progress,
                "utility": reward,
            },
        )

    def record_experience(
        self,
        *,
        event_id: str,
        candidate_id: str,
        utility: float,
        now: float,
        source: str = "synthetic",
    ) -> bool:
        """Controlled benchmark input; forbidden in normal live stores.

        Arbitrary human ratings, simulator truth and hidden utility are not
        accepted by the live outcome API. This hook deliberately marks its data.
        """
        if source != "synthetic" or not self.store.allow_synthetic:
            raise ValueError(
                "Synthetic experience requires an explicitly synthetic store"
            )
        utility = _number(utility, "utility")
        if not -1 <= utility <= 1:
            raise ValueError("utility must be in [-1,1]")
        return self._record(
            event_id,
            candidate_id,
            now,
            {
                "source": "synthetic",
                "outcome": "synthetic_experience",
                "utility": utility,
            },
        )

    @_serialized
    def _record(
        self, event_id: str, candidate_id: str, now: float, fields: dict[str, Any]
    ) -> bool:
        _identifier(event_id, "event id")
        _identifier(candidate_id, "candidate id")
        now = _number(now, "now", minimum=0)
        payload = json.dumps(
            {"candidate_id": candidate_id, "now": now, **fields},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        db = self.store.db
        with db:
            # Acquire the write lock before reading duplicate/update state.
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT payload FROM events WHERE agent=? AND map=? AND id=?",
                (*self.store.scope, event_id),
            ).fetchone()
            if old:
                if old[0] != payload:
                    raise ValueError("Event id reused with conflicting payload")
                return False
            row = db.execute(
                "SELECT * FROM candidates WHERE agent=? AND map=? AND id=?",
                (*self.store.scope, candidate_id),
            ).fetchone()
            if row is None:
                raise ValueError(
                    "Outcome candidate is not registered in this map scope"
                )
            db.execute(
                "INSERT INTO events(agent,map,id,candidate,payload) VALUES(?,?,?,?,?)",
                (*self.store.scope, event_id, candidate_id, payload),
            )
            reward = fields["utility"]
            if reward is not None:
                # A fixed learning rate can adapt when consequences reverse.
                delta = 0.2 * (reward - row["value"])
                value = row["value"] + delta
                lp = 0.8 * row["learning_progress"] + 0.2 * abs(delta)
                db.execute(
                    "UPDATE candidates SET visits=visits+1, learned=learned+1, "
                    "value=?, learning_progress=?, arrivals=arrivals+?, rejections=rejections+? "
                    "WHERE agent=? AND map=? AND id=?",
                    (
                        value,
                        lp,
                        int(fields["outcome"] == "arrived"),
                        int(fields["outcome"] == "rejected"),
                        *self.store.scope,
                        candidate_id,
                    ),
                )
        return True

    @_serialized
    def snapshot(self) -> dict[str, Any]:
        preferences = []
        for row in self._rows():
            attempted = row["arrivals"] + row["rejections"]
            preferences.append(
                {
                    "candidate_id": row["id"],
                    "label": row["label"],
                    "goal": [row["x"], row["y"]],
                    "value": row["value"],
                    "visits": row["visits"],
                    "arrivals": row["arrivals"],
                    "rejections": row["rejections"],
                    "proposals": row["proposals"],
                    "learning_progress": row["learning_progress"],
                    "competence": row["arrivals"] / attempted if attempted else None,
                }
            )
        events = self.store.db.execute(
            "SELECT COUNT(*) FROM events WHERE agent=? AND map=?", self.store.scope
        ).fetchone()[0]
        decisions = self.store.db.execute(
            "SELECT decisions FROM scopes WHERE agent=? AND map=?", self.store.scope
        ).fetchone()[0]
        return {
            "agent_id": self.store.agent_id,
            "map_version": self.store.map_version,
            "strategy": self.strategy,
            "episodes": events,
            "decisions": decisions,
            "evidence": "synthetic"
            if self.store.allow_synthetic
            else "navigation outcomes",
            "model": "nonstationary activity-value learner; no language model",
            "preferences": sorted(
                preferences, key=lambda p: (-p["value"], p["candidate_id"])
            ),
        }
