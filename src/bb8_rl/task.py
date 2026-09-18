"""Versioned scalar navigation task and fixed, disjoint layout seed domains."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SPLITS = {"train": (0, 10000), "validation": (10000, 20000), "test": (20000, 30000)}


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    world_config: str
    observation_mode: Literal["oracle"] = "oracle"
    layout_mode: Literal["procedural", "authored"] = "procedural"
    grid_resolution: float = Field(default=0.1, gt=0, le=0.25)
    clearance_margin: float = Field(default=0.11, ge=0, le=0.3)
    local_map_size: int = Field(default=21, ge=5, le=51)
    history_length: int = Field(default=4, ge=1, le=16)
    min_goal_distance: float = Field(default=0.6, gt=0)
    max_goal_distance: float = Field(default=2.5, gt=0)
    position_tolerance: float = Field(default=0.1, gt=0)
    speed_tolerance: float = Field(default=0.03, gt=0)
    dwell_seconds: float = Field(default=0.5, gt=0)
    max_episode_steps: int = Field(default=600, ge=1, le=10000)
    collision_penalty: float = Field(default=10, ge=0)
    boundary_penalty: float = Field(default=10, ge=0)
    control_cost: float = Field(default=0, ge=0)
    baseline_max_speed: float = Field(default=0.22, gt=0, le=0.35)
    baseline_gain: float = Field(default=1.6, gt=0)
    baseline_velocity_damping: float = Field(default=0.6, ge=0)
    waypoint_tolerance: float = Field(default=0.045, gt=0)

    @model_validator(mode="after")
    def validate_task(self):
        if self.min_goal_distance >= self.max_goal_distance:
            raise ValueError("Goal distance limits must increase")
        if self.local_map_size % 2 == 0:
            raise ValueError("Local map size must be odd")
        return self


def load_task(path: Path):
    config = TaskConfig.model_validate(yaml.safe_load(path.read_text()))
    return config, (path.resolve().parent / config.world_config).resolve()


def require_split_seed(split: str, seed: int):
    if split not in SPLITS or not SPLITS[split][0] <= seed < SPLITS[split][1]:
        raise ValueError(f"Layout seed {seed} does not belong to split {split!r}")
