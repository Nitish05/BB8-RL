"""Versioned task settings, separate from the durable Studio world document."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bb8_rl.control.contract import PlanarDriveParameters


class DriveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_speed: float = 0.35
    response_time: float = 0.25
    max_acceleration: float = 0.7
    max_deceleration: float = 0.9
    dead_zone: float = 0.05
    latency: float = 0.05
    max_command_age: float = 0.15
    heading_offset: float = 0.0

    def parameters(self) -> PlanarDriveParameters:
        return PlanarDriveParameters(**self.model_dump())


class NavigationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    project: str
    body: str = "bb8_body"
    head: str = "bb8_head"
    camera: str = "navigation_camera"
    obstacle_names: tuple[str, ...] = ()
    arena_half_extent: float = Field(default=1.5, gt=0)
    action_steps: int = Field(default=10, ge=1, le=1000)
    provenance: Literal["synthetic", "estimated", "measured"] = "synthetic"
    notes: str
    drive: DriveConfig = Field(default_factory=DriveConfig)

    @model_validator(mode="after")
    def validate_drive(self):
        self.drive.parameters()
        if self.body == self.head:
            raise ValueError("Body and independently posed head must differ")
        if len(set(self.obstacle_names)) != len(self.obstacle_names) or {
            self.body,
            self.head,
        } & set(self.obstacle_names):
            raise ValueError(
                "Obstacle names must be unique and distinct from the body/head"
            )
        return self


def load_config(path: Path) -> tuple[NavigationConfig, Path]:
    config = NavigationConfig.model_validate(yaml.safe_load(path.read_text()))
    return config, (path.resolve().parent / config.project).resolve()
