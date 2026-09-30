"""World-side, nonphysical native RGB markers and resource indicators.

Only public Genesis visual-vertex APIs are used. These fixed, noncolliding
objects are rendered by ordinary cameras with normal depth occlusion; they are
not image overlays. Rendering and occlusion still require native preflight.
Numeric resource values enter this renderer only, never the pixel observer.
"""

import math
from dataclasses import dataclass

import numpy as np

from .visual_interaction import (
    CELL_SIZE_M,
    GAUGE_BINS,
    ID_COLOR,
    ID_ORIGIN,
    ID_PATTERNS,
    PANEL_COLUMNS,
    PANEL_HEIGHT_M,
    PANEL_ROWS,
    gauge_rect,
)

MARKER_TARGET_OFFSET_M = (0.0, -0.25)
MAGENTA = (1.0, 0.0, 1.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLACK = (0.005, 0.005, 0.005, 1.0)


@dataclass(frozen=True)
class VisualPanel:
    """Renderer configuration, not an observed identity or policy coordinate."""

    visual_id: str
    center_xy: tuple[float, float]

    def __post_init__(self):
        if self.visual_id not in ID_PATTERNS:
            raise ValueError("Unknown visual marker pattern")
        if len(self.center_xy) != 2 or not all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and abs(v) <= 10
            for v in self.center_xy
        ):
            raise ValueError("Panel center must contain two bounded finite coordinates")
        object.__setattr__(self, "center_xy", tuple(float(v) for v in self.center_xy))


DEFAULT_PANELS = (
    VisualPanel("marker-01", (0.0, 1.25)),
    VisualPanel("marker-02", (-0.5, 1.25)),
)


@dataclass(frozen=True)
class VisualCamera:
    """Requested ordinary RGB sensor pose; registration is a separate input."""

    camera_id: str
    position: tuple[float, float, float]
    lookat: tuple[float, float, float]
    resolution: tuple[int, int] = (2560, 1920)
    fov: float = 58.0

    def __post_init__(self):
        if not isinstance(self.camera_id, str) or not self.camera_id.isalnum():
            raise ValueError("Invalid visual camera identity")
        for name in ("position", "lookat"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError("Visual camera needs finite 3D vectors")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if (
            len(self.resolution) != 2
            or any(type(v) is not int or v < 64 for v in self.resolution)
            or math.prod(self.resolution) > 8_000_000
            or not 1 <= self.fov <= 120
            or self.position == self.lookat
        ):
            raise ValueError("Invalid visual camera resolution, field of view or pose")
        object.__setattr__(self, "resolution", tuple(self.resolution))


def quantize_resource(value: float) -> int:
    """Declared display quantization; this is not an RGB observation."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Rendered resource must be finite and in [0, 1]")
    return min(GAUGE_BINS, math.floor(value * GAUGE_BINS + 0.5))


def panel_rect(panel, rectangle, *, top=PANEL_HEIGHT_M, thickness=0.002):
    """Map panel-cell geometry to an ordinary horizontal world-space box."""
    x0, y0, x1, y1 = rectangle
    width, height = (x1 - x0) * CELL_SIZE_M, (y1 - y0) * CELL_SIZE_M
    # Rows run toward negative world y, so an above-plane camera sees the
    # declared glyph rather than its mirror image. Rotation remains arbitrary.
    center = (
        panel.center_xy[0] + ((x0 + x1) / 2 - PANEL_COLUMNS / 2) * CELL_SIZE_M,
        panel.center_xy[1] - ((y0 + y1) / 2 - PANEL_ROWS / 2) * CELL_SIZE_M,
        top - thickness / 2,
    )
    return center, (width, height, thickness)


def panel_boxes(panel):
    """Pure, bounded geometry description shared by native construction/tests."""
    if not isinstance(panel, VisualPanel):
        raise TypeError("Expected VisualPanel")
    result = [
        {
            "role": "backing",
            "color": BLACK,
            "rect": (0, 0, PANEL_COLUMNS, PANEL_ROWS),
            "top": PANEL_HEIGHT_M - 0.002,
            "thickness": 0.006,
        }
    ]
    for rectangle in (
        (0, 0, PANEL_COLUMNS, 0.5),
        (0, PANEL_ROWS - 0.5, PANEL_COLUMNS, PANEL_ROWS),
        (0, 0.5, 0.5, PANEL_ROWS - 0.5),
        (PANEL_COLUMNS - 0.5, 0.5, PANEL_COLUMNS, PANEL_ROWS - 0.5),
    ):
        result.append({"role": "border", "color": MAGENTA, "rect": rectangle})
    for row, bits in enumerate(ID_PATTERNS[panel.visual_id]):
        for column, lit in enumerate(bits):
            x, y = ID_ORIGIN[0] + column, ID_ORIGIN[1] + row
            result.append(
                {
                    "role": "glyph",
                    "color": ID_COLOR if lit else BLACK,
                    "rect": (x, y, x + 1, y + 1),
                }
            )
    for index in range(GAUGE_BINS):
        result.append(
            {
                "role": "gauge",
                "color": GREEN,
                "rect": gauge_rect(index),
                "gauge_index": index,
            }
        )
    return tuple(result)


def _vertices(entity):
    value = entity.get_vverts()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("Expected finite world-space native visual vertices")
    return result


class VisualFixture:
    """Attach before build, initialize after build, update from world effects.

    No motion authority or outcome-learning API is exposed here. A caller must
    separately bind an authorized action to a fresh pixel-only before/after
    observation. Manual ``set_resource`` calls are renderer preflight only,
    and do not by themselves establish action-dependent visual learning.
    """

    def __init__(
        self, panels=DEFAULT_PANELS, *, initial_resource=0.25, camera_specs=()
    ):
        if not isinstance(panels, (tuple, list)) or not 1 <= len(panels) <= 6:
            raise ValueError("Need one to six bounded visual panels")
        if any(not isinstance(panel, VisualPanel) for panel in panels):
            raise TypeError("Expected VisualPanel renderer specifications")
        self.panels = tuple(panels)
        if (
            not isinstance(camera_specs, (tuple, list))
            or len(camera_specs) > 2
            or any(not isinstance(item, VisualCamera) for item in camera_specs)
            or len({item.camera_id for item in camera_specs}) != len(camera_specs)
        ):
            raise ValueError("Need at most two distinct visual camera specifications")
        self.camera_specs = tuple(camera_specs)
        self.cameras = {}
        self._levels = [quantize_resource(initial_resource)] * len(panels)
        self._visible = [True] * len(panels)
        self._parts = []
        self._attached = self._initialized = False

    def attach(self, scene, gs):
        """Add fixed noncolliding native entities before ``scene.build()``."""
        if self._attached:
            raise RuntimeError("Visual fixture already attached")
        self._attached = True
        for panel_index, panel in enumerate(self.panels):
            for part_index, spec in enumerate(panel_boxes(panel)):
                position, size = panel_rect(
                    panel,
                    spec["rect"],
                    top=spec.get("top", PANEL_HEIGHT_M),
                    thickness=spec.get("thickness", 0.002),
                )
                entity = scene.add_entity(
                    name=f"visual_fixture_{panel_index}_{part_index}",
                    morph=gs.morphs.Box(
                        pos=position,
                        size=size,
                        fixed=True,
                        collision=False,
                        visualization=True,
                        enable_custom_vverts=True,
                    ),
                    surface=gs.surfaces.Default(color=spec["color"], roughness=1.0),
                )
                self._parts.append(
                    {
                        "entity": entity,
                        "panel": panel_index,
                        "gauge_index": spec.get("gauge_index"),
                    }
                )
        for spec in self.camera_specs:
            self.cameras[spec.camera_id] = scene.add_camera(
                pos=spec.position,
                lookat=spec.lookat,
                res=spec.resolution,
                fov=spec.fov,
                GUI=False,
                debug=False,
            )

    def initialize_visuals(self):
        """Read visual vertices once after build; never read or set body pose."""
        if not self._attached or self._initialized:
            raise RuntimeError("Attach once and initialize once after native build")
        for part in self._parts:
            part["vertices"] = _vertices(part["entity"])
        self._initialized = True
        self._apply()

    def _apply(self):
        if not self._initialized:
            raise RuntimeError("Native visual fixture has not been initialized")
        for part in self._parts:
            panel = part["panel"]
            vertices = part["vertices"].copy()
            if not self._visible[panel]:
                vertices[:, 2] -= 0.20  # Entire panel hides below opaque ground.
            elif (
                part["gauge_index"] is not None
                and part["gauge_index"] >= self._levels[panel]
            ):
                vertices[:, 2] -= 0.004  # Cell hides beneath opaque backing.
            part["entity"].set_vverts(vertices)

    def set_resource(self, resource):
        """World-side effect to ordinary-renderer pixels; no policy receipt."""
        if not self._initialized:
            raise RuntimeError("Native visual fixture has not been initialized")
        level = quantize_resource(resource)
        if self._levels != [level] * len(self.panels):
            self._levels[:] = [level] * len(self.panels)
            self._apply()

    def set_visible_panels(self, indices):
        """Preflight absence/duplicate/reversal arrangement, not live authority."""
        if not self._initialized:
            raise RuntimeError("Native visual fixture has not been initialized")
        indices = tuple(indices)
        if len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < len(self.panels) for i in indices
        ):
            raise ValueError("Invalid visible panel indices")
        self._visible[:] = [i in indices for i in range(len(self.panels))]
        self._apply()

    def rendering_contract(self):
        """Renderer metadata for audit only; do not send it to a pixel observer."""
        return {
            "schema": "bb8.native-visual-fixture.v2",
            "rendering": "ordinary native RGB; fixed noncolliding visual geometry",
            "overlay": False,
            "debug_camera": False,
            "physics_effect": False,
            "dynamic_api": "RigidEntity.set_vverts (visual vertices only)",
            "panel_cells": [PANEL_COLUMNS, PANEL_ROWS],
            "cell_size_m": CELL_SIZE_M,
            "panel_height_m": PANEL_HEIGHT_M,
            "gauge_bins": GAUGE_BINS,
            "gauge_cell_rectangles": [list(gauge_rect(i)) for i in range(GAUGE_BINS)],
            "glyph_on_rgba": list(ID_COLOR),
            "revision_reason": "Saturated glyphs avoid gray-head proposals; larger gauge cells preserve the minimum raw-pixel evidence requirement.",
            "resource_rounding": "floor(resource * bins + 0.5), clipped to bins",
            "target_offset_m": list(MARKER_TARGET_OFFSET_M),
            "target_scope": "Engineered marker semantics applied to RGB-derived anchor",
            "panels_scoring_only": [
                {"visual_id": panel.visual_id, "center_xy": list(panel.center_xy)}
                for panel in self.panels
            ],
        }
