"""Pure native-API adapter tests; no renderer or weight deserialization."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bb8_rl.visual_fixture import (
    DEFAULT_PANELS,
    VisualCamera,
    VisualFixture,
    VisualPanel,
    panel_boxes,
    panel_rect,
    quantize_resource,
)
from bb8_rl.visual_interaction import GAUGE_BINS, PANEL_HEIGHT_M


class Entity:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        morph = kwargs["morph"]
        center, size = np.array(morph.pos), np.array(morph.size)
        self.vertices = np.array(
            [
                center + size * np.array([x, y, z]) / 2
                for x in (-1, 1)
                for y in (-1, 1)
                for z in (-1, 1)
            ]
        )
        self.updates = []

    def get_vverts(self):
        return self.vertices.copy()

    def set_vverts(self, vertices):
        self.vertices = np.array(vertices, copy=True)
        self.updates.append(self.vertices.copy())


class Scene:
    def __init__(self):
        self.entities = []

    def add_entity(self, **kwargs):
        entity = Entity(kwargs)
        self.entities.append(entity)
        return entity

    def add_camera(self, **kwargs):
        return SimpleNamespace(**kwargs)


def fixture(panels=DEFAULT_PANELS):
    scene = Scene()
    gs = SimpleNamespace(
        morphs=SimpleNamespace(Box=SimpleNamespace),
        surfaces=SimpleNamespace(Default=SimpleNamespace),
    )
    result = VisualFixture(panels)
    result.attach(scene, gs)
    result.initialize_visuals()
    return result, scene


@pytest.mark.parametrize("value", [-1, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_resource_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        quantize_resource(value)


def test_quantization_is_fixed_half_up_and_clipped():
    assert quantize_resource(0) == 0
    assert quantize_resource(1) == GAUGE_BINS
    for level in range(GAUGE_BINS):
        assert quantize_resource((level + 0.5) / GAUGE_BINS) == level + 1


def test_native_adapter_has_no_physics_or_pose_commands():
    rendered, scene = fixture()
    assert len(scene.entities) == 2 * (1 + 4 + 16 + GAUGE_BINS)
    for entity in scene.entities:
        morph = entity.kwargs["morph"]
        assert morph.fixed and not morph.collision and morph.visualization
        assert morph.enable_custom_vverts
    assert rendered.rendering_contract()["overlay"] is False
    assert rendered.rendering_contract()["debug_camera"] is False


def test_fixture_palette_does_not_propose_neutral_robot_heads():
    from bb8_rl.vision import candidates

    # V1 white glyph cells produced extra candidates in native A/B frames.
    # Test the actual unchanged proposal stage, not a replacement threshold.
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[8:15, 8:15] = (255, 255, 255)
    assert len(candidates(rgb)) == 1
    for color in {box["color"] for box in panel_boxes(DEFAULT_PANELS[0])}:
        rgb[8:15, 8:15] = np.rint(np.array(color[:3]) * 255).astype(np.uint8)
        assert candidates(rgb) == []


def test_revised_gauge_remains_separated_inside_unchanged_panel():
    renderer = VisualFixture()
    contract = renderer.rendering_contract()
    assert contract["schema"] == "bb8.native-visual-fixture.v2"
    assert contract["panel_cells"] == [14, 8]
    assert contract["cell_size_m"] == 0.025
    assert contract["target_offset_m"] == [0, -0.25]
    rectangles = contract["gauge_cell_rectangles"]
    assert len(rectangles) == GAUGE_BINS
    assert all(
        0.5 < x0 < x1 < 13.5
        and 5 < y0 < y1 < 7.5
        and x1 - x0 >= 0.75
        and y1 - y0 >= 0.79
        for x0, y0, x1, y1 in rectangles
    )
    assert rectangles[15][3] < rectangles[16][1]


def test_gauge_changes_only_visual_vertices_and_keeps_anchors():
    rendered, scene = fixture()
    initial = [entity.vertices.copy() for entity in scene.entities]
    rendered.set_resource(1.0)
    for entity, before in zip(scene.entities, initial):
        np.testing.assert_array_equal(entity.vertices[:, :2], before[:, :2])
    for panel_index in range(2):
        start = panel_index * 53
        for entity, before in zip(
            scene.entities[start : start + 21], initial[start : start + 21]
        ):
            np.testing.assert_array_equal(entity.vertices, before)
        assert all(
            abs(entity.vertices[:, 2].max() - PANEL_HEIGHT_M) < 1e-10
            for entity in scene.entities[start + 21 : start + 53]
        )
    rendered.set_resource(0.0)
    assert all(
        entity.vertices[:, 2].max() < PANEL_HEIGHT_M - 0.002
        for index, entity in enumerate(scene.entities)
        if index % 53 >= 21
    )
    rendered.set_resource(1.0)
    assert all(
        abs(entity.vertices[:, 2].max() - PANEL_HEIGHT_M) < 1e-10
        for index, entity in enumerate(scene.entities)
        if index % 53 >= 21
    )


def test_rows_are_not_mirrored_and_declared_geometry_is_bounded():
    panel = DEFAULT_PANELS[0]
    upper, _ = panel_rect(panel, (1, 1, 2, 2))
    lower, _ = panel_rect(panel, (1, 4, 2, 5))
    assert upper[1] > lower[1]
    for box in panel_boxes(panel):
        position, size = panel_rect(panel, box["rect"])
        assert size[0] > 0 and size[1] > 0
        assert abs(position[0] - panel.center_xy[0]) <= 0.175
        assert abs(position[1] - panel.center_xy[1]) <= 0.1


def test_missing_duplicate_and_reversed_rendering_are_explicit_preflight_states():
    panels = (*DEFAULT_PANELS, VisualPanel("marker-01", (0.5, 1.25)))
    rendered, scene = fixture(panels)
    rendered.set_visible_panels(())
    assert all(entity.vertices[:, 2].max() < 0 for entity in scene.entities)
    rendered.set_visible_panels((0, 2))
    assert scene.entities[0].vertices[:, 2].max() > 0
    assert scene.entities[53].vertices[:, 2].max() < 0
    assert scene.entities[106].vertices[:, 2].max() > 0
    with pytest.raises(ValueError):
        rendered.set_visible_panels((0, 0))


def test_lifecycle_invalid_panels_and_bounds():
    with pytest.raises(ValueError):
        VisualPanel("unknown", (0, 0))
    with pytest.raises(ValueError):
        VisualPanel("marker-01", (float("nan"), 0))
    with pytest.raises(ValueError):
        VisualFixture(())
    with pytest.raises(ValueError):
        VisualFixture(DEFAULT_PANELS * 4)
    unbuilt = VisualFixture()
    with pytest.raises(RuntimeError):
        unbuilt.initialize_visuals()
    rendered, _ = fixture()
    with pytest.raises(RuntimeError):
        rendered.initialize_visuals()
    with pytest.raises(RuntimeError):
        rendered.attach(None, None)


def test_semantic_camera_is_ordinary_rgb_at_declared_pose_and_resolution():
    spec = VisualCamera("V", (2.2, -3.8, 3.2), (0, 0, 0.1))
    renderer = VisualFixture(camera_specs=(spec,))
    scene = Scene()
    gs = SimpleNamespace(
        morphs=SimpleNamespace(Box=SimpleNamespace),
        surfaces=SimpleNamespace(Default=SimpleNamespace),
    )
    renderer.attach(scene, gs)
    camera = renderer.cameras["V"]
    assert camera.debug is False and camera.GUI is False
    assert camera.res == (2560, 1920) and camera.pos == spec.position
    with pytest.raises(ValueError):
        VisualFixture(camera_specs=(spec, spec))
    with pytest.raises(ValueError):
        VisualCamera("V", (0, 0, 1), (0, 0, 1))


def test_resource_and_visibility_require_initialized_renderer():
    renderer = VisualFixture()
    with pytest.raises(RuntimeError):
        renderer.set_resource(0.75)
    with pytest.raises(RuntimeError):
        renderer.set_visible_panels((0,))


def test_preflight_preserves_installed_layout_seed_and_selects_validation_domain():
    from bb8_rl.task import require_split_seed

    script = Path(__file__).resolve().parents[1] / "scripts/preflight-visual-fixture.py"
    spec = importlib.util.spec_from_file_location(
        "visual_fixture_preflight_contract", script
    )
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    demo = {"task": "scene/task.yaml", "layout_seed": 11178}
    cameras = {name: {"position": (0, 0, 3)} for name in "ABC"}
    fixture = object()

    class EnvironmentContract:
        def __init__(self, task_path, *, split="train", **kwargs):
            self.split, self.kwargs = split, kwargs
            self.task_path = task_path

        def reset(self, *, options):
            require_split_seed(self.split, options["layout_seed"])

    env = EnvironmentContract(
        **preflight.environment_kwargs(Path("assets"), demo, cameras, fixture)
    )
    env.reset(options={"layout_seed": demo["layout_seed"]})
    assert env.split == "validation"
    assert env.task_path == Path("assets/scene/task.yaml")
    assert env.kwargs["visual_fixture"] is fixture
    assert env.kwargs["render_mode"] == "rgb_array"
    with pytest.raises(ValueError):
        preflight.environment_kwargs(
            Path("assets"), dict(demo, layout_seed=926101), cameras, fixture
        )


def test_preflight_freezes_two_registered_unoccluded_development_views(
    tmp_path, monkeypatch
):
    script = Path(__file__).resolve().parents[1] / "scripts/preflight-visual-fixture.py"
    spec = importlib.util.spec_from_file_location("visual_fixture_freeze", script)
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    cameras = {name: {"position": (0, 0, 3)} for name in "ABC"}
    demo = {"task": "scene/task.yaml", "layout_seed": 11178, "cameras": cameras}
    (tmp_path / "bundle.json").write_text("{}")
    monkeypatch.setattr(preflight, "validate_assets", lambda _: demo)
    monkeypatch.setattr(preflight, "SOURCES", ())
    protocol = preflight.make_protocol(tmp_path)
    preflight.validate_protocol(protocol, tmp_path)
    assert protocol["schema"] == "bb8.visual-fixture-preflight.v2"
    assert protocol["semantic_camera_ids"] == ["B", "C"]
    assert protocol["registered_cameras"] is cameras
    assert protocol["semantic_resolution_scale"] == 2
    assert protocol["navigation_camera_ids"] == ["A", "B", "C"]
    assert len(protocol["cases"]) == 40
