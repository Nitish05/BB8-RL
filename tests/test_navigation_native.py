import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.native_gui,
    pytest.mark.skipif(
        os.getenv("RUN_NATIVE_GUI") != "1", reason="Opt-in native BB-8 viewer"
    ),
]


@pytest.mark.parametrize(
    "project", ["empty-floor.genesis.json", "synthetic-room/room.genesis.json"]
)
def test_saved_world_reopens_in_studio(project):
    from genesis_studio_desktop.app import run as studio_run

    assert (
        studio_run(
            [
                "--project",
                str(ROOT / "projects/bb8" / project),
                "--smoke-seconds",
                "2.5",
            ]
        )
        == 0
    )


def test_controllable_world_native_viewer():
    from bb8_rl.cli import run

    assert (
        run(
            [
                "simulate",
                "--config",
                str(ROOT / "configs/navigation/bb8-state.yaml"),
                "--viewer",
                "--seconds",
                "1.5",
            ]
        )
        == 0
    )
