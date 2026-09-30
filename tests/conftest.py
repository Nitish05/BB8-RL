"""Explicit private-dependency coverage gate; ordinary pytest still runs all tests.

The public-package CI job discovers every test except the listed Studio API
contracts and opt-in native suites. It never turns unexpected ImportError into a
skip. A separate job with read access to the pinned private dependency runs the
complete non-native suite.
"""

import pytest

STUDIO_MODULES = {
    "test_dreamer.py": "Dreamer adapter imports Studio-backed environment/diagnostics",
    "test_guided.py": "waypoint environment imports Studio-backed navigation",
    "test_navigation.py": "Studio Project and scene configuration contracts",
    "test_room.py": "Studio object schemas and authored-room environment",
    "test_task.py": "Studio-backed task and environment configuration",
    "test_training.py": "SAC/HER environment and Studio-backed diagnostics",
}
NATIVE_MODULES = {
    "test_guided_genesis.py",
    "test_navigation_genesis.py",
    "test_navigation_native.py",
    "test_room_genesis.py",
    "test_task_genesis.py",
    "test_task_native.py",
    "test_training_genesis.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--without-studio",
        action="store_true",
        help="Report and omit explicit private Studio/native dependencies before import",
    )


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--without-studio"):
        return None
    if collection_path.parent != config.rootpath / "tests":
        return None
    if collection_path.name in STUDIO_MODULES or collection_path.name in NATIVE_MODULES:
        return True
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in STUDIO_MODULES:
            item.add_marker(pytest.mark.studio)
    if config.getoption("--without-studio"):
        excluded = [item for item in items if item.get_closest_marker("studio")]
        if excluded:
            items[:] = [item for item in items if item not in excluded]
            config.hook.pytest_deselected(items=excluded)


def pytest_terminal_summary(terminalreporter, config):
    if not config.getoption("--without-studio"):
        return
    terminalreporter.section("Coverage limitation: private Studio dependency omitted")
    for name, reason in sorted(STUDIO_MODULES.items()):
        terminalreporter.write_line(f"tests/{name}: {reason}")
    terminalreporter.write_line(
        "Studio-marked camera world-validation and generated-room fixture cases "
        "are deselected; "
        f"{len(NATIVE_MODULES)} opt-in native modules are not imported."
    )
    terminalreporter.write_line(
        "Full non-native gate: pytest -m 'not genesis and not native_gui' "
        "with the pinned Genesis Studio package installed."
    )
