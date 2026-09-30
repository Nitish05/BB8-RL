"""Prove discovered audit coverage and reject deliberate in-process regressions.

Mutations live only in fresh pytest subprocesses. No source file, existing memory,
model or native simulator is modified or opened. An ordinary passing baseline is
required before the corresponding deliberate mutant must fail an assertion.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "unauthorized-motion": (
        "tests/test_interactive_audit.py::test_no_motion_after_goal_cancel"
    ),
    "missed-solid-overlap": (
        "tests/test_occluded_control_audit.py::"
        "test_cell_center_clear_but_solid_crosses_cell_is_false_free"
    ),
    "unchecked-config-reference": (
        "tests/test_demo_assets.py::"
        "test_nested_configuration_cannot_reference_outside_bundle"
    ),
}


class Mutant:
    def __init__(self, name):
        self.name = name

    def pytest_runtest_setup(self, item):
        module = item.module
        if self.name == "unauthorized-motion":
            original = module.audit.score_rows

            def score_rows(*args, **kwargs):
                result = original(*args, **kwargs)
                result["errors"] = [
                    error
                    for error in result["errors"]
                    if error["reason"] != "motion_without_goal"
                ]
                return result

            module.audit.score_rows = score_rows
        elif self.name == "missed-solid-overlap":
            original = module.audit.solid_intersections

            def solid_intersections(*args, **kwargs):
                overlaps, contacts = original(*args, **kwargs)
                overlaps[:] = False
                return overlaps, contacts

            module.audit.solid_intersections = solid_intersections
        else:
            module.validate_assets = lambda *args, **kwargs: {"valid": True}


def run(arguments):
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-studio", action="store_true")
    parser.add_argument("--run-mutant", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    options = ["-q", "-p", "no:cacheprovider", "-m", "not genesis and not native_gui"]
    if args.without_studio:
        options.append("--without-studio")
    if args.run_mutant:
        import pytest

        return int(
            pytest.main(
                [*options, CASES[args.run_mutant]], plugins=[Mutant(args.run_mutant)]
            )
        )
    collected = run(["-m", "pytest", "--collect-only", *options])
    if collected.returncode:
        print(collected.stdout)
        return 1
    nodeids = {
        line.strip()
        for line in collected.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }
    modules = {node.split("::")[0] for node in nodeids}
    required = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests").glob("test_*audit*.py")
    } | {"tests/test_demo_assets.py", "tests/test_purpose_runtime.py"}
    missing = sorted(required - modules)
    if missing:
        print(f"Required validator suites were not collected: {missing}")
        return 1
    baseline = run(["-m", "pytest", *options, *CASES.values()])
    if baseline.returncode:
        print("Negative-control baseline failed:")
        print(baseline.stdout)
        return 1
    results = []
    for name in CASES:
        command = [str(Path(__file__).resolve()), "--run-mutant", name]
        if args.without_studio:
            command.append("--without-studio")
        result = run(command)
        # pytest exit 1 is test failure; import/usage/no-tests errors do not count.
        rejected = result.returncode == 1 and (
            "AssertionError" in result.stdout or "DID NOT RAISE" in result.stdout
        )
        results.append({"mutation": name, "rejected_by_test": rejected})
        if not rejected:
            print(result.stdout)
    report = {
        "scope": "public dependencies" if args.without_studio else "full non-native",
        "collected_tests": len(nodeids),
        "collected_modules": len(modules),
        "required_validator_modules": sorted(required),
        "negative_controls": results,
        "source_files_modified": False,
    }
    print(json.dumps(report, indent=2))
    return 0 if all(result["rejected_by_test"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
