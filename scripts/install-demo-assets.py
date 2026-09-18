"""Install the pinned private GitHub demo release, or a supplied release archive."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from bb8_rl.demo_assets import install_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded release ZIP"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work/interactive-assets")
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/interactive/demo-assets.json").read_text())
    if args.archive:
        result = install_bundle(
            args.archive, args.output, expected_sha256=spec["sha256"]
        )
    else:
        with tempfile.TemporaryDirectory(prefix="bb8-demo-") as directory:
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    spec["release"],
                    "--repo",
                    spec["repository"],
                    "--pattern",
                    spec["asset"],
                    "--dir",
                    directory,
                ],
                check=True,
            )
            result = install_bundle(
                Path(directory) / spec["asset"],
                args.output,
                expected_sha256=spec["sha256"],
            )
    print(f"Verified demo assets installed: {result}")


if __name__ == "__main__":
    main()
