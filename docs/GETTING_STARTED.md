# Run the BB8-RL control room

## Existing prepared workspace

From the BB8-RL checkout, run `./scripts/launch-control-room.sh`. It selects the
prepared `.venv-dreamer` runtime when present and opens `http://127.0.0.1:8765`.
Keep the terminal open; Ctrl-C shuts down the worker and local server.
On macOS, double-click `BB8-Control-Room.command` in Finder to use the same launcher.

The first native scene build can take longer than later runs. The page remains
available while the worker loads. Camera feeds start when calibration, model and
scene checks have finished.

## Fresh checkout

The tested platform is Apple Silicon macOS with native Python 3.12. Other
platforms need their own native Genesis validation. You need access to this
private repository and its Genesis Studio dependency. Use the GitHub CLI with
your own authenticated account; never paste tokens into configuration files.

```bash
gh repo clone Nitish05/Genesis-Studio
gh repo clone Nitish05/BB8-RL
cd Genesis-Studio
git checkout cd20c8c685f2b51263814fd0a6296bfd849bdce6
cd ../BB8-RL
python3.12 -m venv .venv
.venv/bin/python -m pip install -r configs/navigation/requirements-macos-py312.lock.txt
.venv/bin/python -m pip install --no-deps -e ../Genesis-Studio -e .
./scripts/python.sh scripts/install-demo-assets.py
./scripts/launch-control-room.sh
```

The lock records the prepared macOS environment; it is not a cross-platform lock.
The explicit asset installer downloads the pinned release through `gh`, verifies
its checksum and installs it under ignored `work/interactive-assets`. Nothing is
downloaded by an import or by opening the interface. To install a ZIP obtained
separately, pass `--archive /path/to/bb8-demo-assets-v2.zip`.

## Update an existing demo installation

After updating the checkout, stop the app with Ctrl-C and retain the old assets
before installing the newly pinned release. The installer deliberately requires
a fresh destination:

```bash
mv work/interactive-assets work/interactive-assets-v1-backup
./scripts/python.sh scripts/install-demo-assets.py
./scripts/launch-control-room.sh
```

Choose another backup name if that directory already exists. The models and live
camera registration are unchanged; the new bundle improves the saved scan map.
The 48-view scan was prepared once. Launching or switching camera modes reuses it.
See the [scan-coverage report](M7_12_SCAN_COVERAGE.md) for reconstruction evidence
and limitations.

## Drive with the interface

1. Wait for a current camera estimate. Use **Demo route** for the first run; it
   resets the scene and submits the known development destination.
2. Click observed clear map space, or enter metric X/Y coordinates. The runtime
   checks the robot footprint, route, uncertainty and stopping clearance before
   issuing a drive request. A clear-looking cell alone does not guarantee a route.
3. Select **2 cameras** or **3 cameras** to rebuild the episode in that mode. The
   fixed views share the same estimated scan memory; this does not trigger a new
   physical or synthetic scan. Camera estimates are fused only when compatible.
4. **Stop** or Escape cancels the target and clears queued motion. **Reset** creates
   a fresh episode at the demo start. A lost browser heartbeat also cancels motion;
   reconnecting alone does not resume a cancelled target.

Unknown/occupied space stays blocked. The route can stop as uncertainty grows or
visual evidence becomes stale. The interface distinguishes measured and predicted
positions; it never displays simulator truth as the navigation estimate.

## Launch options and evidence

```bash
./scripts/launch-control-room.sh --mode 2
./scripts/launch-control-room.sh --mode 3 --port 8766
./scripts/launch-control-room.sh --no-browser --output work/my-interactive-session
./scripts/launch-control-room.sh --control-profile baseline
```

The default `reserve-3cm` profile uses a **3 cm fixed margin**, a **15 cm/s speed
cap** and **4 cm of extra route-search clearance** to leave room for braking.
This is not 3 cm total clearance: the robot radius, camera uncertainty and
stopping motion are still included. The wider search can reject tight goals.
The previous settings remain available as `baseline`; all seven experimental
profiles and their outcomes are in the [configuration comparison](M7_11_RESULTS.md).

Choose a fresh output directory. Each worker session records issued requests,
command acknowledgements, source/model hashes and a separately labeled physics
trace for independent scoring. These logs remain ignored by Git.

This is lockstep simulation: physics pauses during camera rendering/inference.
The displayed processing time does not establish a wall-clock control guarantee.
The application binds only to loopback and never connects to robot hardware.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing demo assets | Run the explicit asset installer; confirm access to the private release |
| Port already in use | Choose `--port 8766` or stop the earlier terminal session |
| No current pose | Wait for initialization; inspect camera status; Reset if the worker reports an error |
| Destination rejected | Choose a point with more observed clearance; unknown cells cannot be overridden |
| Route not certified | The complete route needs room for the robot, margin and current camera uncertainty. The planner searches with that radius; if no route passes, try a more open destination or another camera mode |
| Motion cancelled after tab closes | Reopen the app and submit a new goal; heartbeat cancellation is intentional |
| Graphics/runtime error | Run the existing doctor, verify the native Python environment and local Genesis installation |

## Verification commands

```bash
./scripts/python.sh -m pytest -m 'not genesis and not native_gui' -q
./scripts/python.sh -m ruff check src tests scripts
```

The CI job tests pure camera/control/application contracts without graphics or
demo models. Native motion and browser acceptance are separate local tests; their
measured outcomes and limitations belong in [the validation report](M7_9_VALIDATION.md).
