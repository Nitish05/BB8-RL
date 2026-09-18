# Project milestone history

Archived project overview through M7.8. For current launch instructions, see [the user guide](GETTING_STARTED.md).

# BB8-RL

A separate BB-8 navigation and reinforcement-learning project that uses
**Genesis Studio** for scene authoring and construction, and **Genesis World** for
physics and rendering.

M0–M1 provide native runtime checks and a controllable, resettable synthetic world.
M3 adds a Gymnasium navigation task, seeded obstacle courses, an A* route controller,
and paired baseline/random evaluation. M4 adds SAC + HER training, checkpoint/resume
and checkpoint-policy evaluation. Perception, click-to-go and hardware integration
are later work. Dimensions, mass, appearance and response
parameters are synthetic. M2 now uses a user-authorized arbitrary room with random
obstacles and an elevated fixed camera; physical measurements are deferred until
transfer work. See [the synthetic room setup](M2_SYNTHETIC.md).

## Run with the existing Studio installation

Keep this checkout beside the unmodified `Genesis-Studio` checkout:

```text
Codex/
  Genesis-Studio/       dependency: tested at cd20c8c, version 0.2.0
  BB8-RL/              this independent project
```

From **BB8-RL**:

```bash
./scripts/run-bb8.sh world validate
./scripts/run-bb8.sh simulate --viewer --drive 0.5 0 --seconds 4
./scripts/run-bb8.sh doctor --backend metal --device mps --output work/doctor
./scripts/open-studio.sh
```

For the navigation task, install the navigation extra into the selected runtime
once, then run:

```bash
./scripts/python.sh -m pip install 'gymnasium>=1.2,<2'
./scripts/run-bb8.sh env-check --output work/gym-check.json
./scripts/run-bb8.sh evaluate --suite validation --episodes 12 --controllers baseline random --output work/validation
./scripts/run-bb8.sh evaluate --suite train --episodes 1 --controllers baseline --viewer --output work/viewer-demo
```

Choose a fresh output directory for each evaluation. It saves a machine/configuration
manifest, per-episode trajectories, a summary with confidence intervals, a path plot,
and the exact sampled worlds as ordinary Genesis Studio Project JSON. See
[M3 task contract and evaluation](M3.md) for the observation/reward contract,
reserved final test suite and comparison limits.

`open-studio.sh` opens this project's saved world in the existing Studio editor.
Saving it keeps it in BB8-RL. Planar drive and head following are supplied by this
project's simulation command; the normal Studio UI edits the initial scene.

The launcher uses this project's `.venv` when available, otherwise the existing
Studio Python environment. In the latter case it temporarily adds this project's
`src` to the child process's module path. It does not install BB8-RL into Studio,
patch Studio, or copy BB-8 files into the Studio repository. Override the runtime
with `BB8_PYTHON`, or the dependency checkout with `GENESIS_STUDIO_ROOT`.

## Install in an independent environment

Use native Python 3.12 and the Studio checkout at the tested revision. From BB8-RL:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r configs/navigation/requirements-macos-py312.lock.txt
.venv/bin/python -m pip install --no-deps -e ../Genesis-Studio -e .
.venv/bin/bb8-rl doctor --backend metal --device mps --output work/doctor
```

The lock is a snapshot of tested package versions, not a cross-platform or hashed
wheel lock. Re-run the doctor after installation. The current migration was tested
using the existing Studio Python runtime; the independent environment commands
are an optional installation route.

## Ownership and integration

The synthetic 4 × 4 m room is directly usable by the navigation environment:

```bash
./scripts/run-bb8.sh room preview --output work/room-camera
./scripts/run-bb8.sh evaluate --task projects/bb8/synthetic-room/task.yaml --suite train --episodes 1 --controllers baseline --viewer --output work/room-demo
```

Use Studio's Open Project flow for `projects/bb8/synthetic-room/room.genesis.json`.

| BB8-RL owns | Genesis Studio supplies |
| --- | --- |
| BB-8 worlds, assets and task configuration | Schema-v4 Project JSON and validation |
| Planar command contract and simulation adapter | Existing scene builder and compatibility check |
| Gymnasium environment, maps, route baseline and evaluation | Native editor for opening this project's worlds |
| SAC + HER training, checkpoints, diagnostics and tests | General-purpose authoring services |

The [M5 SAC/DreamerV3 comparison](M5.md) adds the pinned authors' DreamerV3
implementation through an optional CPU adapter. See the [experiment protocol](M5_PLAN.md)
for budgets, shared observations and comparison limits. Both M5 learners use the
same isolated dependency overlay; Studio source and its runtime remain unchanged.

The [M6 learned waypoint controllers](M6.md) address the failed navigation
pilot with full-map A* waypoints, successful demonstrations, progress rewards,
and teacher-regularized SAC/Dreamer training. Both policies generate all evaluated
drive commands. This is synthetic hybrid navigation with oracle state/map input;
M6 itself has no camera input. See the [frozen evaluation plan](M6_PLAN.md).
The recorded M6 checkpoints each reached 200/200 held-out synthetic goals with
zero collisions, retaining the original distance/speed/dwell arrival gate.

[M7.1 camera control](M7.md) now provides calibrated RGB localization,
temporal state estimation, a camera-derived floor map and tracking-loss braking
feeding the frozen SAC policy. The first synthetic color baseline achieved only
1/12 development arrivals; learned segmentation and occlusion handling remain
necessary. The [research review and selected approach](M7_RESEARCH.md) choose
SAM 3.1 teaching and a compact temporal student, with TAPNext++ as a tracking
comparator. Those teacher/tracker integrations and physical camera control remain
pending; M7.6 adds a separate foundation model for offline geometry.

[M7.2 learned vision](M7_2.md) trains small RGB floor/head networks from
synthetic renderer labels. Correcting robot self occupancy improved the same
camera development suite from 1/12 to 6/12 arrivals, with zero collisions; six
map/route aborts remain. A five-frame camera dropout test stopped, reacquired and
arrived. This is a single-seed synthetic pilot, not reliable real-room control or
completion of the full temporal/foundation-model research plan.

[M7.3 camera map refinement](M7_3.md) improves that development suite to
8/12 arrivals with zero collisions. It adds optional RGB edge refinement,
continuous endpoint clearance checks and guarded waypoint advancement. The
four remaining cases stop for insufficient visible clearance or conservative
perception; one goal is occluded by an obstacle. The vision/SAC weights and
original arrival gate are unchanged. This remains development evidence from
one synthetic room, with fresh randomized-camera acceptance still pending.

[M7.5 multi-view observation and room memory](M7_5.md) adds one-to-three-camera
RGB capture, conservative measured-position fusion, room-disjoint multi-angle
training/audits and a serializable 3D evidence contract. These interfaces do not
enable multi-camera driving, reconstruct a room from RGB or drive through occlusion.

[M7.6 parallel camera tracks](M7_6.md) adds actual pretrained MapAnything RGB
reconstruction, explicit registration to known metric scan-camera poses, estimated
surface memory and held-out fixed-camera registration experiments. It also adds a
position-only observer, a transactional 50 ms deadline gate, actual box-occlusion
capture and causal velocity/acceleration prediction replay. The isolated two-camera
fast path measured 29.26 ms p95 with no deadline misses in 40 cycles. Reconstruction
accuracy, final-camera registration and free-volume certification still gate
scan-based driving; prediction replay does not authorize hidden motion.

[M7.7 scan refinement](M7_7.md) fixes fixed-camera registration on four
reserved views using SuperPoint/LightGlue tracks and metric triangulation.
Independent position errors are 4.9–8.7 mm. RGB-anchor calibration reduces dense
map p95 surface error from 16.55 to 9.89 cm with every previously accepted sample
retained and about 99% obstacle completeness within 5 cm. A separate multiview
photometric layer improves precision further, with its coverage reported
separately. The scan cameras still have known metric poses; real handheld scan
calibration and general occluded navigation remain unvalidated. The bounded
scan-memory control experiment below adds native driving evidence.

[M7.8 scan-memory control](M7_8.md) adds explicit RGB-supported clear-volume
memory, command-conditioned occlusion prediction, stopping-envelope checks and
an independent native audit. The accepted development map has zero falsely clear
prisms in the authored-room geometry check. One supplemental route now passes:
6.7 cm of native movement between fully hidden captures, maximum hidden position
error 1.36 cm, visual reacquisition and verified arrival without contact. All three
supplemental dropout/invalid-memory controls pass. The original six routes remain
blocked (0/6), and their denominator is preserved. This remains a lockstep
synthetic experiment with known metric scan poses and a selected development route.

Runtime code is under `src/bb8_rl`. `control/genesis_backend.py` owns this project's
Genesis actuation; no new control capability is added to Studio. `world.py` calls
Studio's existing scene builder with the world JSON stored here. Hardware packages
are never imported by simulation or ordinary package import.

## Test

See [the M4 plan](M4_PLAN.md) and [training/resume guide](M4.md) to train
the first policy or run the three-seed pilot. Training is optional and uses
`stable-baselines3==2.9.0`; simulation commands do not import the learner.

```bash
./scripts/python.sh -m pytest -m 'not genesis and not native_gui' -q
RUN_GENESIS_INTEGRATION=1 RUN_GENESIS_METAL=1 ./scripts/python.sh -m pytest -m genesis -q
RUN_NATIVE_GUI=1 ./scripts/python.sh -m pytest -m native_gui -q
./scripts/python.sh -m compileall -q src
```

See [M0–M1 behavior and limitations](M0_M1.md) and [M3](M3.md). Camera and BLE validation remain
pending. Pilot checkpoints are simulation-only and their measured evaluation
results determine what they can do; successful optimization alone is not navigation success.
