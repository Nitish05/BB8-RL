# Original-reference recovery and RGB information preflight

27 September 2026 continuation. The implementation plan was written before changes; the existing checkout, installed assets, environments, learned memory and prior evidence were retained.

## Implemented behavior

With `--scene-validity`, **Recheck scene** can now clear a confirmed scene fault after the original camera positions and visible scene have been restored. The same-session reference images remain immutable. The check requires three fresh matching structural checks spanning at least one simulated second, an unchanged map/calibration, a live heartbeat and a complete zero applied-command history. Stop or a newer command cancels an attempt. Failed attempts do not retry themselves.

The worker and Supervisor acknowledge the exact command generation and fault epoch. A delayed result cannot clear a newer fault or replay a cancelled route. Recovery leaves the destination empty and learning paused; bounded local RGB reacquisition and a new explicit request are still required. Reset cannot replace a locked reference. This is restoration to the original configuration, not arbitrary camera calibration, global localization or map repair.

## Native validation inventory

The three case requests were frozen before evaluation. The known installed room, policy, detector, map and control limits were unchanged. Camera interventions affect the renderer only; control consumes RGB and its original declared calibrations. Truth and segmentation labels are restricted to the post-decision scorer.

| Case | Retained attempt | Result |
|---|---|---|
| One camera, fault during active motion | v1 | Recovery and actual arrival occurred; the driver expected the wrong arrival status and hit its deadline. Preserved as an unsuccessful complete attempt. |
| One camera, same case repeated | v2 | Restored-reference acknowledgement at 9.40 s; arrival at 19.90 s after a new goal. |
| Two cameras, Stop during recheck | v2 | Stop cancelled the attempt at 8.65 s; explicit retry recovered at 9.80 s; arrival at 16.35 s. |
| Three cameras, heartbeat interruption | v2 | Recheck finished before the three-second wall-clock heartbeat expiry; the planned interruption was not exercised. Preserved as a failed scheduling attempt. |
| Three cameras, heartbeat timing refined | v3 | A declared 2.8-second heartbeat silence preceded recheck. Expiry cancelled it at 9.60 s; explicit retry recovered at 10.75 s; arrival at 16.90 s. |

Times are simulated seconds from episode start, not physical-time performance. There are three successful complete attempts out of five attempts for three requested cases. The two unsuccessful attempts remain in the denominator. The heartbeat refinement changed test scheduling only; it did not change production timeouts, recovery thresholds, scene poses or destinations.

All five native attempts recorded zero contacts, boundary failures and worker errors. The successful attempts rejected recheck while the camera remained displaced, retained the original image hashes, kept zero command authority through fault/recovery hold, and admitted motion only after a higher-generation goal. Zero actuator requests do not imply mathematically zero body velocity during braking. Separate raw trace and image audits accompany these results.

The active case had nonzero acknowledged actuator targets before the camera
change for 0.10 simulated seconds. The maximum sampled true speed through the
2.5 s change boundary was 0.0137 m/s when all physics substeps are included
(the coarser frame captures show 0.00385 m/s). It validates command revocation
and recovery at that operating point, not high-speed braking.

A supplemental live control-room session exercised the actual browser button through CUA. The UI showed locked controls, then **Checking scene…**, then **Localized · choose a goal**, with learning paused and no destination. Stop remained available. All 1,936 recorded rows in that session had zero drive actions, no goal and no enabled agency. This UI session is separate from the three-case navigation inventory.

## Additional RGB geometry evidence

A stationary native acquisition added one declared point light at six positions, with intensity 0/10/20 controls, while preserving geometry, materials, existing lighting, camera and physics state. It produced 28 RGB frames with zero physics steps. Capture imported a frozen BB8 source snapshot, with installed Genesis/Studio source identities bound separately. No depth, normal or segmentation buffer entered this experiment.

All 16 zero-light controls exactly reproduced the original RGB image. All six preregistered intensity-response checks passed; independent analysis reproduced the saved results from pixel data. This establishes repeatable controlled illumination, not a metric depth measurement.

Near the original new-room destination, the best light's median quantization halfwidth was about 14.3% of the added signal. An ideal calculation allowing unknown normal and reflectance predicts only about 1.04% residual for a competing 12 cm height. The current acquisition therefore does not establish height resolution. A prospective eight-light arrangement improves ideal conditioning to about 4.24% at that height, but has not been captured or validated against native PBR, shadows and calibration uncertainty. Its exact positions are saved for the next bounded experiment.

No FREE cells, map or replacement bundle were installed. All six original new-room driving requests remain unattempted behind the same full-volume, uncertainty, clearance and route gates. The floor, targets and safety margins were not altered.

## Verification and preservation

The full non-native suite passes: 1,591 tests across the main run and two localhost-bind retries; 20 native/GUI tests were excluded from that command and the native cases above were run separately. Ruff and all three frontend suites pass. CI contract checks collect 73 modules and reject all three injected negative controls. The sensing helpers have 25 passing pure tests.

The preservation inventory checks 255 files: 246 unchanged and nine authorized production/test changes, with no missing or unexpected changed files. All 237 files in the native source freeze remain unchanged after evaluation. Installed assets and production memory retain their hashes; Genesis-Studio remains clean at its original revision. No model or environment was installed or replaced.

The independent native scorer has 36 passing tests, including injected missing
actuator partitions, altered reference/ticket evidence, premature motion,
incorrect destinations and invalid cancellation histories. It separately
reconstructs reference hashes from saved RGB, checks source/image provenance,
verifies all 10 physics intervals per frame, and scores arrival distance, speed
and dwell from post-decision truth.

Launch the optional guard and recovery control from the existing checkout:

```sh
./scripts/launch-control-room.sh --visibility-planning --scene-validity
```

Regression commands:

```sh
./scripts/python.sh -m pytest -q -p no:cacheprovider --strict-markers -m 'not genesis and not native_gui'
./scripts/python.sh -m ruff check src tests scripts
./scripts/python.sh scripts/check-ci-contracts.py
node --check src/bb8_rl/web/app.js
node tests/test_web_coordinates.js
node tests/test_web_localization.js
node tests/test_web_agency.js
```

Native commands and the immutable v1/v2 protocol identities are preserved with
each run under `native-recovery/`; `.venv-dreamer` supplied the existing runtime.
The initial sandbox test command passed 1,589 tests and could not bind two local
HTTP fixtures. Those exact two tests passed when rerun with localhost access.

The [compact validation record](evidence/geometry-recovery-validation.json)
contains the result counts, preservation checks and hashes for the local evidence.
Raw evidence, plans, source snapshots, protocols, tests, independent audits and failed attempts are retained under `work/continuation-20260927-geometry-recovery/`. User-facing results and the continuation checkpoint are copied to the chat's `outputs/` directory. The broader navigation and general visual-learning task remains incomplete.
