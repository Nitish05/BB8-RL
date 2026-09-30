# Capability continuation plan and completion ledger

## Git checkpoint — 2026-09-29

This checkpoint collects the verified September continuation implementation,
tests and selected reports for the existing private repository. The current
source matches all 237 files in the last native evaluation freeze. The latest
[compact validation record](evidence/geometry-recovery-validation.json) retains
the 1,591 passing non-native tests, all five recovery attempts and the six
unattempted new-room requests. Raw evidence, installed assets, environments and
production memory remain local and unchanged. No release or replacement asset
bundle accompanies this source checkpoint.

The next implementation gate is stationary RGB acquisition at the predeclared
eight-light arrangement, followed by native response and geometric uncertainty
checks. Useful trusted new-room mapping and the six original driving requests
remain unfinished. The live marker/gauge learning slice and original-reference
recovery are verified; general object understanding, arbitrary moved-camera
calibration, map repair, global relocalization and the broader personality and
hardware work remain open. Dated sections below preserve their historical
statuses; they are not the current task completion summary.

## Latest verified continuation — 2026-09-27

Explicit original-reference recovery is implemented and validated in the native
control room. Three requested camera modes completed across five preserved
attempts; Stop and heartbeat cancellation require explicit retry and a fresh
goal. All 1,591 non-native tests pass. A separate 28-frame point-light RGB
preflight establishes acquisition repeatability, but does not resolve metric
height or admit a new map. The original six new-room driving requests remain
unexecuted. Arbitrary camera recalibration, map repair/global relocalization,
general visual entities/cross-room identity, richer personality and hardware
remain open. See [the current results](SCENE_RECOVERY_RESULTS.md) and the
checkpoint in the current chat's `outputs/` directory. Earlier entries below
are retained as historical plans and evidence.

## Navigation continuation — 2026-09-26

The next run starts with all 37 prior changed/new files present, Studio clean,
and no listener on port 8765. Prior repairs remain local and must be preserved.
New evidence: `work/continuation-20260926-navigation/`; `start-state.json` records
the incoming source hashes and previous preservation inventory.

Execute in this order:

1. Freeze a held-out protocol before selecting/tuning planner or validity
   parameters. Separate the known difficult development route from held-out
   layout, camera, appearance, and rejected-request cases. Unbuilt or unexecuted
   cases must remain explicit in the denominator rather than disappear.
2. Implement a bounded visibility-aware route candidate from frozen scan geometry
   and registered cameras, preserving the existing full clearance certification.
   Predict blind exposure as a planning heuristic, never as permission to exceed
   the live uncertainty, stopping, or arrival gates.
3. Implement a latched RGB change guard for bounded scene/camera validity. Mask
   observed robot motion, verify freshness, and require explicit reconstruction
   or re-registration after invalidation. A startup image reference establishes
   same-session change evidence, not proof that an old map is correct.
4. Integrate under the existing command/Stop/generation lifecycle and expose
   honest status/evidence in the app. Unit gates cover rejected routes, failed
   evidence, changes, Stop, stale results, and no automatic authority restoration.
5. Run complete regression and independent audit checks. Freeze source before
   serialized native validation; retain every attempt and exercise the actual
   app. Keep defaults opt-in until broader validation passes. Report
   arrival failures independently from successful protective rejection.
6. Record measured capability, limits and next gate. The live visual interaction
   experiment remains a following stage; navigation safeguards alone do not
   complete visually grounded learning or personality.

Parallel ownership: evaluation agent owns the new frozen protocol and evaluation
script/tests; visibility agent owns the new pure planner and its tests; validity
agent owns the new RGB guard and its tests. Integration owner owns runtime/app
integration, shared documentation, final review and every native simulation.
No agent changes Studio, existing bundles, environments or production memory.

Started 2026-09-25 from clean BB8-RL `ef39d3dfcd67eeef4e2d7be4a14fdba4ddfa59e9`
on `main` (tracking `origin/main`). Genesis Studio is an unchanged external
dependency at `cd20c8c685f2b51263814fd0a6296bfd849bdce6`.

## Preservation and integration rules

Reuse the existing runtimes, assets, model weights and evidence. No existing
SQLite memory is used for tests; test memories live in fresh evidence directories
or temporary directories. Do not reset the checkout, modify Studio, replace
environments, delete experiments or download replacement assets. Only one native
simulation may run at a time; the integration owner coordinates native runs.
At initial inspection port 8765 had no listener. Keep actuator authority, Stop,
heartbeat cancellation, command generations, fresh telemetry, localization loss,
uncertainty/clearance limits and disabled restart behavior intact.

Evidence root for this continuation:
`work/continuation-20260925-audit-repair/` (ignored, created fresh).
The original purpose and recursive-asset repros both ran successfully against
unchanged HEAD; their JSON outputs are `purpose-before.json` and
`assets-before.json`. All three defects are still present.

## Stages and acceptance gates

| Stage | Implementation | Acceptance gate | Status |
|---|---|---|---|
| Learning and validation repair | Bounded idle recovery (F1), evidence-triggered early/late adaptation (F2), complete checked configuration graph (F3) | Original repros close; regression, noise/restart/dedup and bundle tests pass; native 1/2/3 modes pass | Complete for tested contracts |
| Dependable operation and verification | Bounded default logs and full audit mode (F4); discovered CI contracts (F5); complete run identity (F6); portable asset input manifests and wheel smoke | Bounded logs, full-audit counts, rejected mutants, source/config identity, native/app checks and packaging review | Locally implemented/verified; hosted private CI access, clean-machine native install and release/archive publication remain open |
| Navigation reliability | Frozen held-out protocol, bounded visibility-aware route candidates and same-session RGB validity latch implemented as opt-in prototypes | Original difficult one-camera regression passes; 24/24 development requests executed; native/app checks below | Development prototype verified; all 48 held-out requests, native moved-obstacle coverage and broad changing-room reliability remain pending |
| Visual interaction and persistent experience | One small end-to-end RGB-grounded entity experiment with persistent IDs, action-dependent observed outcomes, context and uncertainty; asynchronous semantics only if justified by existing model evidence | Predetermine entity/history/reset rules; compare opposite histories, memory ablation, random/nearest/fixed choices, reversed consequences and useless objects; authorized interactions train once; Stop and stale-result rejection under load; restart retains knowledge without motion | Pending |

For every completed stage, run targeted tests plus the complete applicable
non-native suite, Ruff and all three frontend scripts. For runtime integration,
run serialized native Genesis tests with separate memories and exercise the
actual control room. Preserve failures. Record implementation, evidence,
limitations and the next gate below. Do not treat passing unit tests as native,
held-out or physical evidence.

## Parallel ownership

The integration owner owns this plan, shared documentation, integration, CI,
operational work and native scheduling. Independent repair agents own only:

- F1: `src/bb8_rl/purpose_runtime.py`, `tests/test_purpose_runtime.py`.
- F2: `src/bb8_rl/purpose.py`, `tests/test_purpose.py`, a narrowly scoped new
  purpose-adaptation regression/experiment file if necessary.
- F3: `src/bb8_rl/demo_assets.py`, `tests/test_demo_assets.py` and a narrowly
  scoped dependency-validation helper if necessary.

Agents must not edit shared files, run native simulations, commit, publish or
modify the installed asset bundle. Integration reviews their changes and records
commands/results before advancing a stage.

## Completion ledger

| Capability/finding | Evidence/status | Remaining limitation / next gate |
|---|---|---|
| Current truth and preservation | Clean audited HEAD; Studio clean at recorded revision; installed bundle and `.venv-dreamer` retained; launcher and fallback runtime inspected; original repro JSON saved | Record final source/runtime identities and check no concurrent changes before delivery |
| F1 idle route recovery | Closed: original repro fixed; bounded retries, quiet idle and revocations pass | Failed dispatched goals still require changed evidence |
| F2 early effect reversal | Closed: 120 reversed histories, restart/dedup/noise and original benchmark pass | Ambiguous nonzero-epoch legacy histories left unchanged; above-threshold false resource readings remain indistinguishable from gain |
| F3 recursive bundle validation | Closed: graph tests and original escape rejected; installed bundle accepted | Integrity gate is separate from full runtime semantic validation |
| F4 default log growth | Closed: bounded telemetry; historical replay 151,943,905 → 361,792 bytes; actual app verified | Quota is per worker session; explicit audit logs intentionally unbounded |
| F5 CI coverage | Independent validators mandatory; three mutants rejected; 1,041 tests pass with private/native imports blocked | Hosted complete job needs GENESIS_STUDIO_READ_TOKEN; all 1,083 non-native tests pass locally |
| F6 run identity | Closed: 251 source snapshots, actual config/assets/runtime and end checks in native runs | Not an atomic freeze or portable full environment image |
| Reproducible packaging | Portable 14-input manifest, deterministic fixture ZIPs, real preflight, final 0.1.2.dev0 wheel smoke pass | Clean-machine native install, production archive rebuild and recent evidence release remain unverified/unpublished |
| Difficult one-camera route | Prior failure retained; new visibility route arrives at exact `(0.5, 1.5)` target at 12.25s with zero contact/loss/intervention. Arrival comparison 6/6 candidate versus 5/6 baseline across modes 1/2/3 | Planning improvement in existing synthetic development room; 48 held-out requests remain unexecuted |
| Changing rooms/cameras | Opt-in same-session RGB guard revokes and latches; 3/3 startup camera disturbances detected; concurrent Reset cannot discard worker fault | Startup map correctness, hidden/small changes, native moved-obstacle cases, automatic rebuilding/recalibration and global relocalization remain open |
| Visually grounded interaction | Current purpose inputs are declared virtual stations and explicit simulated resource receipts | Implement and evaluate one live visual slice |
| Persistent preferences/personality | Existing learner estimates station response/gain under engineered resource motivation | Controlled history/memory/context comparisons; no personality claim |
| Social interaction/expression | Not implemented | Longer-term work |
| Rich recurrent decisions/cross-room identity | Not implemented | Longer-term work |
| Physical hardware and non-lockstep timing | Not validated | Calibration and supervised physical transfer after simulation gates |

Synthetic room measurements and SAC/Dreamer are accepted project choices.
Do not reintroduce measured-room acquisition or TD-MPC2 as immediate obligations.
The 200/200 M6 results use oracle state/maps and one training seed, with a
100/100 behavior-cloning baseline; they do not prove camera or isolated RL gains.

If the entire scope cannot be finished in one run, finish a coherent verified
stage and write the exact continuation checkpoint here. This ledger must retain
unfinished navigation, visual grounding and personality work explicitly.

## Validation journal

- Baseline: 763 non-native tests passed, 20 native/GUI tests deselected. The
  sandboxed first attempt passed 761 and failed two HTTP fixture setups because
  binding loopback sockets was denied; the authorized rerun passed all 763.
  Both logs are retained (`baseline-tests.log`, `baseline-unsandboxed-tests.log`).
- Learning/validation integration: 1,038 non-native tests passed, 20 deselected
  (`phase1-regression.log`). All three frontend scripts and JavaScript syntax
  passed. The unchanged original purpose repro functions now report both defects
  absent (`purpose-after.json`); the original asset escape is rejected while the
  installed bundle remains accepted (`assets-after.json`). Native acceptance is
  still pending at this journal entry.
- Logging: 20,000 synthetic stopped frames over 1,000 simulated seconds retain
  bounded logs, 1,000 samples and two event records under an 8 KiB test quota.
  Separate tests preserve unsampled outcomes/Stop/faults, account for evictions,
  and retain full exact unrotated rows in explicit audit mode. Re-encoding all
  16,212 historical native rows (810.6 simulated seconds) reduces retained JSONL
  from 151,943,905 to 361,792 bytes. This is **offline logging replay**, not a new
  native run; source hash and counts are in `log-replay/comparison.json`.
- Final: **1,083 passed / 20 deselected** (`final-tests.log`), **1,041 passed /
  5 deselected** with Studio/native/training imports blocked
  (`f5/public-dependency-final-review-after.txt`). Three deliberate CI mutants
  are rejected (`ci-post-review.json`). Ruff, frontend scripts and diff checks
  pass. The initially failed isolated builder tests are retained; a shared
  data-only dependency gate removed the unnecessary private import.
- Native: all six checks pass in each of modes 1/2/3, with three learned receipts
  per mode, satisfied idle and disabled durable restart. No contacts or invalid
  arrivals. An additional final two-camera learning/restart smoke passes after
  the builder/validator factoring. All completed workers record unchanged run
  identity. The difficult target still fails arrival and is retained.
- Browser: actual one-camera Start learning → 100% / three experiences → Stop
  → Reset → 35% / three experiences / disabled. Both ordinary telemetry sessions
  finalized with unchanged identity; the owned test server was closed gracefully.
- Packaging: final wheel installed outside source import paths and passed module,
  web asset, SQLite and installed-bundle checks. Exact SHA and scope are in
  `wheel-final-smoke.json`. No existing runtime or production memory changed.
- Navigation continuation: 1,200 non-native tests pass / 20 deselected; public
  dependency isolation passes 1,158 / five deselected. CI discovers all ten
  required validator modules and rejects all three deliberate mutants. Planner,
  guard, goal/provenance checks, Reset-race protection and frontend gates pass.
- Frozen development comparison: 24/24 requests executed; candidate arrives on
  all six arrival requests, baseline on five. Original difficult one-camera
  candidate arrives at 12.25s without contact/loss/intervention. All six occupied
  requests reject; candidate startup camera disturbance latches in all three
  modes. Across all 24: zero contacts, unexpected rejections, worker/audit errors
  or manual interventions. Failures and the short no-arrival pilot are retained.
- Held-out protocol: fixed before tuning; all 48 requests remain explicitly
  unexecuted because new scan/registration fixture families are not provisioned.
- Navigation app: Demo arrives, learning records two useful station interactions
  and reaches satisfied idle, Stop disables it, Reset retains both experiences
  at 35% resource with no motion authority. All three completed worker identity
  checks pass. Actual station effects remain direct simulation telemetry.
- Preservation review: baseline memory, bundle, policy and vision hashes,
  all 13 installed bundle members and all 4,254 prior indexed evidence files
  unchanged; Studio remains clean and all existing runtimes are retained.
- Active-motion supplement: all three modes pass 160-frame independent audits,
  moving before the 3.0s disturbance, latching at 3.5s, zero commands thereafter,
  and rejecting all three new goals. Five attempts include two retained startup
  failures (sandbox abort and unexplained pre-frame IndexError). Initial-request
  rejection is 0/3; post-latch probe rejection is 3/3. No active-autonomy claim.
- Final display fix: stale warmup reason replaced only after current readiness;
  all frontend contracts pass and the actual app shows the corrected message
  after Reset, retaining two experiences with autonomy disabled. Five app worker
  manifests across both browser runs finalize with unchanged identities. INT
  finalized workers but app parents needed TERM; all known PIDs/ports are closed.

## Precise continuation checkpoint

Read `NAVIGATION_VISIBILITY_RESULTS.md`, `CONTINUATION_RESULTS.md` and this ledger,
then recheck Git status and the evidence indexes. Current changes are local on
`main`, based on `ef39d3d`; do not
reset them. No PR/release was created. Reuse installed v2 assets and
`.venv-dreamer`; the semantic model remains under `work/semantic-trial`.

The **navigation visibility/validity prototype** is implemented and verified on
the existing development room; it remains opt-in. Provision the four new fixture
families in frozen `configs/interactive/navigation-heldout-v1.json` using their
own RGB scans and estimated registrations, then execute all 48 requests without
tuning on the held-out outcomes. Do not substitute the bundled development map.
Retain the original failed baseline and short pilot. Add native moved-obstacle
and appearance nuisance cases before broader changing-room claims.

Next implement the **live visual interaction slice** designed under
`work/continuation-20260926-navigation/next-slice/design.md`. First verify actual
native RGB-rendered entity response states and occlusion; a HUD label or telemetry
receipt is not a visually observed consequence. Then connect persistent entity
IDs, authorized interactions, RGB outcomes and the existing learner with
opposite-history, ablation, fixed/random/nearest and reversal/idle comparisons.
This vertical slice remains unimplemented. Do not add a disconnected framework,
label direct simulated telemetry as vision or claim personality.

Separate operational follow-ups are private Studio CI access, clean-machine
native installation, intermittent pre-frame native startup IndexError, app parent
shutdown behavior and portable publication of recent full evidence. Current
native evidence uses this prepared Mac and a development room, not held-out
camera generalization or physical real-time operation.
