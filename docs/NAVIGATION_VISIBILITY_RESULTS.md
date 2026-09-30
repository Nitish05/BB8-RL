# Navigation visibility and scene validity results — 26 September 2026

The opt-in visibility planner completed the original difficult one-camera route
to `(0.5, 1.5)` without contact, localization loss or intervention. Independent
scoring found a valid arrival at 12.25 simulated seconds, 2.65 cm from the exact
requested goal, at 0.0033 m/s with 0.765 seconds of stationary dwell. The baseline
still failed this route. The target, start, map, controller bounds and policy were
preserved. This is a development result in the existing synthetic room with
lockstep rendering, not held-out or physical real-time validation.

Both features remain **opt-in**. The complete frozen held-out set has **48
unexecuted requests**; its four new room/camera/appearance fixture families still
need their own RGB scans and estimated registrations. A frozen request inventory
does not itself establish generalization.

## What changed

- Bounded visibility-aware A* proposes routes from the checked scan and
  registered cameras. Every segment retains the existing clearance certificate;
  rejected candidates cannot fall back to unrestricted paths. The worker's
  10 cm consecutive predicted blind-distance budget is a planning heuristic.
- A bounded RGB guard establishes a stable same-session reference, revokes goals
  and autonomy on suspicious change, and latches confirmed invalidity. A shared
  event preserves the fault across a concurrent Reset or dropped UI update.
  Reset and mode changes cannot clear the latch; Stop continues to work.
- Runtime/app status, audit recording and exact source/config identities cover
  both optional features. Independent scorers require the actual requested goal,
  fresh RGB change evidence, recorded renderer disturbances and a rejected fresh
  command after invalidation.

See [mechanisms and limitations](NAVIGATION_VISIBILITY.md). The prior learning,
asset, logging, CI and identity repairs are retained; their distinct results are
in [the audit repair report](CONTINUATION_RESULTS.md).

## Frozen development comparison

All **24/24 requested runs** executed: four cases × three camera modes × two
variants. Each arrival request allowed 30 simulated seconds; occupied rejection
allowed five seconds and startup camera change eight. Runs terminated early only
after announced arrival, which still had to pass independent truth-only scoring.

| Request category | Clearance-only baseline | Visibility + validity |
|---|---:|---:|
| Short-goal valid arrivals | 3/3 | 3/3 |
| Original difficult-goal valid arrivals | 2/3 | 3/3 |
| Arrival localization-loss episodes | 1 across 6 requests | 0 across 6 requests |
| Occupied-goal protective rejection | 3/3 | 3/3 |
| RGB invalidation after startup camera change | Feature disabled | 3/3 |
| Contacts | 0 across 12 requests | 0 across 12 requests |
| Manual interventions | 0 across 12 requests | 0 across 12 requests |

The camera disturbance is applied only to rendering; controller calibration
remains unchanged. In all three candidate startup cases, RGB suspicion appears
at 1.0 seconds and invalidation latches at 1.5 seconds. All 130 subsequent rows
per case retain zero selected, applied and acknowledged motion and revoked
navigation/interaction authority. The fresh post-latch goal is rejected. These
cases start before the reference is ready and do **not** establish active braking.
The classifier reports scene-content change after camera movement; reliable
camera-versus-object classification is not demonstrated.

Across all 24 runs: 11 valid arrivals, seven localization-loss episodes (one
baseline arrival failure and six deliberately disturbed camera cases), nine
rejected initial requests (six occupied goals and three candidate startup goals),
three scene invalidations, zero unexpected rejections, zero worker/audit errors,
and zero interventions. Twenty-three requests meet their respective criteria.
Three additional generation-2 post-fault probes also reject: among all 27
recorded goal commands, 12 reject and 15 receive preliminary acceptance. The
summary's rejection metric counts initial requests, not the later probes.
That aggregate is not a shared safety score: camera-case criteria differ when
the guard is disabled. The meaningful arrival comparison is **6/6 versus 5/6**.

An earlier 12-second candidate pilot is retained as a failed arrival: it reached
the vicinity but completed insufficient arrival dwell. Both variants subsequently
used the same 30-second development window, fixed before the main family. The
pilot remains separate from the main denominator. Executed entries in the frozen
summary retain an obsolete inventory `reason: not_yet_run`; their `status`, saved
reports and exact run evidence establish execution. The independent review
recomputed every count and checked all 24 unique requests and provenance records.

## Additional acceptance checks

The actual app completed Demo, then learned two Station B outcomes and reached
100% resource with satisfied idle. Stop disabled learning. Reset restored 35%
resource, retained both experiences, and left motion/learning disabled. The
station effects still come from explicitly labelled simulation telemetry, not
visual interpretation. Bounded ordinary logs cannot support the full physics
audit used for native benchmark runs.

All **three supplementary active-motion camera cases** (modes 1/2/3) passed full
160-frame independent audits. Each accepted the exact difficult goal, recorded
authorized motion and scoring-only speed above 0.03 m/s before the camera moved
at 3.0 seconds, and latched RGB invalidity at 3.5 seconds. Every post-latch row
has zero requested action, no goal, disabled agency and no interaction request.
All three new generation-2 requests reject. Initial-request rejection is 0/3;
probe rejection is 3/3. There are zero contacts or manual interventions. These
manual-goal cases do not demonstrate revocation of already-active autonomy.

There were **five startup attempts for those three cases**. The first mode-1
attempt aborted in the sandbox without frames; an authorized attempt then
reported an `IndexError` before frames. Both remain failed attempts. A diagnostic
retry and modes 2/3 passed without a controller change. The underlying startup
exception has not been reproduced or explained; no repair is claimed. The
work-only diagnostic wrapper was frozen with its own identity, and its bounded
exception log filled with expected empty-queue events, so it is not exhaustive.
These cases remain separate from the frozen 24-request family.

The final display-only patch replaces the obsolete scene-warmup pause reason
once current scene/localization evidence makes Start learning available. It
does not enable learning, change remembered experiences or suppress current
warnings. Its regression failed on the old display code and all three frontend
suites passed after the patch. Native controller/guard/planner sources are
unchanged from the recorded comparison.
The running app then displayed the corrected ready message after Reset, with
two experiences retained and learning disabled. Its two worker manifests also
finalized with unchanged identities. Test ports and all known test PIDs were
closed; the app parents needed TERM after INT finalized their workers, so parent
shutdown behavior remains a follow-up rather than a claimed repair.

## Validation and reproducibility

- Complete non-native regression: **1,200 passed, 20 deselected**.
- Public-dependency isolation (native/private/training imports blocked):
  **1,158 passed, five deselected**.
- Ruff, JavaScript syntax and all three frontend contract scripts pass.
- Guard, planner and independent audit negative tests cover stale evidence,
  bounded search, clearance, duplicate/mismatched goals, generic unavailable
  camera failures, exact disturbance events and post-latch command generations.
- A separate agent verified 960 saved RGB hashes and 255 source snapshots for
  each startup camera case. One-camera RGB replay reproduced every guard decision
  through invalidation without scoring truth or the disturbance schedule.

Commands for this prepared checkout:

```sh
cd /Users/rrnitish/Documents/Codex/BB8-RL
./scripts/launch-control-room.sh --visibility-planning --scene-validity
PYTHONDONTWRITEBYTECODE=1 ./scripts/python.sh -m pytest -q -p no:cacheprovider -m 'not genesis and not native_gui'
./scripts/python.sh -m ruff check --no-cache src tests scripts
node --check src/bb8_rl/web/app.js
node tests/test_web_coordinates.js
node tests/test_web_localization.js
node tests/test_web_agency.js
```

All new evidence is under `work/continuation-20260926-navigation/`:
`start-state.json`, `comparison-design.json`, `native-pilot/`,
`native-comparison/`, `development-summary/summary.json`,
`evaluation/main-summary-independent-review.json`,
`validity/native-camera-change-independent-review.json`, `live-app/`,
`native-active-camera/`, `evaluation/active-camera-protocol.json`,
`evaluation/live-app-independent-review.json`, `preservation-review.json`,
`evaluation/active-camera-independent-review.json`, `live-app-display-fix/`,
`final-regression.log`, `public-dependency-tests-final.log` and `lint-final.log`.
The frozen held-out protocol is
`configs/interactive/navigation-heldout-v1.json`, SHA256
`82ccfbe9e65c8a65b1ef454b69317087994a4b6257a56cf4a07af9f6ddd875fe`.
Native runs keep full frames/rows, separate test memories, source snapshots,
runtime/config/asset identities and end checks. Failures remain alongside passes.

## Remaining acceptance gates

The scan has limited height coverage; predicted visibility does not guarantee
detector success or bound actual unseen time. The RGB guard cannot detect every
hidden, small or visually indistinguishable change and does not prove the saved
map valid at startup. Automatic map rebuilding, camera recalibration and global
relocalization are still absent. Moved-obstacle native coverage and fresh-room
held-out testing remain open. Defaults must not be promoted from these examples.

Next provision and evaluate the frozen held-out fixtures without tuning on their
outcomes, then advance the designed small live RGB entity/consequence experiment.
Persistent visual identities, RGB-observed outcomes, opposite-history and memory
ablation comparisons are not implemented by this navigation phase. The existing
station learner and engineered resource objective do not demonstrate personality.
Social interaction, expression, cross-room identity and hardware remain longer-term.
