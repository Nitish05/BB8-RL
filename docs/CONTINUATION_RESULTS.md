# Audit repair and dependable operation

25 September 2026. Local implementation based on clean
`ef39d3dfcd67eeef4e2d7be4a14fdba4ddfa59e9`; Studio remains unchanged at
`cd20c8c685f2b51263814fd0a6296bfd849bdce6`. Changes remain local and unreleased.
This completes the reproduced-defect repair and adds operational foundations.
It does **not** complete navigation generalization or visually grounded personality.

## Findings and implementation

| Finding | Result | Evidence and limits |
|---|---|---|
| F1 resource-only idle cache | Closed | Failed dry routes retry with 2/4/8/16/30-second backoff. Significant measured pose/clearance evidence can reopen a blocked goal; unchanged successful dry plans cannot cause endless re-dispatch. Satisfied/learned-useless idle stays quiet; authority loss still pauses. |
| F2 early effect reversal | Closed for tested histories | Two corroborated responses then three failures trigger one alternative probe. Initial four-probe budget, exact receipt deduplication and disabled restart remain. Epoch-zero legacy stuck histories recover on explicit selection; ambiguous nonzero legacy epochs do not receive speculative extra probes. |
| F3 unchecked configuration dependencies | Closed | Containment/checksums cover task → world → primitive project and implicit/declared map inputs. Missing, escaped, symlinked and unlisted dependencies fail before model loads. Runtime task override cannot bypass the bundle. Full world/map semantic validation remains separate. |
| F4 excessive ordinary logs | Closed for bounded worker sessions | Default compact samples plus separate events rotate at 8 MiB total JSONL per worker session, with explicit eviction/truncation metadata. Explicit audit mode records every row without rotation. Separate resets create separate sessions; there is no destructive global cleanup. |
| F5 omitted independent validators in CI | Audit-suite omission repaired; hosted full gate pending access | Automatic discovery plus mandatory auditor checks and three rejected mutants. Full Studio contracts run locally; hosted full job needs `GENESIS_STUDIO_READ_TOKEN`, otherwise reports NOT RUN. |
| F6 partial run identity | Closed within stated provenance scope | 251 BB8/Studio source files in native runs, relevant revisions/dirty patches, complete config graph, 14 asset entries, package and loaded-module origins/versions, effective reset setup and end verification. Snapshots are not an atomic filesystem freeze or cryptographic publisher authentication. |

No actuator, dynamics, camera uncertainty, arrival dwell, clearance margin,
map free-space classification or SAC weights changed. No hardware connection is
created by imports or simulation. Existing assets, prepared environments, memories
and previous experiment evidence were retained.

## Regression and controlled learning checks

The unmodified original audit functions reproduce both learning defects before
the change and neither afterwards. The original escaped-world asset repro is
rejected; the installed published bundle still validates without deserialization.

The final integrated non-native suite passed **1,083 tests**, with
20 opt-in native/GUI tests deselected. Ruff, JavaScript syntax and coordinate,
localization and agency frontend scripts passed. The isolated public-dependency
suite passed **1,041 tests**, with five explicitly Studio-marked cases deselected
and six Studio-dependent modules omitted before import. Full local testing
includes those contracts. Post-review results are in the ledger/evidence index.

Early/late adaptation tests cover 120 reversed histories: ten seeds, both station
assignments and 2/3/5/6/8/24 prior useful outcomes. All recover within five trials.
Static sub-threshold noise settles after eight total trials. Repeated failures,
restarts and duplicate receipts do not replenish the consumed change hypothesis.
The existing 20-seed comparison retains all seven passing checks, including
1.021 acquisition and 1.167 reversal interactions per need for the learner.
These are designed synthetic consequence histories, not perception experiments.

The mandatory CI validator gate rejects deliberate unauthorized-motion,
missed-solid-overlap and unchecked-config-reference mutants in separate test
processes without modifying the checkout. A final dependency-isolation review
exposed a private Studio import in the new asset builder; that failed run is
retained and the builder now shares the data-only project dependency gate.

## Native purpose acceptance

Fresh memories were used for each camera mode, followed by a separate native
restart using only that mode's new test memory. All six acceptance checks pass
for each mode: learning physics/Stop, valid stationary effects, learning from
receipts rather than arrival, satisfied idle, restart physics/Stop, and durable
knowledge with autonomy disabled. Full record counts and provenance also pass.

| Cameras | Learning captures / simulated seconds | Delivered / learned outcomes | Restart captures / seconds | Contacts / invalid arrivals |
|---|---|---|---|---|
| 1 | 321 / 16.05 | 3 / 3 | 43 / 2.15 | 0 / 0 |
| 2 | 322 / 16.10 | 3 / 3 | 43 / 2.15 | 0 / 0 |
| 3 | 322 / 16.10 | 3 / 3 | 43 / 2.15 | 0 / 0 |

All three learning runs recorded A's zero effect, B's +45 percentage points,
then B's remaining approximately +21.25 percentage points. Resource reached
100%; the independent auditors verified each requested one-second stationary
effect from the full 5 ms physics trace. Learning and restart runs included Stop
observation windows. All six run identities remained unchanged through completion.

After extracting the unchanged project dependency rule into a shared helper for
the builder's pure CI path, an additional final **two-camera learning/restart
smoke** passed all six gates with unchanged identity. It is retained separately
under `native-final-tooling-smoke/`; it is not counted as a new room or added to
the three-mode table's denominator.

This is **3/3 purpose scenarios plus 3/3 disabled restart scenarios in the same
development room**, not a 100% general navigation rate. These are lockstep native
Genesis captures: physics pauses during rendering/inference. Station positions
and resource receipts remain explicitly simulated task inputs, not RGB object
recognition or observed physical charging.

## Difficult route and actual application

The original one-camera target `(0.5, 1.5)` was retained and requested directly
with unchanged `reserve-3cm` limits. Across that **one request**, the six-second
run records **zero arrivals, one localization loss, zero contacts, zero boundary
violations and zero false arrivals**. Loss occurs at 3.20 simulated seconds;
reacquisition starts at 3.55 seconds, with the old goal cancelled and no automatic
resumption. There was no manual recovery intervention. This remains a failed
arrival regression, even though its stopping/provenance audit passes. The earlier
purpose-harness failure at 3.30 seconds is also retained. Targets were not moved.

The actual loopback control room was opened and driven through its UI using a
separate test memory. One-camera RGB and measured pose displayed correctly;
Start learning produced the station intention, three remembered outcomes and
100% resource with "Need satisfied · waiting". Stop disabled learning. Reset
restored 35% resource while keeping three experiences and leaving learning paused.
The owned test server was then shut down gracefully to finalize both manifests.
Both live sessions used bounded telemetry and passed end identity checks.

## Logging and packaging evidence

Offline re-encoding of the audit's historical 16,212 rows / 810.6 simulated
seconds retained **361,792 bytes**, compared with **151,943,905 bytes** in the
original full physics log. This is an offline logging comparison, not a fresh
810-second simulation. A separate 20,000-frame quota test checks long idle,
rotation accounting, unsampled outcomes/Stop/faults and explicit truncation.
Independent review caught and fixed oversized-marker overflow, final-log failure
masking the original fault, and missing-tail acceptance in full-audit validation.

The builder accepts a versioned checksummed manifest covering all 14 actual
source inputs. Relocation and changed filesystem mtimes produce identical
archives in opaque-weight fixture tests on this runtime. Real installed input
preflight passes. A production archive rebuild was not needed; the existing
published bundle remains installed. Third-party weights were not loaded or copied.

An unreleased **0.1.2.dev0** wheel was built and installed in a fresh ignored
target directory, then imported from outside the checkout. New modules, bundled
web assets, SQLite choice and installed-bundle validation pass using existing
prepared dependencies. This is not a clean-machine native installation. Build
tools were installed only in a fresh ignored test directory; neither prepared
runtime was modified. No commit, PR, release or visibility change was published.

## Evidence and reproduction

All new raw evidence is under
`work/continuation-20260925-audit-repair/` in the BB8-RL checkout. Key artifacts:

- `purpose-before.json`, `purpose-after.json`, `assets-before.json`, `assets-after.json`.
- `integrated-tests.log`, final test/CI records, and `f5/` retained failure/review checks.
- `f2-purpose-benchmark-legacy-recovery/`, `log-replay/comparison.json`.
- `native-purpose/summary.json` and each mode's complete learning/restart records.
- `difficult-route-protocol.json`, `difficult-route-native/summary.json` and raw failed route.
- `browser-acceptance.json`, `live-app/`, and its separate `live-app-memory.sqlite3`.
- `asset-inputs.json`, `identity-after.json`, `identity-validation/`, and `wheel-smoke.json`.

Each native worker stores full source/config snapshots, runtime/asset identity,
recording metadata, effective setup and end-of-run verification. The evidence
index records hashes without deleting or replacing old evidence.

```bash
./scripts/launch-control-room.sh
# Full per-step audit evidence when explicitly needed:
./scripts/launch-control-room.sh --recording-mode audit --output work/fresh-audit

PYTHONDONTWRITEBYTECODE=1 ./scripts/python.sh -m pytest -q -p no:cacheprovider -m 'not genesis and not native_gui'
./scripts/python.sh -m ruff check --no-cache src tests scripts
node --check src/bb8_rl/web/app.js
node tests/test_web_coordinates.js
node tests/test_web_localization.js
node tests/test_web_agency.js

BB8_PYTHON="$PWD/.venv-dreamer/bin/python" ./scripts/python.sh \
  scripts/benchmark-purpose-navigation.py --modes 1 2 3 --output work/fresh-purpose-validation
```

## Unfinished work and next gate

1. Freeze a new held-out camera evaluation definition **before tuning**: layouts,
   camera poses, appearance/lighting variants, modes 1/2/3, reachable and rejected
   requests. Report every request and intervention. The old oracle 200/200 M6
   result and its 100/100 behavior-cloning baseline are separate evidence.
2. Investigate camera visibility and predicted blind duration for the retained
   failed route. Keep existing uncertainty/braking/clearance bounds. Implement
   bounded image-evidence map/camera invalidation before claiming changed-room
   operation. Global relocalization remains deferred beyond the local 12 cm gate.
3. Implement one live RGB-grounded interaction slice with distinguishable
   persistent entities, authorized action-dependent visible consequences and
   explicit identity/reset/map-change rules. No such slice was added in this run.
   Existing Qwen/MLX weights and offline trial remain available and disconnected.
4. Evaluate opposite histories, memory ablation, random/nearest/fixed choices,
   changed consequences and useless objects. Keep learned predictions distinct
   from engineered motivation. Any semantic integration must be asynchronous,
   use offered evidence-backed IDs and survive Stop/stale-result tests.
5. Configure private Studio CI access; validate a clean-machine native install
   and portable recent-evidence replay before publishing a new release.

Social interaction, expression, richer recurrent decisions, cross-room identity,
non-lockstep timing and physical hardware are still open. Synthetic room
measurements and SAC/Dreamer remain the accepted choices; no TD-MPC2 or physical
room-acquisition obligation has been reintroduced. The capability ledger in
`CONTINUATION_PLAN.md` is the next run's entry point.
