# M7.9 independent interactive validation

This protocol was written before the interactive native trials. Its purpose is
to verify the actual worker/API integration; the short path is inherited from
M7.8's selected development fixture. It is not an unseen-room reliability test.
The original M7.8 six-case failures remain separate.

## Fixed checks

`scripts/audit-interactive.py` reads immutable completed-run evidence and never
passes physics, labels or scoring decisions to the controller. Arrival requires
the original 10 cm position, 3 cm/s true speed and 0.5 s continuous sampled dwell,
plus a current accepted RGB observation from a camera with at least three native
head pixels. Every declared arrival is scored; a later valid arrival cannot erase
an earlier premature declaration. Dwell resets on a changed goal or generation.

The scorer verifies all 5 ms physics intervals, continuous capture timing,
normalized commands, exact causal command-acknowledgement partitions and source
IDs consistent with the selected camera count. Stop and rejected-goal events
must clear the target and issue zero until a newer accepted goal arrives. A
stationary zero command is not credited as braking from motion. Native head
invisibility, injected stream loss and detector misses are distinct conditions.
Predicted-position error and uncertainty coverage are descriptive, not confidence
calibration or a physical safety guarantee.

Every run records model/map/source hashes and source snapshots. Image/mask hashes
are verified when saved; missing mask evidence is explicitly unavailable. A
manifest's no-truth declaration is not itself proof: its saved runtime source is
reviewed for decision-before-scoring ordering. Reset and camera-mode replacement
cross worker boundaries and require separate supervisor/browser tests.

## Predeclared native cases

All cases use the portable bundle's unchanged start, goal, map and registered
camera poses. Camera A is query20, B is query23 and C is query14; all three were
excluded from the map's source views. A metric goal command is sent through the
same queue as an accepted map click. Heartbeats keep the session alive. Commands
depend only on fixed times and reported UI state, never on scoring truth.

| Case | Intent |
| --- | --- |
| goal-mode1 | A-only RGB navigation and independently visible arrival |
| goal-mode2 | A+B fusion connected to navigation and visible arrival |
| goal-mode3 | A+B+C fusion connected to navigation and visible arrival |
| stop-moving | Stop at the first reported simulation time at/after 2 s; enqueue a stale goal and verify it cannot resume |
| handover-mode2 | Omit A from observation during 2–3 s; remaining camera must support observed handover without false identity acceptance |
| all-lost-mode2 | Omit A+B during 2–5 s; verify automatic braking after the existing blind-time bound, then reacquisition |
| invalid-memory | Invalidate map version at 2 s; verify cancelled motion and physical stopping |

The native harness is retained under ignored `work/m79/run-native-validation.py`.
All attempted runs remain in separate directories. Early initialization failures,
route rejection and failed arrivals remain recorded; repeated variants are not
combined by selecting their best outcome. Interactive browser checks additionally
cover occupied/unknown goal rejection, Stop, Reset, camera-mode changes, and
heartbeat/disconnection behavior.

## Evidence status

The independent scorer currently passes 23 focused regression tests, including
false arrival, single fast physics samples, goal/generation dwell leakage, stale
poses, wrong-camera visibility, all-hidden false acceptance, missing ticks, Stop
ordering, stale goal resumption, NaN inputs and artifact path escape.

The initial mode1 launch failed before native physics because the portable map
manifest referenced an omitted `floor-masks.npz`. It is retained under
`work/m79/native-initial/goal-mode1`; no movement or navigation outcome is claimed.
The corrected bundle is independently verified through actual worker loading.
Its final seven-case worker suite passes **7/7 intended checks**, with 758 camera
captures, 2,822 RGB/head-mask PNGs and 7,580 physics/acknowledged-command intervals.
All source snapshots, bundle artifact hashes, frame hashes, label counts,
timestamps and command partitions pass. There are zero collisions, boundary
violations, false all-hidden observations or premature arrival declarations.

| Native worker case | Recorded outcome |
| --- | --- |
| One camera | Arrives at 4.95 s; 3.380 cm true distance, 0.5475 cm/s speed, 0.595 s true dwell. Ten captures have zero native head pixels; all 16 predicted samples stay within the assumed radius (maximum error 1.356 cm). |
| Two cameras | Arrives at 4.90 s; 2.992 cm, 0.6020 cm/s, 0.560 s true dwell. At least one accepted RGB source remains visible throughout. |
| Three cameras | Arrives at 4.95 s; 2.890 cm, 0.5428 cm/s, 0.595 s true dwell. At least one accepted RGB source remains visible throughout. |
| Stop while moving | Stop is consumed at 2.05 s. All 64 following frames clear the goal and request zero; the stale queued goal never resumes. Final true speed is 0.00001432 m/s. |
| A-stream handover | All 20 omitted-A captures during 2–3 s use B alone and continue nonzero commands; A is accepted again at 3.20 s. Visible arrival at 4.90 s passes with 0.560 s dwell. |
| Both streams missing | All 40 expired-blind captures request zero, reducing true speed to 0.0006287 m/s before reacquisition at 5.0 s. First arrival at 6.90 s has 3.343 cm distance, 0.5282 cm/s speed and 0.610 s true dwell. |
| Invalid memory | All 65 frames at/after 2 s cancel the goal and request zero. Motion precedes invalidation; final true speed is 0.00001419 m/s. |

Authoritative reports are under `work/m79/audit-final/`; `summary.json` references
and hashes all seven individual reports. Earlier scoring runs and the initial
incomplete-bundle failure are preserved. These are seven correlated development
cases on one short selected route, not seven independent reliability trials.
The all-stream-loss run keeps recording after arrival; its twelve valid arrival
**declarations** are one case, not twelve successes.

Worker tests issue click-equivalent metric goals through the actual command
queue. Browser clicks, Reset and camera-mode replacement are separate checks of
the supervisor/frontend; worker success does not substitute for those checks.

## Browser and source acceptance

The actual local native application was exercised in a browser on 17 September
2026. Map clicks reached the selected goal in one- and three-camera modes; the
coordinate form and Demo route reached it in two-camera mode. Mode switches
cleared the target, reset the episode and displayed the selected native RGB feeds.

Stop and Escape were each tested after motion began; both cancelled the target
and cleared the route. Reset returned to a fresh idle episode with waiting-frame
placeholders during startup. Clicking an unknown map cell was rejected without
setting a goal. Leaving the page while running expired the heartbeat and cancelled
motion; reopening the page did not resume it. Reset restored operation.

Missing assets produced explicit unavailable-state feedback. Desktop viewports
of 1280×720 and 1440×1100 and a 390×844 mobile viewport were checked; the mobile
layout had no horizontal overflow and retained its Stop control. The browser
console reported no warnings or errors during these checks. The
[control-room screenshot](media/control-room.png) is an unmodified 1440×1100
viewport capture with three native feeds and no active goal.

The full non-native test selection passed **423 tests, with 20 native tests
deselected**. Ruff and JavaScript syntax checks passed. Coordinate assertions
cover the map y flip, world/canvas round trips, half-open bounds, cell states and
letterboxing. A built wheel includes all three frontend assets. The portable
archive was installed into a fresh directory and its complete scan memory loaded
successfully. Browser checks do not establish physical stop latency or real-time
control; simulation remains lockstep.

## Reproduce the independent score

```bash
BB8_PYTHON="$PWD/.venv-dreamer/bin/python" ./scripts/python.sh \
  scripts/audit-interactive.py \
  --run work/m79/example/worker --output work/m79/example-audit
```

Output paths must be fresh. Physics runs require the prepared native Genesis
environment; the scorer itself does not initialize Genesis or issue commands.

## Publication scope review

The schematic robot is made from original native primitives; no manufacturer
mesh is bundled. Keep upstream model weights, vendor source, environments,
training datasets and generated run directories outside the source archive.
The runnable bundle uses this project's trained SAC/vision checkpoints and
derived synthetic map/calibration evidence, with their hashes and provenance.
Do not bundle SuperPoint's separately licensed research weights. Dependency
licenses remain distinct from any project license chosen by the author.

Portable metadata must not depend on a developer's absolute local paths. The
archive must include every file required by the runtime and every artifact named
by its executable checksum manifests. Installation checks alone are insufficient:
the native worker must load the installed bundle in the final test.
