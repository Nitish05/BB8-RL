# Native autonomous navigation: first integration results

**Historical result, corrected interpretation:** the original three-arrival
criterion accepted a two-point shuttle. These logs pass movement checks but
**fail the revised exploration criterion** of three distinct verified targets
with no successful revisit. See the [coverage correction](AGENCY_COVERAGE_FIX.md).

Recorded 24 September 2026 on Apple M5 Pro, 48 GB unified memory. Three-camera
exploration completed **three independently verified arrivals**. The single-camera
case lost localization during its first self-selected trip and remained paused;
it **failed the three-arrival objective**. Both cases preserved the motion guards.

These are two bounded development runs in the existing synthetic room, not a
success-rate estimate across rooms or evidence of mature personality.

## Frozen protocol and independent scoring

`scripts/benchmark-agency-navigation.py` launches the native `worker_main` using
`work/interactive-assets`. Camera modes 1 and 3 run serially, each with a fresh
run directory and separate empty SQLite memory. The harness sends heartbeats
every 0.2 wall seconds, enables exploration once after idle valid localization,
then lets the agent choose its own destinations. It supplies no target coordinates.

The stopping limit is three reported arrival episodes, 120 active simulation
seconds, or 600 wall seconds, whichever occurs first. It then sends Stop at
generation 2 and records about one further simulation second before graceful
worker shutdown. No implicit restart follows localization loss. Native source
hashes were frozen and remained unchanged through both runs.

Every control step and physics tick is logged. Native RGB and scoring-only head
masks are saved every twentieth frame. Truth and segmentation are excluded from
goal selection; the completed logs are scored afterward by
`scripts/audit-interactive.py::score_rows`.

An arrival requires fresh visible RGB, true distance at most 10 cm, speed at most
3 cm/s, and a continuous 0.5-second physics dwell. Repeated arrived frames count
as one arrival episode. The Stop audit requires zero requested actions and
disabled exploration throughout the observation window, followed by speed at
most 1 mm/s in its final half second. This permits the modeled braking transient.

## Results, including the failed case

| Measure | One camera | Three cameras |
|---|---:|---:|
| Independently valid arrival episodes | **0 — objective failed** | **3 — objective passed** |
| Unique intentions | 1 | 3 |
| Distinct places selected | 1 | 2 |
| Recorded simulation duration | 121.10 s | 24.40 s |
| Worker wall duration, including startup | 75.18 s | 60.17 s |
| Control frames / physics ticks | 2,422 / 24,220 | 488 / 4,880 |
| Saved RGB images | 122 | 75 |
| Recorded collisions / boundary failures | 0 / 0 | 0 / 0 |
| Audit integrity errors / invalid arrived frames | 0 / 0 | 0 / 0 |
| Stop observation duration | 1.05 s | 1.05 s |
| Final half-second maximum physical speed | 0.00000000343 m/s | 0.000623 m/s |
| Zero requested actions and disabled agency after Stop | Passed | Passed |
| Native worker / cleanup errors | 0 | 0 |

The three-camera agent selected these destinations without a user-supplied goal:

| Arrival time, simulation | Destination, metres | Independent arrival gate |
|---|---|---|
| 7.25 s | (0.5, 1.5) | Passed |
| 15.25 s | (0, 1) | Passed |
| 23.30 s | (0.5, 1.5) | Passed |

The single-camera agent selected (0.5, 1.5), lost localization, stopped, then
reacquired a camera position while remaining paused. Its recorded episode is
`localization_lost`; it received no successful-visit utility update. The harness
retained the entire fixed-duration failed run and did not retry with a favorable
start or weaken the loss guard. Its explicit final Stop occurred after loss had
already paused motion; it is not a mid-motion interruption test.

Geometry clearance alone does not establish that an entire trip will remain
observable to one fixed camera. This first result identifies camera visibility
and goal feasibility as remaining limits on autonomous trip selection. Three
short arrivals between two places do not establish general exploration coverage.

## Actual RGB replay

![Three-camera native autonomous exploration](media/agency-exploration.gif)

Left to right: **Camera A, B, C**. These are synchronized, fixed crops of the
actual recorded RGB frames, resized for visibility. The replay samples every
1 simulation second and displays each frame for 250 ms—approximately **4×
simulation speed**, not real-time camera performance. It covers simulation
times 0–24 s, including the return visit and final stop. There is no generated
scene imagery or synthetic robot overlay.

The 25-frame GIF is 540,546 bytes. Its SHA-256 is
`070305c046afe3a091789e4e12d4105f1742afad2823528eedfa5db669f34160`.
The render script, fixed crop rectangles, source image hashes and source log hash
are retained with the audit evidence in `render-gif.py` and `gif-metadata.json`.

## Additional browser verification

A separate live browser sequence exercised Start, Stop, camera switching and
Reset. Its completed one-camera worker recorded no arrival and no collision or
audit integrity error. Its three-camera worker recorded two distinct arrival
episodes at simulation times 26.35 s and 34.35 s; all reported arrival frames
passed the independent dwell gate, with no recorded contacts or audit errors.

The browser Stop was accepted at 38.7 s while the robot had been moving at up to
0.149729 m/s during the preceding 0.3 s. All subsequent requested actions were
zero and exploration remained disabled. Physical speed was 0.008452 m/s one
second later: this verifies cancellation and braking, **not instantaneous zero
physical velocity**. It complements the bounded harness's Stop after its arrival
limit and already-paused single-camera case.

Browser evidence is under `work/agency-live-20260924/session-0000/` and
`session-0003/`, including `browser-independent-audit.json` and
`browser-stop-audit.json`. These are separate sessions and are not pooled with
the two fresh-memory protocol runs to manufacture a success percentage.

Browser Reset started a new three-camera worker with valid measured localization,
exploration unchecked and no automatic motion. It retained five recorded
experiences: two localization losses, two arrivals and one cancellation. This
checks storage across a worker restart, not long-term behavioral individuality.

## Reproduction and evidence

Run with native Metal/graphics access from the BB8-RL repository:

```bash
scripts/python.sh scripts/benchmark-agency-navigation.py \
  --output work/agency-native/new-independent-run \
  --modes 1 3 --arrivals 3 --sim-seconds 120 --wall-seconds 600
```

The output directory must be new. Each mode gets its own memory database,
configuration, worker log, full command/control log, source hashes, independent
audit and summary. A failed mode remains in the combined result; no evidence is
deleted to rerun a favorable variant. The first recorded batch is
`work/agency-native/first-integration-20260924/` and its combined result is
`all_passed: false`, because the single-camera case did not complete its trips.

The high-level learner in these native runs is the new activity-value model;
SAC still follows admitted navigation goals. The optional local Qwen trial is
separate and was not involved in these decisions. Learning useful visit values,
semantic grounding and persistent character require separate evaluations; this
audit establishes their first guarded navigation integration only.
