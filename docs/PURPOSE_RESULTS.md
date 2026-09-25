# Learning useful interaction outcomes

24 September 2026. Synthetic experiment; no hardware or personality evaluation.

## What changed

The default autonomy panel now addresses an explicit simulated resource need.
BB-8 predicts which virtual station will help, navigates there with the existing
camera/SAC stack, requests a stationary interaction, and remembers the observed
resource change. Visiting a coordinate does not itself provide a learning reward.

The simulator owns station mechanics. The chooser receives only declared station
identities/coordinates, resource telemetry, and camera-derived navigation state.
It cannot read hidden station efficacy or substitute physics truth for its pose.
Interaction receipts include a request ID, station ID, before/after readings,
simulated timestamp and provenance. A receipt must match the current authorized
interaction and its timing; stale, interrupted or unmatched evidence cannot train
the model. Repeated requests cannot apply an effect twice.

## Decision model and designed parameters

The learned quantities are station response probability and conditional resource
gain. A bounded recent evidence window supports adaptation; an explicit change
detector can reopen alternatives after a formerly reliable station stops working.
The decision compares expected reduction in the current deficit, bounded
information value, travel/interaction cost, and idle with value zero.

| Parameter | Value | Role |
|---|---:|---|
| Resource at new episode | 35% | Explicit simulation state |
| Resource target | 80% | Designed operational objective |
| Station A / B effect | 0 / +45 percentage points | Hidden world-side mechanics |
| Shared station coordinates | A `(0, 1)` / B `(-0.5, 1)` metres | Declared virtual task fixtures in both camera modes |
| True travel consumption | 2.5 percentage points per metre | World-side path accounting |
| Interaction radius / speed | 14 cm / ≤3 cm/s | True continuous sampled dwell gate |
| Dwell | 1 second | Checked at every 5 ms physics step |
| Meaningful gain | ≥4 percentage points | Learner response threshold |
| Recent evidence window | 12 outcomes | Outcome adaptation |
| Initial ineffective probe budget | 4 per station | Prevents endless useless probing |
| Choice interaction cost | 0.005 | Designed utility cost |
| Choice distance cost | 0.015 per metre | Straight-line cost approximation |
| Maximum information value | 0.03 | Bounded decision-relevant uncertainty value |

The cost model is an approximation, not learned battery physics. Idle causes no
resource drain. Knowledge persists in independent SQLite tables scoped to the
robot, checked map bytes and station fixture identities. Resource resets with a
new simulation episode; authority always resets to paused.

## Controlled synthetic comparison

Twenty seeds, both initial station assignments, three choice rules, and 24 needs
per acquisition/reversal phase produce 120 comparison cases. Another 20 cases
exercise ineffective sensor noise. These tests have designed consequences and
do not simulate navigation, occlusion, telemetry delay or perception error.

| Choice rule | Interactions per need: acquisition | After station effects reverse | Late useful first choice: acquisition / reversal |
|---|---:|---:|---:|
| Learned outcomes | 1.021 | 1.167 | 100% / 100% |
| Random | 1.982 | 1.971 | 51.5% / 51.0% |
| Nearest without memory | 4.500 | 4.500 | 50% / 50% |

All seven benchmark checks pass: exact event deduplication, restart retention,
synthetic need restoration, reversal, fewer interactions than random, improvement
over nearest-without-memory, and finite noise probing. All rules know when a need
is satisfied; only the learned chooser uses past outcomes to select a station.
Each synthetic episode has a bounded interaction count, so the nearest baseline's
4.5 average includes capped failures and should not be read as universal efficiency.

Reproduce with:

```bash
./scripts/python.sh scripts/benchmark-purpose.py --output work/purpose/my-comparison
```

Source evidence: `work/purpose/outcomes-final-20260924/results.json` and `results.md`.
The [compact checked-in summary](evidence/purpose-outcomes.json) retains the
parameters and aggregate checks without copying the databases.

## Native camera integration

Native validation uses separate fresh memories for one and three cameras, then
restarts each with its learned memory and no autonomy command. It independently
checks resource accounting at every physics step, requested stationary effects,
effect-to-learning counts, navigation arrivals, contact flags, satisfied idle,
Stop and disabled restart. A return to a useful station is allowed when the
remaining need makes it worthwhile; repeated trips alone are not success.

```bash
BB8_PYTHON="$PWD/.venv-dreamer/bin/python" ./scripts/python.sh \
  scripts/benchmark-purpose-navigation.py \
  --output work/purpose-native/my-validation --modes 1 3
```

The original fixture used A `(0.5, 1.5)` and B `(-0.5, 1.5)`. Three-camera
navigation and learning passed there, but the one-camera approach to A lost
localization at 3.30 simulated seconds after more than one second without a
usable detection. Autonomy correctly paused with zero learned station outcomes.
The complete attempt, Stop and disabled restart remain under
`work/purpose-native/first-outcomes-20260924`.

Both modes now use the same revised fixture, A `(0, 1)` and B `(-0.5, 1)`.
Existing nearby native RGB supported this corridor, and both approach capsules
were clear in the saved scan at 13 cm and 16 cm radius before retesting. This is
an explicit task-fixture revision, not an improvement in occlusion tracking.
No camera uncertainty limits, braking rules or navigation clearances changed.

Both revised first attempts passed all six acceptance checks per mode:

| Camera mode | Captures / simulated duration including Stop | Station effects | Final resource | Invalid arrivals / contacts |
|---|---|---|---:|---:|
| One fixed camera | 321 / 16.05 s | A: 0; B: +45 pp; B: +21.25 pp | 100% | 0 / 0 |
| Three fixed cameras | 322 / 16.10 s | A: 0; B: +45 pp; B: +21.26 pp | 100% | 0 / 0 |

Each effect had its own request and 200 valid physics samples spanning the full
one-second dwell. Three recorded interaction outcomes matched exactly three
delivered receipts, including A's zero effect. Neither arrival nor duplicate
receipt delivery added an outcome. The learned expected gain for B exceeded A.

Both runs retained 1.05 seconds of satisfied idle and 1.05 seconds after Stop
with zero requested actions. Separate native restarts retained exact learned
preferences/counts and remained disabled with the new episode resource at 35%.
Source hashes stayed unchanged. Full evidence is under
`work/purpose-native/revised-fixture-20260924`; the
[compact native summary](evidence/purpose-native.json) records the gates and effects.

Final regression verification: **763 non-GUI tests pass**, with 20 opt-in native
tests deselected. Those tests are separate from the actual native runs above.
Ruff and all three frontend test scripts pass. UI tests include simulated resource
display, learned response estimates, satisfied idle and loss-safe pause.

## Limits

This is an inspectable learned consequence model with an engineered motivation,
not a recurrent high-level RL policy, self-created objective, or social personality.
The virtual station zones are task fixtures, not objects recognized from RGB.
Their effects are explicitly simulated telemetry, not camera-observed charging.
The existing Qwen semantic trial remains separate from live movement.

Negative resource changes are treated as ineffective; the model does not estimate
harm magnitude. Its current domain is zero-or-positive station effects. The
synthetic reversal/noise benchmarks do not establish broad robustness in new
rooms or with unmodeled interaction costs. The next extension requires richer
interactive objects, observed consequences and held-out contexts before any
claim about persistent personality.
