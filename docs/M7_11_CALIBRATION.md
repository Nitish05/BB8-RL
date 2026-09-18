# Historical control calibration and frozen validation

The historical evidence supports keeping the current measurement and dynamics
bounds. It points to earlier speed reduction as a useful optimization: lowering
a newly requested speed cannot immediately remove existing momentum or pending
commands. Fixed clearance margins remain explicit experimental settings; the
historical residuals do not establish a smaller map-error bound.

`scripts/analyze-control-calibration.py` scores ten completed M7.9/M7.10 runs:
2,364 captures with command acknowledgements and 5 ms physics traces. All source
snapshots match their recorded hashes. The script reads scoring truth offline
and does not import or modify the controller. These are correlated trajectories
in one synthetic room, including repeated demo paths, not independent trials.

## Position and velocity

Moving frames have true speed above 3 cm/s. Stationary and slow frames are
reported separately; an additional summary gives each run/20 cm position bin
equal weight to avoid letting long waits dominate the result. Per-mode results
have different route coverage and cannot be interpreted as a controlled camera
count comparison.

| Camera count | Moving captures | Visible RGB samples | RGB error p95 / max | Velocity-vector error p95 / max |
| --- | ---: | ---: | ---: | ---: |
| 1 | 802 | 770 | 9.90 / 28.99 mm | 13.12 / 18.13 mm/s |
| 2 | 228 | 195 | 6.85 / 9.14 mm | 17.84 / 18.13 mm/s |
| 3 | 685 | 685 | 2.73 / 5.98 mm | 9.90 / 18.13 mm/s |

All 2,235 timestamp-current position and velocity estimates lie within their
respective declared radii. This includes 92 predicted samples: 32 one-camera
samples with maximum position error 13.56 mm and 60 two-camera samples with
maximum error 13.13 mm. The 129 frames after Stop/invalid-memory clearing have
no active controller state and are excluded from estimate-error denominators,
while remaining in capture and measurement counts. These coverage figures are
descriptive; they do not calibrate confidence or justify shrinking uncertainty.

## Physical braking and repeated guards

Zero-target bouts are determined from command acknowledgements captured before
each physics tick. Requested actions and end-of-step commands are insufficient
because actuator latency can leave a previous target active. Initial stationary
warm-up is excluded. Every bout must follow a nonzero target and begin above
3 cm/s; interruption by another nonzero target remains right-censored.

There are 126 qualifying bouts, including 101 interrupted before reaching
3 cm/s. The two long routes account for 118 bouts. Across them, the median
per-bout empirical exponential time constant is approximately **0.3479 s**.
The current nominal model uses 0.25 s and its conservative braking tail uses
0.35 s. The latter should remain unchanged; the measured result is already
close to it. No parameter is fitted back into online control.

| Existing control | Time from zero acknowledgement to ≤3 cm/s | Travel to ≤3 cm/s |
| --- | ---: | ---: |
| Stop during motion | 0.540 s | 3.852 cm |
| All-camera loss | 0.560 s | 4.090 cm |
| Invalid memory | 0.540 s | 3.816 cm |

Threshold crossing is sampled every 5 ms. Small positive differences from the
continuous conditional braking formula are at most one sampling tick; the
recorded XY travel includes that last tick. The assumed braking tail is not a
proof against unmodeled disturbances. Preserve the additional V8 settling wait
before the unchanged 0.5 s visible arrival dwell.

The M7.10 long one-/three-camera routes contain 136/150 guarded frames across
63/70 guarded episodes. Every lowest-speed failed trial remains dominated by
the current velocity enclosure or acknowledged/unexpired target speed. Their
median total envelope radii are 16.37/14.69 cm; median stopping-distance terms
are 3.41/3.40 cm. This describes the source of guard pressure. It does not prove
any counterfactual rejected motion would have been safe.

## Predeclared separation

The ignored `work/m711/protocol.json` was frozen before candidate native
outcomes (SHA-256
`18d05f69ce4dfa1720fb24848fe1e097051f19138a14430290071dfd190c8d22`).
The reported forward route and reverse target `(-1.59, 1.41)` are tuning cases.
Three different validation routes were chosen using only the frozen estimated
map, with exact whole-capsule route checks at radii 14, 16, 18 and 20 cm:

| Validation | Start | Goal |
| --- | --- | --- |
| One camera, north | (-1.40, 1.25) | (0.20, 1.55) |
| Two cameras, center | (-0.30, -0.75) | (-0.35, 0.70) |
| Three cameras, north to center | (0.20, 1.55) | (-0.20, -0.70) |

The original one-camera demo is a regression, not a new held-out route. Stop,
all-stream dropout and invalid-memory controls use the original demo and must
show actual motion before their event. Failed routes and control failures stay
in the denominator. Parameter selection uses tuning outcomes only. Arrival
retains true distance ≤10 cm, speed ≤3 cm/s and continuous 0.5 s dwell, plus
current accepted RGB and renderer-visible head evidence. Validation does not
establish unseen-room or physical-robot reliability.

Detailed immutable input hashes, per-run/per-mode/motion summaries, each braking
bout and censoring status are in ignored `work/m711/calibration.json`. Five
focused tests cover initial-idle exclusion, interrupted bouts, path distance,
dead-zone commands, timeline gaps and the fixed conditional-tail equation.
