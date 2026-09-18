# Clearance and speed configuration study

## Objective

Reduce avoidable clearance-related stops in the interactive camera controller
while preserving successful navigation, contact avoidance and the original
arrival checks. Compare configurations rather than choose a smaller number by
inspection. Results remain conditional on the existing synthetic room.

## Parallel work

1. Independently measure RGB localization error and physical braking from
   existing completed runs. Keep scoring truth out of online control. Report
   coverage and limitations before recommending uncertainty/model changes.
2. Implement an optional command-based lookahead speed strategy. It may slow
   SAC's requested motion early but must retain the current independent
   stopping-envelope check, command history, map and visual-loss guards.
3. Compare fixed margins and planning clearance on a frozen set of routes.
   Evaluate complete native runs and keep every attempt, including timeouts.

## Candidate comparison

Keep the current 4 cm margin / 15 cm/s speed configuration as the baseline.
Compare 3 cm and 2 cm fixed margins, a slower 10 cm/s configuration, and the
lookahead strategy with a 3 cm margin. Robot geometry, map evidence, camera
calibrations, learned checkpoints, measurement uncertainty assumptions, actuator
and physics stay fixed. An additional configuration is allowed only as a labeled
second-round experiment if the first comparison identifies a concrete problem.

Use the previously reported forward route and its reverse as tuning cases. Freeze
separate validation endpoints and camera modes before inspecting candidate
outcomes. Include the original occlusion demo, a held-out multi-camera route and
Stop, all-view loss and invalid-memory controls for the selected configuration.

## Selection rule

Exclude a candidate with a collision, boundary violation, premature/unobservable
arrival, corrupted command provenance or motion beyond the existing visual-loss
bound. Prefer more valid completed goals, then lower total elapsed simulation
time and fewer guarded-stop episodes. Report speed and stop metrics together;
reaching fewer goals cannot count as improved efficiency. Do not relax the
10 cm / 3 cm/s / 0.5 s independently scored arrival gate.

Tuning runs retain commands, complete physics ticks, renderer visibility counts,
source hashes and configuration. Full image evidence is retained for final
validation. Missing tuning images are labeled unavailable, never verified.
Parameter selection uses tuning cases only; validation failures remain failures
and any subsequent tuning requires fresh validation cases.

## Delivery

Record the comparison table, calibration limitations and exact selected settings.
Add meaningful regression tests, rerun the full non-native suite, audit the native
validation cases, verify the browser interface and restart the local app. Publish
the source change and compact evidence to the existing private repository; keep
bulk logs and checkpoints outside Git. Genesis Studio remains unchanged.

## Predeclared second round

All six first-round profiles completed the forward route but timed out in the
reported reverse direction. The 3 cm margin profile had the lowest combined
elapsed simulation time. Offline estimated-map analysis then identified a
minimum offered-command clearance deficit of 1.27 mm at the baseline's stalled
corner. A 4 cm planning reserve removes the sampled resting-envelope deficits
without changing the physical certificate. The single second-round candidate
therefore copies `margin-3cm` and adds only `planning_reserve_m=0.04`. Compare it
on the same two tuning routes before consulting the untouched validation cases.
This wider search can reject additional tight destinations; report that tradeoff.
