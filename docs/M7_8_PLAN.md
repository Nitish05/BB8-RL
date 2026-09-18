# M7.8 — Scan once, fix one camera, drive through bounded occlusion

The user requested execution of the pending occluded-driving step on 17 September
2026. The implementation stays in BB8-RL and uses the existing Genesis actuator
and frozen SAC policy. Hardware is outside this synthetic experiment.

## Implementation and boundaries

1. Build explicit clear-space evidence from the twenty allowed static RGB scan
   views. All pixels covering the projected body-height prism must be classified
   as floor in at least two separated views. Combine this with the M7.7
   photometric occupied-surface layer; unknown and conflicts block movement.
   Later occupied refinements remain separate, versioned estimates with source
   samples and rejected variants preserved. The palette is only a candidate
   filter because its colors overlap obstacles. The grounded-obstacle/no-overhang model and known metric
   scan poses are declared assumptions, not general real-room certificates.
2. Use the existing M7.7 RGB-estimated pose of final camera 23 for current RGB
   localization. The true pose sets the rendered fixture and is never a control
   observation. Other scan views are unavailable while driving.
3. Predict position and velocity from the previous acknowledged drive commands
   and a first-order bounded-response model. Keep predictions separate from
   visual observations. Missing RGB cannot refresh last-seen time or establish
   arrival. Reject inconsistent reacquisition and stale frames. Complete
   command-only acknowledgement timelines can establish historical transition
   times; they do not provide measured position or velocity.
4. Let frozen SAC produce waypoint commands from estimated state, then restrict
   their speed and verify the whole uncertainty/reaction/braking envelope against
   remembered clear space. Brake on timeout, uncertainty, missing route evidence,
   or map/calibration version mismatch.
5. Independently score native trajectories using renderer visibility masks and
   every 5 ms physics sample. Those truth values never enter map production,
   localization, route planning or policy action selection.

The first frozen controller has a 0.15 m/s cap, one-second visual-loss bound,
eight-centimeter position-radius limit and original visible arrival requirements:
10 cm position, 3 cm/s speed, 0.5 seconds dwell. Error and braking bounds are
engineering assumptions pending the native tests; confidence coverage is measured
descriptively and is not presented as a statistical guarantee.

## Frozen development cases

`work/m78/audit/protocol.json` freezes the six-case denominator before native
control trials. The two real box-shadow traversals were selected using static
geometry for scenario authoring, not controller outcomes. A route rejected by
the estimated map remains a result; it is not replaced by an easier endpoint.

| Case | Expected evidence |
|---|---|
| natural_0 | Nonzero control while the entire rendered head is hidden, followed by visible reacquisition and original-gate arrival |
| natural_1 | A second box-shadow traversal, with rejection/failure retained |
| visible_baseline | Visible arrival on the shared control corridor |
| brief_dropout | Bounded prediction through an injected 0.3-second omission, then reacquisition; labeled separately from geometric occlusion |
| long_dropout | Braking before the blind-motion bound expires during a three-second omission; continue logging through return |
| invalid_memory | Zero requests immediately after a map-version mismatch, with physical braking observed |

The independent scorer checks false free-space claims against all authored
obstacles, contacts, truth position and velocity errors, uncertainty coverage,
actual zero-head-pixel intervals, prediction-supported actions, reacquisition,
visible-only arrival and stopping behavior. All frames/actions and failed runs
are retained. Native simulation runs in lockstep; render/inference wall times
are recorded but do not establish real-time operation.

## Promotion criteria

Free-space false positives block promotion. Native geometric occlusion must be
observed, not inferred from a blacked-out image. Successful predicted movement
must remain collision-free and reacquire before a visible arrival. Long loss and
invalid memory must trigger braking. Any implementation correction or model
revision is written as a new artifact/run; earlier outcomes and the original
denominator remain visible. A synthetic demonstration does not establish a
general navigation success rate or physical deployment readiness.

Supplemental development fixtures may select another already-registered held-out
fixed camera and a corridor supported by the estimated map. Their selection is
declared after map inspection, their protocol is frozen before the corresponding
native test, and their results remain separate from the original six cases.
