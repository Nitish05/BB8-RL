# Occlusion uncertainty and lost-localization recovery

Prepared 24 September 2026 before implementation and new native experiments.
The user called this next step M4, then clarified that it concerns occlusion
uncertainty and lost localization. This camera-control follow-up is recorded as
M7.13 to preserve the historical M4 SAC/HER training reports.

## Problem and research decision

The reported UI showed a 719.2 cm position bound while the robot was stopped and
RGB localization was missing. The current command-conditioned predictor permits
persistent acceleration mismatch of 0.12 m/s². At 5 ms physics intervals its
velocity-error radius converges to about 0.030301 m/s; integrating that envelope
adds about 1.836 m per simulated minute. This is an engineering error enclosure,
not measured robot motion or a calibrated confidence interval. It continues after
the one-second blind-motion or eight-centimetre position-error limit has expired.

The [research review](M7_13_RESEARCH.md) compares primary papers and official
implementations. The selected approach separates usable visual localization,
expired predictions, and conditional braking predictions. A commanded stop is
not a zero-velocity observation. Fixed room cameras cannot establish robot rest
from a stationary background. New tracking networks or a full SLAM replacement
would not by themselves correct this state-validity problem.

## Fixed boundaries

- Keep the separate BB8-RL project and unchanged Genesis Studio dependency.
- Preserve the SAC/vision checkpoints, scan map, actuator, speed cap, position
  uncertainty limit, one-second occlusion limit, stopping certificate and arrival
  tolerances. No simulator pose/velocity/segmentation enters navigation.
- Preserve the existing conservative predictor and its raw error radius in
  diagnostics. Do not clamp uncertainty, invent a zero-velocity measurement, or
  use a smaller advisory region to authorize motion or arrival.
- A prolonged loss cancels the prior goal. Reacquiring a visual position does not
  silently restart old motion. A new user goal requires valid localization.
- Retain failed runs and rejected measurements. Report one-room synthetic
  evidence separately from physical-robot or general statistical guarantees.

## 1. Explicit localization lifecycle

Expose `uninitialized`, `measured`, `predicted`, `lost`, and `reacquiring` states.
The existing time/radius limits define usable prediction. Also reject stale,
invalid or incomplete command/measurement timing. Store the last accepted visual
position, its timestamp and radius independently of the predicted position.

On loss, cancel motion through the existing actuator stop, discard the route and
pending goal, and retain the raw prediction only as a diagnostic. Goal requests
while lost must be rejected rather than queued for an unexpected later start.
Stop, Reset, and camera-mode reset remain available.
The HTTP/user interface also rejects goals before initial localization. Internal
demo/bootstrap fixtures may retain their existing startup goal until the first
accepted fix; they cannot move before the existing visible initialization gates.

Recovery requires a sequence of fresh, finite, consistent visual observations,
the existing position/innovation limits, and a fresh zero-command velocity
initialization. A single returning detection is insufficient. Unsupported large
position jumps remain rejected; this is bounded reacquisition, not global
relocalization or object-identity proof. After recovery show a stopped, localized
robot and require a new destination.

### Revision before the second complete native family

The first complete family passed eight of nine intended outcomes. During the
five-second all-camera outage, the robot moved within the existing permitted
blind-motion interval and stopped about 21 cm from its last visual fix. A gate
centred on that historical visual point prevented otherwise consistent recovery.
The returning observation was 6.11 cm from the last still-valid predicted pose.

Freeze that prior valid prediction as the association reference when loss begins,
before processing any returning measurement. Keep the existing 12 cm association
threshold and fresh-track warm-up. Never move this anchor with later detections,
expired uncertainty or the conditional braking model; invalidate it if map or
calibration versions change. This remains bounded reacquisition. Also publish
pose, measured/predicted labels and last-seen history from the persistent observer
consistently; the controller keeps its independent, stricter expiry checks.
Retain the failed family and rerun all nine unchanged cases after these fixes.

## 2. Conditional braking prediction

Implement a separate, non-authoritative braking-region diagnostic. At the start
of a complete, contiguous, all-zero acknowledged interval, anchor the current
position enclosure and speed upper bound `norm(v) + velocity_radius`. Integrate
the existing conditional braking assumptions (0.5 m/s² lower deceleration and
0.35 s upper response time), including a declared integration-tick allowance.

The region has a finite limiting radius around its fixed anchor because this
conditional model assumes dissipative unforced stopping. Keep its assumption
label and `motion_authority: false` in every record. Invalidate it on a nonzero
command, missing/contradictory timeline, invalid state, or map/calibration change.
It must never replace the adversarial process-noise enclosure in the controller.

This reconciles the two existing modeling assumptions explicitly instead of
claiming that persistent unknown acceleration and a finite stopping distance
hold simultaneously without conditions. Native validation can support the
assumption for the tested synthetic drive; it cannot establish real robot rest.

## 3. UI and API behavior

When localization is lost, publish no current usable pose/radius. Show
`Localization lost`, the last observed position as a clearly marked ghost, and
the elapsed simulation time since it was observed. Do not flood the map with an
expired prediction circle or present the last position as a current measurement.
During recovery show `Reacquiring position`. Disable click-to-go and coordinate
submission until localization is usable; keep reset/mode/stop controls available.

Continue logging the raw predictor and conditional braking diagnostic. A new
camera observation may reduce the measurement radius; UI formatting alone must
not change any controller decision.

## 4. Verification fixed before implementation

1. Unit and API regressions: short occlusion remains usable; long loss and radius
   expiry cancel motion; idle and pending-goal loss also expire; long zero-command
   intervals do not publish a current pose; stale/missing histories fail closed;
   one spurious or repeated detection cannot recover; stable fresh observations
   can recover without resuming a cancelled goal; an explicit new goal can move.
2. Braking-model checks: analytic saturated/exponential phases, continuity,
   integration over 5/2.5 ms ticks, bounded limiting region, pre-interval anchor,
   nonzero/invalid input invalidation, and no effect on existing policy/guard
   outputs. Include an adversarial sustained-disturbance example to demonstrate
   why the conditional region cannot serve as a general bound.
3. Frontend checks: lost/reacquiring views, last-seen age, absent current circle,
   disabled navigation and enabled recovery controls. Inspect the actual browser.
4. Native regression: run the one-camera demo, reported destination in one/two/
   three-camera modes, Stop, invalid map, short all-view loss, prolonged all-view
   loss and reacquisition, and a multi-camera case with one view unavailable.
   Require zero contacts/boundary violations and the unchanged physical arrival
   dwell/speed/distance gate for every arrival claim.
5. Prolonged-loss experiment: preserve at least 240 simulated seconds of blind
   stationary bookkeeping (offline replay of captured command/measurement records
   is allowed and must be labeled); current-pose presentation must stay expired
   while raw uncertainty remains honest. Independently score conditional stopping
   regions against native truth only after controller decisions are recorded.

## 5. Delivery

Record research, implementation choices, all validation results and limitations.
Update the README, setup/user guide and M0/M1 index. Restart the local control
room only after validation, then publish the code/docs to the already authorized
private repository. The existing model/map release bundle remains unchanged.
