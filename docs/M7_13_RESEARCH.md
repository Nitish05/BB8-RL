# M7.13 research: occlusion, lost localization, and reacquisition

Prepared and sources accessed: **24 September 2026**.

This review addresses the reported missing-camera state and large position bound. It does not propose retraining SAC. The recommended change is an explicit loss/relocalization state machine, fresh validated RGB reacquisition, and clear separation between authoritative uncertainty and any conditional stopping diagnostic. Preserve the existing conservative controller bound and motion timeout.

## What the present failure means

The current command belief continues to inject acceleration uncertainty during prolonged missing observations. In its unsaturated zero-target regime, for native tick duration h = 0.005 s, response time tau = 0.25 s, and acceleration uncertainty q = 0.12 m/s²:

```
velocity_radius_next = exp(-h / tau) * velocity_radius + q * h
limiting_velocity_radius = q * h / (1 - exp(-h / tau))
                         = 0.03030099999 m/s

position_radius_increment / h = velocity_radius + 0.5 * q * h
asymptotic_position_radius_growth = 0.03060099999 m/s
                                 = 1.83606 m per simulated minute
```

These are calculations from the current local model, not measurements of physical motion. Persistent admissible disturbance can keep a reachable position set expanding even while the nominal trajectory has stopped. The UI must distinguish a last accepted position, an extrapolated position, its observation age, and a current accepted measurement.

The existing braking-tail calculation uses a stronger dissipative zero-target assumption than indefinitely applying adversarial q. That assumption can support a separate conditional diagnostic, but it does not justify silently shrinking the authoritative belief. Local implementation reviewed: [occluded_control.py](../src/bb8_rl/occluded_control.py), particularly `CommandBelief._advance_interval`, `observe`, and `OccludedController._arrival_braking_time`. This research records the pre-M7.13 behavior; subsequent implementation may change lifecycle handling.

## Primary-source comparison

The recommendations in the final column are project-specific inferences, not claims that the source has validated BB8.

| Source and verified implementation | Evidence from the source | Transfer to this project and limits |
| --- | --- | --- |
| **robot_localization**: [official state-estimation documentation](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/state_estimation_nodes.rst), [dynamic Q implementation](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/src/filter_base.cpp), [EKF update](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/src/ekf.cpp) | Optional dynamic process noise scales the pose-Q block using the norm of the estimated state's twist. Control inputs affect prediction; they are not measured velocity. The EKF uses innovation gating and covariance propagation. | Motion-dependent noise is a useful later calibration experiment. Estimated velocity can be wrong after loss. Scaling Q does not erase inherited velocity uncertainty, and a covariance is not the current deterministic engineering radius. Do not copy this switch as a bound-shrinking fix. |
| **OpenVINS ZUPT**: [official derivation](https://docs.openvins.com/update-zerovelocity.html), [official implementation](https://github.com/rpng/open_vins/blob/master/ov_msckf/src/update/UpdaterZeroVelocity.cpp) | Its inertial rest test uses actual IMU evidence, with a velocity-magnitude check because zero acceleration can also mean constant nonzero velocity. Visual disparity is an additional fallible rest cue. | We have fixed external cameras and no observer IMU. Background disparity remains small when BB8 moves, so it cannot establish robot rest. Visible BB8-specific temporal measurements may support a rest hypothesis; a zero command or hidden robot cannot supply that evidence. |
| **ORB-SLAM3**, Campos et al., 2021: [paper](https://arxiv.org/abs/2007.11898), [official tracking/relocalization implementation](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Tracking.cc) | Distinguishes tracking, recently lost, and lost states. Relocalization searches candidates and applies feature matching and geometric pose validation before declaring tracking recovered. | Adopt explicit lifecycle and independently validated recovery. Do not transfer its PnP/inlier thresholds: it estimates a moving camera against mapped landmarks, whereas our cameras are fixed and BB8 is the moving target. Starting a new SLAM map is not the required recovery behavior here. |
| **OC-SORT**, Cao et al., CVPR 2023: [paper](https://arxiv.org/abs/2203.14360), [official filter implementation](https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/kalmanfilter.py) | Uses real observations to correct accumulated filter error after occlusion, including a virtual trajectory constructed after reacquisition. | Preserve observation history and rebuild velocity from accepted measurements after loss. A retrospectively reconstructed hidden trajectory is not evidence available when the hidden action was issued. Never mark it as measured or let it certify blind motion. This is bounding-box tracking, not metric navigation safety. |
| **CoTracker3**, Karaev et al., 2024: [paper](https://arxiv.org/abs/2410.11831), [official code and checkpoints](https://github.com/facebookresearch/co-tracker) | Provides learned point tracks and predicted visibility, with offline and online interfaces. The official implementation recommends a GPU; small CPU tasks are possible. Its majority license is CC-BY-NC. | Optional later comparison for BB8-specific temporal association, using only causal outputs and retaining source timestamps. Predicted visibility is not renderer truth or a metric position certificate. Online chunks have buffering/latency that must be measured on this Mac. No new dependency, checkpoint, or licensing change is needed for the selected fix. |

A foundational precedent for command-conditioned noise is the velocity model in Thrun, Burgard, and Fox's [author-hosted Robot Motion chapter](https://robots.stanford.edu/probabilistic-robotics/corrections1/pg122-139.pdf). It makes noise depend on motion variables while distinguishing command-based prediction from odometric sensing. Our inertia, latency, and collision model require separate residual calibration; setting process uncertainty to zero whenever a requested command is zero is not established by that kinematic model.

This is a targeted comparison of relevant primary work and current official implementations, not an exhaustive ranking of all tracking models. Recent model benchmark accuracy alone does not resolve the present state-management and evidence-semantics problem.

## Selected implementation direction

### 1. Give localization a lifecycle independent of goals

Use explicit `uninitialized`, `tracking`, `predicting`, `lost`, and `relocalizing` states. Keep the existing short-occlusion motion allowance and its complete stopping certificate. After its existing timeout, issue zero and keep motion blocked.

A new goal, Stop, goal rejection, or creation of a new controller must not manufacture a new pose or reset observation age. The localization lifecycle needs to survive those transitions. Missing images, absent detections, rejected measurements, and stale/invalid calibration should have separate diagnostic reasons.

When lost, show the last accepted RGB position and its time as historical evidence. If showing an extrapolated position, label it predicted. The growing authoritative bound may remain available in diagnostics; it should not be presented as a new physical displacement measurement. This is a presentation and lifecycle correction, not a numerical cap.

### 2. Reacquire using fresh target evidence

Keep the ordinary short-gap innovation gate for normal tracking. Long-loss recovery should collect a separate candidate track from fresh, calibrated, finite, detector-qualified RGB measurements while commands remain zero. Require consistent timestamps, target identity, plausible temporal motion, and multiple accepted frames. When two or three views are available, add synchronized cross-view metric consistency; a single-view mode must remain usable with stronger temporal confirmation.

Do not accept an isolated high-confidence box merely because the old position bound became huge. Conversely, do not require every long-loss candidate to stay within the old fixed tracking gate forever: a separate, explicitly logged relocalization transition can resolve that deadlock. Freeze recovery thresholds before held-out testing.

On acceptance, initialize position from fresh measurements, reset stale velocity/history and model enclosure state, and re-enter existing warm-up and velocity-initialization gates. Preserve the old trajectory for audit. Do not splice synthetic hidden points into the measured history. Recompute any route from the new accepted pose and keep the original map, clearance, stopping, and visible-arrival conditions.

This is the selected practical approach. It requires small state and validation logic around the existing detector, not an additional large vision model or RL policy.

**Implementation boundary:** the first implementation used a 12 cm association
gate around the last visual fix. The first native family exposed its failure to
account for motion during the permitted blind interval. The revised plan retains
the 12 cm threshold around the last valid prediction, frozen before processing
the frame that triggers loss. It implements bounded reacquisition, not the broader
global relocalization described above. A returning target beyond that gate needs
an explicit episode reset. Target identity is inherited from the existing BB8
detector and calibrated rig; this change adds temporal consistency, not a new
identity verifier. The final state names are `measured`, `predicted`, `lost`, and
`reacquiring`, alongside `uninitialized`. The concrete plan fixes nine native
cases plus a 240 s offline prolonged-loss experiment; the larger research matrix
below describes future coverage and must not be read as completed validation.

### 3. Keep a conditional stopping anchor diagnostic separate

A finite stopping region can be displayed or scored under the existing declared braking-tail assumption, after complete acknowledged zero commands and expiration of all potentially active nonzero targets. Requested zero alone is insufficient.

Let s be a conservative speed bound at that event, a the assumed minimum braking acceleration, and tau the assumed maximum response time. If speed obeys the dissipative bound ds/dt <= -min(a, s/tau), the remaining path-length bound is:

```
D(s) = s * tau                                      if s <= a * tau
D(s) = (s*s - (a*tau)**2) / (2*a) + a*tau*tau       otherwise
```

This is our derivation from the project's stated model, not an empirical guarantee from the cited papers. Anchor the diagnostic to the position enclosure at the zero-acknowledgment event and include all preceding latency/motion uncertainty. Do not restart it at a later predicted point with discarded uncertainty.

Name its assumption, retain the authoritative bound separately, and set `authorizes_motion = false`. Invalidate the diagnostic on an acknowledged nonzero target, incomplete command history, or version/time inconsistency. Even a finite diagnostic region does not prove that BB8 is observed or exactly stationary. Sustained external forcing, slip outside the model, slopes, or unmodelled impacts invalidate this dissipative assumption.

### 4. Defer dynamic-Q tuning and learned tracking

First resolve lost-state semantics and recovery using the frozen camera model. A later experiment may fit a command/speed-dependent residual model on independent calibration sequences. Keep probability covariance and worst-case engineering bounds explicitly different; do not replace one with the other by changing units or labels.

CoTracker3 or observation-centric track refinement can then be evaluated as association aids. No rollout claim should use future frames, offline smoothing, or learned hidden coordinates as current measured position. Benchmark inference and buffering on the actual Mac before adding a runtime path.

## Frozen validation recommended before promotion

Freeze cases, thresholds, code/model/map hashes, and expected outcomes before running them. Retain every failed attempt. Synthetic physics and renderer labels are independent scorers read after each decision; neither enters localization, the stopping diagnostic, or route selection.

| Case family | Required evidence |
| --- | --- |
| Short loss from several nonzero speeds | Existing allowed prediction behavior; timely zero after expiration; unchanged stopping and map certificates; fresh recovery. |
| 30 s and 300 s missing-camera intervals | Explicit LOST throughout the stale period; zero commands; new goals cannot bypass loss; historical and predicted displays remain distinguishable. Keep physical stopping evidence separate from command acknowledgments. |
| Genuine geometric occlusion and injected image loss | Separate denominators. For geometric loss, record target visibility labels independently; for injection, label the failure artificial. |
| Delayed/pending nonzero commands and malformed timelines | No premature stopping anchor, no false rest, and motion blocked on incomplete or contradictory history. |
| False detections, stale frames, inconsistent cameras, and an out-of-gate true reappearance | Reject bad candidates; accept a stable validated fresh track via explicit recovery; reset stale velocity state; do not resume on one frame. |
| Adversarial persistent disturbance | Demonstrate why the conditional stopping region cannot be treated as the authoritative bound. A zero acceleration/constant nonzero velocity case must not trigger a sensorless ZUPT. |
| Modes 1, 2, and 3; Stop; reset; goal replacement; invalid memory | Localization age/state survives each lifecycle correctly; valid reset is explicit; invalid memory still brakes. Existing original arrival gate remains 10 cm, 3 cm/s, and 0.5 s of visible dwell. |

Report time to LOST, time to valid reacquisition, false-reacquisition counts, blocked/unintended commands, position and speed error, authoritative-bound coverage, conditional-diagnostic coverage under declared assumptions, collisions, braking time/distance, and end-to-end latency. Split fitting/development and validation by complete episode or session, not adjacent correlated frames. Stratify motion and occlusion duration so long stationary idle does not dominate averages.

For a probabilistic filter experiment, report consistency as well as RMSE; [OpenVINS' official evaluation guidance](https://docs.openvins.com/eval-metrics.html) discusses error and NEES. For the present engineering radii, report violations/coverage directly rather than calling the radius a calibrated confidence interval.

No cited method demonstrates universal recovery for this BB8 fixture. Promotion should mean that the frozen local cases pass with the stated assumptions and that existing successful navigation/control cases remain valid.
