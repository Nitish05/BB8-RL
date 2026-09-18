# M7: fixed-camera navigation — research decision

Literature/implementation review cutoff: 16 September 2026. Selection is for a
single elevated, stationary RGB camera seeing a small BB-8 in a room. There is no
published benchmark establishing a universally best model for this exact setup.
The judgments below are engineering inferences, not a cross-paper leaderboard.

## Decision

Build **calibrated ground-position belief control with a distilled temporal visual
front end**. Use SAM 3.1 as the planned segmentation/annotation teacher and
TAPNext++ as the planned causal tracking/re-detection comparator. Distill a small
task-specific model that predicts BB-8 ground position, visibility and uncertainty,
plus visible free floor. Keep the tested SAC waypoint controller as the first
downstream policy; evaluate Dreamer through the same observation interface later.

A calibrated camera turns pixels into metric coordinates. Temporal estimation
provides velocity and uncertainty. Unknown/occluded floor stays blocked, and lost,
stale, ambiguous or uncertain localization commands braking. The controller gets
camera-derived waypoints and velocity, not simulator pose or an oracle map.
The current policy's calibration and all original arrival gates stay fixed.

The proposed research contribution is **control-aware ground-position/visibility
distillation**: train perception for calibrated position, velocity, uncertainty
coverage and downstream clearance/arrival, rather than segmentation IoU alone.
This is a proposed combination and testable hypothesis, not a verified claim of
first-of-its-kind novelty. A systematic novelty review and ablations remain needed.

## Papers and available implementations

| Work | Evidence and role | Fit to this project |
| --- | --- | --- |
| [SAM 3 paper, Nov 2025 / Mar 2026 revision](https://arxiv.org/abs/2511.16719), [SAM 3.1 release, 27 Mar 2026](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) | Promptable object segmentation and video tracking; released code/checkpoints. The 3.1 update jointly processes tracked objects. | Preferred offline semantic teacher; does not supply metric ground position or a navigation guarantee. Official setup requests CUDA and checkpoint access. Its multi-object GPU speed figures do not establish Mac latency. |
| [TAPNext++, Apr 2026](https://arxiv.org/abs/2604.10582), [official TAP implementation](https://github.com/google-deepmind/tapnet) | Causal recurrent point tracking, long sequences and re-detection after occlusion; released checkpoints. | Preferred modern tracking comparator. Track the stable head/shape and visibility; a rolling shell texture point is not the robot center. Predicted occluded tracks are not observed free space. |
| [CoTracker3, Oct 2024](https://arxiv.org/abs/2410.11831), [official code](https://github.com/facebookresearch/co-tracker) | Teacher-generated pseudo-labels on real video; online/offline variants. | Strong distillation precedent and secondary baseline. Measure causal buffering/latency rather than using offline future frames in control. |
| [VGGT, CVPR 2025](https://github.com/facebookresearch/vggt), [VGGT-Ω, CVPR 2026 project](https://vggt-omega.github.io/) | Feed-forward scene geometry and camera/depth estimation; implementation links. | Useful calibration/reconstruction comparator when camera placement changes. One fixed view still needs a metric reference and cannot reveal arbitrary hidden obstacle geometry. Later-dated notices on the live Ω page are excluded from this cutoff. |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Released monocular/multi-view geometry models. | Depth prior for a later uncalibrated-camera ablation; learned depth alone does not establish centimetre-scale accuracy in this room. |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Released dense visual feature backbones. | Candidate student backbone if a small from-scratch model fails appearance generalization; features still require task heads and calibration. |
| [V-JEPA 2 / 2.1 implementation](https://github.com/facebookresearch/vjepa2), [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) | Video representations; action-conditioned world-model planning demonstrated on manipulation. | Research comparator after collecting camera/action data. Its arm results do not transfer directly to BB-8 planar drive or establish this task's reliability. |
| [NoMaD paper](https://arxiv.org/abs/2310.07896), [ViNT/NoMaD implementation](https://github.com/robodhruv/visualnav-transformer) | Goal-conditioned mobile-robot visual navigation and released deployment stack. | Valuable navigation baseline, but the onboard view/action setup differs from a fixed external camera. Not a drop-in replacement. |
| [Visual Servoing on Wheels, 2023](https://arxiv.org/abs/2306.14848) | Explicit remote-viewpoint robot control and orientation estimation. | Closely related task formulation. Our world-frame planar command model does not currently require shell heading, but physical calibration may change that. |

## Why this choice

BB-8 occupies only a few pixels in some current room views; occlusion and subpixel
localization can dominate policy error. A larger action-generating model does not
remove those observability limits. M6 already provides a strong controller, and
its imitation-only ablation also succeeded. The first experiment should isolate
perception, not simultaneously replace the successful control layer.

A floor homography is valid only for floor points. Mapping a head/body mask's
centroid through that homography biases position. The first implementation uses
a declared head-height plane; the student will explicitly predict the ground
projection and handle visibility. Obstacle silhouettes projected onto the floor
are treated as blocked/unknown, not asserted to be exact footprints. A camera
mount/calibration change must invalidate the existing map.

## Executable stages and acceptance criteria

1. **M7.1 — interface and geometry (start now).** Implement timestamped RGB-only
   perception packets, calibrated image/plane transforms, position/velocity belief,
   conservative floor occupancy, image-click goals and the saved SAC adapter.
   Establish a clearly labeled synthetic appearance baseline. Test stale frames,
   missing/ambiguous detections, calibration, blocked goals and arrival gates.
   Run real Genesis RGB closed-loop development cases and report failures.
2. **M7.2 — dataset and learned perception.** Render randomized room/camera/light/
   appearance/occlusion sequences. Simulator segmentation, depth and pose may
   label training/evaluation records, but must never enter the deployed observer.
   Split entire room/camera/appearance sequences, not neighboring frames. Label
   real calibration videos using SAM 3.1 with reviewed masks; compare TAPNext++
   causal tracks. Train the compact temporal ground-position/free-floor student.
3. **M7.3 — uncertainty calibration and control adaptation.** Fit uncertainty on
   validation only; measure 95% position-region coverage, visible-position p95
   error target 2 cm, velocity error and false-visible rate. Inject recorded
   perception delays/noise into policy training only if the frozen controller
   fails. Evaluate guard-enabled and guard-disabled ablations; stopping is not
   counted as arrival. Target <=50 ms observer+policy p95 on the chosen runtime;
   record capture/transport latency separately.
4. **M7.4 — frozen camera acceptance.** Select on >=100 development episodes,
   then register fresh 200-case tests before freezing. Include lighting/camera
   perturbations, occluders, distractors, dropout, blur and calibration drift.
   Target >=95% arrivals; report collisions, aborts and timeouts separately.
   Compare color baseline, student, teacher, tracking-only and oracle reference
   on the same cases. Include multiple training seeds and confidence intervals.
5. **M7.5 — real video before actuation.** Calibrate a real fixed camera with
   measured floor correspondences and lens distortion; evaluate recorded video
   without issuing hardware commands. Physical drive remains a separate gate.

M7.1 is an integration baseline, not completion of M7.2–M7.5. No SAM/TAP weights
are represented as running until checkpoint access, local compatibility, latency
and an actual inference test are verified. The initial baseline deliberately
depends on the synthetic room's color/shape and is not a real-room detector.

## First student experiment specification

This is a proposed experiment, not an implemented or benchmarked model. Use two
scales: a room-level encoder/decoder for visible free floor and a native-resolution
robot crop for ground-projection/visibility. Do not resize the whole 1280×960 view
to a small input and discard the five-pixel head. Generate crop proposals from
the previous **predicted** track and periodically reacquire over overlapping image
tiles; include proposal failures during training rather than always giving perfect
ground-truth crops. Full-resolution teacher masks require small-object review.

Start with a small convolutional encoder and recurrent crop head, using the
existing PyTorch runtime. Predict (1) ground-projection heatmap, (2) visible/not
visible probability, (3) a positive-definite planar covariance and (4) floor,
occupied and unknown probabilities. Calibrated projection and a temporal filter
remain explicit. SAM 3.1 supplies reviewed semantic masks; synthetic labels supply
metric targets, and TAPNext++ supplies the independent causal tracking baseline.

Train with position negative log likelihood, visibility classification, asymmetric
occupancy loss penalizing false-free more than false-blocked cells, and temporal
position/velocity consistency. Compare that objective against ordinary mask-only
training with the same data/model budget. Select probability thresholds and
covariance calibration on validation only. Assess actual policy rollouts, not just
pixel accuracy; include the current shaded-face failures and real missed tracks.

Begin with a bounded data pilot of 40 sequences × 128 frames, partitioned by whole
room/camera/appearance setup (24 train, 8 validation, 8 development evaluation).
Randomize lighting, textures, object colors, camera pose, shutter blur, noise,
dropout and distractors. The eight development sequences are not the final
acceptance test. Increase diversity before model size if generalization fails.
Run three training seeds for the selected configuration before freezing the fresh
camera acceptance suite. These starting budgets are engineering choices to be
revised using development evidence, not numbers taken from a paper.

## Data boundary and evaluation protocol

The observer receives only RGB, capture timestamp, calibration, known robot
geometry and a goal pixel. It has no environment, entity IDs, depth buffer,
simulator pose, velocity, obstacle coordinates or success flag. The evaluation
harness may read those fields for scoring, never for selecting actions or routes.
Scene rendering/reset and physics scoring remain in a separate module. No oracle
fallback is allowed when tracking or planning fails.

Known calibration is an explicit assumption, not camera-only self-calibration.
RGB simulation latency must be measured before claiming real-time performance.
Development cases may be adjusted while building the pipeline; they are never
reported as held-out acceptance. The original M6 test remains a historical state-
based result and says nothing yet about camera success.

## Implementation progress

[M7.2](M7_2.md) now implements a smaller first step: two convolutional segmentation
networks trained on native synthetic labels, gray RGB crop proposals, known
projection and the existing analytic temporal filter. Its experiment log records
the ordinary floor-mask failure and the correction for robot self occupancy.
The research specification above remains broader: recurrent tracking, learned
uncertainty, three semantic classes, reviewed SAM masks, TAPNext comparison,
texture/blur/distractor randomization, three training seeds and a fresh acceptance
suite are not completed by this initial experiment.
