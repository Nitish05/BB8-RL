# M4 execution plan: first SAC + HER pilot

## Objective and scope

Implement, run and evaluate a resumable goal-conditioned SAC learner in the
independent BB8-RL project. Use the synthetic 4 × 4 m room, existing bounded drive,
Gymnasium contract and sparse goal reward. Studio remains an unchanged dependency.
This delivery establishes a tested training pipeline and measured pilot results;
it does not assume the pilot will solve navigation or meet final acceptance targets.

## 1. Freeze the experiment

- Use the existing authored room, seed 42, seven obstacles and oracle observations.
- Preserve position/speed tolerances, 0.5 s arrival dwell, collision penalties and
  goal-independent failure/time-limit flags. Do not add goal-dependent termination
  to make learning curves look better.
- Training seeds: **17, 29, 43**. Train each for at least **12,000 transitions**.
  Complete the current episode at checkpoints; disclose any budget overshoot
  (at most 1,199 transitions for this task's 1,200-action horizon).
- Evaluate before training and at the checkpoint near 6,000 steps on four fixed
  validation cases, then evaluate final checkpoints on all 12 validation cases.
  Run A* and random actions on the same final cases once for paired references.
- Use no final-test cases for model selection or reported pilot performance.
  Within this authored room the splits vary endpoints, not room geometry.

## 2. Implement SAC + HER

- Pin the installed Stable Baselines3 release in the environment snapshot.
- Use SAC with a dictionary policy and `HerReplayBuffer`, future-goal sampling,
  four relabeled goals per real goal, and `copy_info_dict=True` so collision and
  control penalties survive reward recomputation.
- Start with a modest MLP (128/128), batch 128, replay capacity 20,000,
  learning rate 0.0003, gamma 0.98, tau 0.005 and automatic entropy adjustment.
- Start optimization after 1,500 transitions, ensuring the buffer contains a
  complete episode even when the first rollout reaches its full 1,200-step limit.
- Use fixed feature scaling and a compact pooled representation of the local map.
  Compute relative goal features inside the policy, after HER relabeling; never
  bake the original desired goal into the stored ordinary observation.
- Keep observations, reward and action mapping identical in training/evaluation.
  The policy receives no A* waypoints or full occupancy map. A*'s information
  advantage is a comparison limitation, not a hidden learning input.
- Run real SAC update probes on CPU and MPS. Select the supported faster learner
  for this small network; record actual device and timings. Physics remains CPU
  for the first comparison. Unsupported device requests must fail explicitly.

## 3. Checkpoint, resume and cancellation

- Save the SAC model/optimizers, inference policy, compressed replay buffer,
  Python/NumPy/Torch random states, action RNG and the next training episode seed.
- Record source, task, dynamics/world and dependency fingerprints, fixed feature
  normalization, reward version, steps, gradient updates and resume lineage.
- Publish checkpoints atomically only at episode boundaries. Genesis solver state
  is not serialized; resume starts a clean, deterministically seeded episode.
  Do not claim bit-identical GPU or mid-episode replay.
- Retain copied HER info and timeout metadata in replay. Verify that saved policy
  actions match loaded actions, replay survives, optimizer state resumes, and
  training step/update counters advance after restart.
- Treat first cancellation as a request to finish the current episode and save;
  abrupt termination retains the previous complete checkpoint. Failures are
  recorded and are not silently retried as successful training.
- Explicitly restart seed 17 from its midpoint checkpoint for the second half.
  This makes resume verification part of the actual pilot, not only a unit test.

## 4. Evaluate and inspect learning

- Add checkpoint-policy inference to the existing evaluator and preserve every
  episode outcome, trajectory and elapsed action time.
- Report successes with Wilson intervals, contacts, timeouts, goal error, arrival
  time, path length, and training throughput/gradient-update counts separately.
- Produce per-seed learning curves, a final comparison table and trajectory plots.
  Report weak or zero learning honestly. Do not pool correlated same-room episodes
  into a spurious claim of broad generalization or choose the best seed silently.
- Preserve initial/mid/final checkpoints and full logs. New performance tuning or
  a larger run follows evidence from this pilot rather than changing its budget
  or reward midway.

## 5. Verification and delivery

- Unit checks: configuration/contract mismatch rejection, fixed feature scaling,
  goal relabeling, retained penalties, replay timeouts and checkpoint integrity.
- Real-engine checks: finite SAC updates, bounded actions, terminal observations,
  save/load and episode-boundary resume. Run the existing regression suite.
- Provide this plan, the implemented source, launch/resume/evaluation commands,
  the three-seed report, checkpoint locations and usable inference policies.
- Keep run artifacts ignored by Git; no Studio patches, hardware commands, cloud
  jobs, commits or publication are required.

## Completion criteria

The pipeline must execute real gradients with HER, produce loadable checkpoints,
resume a real run and finish all declared pilot/evaluation cases. Numerical
failures or unsupported devices block that configuration, not the whole simulator
project. Navigation success is a measured outcome, not a prerequisite for honestly
completing this initial experiment. The next decision is whether to extend training,
adjust the task/architecture in a new experiment, or introduce held-out room layouts.

Implementation references: [SB3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html),
[SB3 HER](https://stable-baselines3.readthedocs.io/en/master/modules/her.html),
and [save/load behavior](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html).

## Evidence-driven amendment: terminal failure incentive

The first completed seed failed all 12 validation cases through contact. Inspection
identified an objective mismatch: with gamma 0.98, remaining outside the goal has
an infinite discounted cost of 50, while immediate collision costs only 11 and
ends future costs. An early failure can therefore be preferable to continuing.
This is a reward-design problem, not proof that SAC/HER cannot navigate.

Finish and preserve the original three-seed experiment exactly as declared. Add
a separate `task-penalty-v2.yaml` with collision/boundary penalties of 100, preserving
the reward formula, dynamics, observation and arrival contract. This exceeds the
discounted cost of continuing indefinitely and removes the early-failure advantage
for the present gamma and zero control cost. Add a regression check of that bound.

Run seed 17 for at least 6,000 transitions under this corrected task and evaluate
four validation cases. Label this an additional diagnostic screen, not an equal-
budget three-seed comparison or a reliable learned policy. Report it separately
from the original pilot. Do not expand to a large training campaign on the original
misaligned reward. Future runs must reconsider the bound if gamma/control cost change.
