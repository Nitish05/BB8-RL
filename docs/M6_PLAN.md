# M6: reliable learned waypoint control

Target: at least 95% arrivals on a frozen 200-case test for each learned controller,
with collision counts reported separately. This is synthetic state-based navigation.
The camera/perception and physical robot remain outside this claim.

The M5 agents had zero training successes, sparse rewards, a 50-action discount
horizon, and no global route. M6 explicitly changes the architecture: A* over the
oracle room map supplies collision-clear waypoints, while learned SAC/Dreamer
policies emit every continuous drive command. No expert action fallback is allowed
in learned-controller evaluation. This is a hybrid planner + learned controller,
not proof that end-to-end RL was solved with the original local observations.

Train on procedural 3 m rooms (empty, barrier, corridor, scatter) with disjoint
training seeds. Collect successful expert rollouts, pretrain actors with imitation,
then perform genuine RL updates with an auxiliary teacher loss on training states.
Use relative waypoint displacement, velocity and a final-waypoint flag; use a
progress reward, explicit success termination and a longer discount horizon.
Preserve the original arrival gate: 10 cm, 3 cm/s and 0.5 s dwell. Do not relax it.

Both algorithms remain recognizable: SAC twin critics, entropy term and target
updates; DreamerV3 recurrent world model, imagined rollouts, actor and value losses.
Dreamer evaluation uses its mean action. The authors' code is pinned and unmodified;
local extensions implement supervised actor regularization. Record the substantial
role of planning and imitation and include a behavior-cloning-only ablation.

Development: fixed validation seeds (never test seeds), initially 12 cases per room
family, then 50 authored and 50 procedural cases before freezing candidates. Train
and adjust only using training/validation. Save failed runs alongside successes.

Final test: after each candidate passes >=95/100 validation cases, freeze its
checkpoint hash and evaluate once on 200 previously unused cases: 100 in the 4 m
seven-obstacle authored room and 100 procedural 3 m layouts. Both use test layout
seeds starting at 20000 and episode seeds at 25000, disjoint from development.
Do not tune using final test outcomes; if a candidate misses the target, report it
and use a new preregistered test set for any future candidate.

Record success, contact, timeout, path length, final error, arrival time, all cases,
checkpoint/source hashes, seed, actual training steps and timings. Validate native
Genesis dynamics, adapter reward/termination, teacher data separation and reload.
