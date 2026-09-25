# Persistent autonomous exploration: first experiment

Prepared 24 September 2026. This implements the first bounded step from the
personality research, not a claim that BB-8 has subjective feelings or a fully
learned personality.

1. **Separate intention from movement.** Add a persistent place-choice learner
   above the existing RGB localization, map certification, A* and SAC controller.
   It can select only estimated-map anchors with a currently certified route.
   It receives navigation outcomes, never simulator truth or segmentation.
2. **Make experience durable.** Store candidate identity, proposals, terminal
   outcomes and learned values in SQLite, partitioned by robot and map version.
   Count each terminal event once. Preserve memory across resets and process
   restarts, but require an explicit Start exploring command every session.
3. **Keep control authority explicit.** Manual goals, Pause, Stop, reset, camera
   changes, heartbeat expiry and lost localization end autonomous activity.
   Reacquisition cannot restart it. New choices require a fresh measured pose;
   existing navigation uncertainty and braking checks remain authoritative.
4. **Compare choice rules.** Run fixed-profile, memory-only novelty and learned
   value baselines on matched synthetic outcomes, including preference reversal,
   noisy outcomes and restart retention. These are non-LLM baselines and the
   initial learner is a bandit, not recurrent RL. Report failures and limits.
5. **Trial a compact semantic model separately.** Verify an available local VLM,
   then measure a bounded candidate-ID proposal on saved synthetic RGB. Malformed,
   stale or timed-out output abstains. Keep it outside the live motor loop until
   grounded semantic evidence and useful closed-loop behavior are demonstrated.
6. **Integrate and verify.** Add an optional exploration panel, intent explanation
   and persistent experience count. Test command priority, map isolation,
   cancellation, restart behavior and route checks. Run native camera sessions
   and independently score their motion logs before reporting success.

Success means BB-8 can explicitly enter exploration, choose a reachable place,
record an actual arrival, reuse that experience, and stop reliably. It does not
mean a synthetic utility benchmark has established a general personality. The
next phase needs semantic experiences and a temporally extended learned policy,
evaluated for stable individuality, adaptation and voluntary activity choice.
