# Purposeful interaction experiment

Prepared 24 September 2026. Scope: a synthetic, embodied learning experiment,
not a claim of personality or subjective needs.

## Problem and objective

The map-coverage prototype rewarded arrival and eventually repeated successful
destinations. Excluding completed targets stopped the loop but gave exploration
no useful purpose. With fixed external cameras, moving BB-8 does not provide a
new camera viewpoint. This experiment instead asks an action-dependent question:
**which interaction can restore a currently depleted simulated resource?**

## Implementation sequence

1. Add two declared virtual station zones, A `(0, 1)` and B `(-0.5, 1)`,
   in the already mapped room. They are
   task coordinates, not newly recognized objects. Camera localization and the
   existing route certification still determine how BB-8 approaches them.
2. Model a resource initially at 35%, a target of 80%, and a small cost per metre
   traveled. Idle does not drain it to manufacture activity. Only an explicitly
   requested, stationary one-second interaction inside a station can change it.
   Station A initially has no effect; station B restores 45 percentage points.
   These mechanics belong to the simulator and are hidden from the chooser.
3. Deliver resource and interaction results as **simulated station telemetry**.
   This is separate from RGB localization. Do not paint a response into camera
   frames or call it visual understanding. Station zones appear on the map.
4. Learn each station's response probability and useful gain from before/after
   evidence, in separate persistent tables. Arrival earns no reward. Compare
   predicted need reduction and bounded information value with travel/interaction
   cost and waiting. Suppress repeatedly unhelpful probes; allow evidence of a
   changed response to reopen a previously ruled-out alternative.
5. Connect this learner above the existing SAC waypoint controller. Every goal
   uses the same clearance, camera, stopping, heartbeat and command-generation
   checks. Stop, manual control, stale telemetry or lost localization cancels an
   interaction without learning a successful outcome. Restart remains paused.
6. Show the current resource, question, prediction, actual outcome and learned
   station response estimates. Preserve map coverage as an explicit diagnostic
   mode, rather than presenting it as purposeful behavior.

## Acceptance evidence

- Matched starting conditions with different experience must produce different
  station choices; reversing station effects must cause adaptation.
- Arrival alone, cancellation, missing evidence and an ineffective interaction
  must not count as positive outcomes. Retried request IDs must not refill twice.
- Satisfied needs lead to idle. Empty/noisy stations must not support endless
  reward-free wandering. Memory survives restart; motion authority does not.
- Native one- and three-camera runs must show camera-guided approach, stationary
  effect delivery, learned outcomes, satisfied idle, and Stop without contacts.
- Report synthetic model benchmarks separately from native navigation evidence.

## Limits and next extension

This implements learned action consequences under an engineered motivation. It
does not learn its own ultimate objective, recognize a person, infer affection,
or establish a personality. The virtual stations and resource make causality
testable before adding actual interactive objects and visual response detection.
Resource resets with a new simulation episode; learned station knowledge stays.

Design follows the project's [research review](research/personality/RESEARCH.md):
typed evidence, persistent outcome memory, decision-relevant learning progress,
explicit operational needs, and evaluation against opposite histories and noise.
