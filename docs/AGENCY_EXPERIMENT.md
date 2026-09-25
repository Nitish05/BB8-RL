# Persistent activity choice: first experiment

This milestone adds durable activity memory and an adaptive goal chooser above
BB8-RL's existing navigation stack. It is a bounded first experiment toward an
autonomous character. It does not establish a human-like personality, emotion,
social preference, or open-ended intelligence. No language or vision-language
model runs in this experiment.

## Plan and implementation

1. Define stable activity anchors on the estimated map. The chooser receives only
   candidate IDs, labels, coordinates and the current estimated position. It has
   no simulator, actuator or hardware import.
2. Store identity, map version, candidate definitions, proposals and terminal
   events in SQLite. Apply an event and its learning update in one transaction.
3. Compare three choice rules with identical evidence access: a seeded fixed
   ranking, a memory baseline that favors less-visited activities, and a learned
   value baseline that adapts to consequences.
4. Check acquired preferences under opposite experience histories, exact restart
   persistence, changed consequences, stochastic distractors and duplicate events.
5. Connect selected goals to the existing estimated-map planner and its clearance,
   localization and stopping gates. The runtime owns execution, interruption and
   terminal outcomes; a proposal is never counted as a completed activity.

The first four steps are implemented in `bb8_rl.agency`, its tests and
`scripts/audit-agency.py`. The live runtime integration is separate so the same
chooser can be tested independently of physics and perception.

The live [coverage correction](AGENCY_COVERAGE_FIX.md) now excludes completed
map targets before invoking this selector. Its no-repeat rule is prescribed and
separate from this synthetic value-learning comparison. A synthetic reversal
score does not measure exploration quality or detect repeated navigation loops.

## What is learned

The learned strategy is a **nonstationary activity-value learner**, a simple
bandit. It is not recurrent reinforcement learning and does not train the SAC
motor policy. Each activity stores an exponentially weighted utility estimate:

```text
value ← value + 0.2 × (observed utility − value)
learning progress ← 0.8 × previous progress + 0.2 × |value change|
score = value + 0.12 / √(1 + visits) + 0.05 × learning progress − distance cost
```

Twelve percent of learned decisions sample an alternative activity. This retains
a way to discover that an initially unhelpful activity has become useful. The
distance cost is 0.01 per metre, capped at 0.04. An anchor within 15 cm of the
estimated current position is omitted from that choice to avoid reselecting an
already-completed stationary destination. Existing navigation gates still decide
whether a proposed route is executable.

The fixed baseline uses a seed-stable arbitrary ranking. It is **not a prompt-only
LLM baseline**. The memory baseline uses visit counts and ignores outcome valence.
All three retain identical evidence and value estimates; the fixed and memory
baselines ignore the stored utility when choosing.

Live utility has a narrow, explicit meaning:

| Runtime event | Utility update | Meaning |
|---|---:|---|
| `arrived` | +1.0 | Navigation to the activity completed. |
| `rejected` | −0.6 | Navigation to this activity was rejected or could not complete. |
| `cancelled` | None | Interruption is remembered, without penalizing the activity. |
| `localization_lost` | None | Estimator loss is remembered, without inventing a preference. |

These are designer-selected navigation rewards. Duration and normalized progress
are retained as evidence, but do not create an extra speed reward. The system
does not infer affection, approval, battery state or other hidden needs from them.
If every destination has the same consequence, the learner has no basis for
claiming distinct acquired social preferences.

## Durable memory contract

```python
from bb8_rl.agency import AgencyEngine, AgencyStore, Candidate

store = AgencyStore("work/agency/live.sqlite", agent_id="bb8", map_version="scan-123")
engine = AgencyEngine(store, strategy="learned", seed=0)
proposal = engine.select(
    [Candidate("west", (-1.0, 0.0), "West activity")],
    now=10.0,
    pose=(0.0, 0.0),
)
# The runtime now attempts the guarded goal. Only after an actual terminal event:
engine.record_outcome(
    event_id="unique-runtime-trip-1",
    candidate_id=proposal["candidate_id"],
    outcome="arrived",
    now=22.0,
    duration_s=12.0,
    progress=1.0,
)
store.close()
```

Use a stable agent ID and map version when reopening memory. The schema is
versioned; unknown versions fail rather than silently reinterpreting data.
Candidate coordinates cannot change under an existing ID in the same map version.
Different map versions and agent identities have separate evidence. Historical
events are not automatically transferred across reconstructed maps.

An event ID identifies an immutable payload within its agent/map scope. Exact
replay returns `False` and applies no update; conflicting reuse raises `ValueError`.
The same-ID payload includes the timestamp, so a replay must retain its original
timestamp. The store serializes access across server threads; SQLite also acquires
a write lock before event comparison and updates. Selection counters persist so a
restart does not reset the exploration sequence when the same seed is supplied.

`select` returns `None` when no candidate is eligible. Its decision and proposal
counts measure requests, not successful visits. `snapshot` provides JSON-safe
identity, scope, evidence type, event count, decisions and per-activity preferences.
Competence is the fraction of arrived outcomes among arrived/rejected outcomes;
it is not a calibrated collision probability or a route certificate.

The optional `record_experience` hook accepts bounded synthetic utility only in an
explicit `allow_synthetic=True` store. A synthetic scope cannot be opened as live
navigation memory or vice versa. The live outcome API accepts only named runtime
events, duration and normalized progress, with no arbitrary reward or truth field.
This is an API provenance boundary, not a replacement for validating the runtime
that supplies those events.

## Reproduce the controlled audit

```bash
scripts/python.sh -m pytest tests/test_agency.py -q
scripts/python.sh scripts/audit-agency.py --seeds 20 --output work/agency-audit
```

The audit writes JSON and Markdown summaries plus the SQLite stores into ignored
`work/`. It runs 20 held-out seeds (100–119), two opposite histories per seed, and
three baselines: **120 independent cases**. Seed 0 is used during development;
the learner's constants are fixed across the audit. Each case receives 120
closed-loop acquisition outcomes, a 60-choice frozen evaluation, a store close
and reopen, another evaluation, 120 reversed-consequence outcomes, and a final
evaluation. Evaluation advances decision counters but does not update values.

There are three activities. One has synthetic utility +0.8, the other −0.5;
five percent of these outcomes have their sign flipped. The third produces
independent −0.7/+0.7 outcomes with equal probability. The positive activity
switches during reversal. The chooser never receives this hidden schedule; it
receives only the consequence of its selected activity. Each terminal event is
delivered three extra times to stress replay handling.

Initial measured results:

| Strategy | Preferred after acquisition | After restart | After reversal | Noisy activity after reversal |
|---|---:|---:|---:|---:|
| Fixed ranking | 30.00% | 30.00% | 30.00% | 40.00% |
| Visit memory | 30.00% | 30.00% | 40.00% | 20.00% |
| Learned value | 91.21% | 92.13% | 91.88% | 4.25% |

All 120 snapshots matched exactly across restart. The audit processed 28,800
unique synthetic events and ignored 86,400 deliberate duplicate deliveries.
Acquisition and reversal both exceeded the preset 85% preferred-choice threshold;
the learned acquisition result exceeded memory by more than 30 percentage points;
the noisy activity remained below the preset 15% threshold.

The different percentages before/after reopening come from fresh exploratory
choices, not lost memory: the complete stored snapshot is compared exactly.
Fixed and memory evaluation can repeatedly choose the same tied activity because
visits are frozen. The reported mean spans both opposing histories, so arbitrary
anchor ranking cannot identify the positive consequence in advance.

These are **synthetic utility results**. They do not measure navigation success,
human preference, character believability, physical BB-8 behavior or language-model
quality. A three-activity bandit with short consequence reversal is deliberately
much simpler than the intended long-term companion.

## Next evidence needed

Run the same interruption and persistence contracts through the live control
room, verifying that Stop, map changes and localization loss prevent further
autonomous goals. Then introduce measurable activity distinctions and a real
prompt-only/memory-only/model-assisted comparison. Benchmark any VLM on the actual
Mac with Genesis running before choosing its deployment configuration. Later
social or human-feedback learning needs its own source-labelled evidence and
held-out interaction evaluation; it must not reuse the synthetic reward hook as
if those rewards were observations from people.
