# M5: SAC + HER versus DreamerV3

Prepared 16 September 2026, before comparison runs.

## Question and scope

Can either learner acquire useful control in the existing synthetic room with
the corrected reward? This is a bounded local pilot, not a claim of algorithm
superiority. BB8-RL remains a separate consumer of Genesis Studio.

Use the authored 4 m room, seven obstacles, task-penalty-v2, oracle observations,
two bounded continuous actions, and identical arrival/failure/timeout semantics.
Both policies receive the same 180 features: scaled state/history, 3 x 7 x 7
average-pooled local map, and relative goal. Neither receives A* routes or the
global map. No camera images or hardware commands are part of this experiment.

## Implementation

1. Pin the authors' DreamerV3 repository to
   `e3f02248693a79dc8b0ebd62c93683888ddaccfe`, under ignored work storage.
   Keep upstream code unchanged. Use an isolated Python overlay for its CPU JAX
   dependencies, leaving the existing simulation runtime unchanged.
2. Adapt Gymnasium reset, termination and truncation to Dreamer's `is_first`,
   `is_last`, and `is_terminal`. A time limit is last but not terminal. Reset
   observations do not count as environment interactions. Preserve transition
   action alignment using the upstream Driver and replay stream.
3. Use the official size1m architecture, float32 CPU, batch 4 x sequence 32,
   replay ratio 16, imagination length 15, and discount horizon 50 (gamma 0.98,
   matching SAC). These are resource-bounded settings, not the paper's benchmark
   training recipe. Both learners warm up for 1,500 physical transitions.
4. Save agent parameters/optimizer counters, replay chunks, configuration,
   upstream revision, source/task hashes and per-episode metrics. Dreamer
   checkpoints initially support verified inference reload; exact training
   continuation is outside this pilot. SAC retains its existing resume support.
5. Integrate Dreamer into the paired evaluator. Retain upstream sampled actions
   in eval mode, with a reproducible RNG counter per case, and report this
   difference from deterministic SAC evaluation.

## Execution and limits

- Run a real Genesis/Dreamer smoke test before the pilot. Check finite updates,
  action bounds, termination mapping, replay and checkpoint reload.
- Train fresh SAC + HER and DreamerV3, seed 17, at least 30,000 physical
  transitions each. Finish the final episode (up to 1,199 extra transitions).
  Save checkpoints around 15,000 and 30,000 transitions.
- Evaluate each final policy on the same first 12 validation cases. Run A* as a
  full-information reference under the same corrected task. Keep the final
  200-case test suite untouched.
- Run the learners sequentially for interpretable wall time. Record startup/JIT
  separately where possible, training time, updates, arrival/contact/timeout
  counts and final distance. Equal environment budgets do not mean equal compute,
  model capacity or replay distributions (SAC has HER; Dreamer has imagination).
- Run non-GUI regression tests and targeted native Genesis integration tests.
  Deliver the results even if neither policy succeeds. One seed, one room and
  12 validation cases cannot establish a robust winner or physical transfer.

If runtime compatibility or throughput requires changing this protocol, record
the change and reason before collecting the comparison results.

## Interpretation and next decision

Use success and collision counts as primary outcomes; training return/loss alone
does not establish navigation. If both fail, investigate curriculum, successful
demonstrations and reward/horizon design before escalating model size. If either
shows promise, repeat with multiple seeds and new layouts before vision transfer.

Sources: [DreamerV3 authors' code](https://github.com/danijar/dreamerv3),
[paper](https://www.nature.com/articles/s41586-025-08744-2).

## Runtime preflight amendment

The isolated CPU runtime imported and passed native probes, but `pip check`
identified inherited OpenCV 5 and tifffile requirements for NumPy >=2, whereas
the pinned Dreamer runtime uses NumPy 1.26.4. The first SAC launch was gracefully
cancelled and preserved separately. Pin OpenCV 4.11.0.86 and tifffile 2024.9.20 in
the overlay, verify dependencies, then restart both learners fresh. These image
utilities are not policy inputs in this state-observation comparison. No task,
reward, model or budget is changed.
