# Clearance and speed configuration results

`reserve-3cm` is the best configuration in this comparison. It uses a 3 cm fixed
margin, a 15 cm/s speed cap and 4 cm of additional clearance for route search.
That search reserve chooses roomier paths; it does not relax the independent
physical, uncertainty or stopping checks. The optional longer-horizon slowing
strategy was tested and is disabled in the selected profile.

## Frozen comparison

The [plan](M7_11_PLAN.md) defined completion-first selection before the runs.
The [protocol](../configs/interactive/benchmark-protocol.json) fixed two tuning
routes, three separate validation routes, the original demo and failure-control
cases. Every run retains the original independently scored 10 cm position,
3 cm/s speed and 0.5 s visible arrival dwell criteria. Simulator truth is recorded
after controller decisions and used for independent scoring; it is never a
controller input.

The first round tested six profiles. All reached the forward goal, but none
completed the reported reverse route within 60 simulated seconds. Offline map
analysis then showed a planned corner with 14.030 cm available clearance versus
14.158 cm required by the smallest offered command. Simply reducing the next
command cannot remove current momentum and queued targets. A second round added
only a 4 cm planning reserve to the best first-round profile, `margin-3cm`.

| Profile | Fixed margin | Speed cap | Longer lookahead | Forward | Reverse | Guard episodes, forward / reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 4 cm | 15 cm/s | Off | 36.65 s | Timeout | 63 / 79 |
| Margin 3 cm | 3 cm | 15 cm/s | Off | 29.10 s | Timeout | 38 / 27 |
| Margin 2 cm | 2 cm | 15 cm/s | Off | 34.30 s | Timeout | 55 / 73 |
| Slower | 3 cm | 10 cm/s | Off | 33.60 s | Timeout | 32 / 87 |
| Lookahead 100 ms | 3 cm | 15 cm/s | 100 ms | 33.95 s | Timeout | 71 / 32 |
| Lookahead 200 ms | 3 cm | 15 cm/s | 200 ms | 52.60 s | Timeout | 151 / 32 |
| **Selected: 4 cm planning reserve** | **3 cm** | **15 cm/s** | **Off** | **21.65 s** | **33.60 s** | **0 / 45** |

The selected profile alone completes **2/2 tuning goals**. Its forward run is
40.9% faster than the baseline and has no guarded-stop episode. A timed-out run
is not a successful slower arrival; lower guard counts during a permanent stop
also do not establish better performance. All 14 tuning attempts remain in the
denominator and have zero contacts, boundary violations or premature arrivals.
Tuning captures retain visibility counts and physics/command traces; camera
images were deliberately not saved for this stage and are not claimed verified.

## Separate validation

The [selection record](evidence/m711-selection.json) froze the winner before any
validation run. The [independent report](evidence/m711-benchmark.json) verifies
the protocol, selection, run hashes and complete seven-case set; it does not
choose a different winner using validation results.

| Fixed validation case | Cameras | Outcome | Arrival, simulation time |
| --- | ---: | --- | ---: |
| North route | 1 | Visible arrival | 13.30 s |
| Center route | 2 | Visible arrival | 12.90 s |
| North-to-center route | 3 | Visible arrival | 19.70 s |
| Original occlusion demo | 1 | Hidden traversal, visible arrival | 5.10 s |
| All cameras missing, 2–5 s | 2 | Stops after blind allowance, recovers and arrives | 7.10 s |
| Stop while moving + stale goal | 1 | Goal cleared; stays stopped | — |
| Invalid memory while moving | 2 | Goal cleared; stays stopped | — |

All **seven intended checks pass**: five arrivals and two braking controls.
Across 1,387 captures and 13,870 physics samples there are no contacts, boundary
violations, premature arrivals or nonzero requests beyond the one-second blind
allowance. All 5,378 recorded image artifacts pass integrity checks. Stop and
invalid-memory cases verify physical motion before the event, zero requests and
cleared goals afterwards, and final speed below the original 3 cm/s threshold.
The three-second dropout is observed in full, with 40 expired-loss captures
issuing zero and a fresh visible arrival after recovery.

There is one attempt per fixed validation case in the same synthetic room.
These checks do not estimate reliability across arbitrary goals or new rooms.
The interactive default is now `reserve-3cm`; `--control-profile baseline`
restores the previous settings. No policy/model weights or actuator changes are
part of this comparison.

The complete non-native regression suite passes **475 tests** (20 opt-in native
tests deselected); Ruff and frontend coordinate/syntax checks pass. After the
default change, the actual browser interface accepted `(-0.21, -0.87)` and then
`(-1.59, 1.41)`, displaying **Arrived visible** for each. The live worker manifest
records `reserve-3cm`. This browser check is additional interface acceptance;
the independently scored native evidence is the fixed validation set above.

## Calibration and limits

The [historical calibration report](M7_11_CALIBRATION.md) analyzes ten completed
runs and 2,364 captures. Its empirical braking time constant is about 0.3479 s,
close to the existing 0.35 s braking bound. Those braking bounds and all camera
uncertainty assumptions remain unchanged. Coverage is descriptive and limited
to correlated trajectories in one room.

The [offline map audit](evidence/m711-map-clearance.json) checks the frozen map
against eleven authored axis-aligned colliders. It supports testing a smaller
fixed margin in this fixture: free-cell centers already lie at least 10.18 cm
from those colliders. This geometry is scoring data only and is never supplied
to the online planner. It does not establish physical or unseen-map safety.

**Wider-path tradeoff:** the reserve can reject tight destinations that the
smaller search accepts. In one fixed-map sweep at a 5.3 cm position bound,
reachable sample targets fall from 72 to 53 out of 204 observed-free targets.
These are map-feasibility counts, not native arrival trials. The tuning path
lengths grow only about 1–2%, while avoiding corners with insufficient stopping
space. This is the best tested configuration under the declared completion/time
objective, not a globally optimal or minimum-clearance controller.

## Reproduce

```bash
./scripts/python.sh scripts/benchmark-control-profiles.py \
  --protocol configs/interactive/benchmark-protocol.json \
  --case tuning-reverse --profile reserve-3cm \
  --output work/my-profile-test --save-frames
```

Use a fresh output directory. The harness validates the frozen source asset
hashes, changes only the synthetic initial pose/goal in a copied demo bundle,
records the chosen parameters and source snapshots, and invokes the independent
audit after the worker exits. Stop/dropout/invalid-memory windows are fixed by
the protocol. The original failed configurations remain under `work/m711`.
