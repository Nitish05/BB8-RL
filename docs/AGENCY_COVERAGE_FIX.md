# Preventing repeated two-point exploration

Validated 24 September 2026. This fixes the observed shuttle between `(0.5, 1.5)`
and `(0, 1)`. The first updated three-camera native run reached three different
destinations without a successful revisit. This is a bounded place-exploration
baseline; it does not establish personality or meaningful autonomous interests.

## Why the shuttle occurred

Every successful arrival supplied utility `+1`. Repeating an easy trip therefore
pushed its learned value toward `1`, while the novelty bonus could contribute at
most `0.12`. The three-second arrival cooldown expired during an ordinary trip.
The runtime also offered only five successful routes among its twelve nearest
anchors, which could prevent farther unvisited places from reaching the selector.

The previous native acceptance rule required three valid arrival episodes. It
therefore accepted `A → B → A`, although that was only two different destinations.

## Updated policy

The runtime excludes previously reached anchors **before** ordering and limiting
route checks. It reads the existing SQLite arrival counts, so completion survives
app restarts and applies to memory created before this change. No migration or
deletion of the user's experience is required. Completion remains scoped to the
existing agent and map identity.

Route search checks at most twelve remaining anchors per decision and continues
with later candidates on subsequent frames when the first page finds no usable
route. When no reachable unvisited target is available from the current position,
exploration pauses explicitly. It does not erase completion history or begin an
automatic second lap. Rejections, cancellation and localization loss do not mark
an anchor completed.

The existing value selector still ranks the offered unfinished targets. The
non-revisit rule is an explicit coverage constraint, not a learned personality
trait. Manual goals remain available for revisiting a place. Route certification,
SAC movement, localization loss, Stop and heartbeat controls are unchanged.

Visiting an anchor does not move the fixed cameras or by itself acquire a new
view of the room. These counts describe completed navigation targets, not a
percentage of map reconstruction or semantic scene understanding.

## Test protocol

Regression tests cover old high-value two-point memory after reopening SQLite,
search beyond twelve blocked nearby routes, persistent exhausted coverage,
unfinished failed targets and six distinct simulated terminal arrivals. Existing
agency tests retain cancellation, Stop, heartbeat and localization-loss checks.

The native harness now stops supervision after the requested number of distinct
reported destinations, then uses the independent physics audit to verify them.
Acceptance requires the requested number of **distinct verified destinations**
and **zero successful revisits**. Episode count and unique intention IDs remain
separate measurements. Eight pure audit tests include `A → B → A` failure,
`A → B → C` success, and failure when a revisit occurs before eventually reaching
a third destination. An unverified reported arrival cannot supply a missing goal.

Native arrival verification retains fresh RGB localization, true distance at most
10 cm, speed at most 3 cm/s and 0.5 seconds of continuous physics dwell. Truth and
segmentation are scoring inputs only, read after decisions. Stop must produce
zero requested actions and disabled exploration throughout the observation
window, with physical speed at most 1 mm/s during its final half second.

## First native result

One run, three cameras, fresh independent memory, unchanged synthetic room. The
bounds were three distinct destinations, 120 simulation seconds or 600 wall
seconds before Stop. No retry or selected successful replacement was used.

| Measure | Result |
|---|---:|
| Independently verified arrival episodes | 3 |
| Distinct verified destinations | 3 |
| Successful revisits | 0 |
| Unique intentions / selected places | 3 / 3 |
| Collision / boundary flags | 0 / 0 |
| Premature or unobservable arrival frames | 0 |
| Capture frames / recorded simulation duration | 490 / 24.50 s |
| Worker wall duration, including startup | 62.78 s |
| Stop observation duration | 1.05 s |
| Final half-second maximum physics speed | 0.000665 m/s |
| Zero requested actions and disabled exploration after Stop | Passed |
| Source files unchanged during the run | Passed |
| Worker errors / audit integrity errors | 0 / 0 |
| Revised exploration acceptance | Passed |

| Arrival time, simulation | Destination, metres | Independent gate |
|---|---|---|
| 7.25 s | `(0.5, 1.5)` | Passed |
| 15.25 s | `(0, 1)` | Passed |
| 23.40 s | `(-0.5, 1.5)` | Passed |

The third trip now visits a different target. This short run verifies the reported
shuttle regression; it does not demonstrate whole-room coverage or single-camera
performance. The earlier single-camera localization-loss limitation remains.

The full non-GUI suite passed **664 tests**, and all three frontend test suites
passed. A separate browser smoke check retained the existing 22 recorded
experiences across the server update. It selected and reached `(-0.5, 0.5)`
before choosing `(-0.5, 1.5)`, instead of returning to the completed shuttle
points. The session was paused after verification. These UI observations are
separate from the fresh-memory native audit above.

## Reassessment of the old result

The retained [first native report](AGENCY_NATIVE_RESULTS.md) recorded three valid
trips to `(0.5, 1.5)`, `(0, 1)`, and `(0.5, 1.5)` again at 23.30 seconds.
Read-only rescoring of its saved arrival episodes with the revised helper gives
**two distinct verified destinations, one successful revisit, and failure**.
Its raw evidence is retained unchanged. Successful movement between two places
was integration evidence, not a successful exploration-coverage result.

The synthetic utility-reversal benchmark remains a separate test of the value
learner. Its fixed observer position and supplied outcome schedule cannot
establish physical exploration quality.

## Reproduction and evidence

Run from the BB8-RL repository:

```bash
scripts/python.sh -m pytest tests/test_agency_coverage.py tests/test_agency_navigation_audit.py -q
scripts/python.sh scripts/benchmark-agency-navigation.py \
  --output work/agency-coverage/new-independent-run \
  --modes 3 --arrivals 3 --sim-seconds 120 --wall-seconds 600
```

The first result is retained under
`work/agency-coverage/first-native-20260924/`, including the fixed protocol,
copied harness source, per-frame rows, RGB evidence, physics audit, Stop report,
worker source hashes and summaries. The old result remains under
`work/agency-native/first-integration-20260924/`. Generated evidence and SQLite
memory stay outside version control.
