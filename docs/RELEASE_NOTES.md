# BB8-RL 0.1.0 — Interactive camera navigation

The first repository release brings together the separate BB8-RL project and its
local control room. Choose one, two or three camera views, click a map destination,
inspect the estimated route and RGB feeds, and use Stop/Reset during native Genesis
simulation. Multi-camera position fusion now feeds the same guarded navigation
controller as the single-camera scan-memory workflow.

## Included

- Browser control room, metric coordinate input, camera mode selection and live RGB.
- Scan-derived free/occupied/unknown map, guarded SAC waypoint control and
  command-conditioned prediction through brief visual loss.
- Stop priority, stale-command cancellation, worker resets, browser heartbeat
  cancellation and loopback-only command endpoints.
- Independent native audit, regression tests, setup guide, screenshots and replay.
- A small portable demo ZIP with project-trained checkpoints, estimated map and
  three registered cameras. Installer checks the pinned archive and file hashes.

## Verified scope

All seven selected native worker cases passed: goal arrival in each camera mode,
Stop from motion, two-camera handover, all-view loss and invalid map memory.
The audit checked 758 captures, 2,822 RGB/mask images and 7,580 physics/command
intervals, with zero collisions or premature arrival declarations. These are
correlated development cases on one short path, not a general success rate.

Actual browser acceptance covered goal arrival in all three camera modes,
Stop/Escape from motion, Reset, unknown-goal rejection and heartbeat expiry with
no automatic resumption. The local non-native suite passed 423 tests; native
checks are reported separately. The README includes the real control-room
screenshot and native single- and two-camera replay GIFs.

The original six M7.8 routes still lack sufficient map evidence and remain
rejected. All results are synthetic, with known metric scan poses and lockstep
simulation. This release does not connect to hardware or claim real-time control.

## Start

Follow the README's source-checkout setup. Then run:

```bash
./scripts/python.sh scripts/install-demo-assets.py
./scripts/launch-control-room.sh
```

The installer requires authenticated access to this private GitHub release.
Models/datasets/third-party dependencies are not committed to the source tree.
The release ZIP does not include upstream reconstruction/matching model weights.
