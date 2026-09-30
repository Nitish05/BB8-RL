# Navigation visibility and scene validity prototype

This opt-in continuation uses the existing checked synthetic room, frozen scan,
RGB observer and SAC waypoint follower. It does not replace the map, policy,
vision weights, Studio dependency or prepared runtime.

```sh
./scripts/launch-control-room.sh --visibility-planning --scene-validity
```

## Route candidates

`visibility_planning.py` projects uncertainty-expanded occupied scan volumes
through registered cameras into possible head-plane shadows. Frustum checks and
bounded A* account for consecutive predicted blind distance. The worker uses a
10 cm predicted blind-distance budget with a 10 cm/s reference speed; the pure
helper defaults to zero. These are declared planning heuristics, not calibrated
limits on actual unseen time. A slower robot, dwell or a different executed path
can remain unseen longer. Live localization expiry and uncertainty limits still
stop motion independently.

The scan covers heights 0–0.3 m. Absent higher geometry does not prove a clear
line of sight, and this calculation cannot guarantee learned-detector success.
Each proposed route still requires the existing full footprint/uncertainty/
reserve certificate. Rejected visibility routes cannot fall back to an unrestricted
planner. Search size, grid dimensions, geometry count and smoothing work are
bounded; stale map or calibration context is rejected.

## Same-session RGB guard

`scene_validity.py` requires three stable checks at 0.5 simulated-second intervals
before allowing navigation. It preserves the first reference rather than adapting
it to later images. Up to 128 image features and residual components at at most
320×240 distinguish distributed image displacement from localized changes.
Global brightness correction and conservative RGB-derived robot exclusions
reduce false alarms. These thresholds are engineered development settings.

Frame identity, calibration version, timestamp and freshness are validated every
capture. A first suspect structural check cancels the current goal and autonomy;
two consecutive changes latch invalidity. A clear subsequent image cannot restore
the cancelled goal. The ordinary bounded local reacquisition rules apply before
a new explicit request can move again. After restoring the original camera and
scene, **Recheck scene** starts an explicit stopped comparison against the same
reference. Three fresh matching structural checks must span at least one
simulated second. A changed or malformed sample rejects the attempt; Stop,
heartbeat expiry or a newer command cancels it. A later retry needs another
explicit request. The reference is never replaced by the altered scene.

The worker retains a persistent fault event separate from the presentation
queue. Recovery acknowledges the exact command generation and fault epoch only
after the post-render command drain and complete zero applied-command history.
The Supervisor clears its lock only for that acknowledgement and a clear event.
Delayed state from an earlier fault cannot undo an acknowledged recovery, while
a new fault relatches it. Reset and mode changes cannot replace a locked
reference. After recovery, local RGB reacquisition must complete; the cancelled
goal and autonomy remain cancelled until a new explicit request.

This is not saved-map validation at startup. Changes hidden from every camera,
visually indistinguishable objects, excluded robot/reference pixels and changes
below the image thresholds can remain undetected. Strong local lighting changes
can conservatively stop the system. There is no automatic reconstruction,
camera recalibration, global relocalization or physical-time safety claim.

The [27 September recovery results](SCENE_RECOVERY_RESULTS.md) cover explicit
restoration, active-motion invalidation and Stop/heartbeat cancellation. All
attempts, including test-driver failures, remain recorded.

## Evaluation contracts

`configs/interactive/navigation-heldout-v1.json` was frozen before tuning at
SHA256 `82ccfbe9e65c8a65b1ef454b69317087994a4b6257a56cf4a07af9f6ddd875fe`.
It defines 24 requests per variant across two new layouts, two camera/appearance
pairings, camera modes 1/2/3 and arrival/rejection requests. Both variants total
48 requests. Those four fixture families need independently scanned maps and
RGB-estimated registrations. The runner refuses substitution of the old map;
all 48 remain unexecuted until provisioning is completed.

A separate 24-request development comparison uses the existing room: original
difficult destination `(0.5, 1.5)`, original short destination, an occupied
destination and an evaluation-only camera move, each in modes 1/2/3 for both
variants. Arrival windows are 30 simulated seconds, occupied rejection 5 seconds,
and camera change 8 seconds. The camera moves during reference startup; that case
proves detection and lockout, not stopping an already moving robot. All renderer
changes stay outside the controller; the guard consumes RGB and RGB-derived
measurements only. No body pose setter is added.

Every run retains complete rows, RGB frames, renderer labels for scoring only,
exact requested goals, applied disturbance events, original/modified bundle
identities, source snapshots and final identity checks. The independent scorer
binds arrival to the requested destination and rejects generic unavailable/stale
evidence as proof of RGB change detection. Contacts, valid arrivals, localization
losses, rejected requests, interventions and unexecuted requests have explicit
denominators. No held-out result is used to tune these parameters.

An earlier 12-second development pilot is retained separately: zero contacts or
localization losses, near the goal but with arrival dwell incomplete, hence no
valid arrival. Its original metadata/source and independent re-audit remain in
`work/continuation-20260926-navigation/`; it is not counted in the main comparison.

Measured development results and all limitations are recorded in
[the navigation results](NAVIGATION_VISIBILITY_RESULTS.md) and the continuation
ledger. Both options remain off by default pending broader validation.
Live RGB-grounded entity interactions and persistent preference experiments are
the following stage; this work does not demonstrate personality.
