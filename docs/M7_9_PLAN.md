# M7.9 — Interactive camera navigation and GitHub delivery

Prepared 17 September 2026. This plan precedes implementation.

## Intended result

Launch one BB8-RL application, choose one, two or three fixed cameras, inspect
live RGB and the estimated scan map, click a clear destination, and watch the
existing SAC-backed guarded controller navigate. Stop and Reset remain available
while simulation or inference runs. The separate Genesis Studio checkout remains
an unmodified dependency. After implementation and verification, publish the
project with an illustrated, reproducible README and honest capability limits.

## Starting evidence and boundaries

- M7.8 demonstrates one short scan-once single-camera occlusion traversal with
  visible arrival; the original six routes still fail the map clearance gate.
- Multi-camera RGB capture, fusion and handover exist, but do not yet drive.
- The existing map has explicit observed-free, occupied and unknown cells. Do not
  turn missing observations into free space or use scoring geometry to repair it.
- The synthetic metric scan, RGB-estimated fixed-camera poses, unchanged SAC and
  vision checkpoints are the initial interactive demo assets. No hardware input
  or output, automatic scanning of a physical room, or real-time claim is added.

## Architecture and work ownership

1. **Application supervisor and local HTTP server (primary agent).** A loopback-only
   standard-library HTTP server serves a browser UI and validates all commands.
   Genesis runs in a spawned worker, keeping Stop, health and page requests
   responsive. Bounded queues/latest-state delivery prevent stale-frame backlogs.
   Same-origin checks, bounded JSON requests and a per-session command token
   protect the local command endpoint. Camera changes reset the episode.
2. **Native runtime and control integration (controller agent).** Connect accepted
   one-to-three-camera RGB estimates to `OccludedController` using the existing
   calibrated `CameraRig`. Feed acknowledged command intervals exactly as in
   M7.8. Runtime control accepts RGB, estimated calibration/map, target and command
   telemetry only; physics truth remains a separately labeled evaluation record.
   Stop cancels the route and clears queued actuation through the existing adapter.
3. **Interactive frontend (interface agent).** A responsive control room with mode
   selector, live feeds, estimated map, robot estimate/uncertainty, route, goal,
   status, Stop and Reset. Map coordinates use documented metric bounds and flip
   the canvas y axis correctly. Show free/occupied/unknown legend. Clicking a goal
   previews/submits it; rejected or unknown targets show the reason. A demo target
   provides a reproducible first run without suggesting every point is reachable.
4. **Independent verification (audit agent).** Exercise runtime/API failure cases,
   map-coordinate conversions, route rejection, stale observations, camera loss,
   Stop, Reset and camera-mode changes. Run actual native single/two/three-camera
   episodes, assess physical arrival independently and retain failures. Capture
   screenshots and GIFs from the real interface and simulator for documentation.

## Interface contract

- `GET /api/state`: session token, phase, mode (1/2/3), simulation time, estimated
  pose/radius, target, route, controller status, source camera IDs, per-view status,
  frame sequence, processing time, error/rejection message and asset readiness.
- `GET /api/map`: metric bounds/resolution and explicit cell classes, loaded from
  the saved scan map only. `GET /api/frame/<id>` returns the latest RGB JPEG.
- `POST /api/command`: `{action, ...}` with session token header; actions `goal`
  (finite metric `x,y`), `stop`, `reset`, `mode` (integer 1–3), `demo`, `heartbeat`.
- Goal/reset/mode generations invalidate stale work. Stop has priority over queued
  goals. Replacing a target first drains/brakes prior motion; no old controller
  resumes after Stop or reset. Browser heartbeat loss cancels motion.
- UI polls latest state rather than replaying an unbounded history; worker errors
  become visible status and cannot leave the interface reporting a running route.

## Assets and reproducibility

Create a portable, hash-checked demo bundle: the accepted map, three registered
camera calibrations/fixture positions, synthetic scene/task and frozen local
SAC/vision checkpoints. Remove machine-specific source paths from the portable
configuration while retaining provenance in a separate manifest. Generated runs,
datasets, environments and third-party dependencies remain ignored. Publish the
small executable demo bundle as a GitHub release asset, rather than committing
training outputs. The application offers a clear missing-assets message and an
explicit asset installation command; imports do not download files or run models.

## Acceptance sequence

1. Pure tests: coordinate round trips, goal schema, map rejection, mode limits,
   command ordering, Stop priority, heartbeat expiry and clean worker failure.
2. Native tests: click-equivalent accepted goal in each camera mode; single-camera
   occlusion; two-camera handover/dropout; all-view loss and invalid memory stop.
   Keep any rejected route or failed arrival visible in the report. Do not merge
   these new selected development cases with M7.8's original six-case denominator.
3. Browser interaction: launch, choose modes, click a goal, inspect route/feed,
   reject unknown/occupied clicks, Stop while moving, Reset, disconnect handling.
4. Full non-GUI suite, relevant native checks and Ruff. Recheck unchanged actuator,
   learned checkpoints and Genesis Studio source. No broad performance claims
   from a handful of correlated room trials.

## Documentation and publication

Rewrite the README around the actual user experience: concise purpose, real UI
hero image, native-motion GIF, quick start, demo assets, camera-mode examples,
architecture, evidence table, clear limitations, troubleshooting and research
links. Keep detailed milestone history in docs. Add a release guide and lightweight
CI for tests that need neither native graphics nor proprietary assets.

Before the first commit, inspect the explicit publish file list for credentials,
machine-specific paths, unwanted datasets, model-license obligations and large
binaries. Do not invent a software license or claim ownership of third-party work.
Create/push the requested repository only after code and documentation are ready;
publish the demo bundle and verify remote source, README media and release links.
If the user does not choose visibility, use a new private repository by default.

## Definition of done

Both camera pipelines operate through the same interactive controls; a valid
clicked goal can issue guarded native motion, while blocked goals and missing
evidence stop explicitly. Tests and native outcomes are reported accurately.
GitHub contains the reviewed project, illustrated README, launch instructions and
portable demo assets, with a verified repository URL returned to the user.
