# Recovering scan-derived passage coverage

The requested destination `(1.3554502369668247, -0.8446386255924172)` is
disconnected from the observed start `(-0.14399692380052198,
-0.062045330040032036)` in the accepted RGB map. Offline geometry diagnosis
finds a 52.38 cm opening; the map retains only a 6 cm free strip there. Removing
the controller's planning reserve cannot repair that missing map evidence.

## Fixed boundaries

- Keep the separate BB8-RL project, existing actuator, SAC/vision checkpoints,
  camera-registration estimates, and `reserve-3cm` controller settings.
- Reconstruction consumes scan RGB and declared synthetic metric calibration.
  Original reserved views 14, 17, 20 and 23 remain excluded from mapping inputs.
  Simulator geometry/depth/labels are never reconstruction or controller inputs.
- Preserve unknown blocking, occupied conflicts, the 5 cm projection guard,
  three separated supporting views, and existing floor-mask thresholds.
- Retain every failed candidate. A whole-map, full-volume independent audit
  must report zero false-FREE prisms before native motion with a candidate.
- Preserve original independent arrival checks: 10 cm distance, 3 cm/s speed,
  0.5 s physical dwell and fresh visible RGB. Count rejected goals as failures.

## Execution

1. **Freeze coverage and driving cases.** Include the exact user request, all
   six original M7.8 cases and the M7.11 validation destinations. Preserve the
   original case IDs even when cases share endpoints or have dropout variants.
   Report map connectivity separately from demonstrated native navigation.
2. **Build one conditional column candidate.** Reuse the frozen intersection of
   learned, palette and floor-plane RGB evidence. Test the complete expanded
   ground footprint instead of requiring every projected elevated voxel to
   look like floor. Extend a verified footprint vertically only under an
   explicit assumption: every obstacle column extends continuously to the
   support plane, with no floating objects or overhangs. This is a synthetic
   box-room assumption, not a general indoor reconstruction guarantee. Keep
   existing occupied conflicts and do not fill missing masks or tune thresholds
   from authored geometry.
3. **Independent promotion gate.** Audit full free prisms, input/source hashes,
   support ancestry, excluded views, occupancy preservation and the fixed
   route set. If the candidate fails, retain it and improve positive scan
   evidence using additional translated views or a better geometric estimator;
   never patch cells from scoring geometry.
4. **Native validation.** Package a fresh candidate bundle. Test the reported
   route with one, two and three cameras, relevant old routes, demo/occlusion,
   Stop and invalid-memory controls. Score all attempts and images independently.
   Investigate any newly exposed controller issue without weakening the gates.
5. **Integrate and publish only the validated result.** Update the local asset
   default through a checksum-verified bundle, preserve the old bundle, verify
   click-to-go in the browser, document assumptions and failures, and publish
   code plus the versioned asset release to the existing private repository.

Work is split between a reconstruction agent, an independent audit agent and
the root agent handling integration, native execution, documentation and release.
Native rendering jobs run serially. No general success-rate claim follows from
these selected routes in one synthetic room.

## Recorded first-candidate failure and second experiment

The first conditional-column candidate adds 4,508 FREE cells, but the unchanged
independent geometry audit finds 184 free prisms intersecting static solids.
It also fails to connect the reported destination. It is rejected; its cells
are not corrected using those geometry findings.

The second experiment returns to complete expanded 3D-prism evidence and adds
12 calibrated RGB captures from one camera moved over a fixed overhead grid.
Their positions are fixed in `capture-plan.json` before rendering. The 20 old
mapping views are retained byte-for-byte; original query views remain excluded.
The new views use the same learned thresholds and geometric agreement gates,
and occupied conflicts remain intact. Supplemental capture hashes are bound
before candidate scoring. This tests stronger scan coverage, not a weaker
clearance rule. The robot stays parked during scanning, then navigation uses
the same one-to-three fixed live cameras as before.

## Stricter supplemental evidence

The first full-prism extension connects the reported route at a 14 cm search
radius, but its independent whole-map audit finds three final free prisms
intersecting solids. This candidate is also rejected before driving. A separate
revision requires four mutually separated image supports and a two-pixel
projection guard for every new free cell. Original accepted free evidence and
occupied conflicts remain intact. The revision is frozen before its build;
no failed-cell coordinates are supplied to the reconstruction agent.

## Filling gaps under the stricter rule

The stricter 32-view candidate has zero audited false-free prisms and preserves
all baseline evidence, but still fails the required 14 cm route-connectivity
gate. It is not driven or installed. A fourth candidate adds a globally uniform
16-view grid offset by half a grid spacing, with x/y coordinates -1.5, -0.5,
0.5 and 1.5 m at 3.0 m height. Its capture plan is frozen before rendering.
The old 32 images, calibration records and masks stay unchanged. New cells retain
the four-view/two-pixel evidence rule. Supporting-view storage is extended from
32 to 64 bits with backward-compatibility and high-bit round-trip tests; no view
may silently disappear from provenance.

## Completion record

The fourth candidate passes the independent static gates and adds 25.2% more
observed free space. A complete first native family exposes three route rejections
from clearance-grid quantization; those failures remain retained. Correcting the
search without reducing the actual requirement produces a complete passing
second family: 12/12 intended checks, including ten valid arrivals. The one-camera
browser flow also reaches the reported destination. The
[results report](M7_12_SCAN_COVERAGE.md) contains metrics, failed attempts, replay
media, installation instructions and the experiment's limits.
