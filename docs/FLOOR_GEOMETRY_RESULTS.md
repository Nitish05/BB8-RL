# Floor-height ambiguity continuation — 27 September 2026

Implemented and tested a rejection-only height-ambiguity filter, then declined to promote it because it removed every useful floor claim from the positive control. The original new-room navigation gap remains open. No new runtime capability or native arrival is claimed by this phase; the earlier live RGB interaction-learning slice remains available.

## What changed

A fresh plan and three parallel independent reviews preceded implementation. `height_ambiguity.py` tests fixed source-camera rays against 14 declared elevated planes, using only RGB and camera calibration. It counts compatible independent partners at each height separately. A demonstrated alternative can remove a candidate floor pixel; absence of a sampled match cannot prove that the pixel is floor. The filter adds no positive evidence and is retained under ignored work, outside the production estimator.

The new evaluator freezes code, runtime, weights and control inputs before execution; invokes the actual production centered estimator as its baseline; intersects each result with the unchanged learned floor model; and scores every claimed cell's complete body volume. Independent review fixed comparator wiring, per-mode staging, mandatory uniform-scene abstention, and RGB/calibration immutability before the run. No experiment result was used to tune thresholds.

## Frozen controls and measured results

Sixteen analytic RGB cases were frozen before candidate evaluation: two exposed information anchors and fourteen new variations. These are ideal synthetic image controls, not native Genesis or physical tests. The strict model-control result is separate from route utility.

| Estimator | Cases executed | Cases blocked | Positive checker FREE cells | False-FREE cells | Result |
|---|---:|---:|---:|---:|---|
| Production centered |16|0|140|0|Passes this bounded control set |
| Previously rejected covered-window |16|0|3,024|258|Rejected |
| Covered-window plus height veto |2|14|0|0|Uninformative positive; not admitted |

The additional fine-checker positive produced 68 centered and 2,446 covered FREE cells. The centered and covered references produced no FREE claims in all three pixel-identical uniform scenes—empty, low raised and high raised. The veto evaluated only the empty uniform anchor; its two raised variants were blocked. The centered rule's control pass does not repair its missing original goal footprint or establish general safety.

The covered rule failed six new cases: low block 53, high block 53, texture gap 38, texture-flush surface 89, thin rail 4, and missing-pixel case 21 false-FREE cells. Their sum 258 counts cell claims across separate synthetic scenes, not unique real obstacles. All failures and raw masks remain preserved.

The new filter ran only the two anchors. Because its positive map was empty, the frozen information gate blocked its fourteen new cases. Those cases are not reported as safe passes, and their reference results now make them exposed development data for future model selection. Neither the original eleven controls nor the original 48-request navigation ledger was overwritten.

## Why the original goal remains blocked

The retained 60-view scan was probed at the unchanged goal (0.8, 1.3). Fifty source views have at least two separated partners compatible with the floor hypothesis. All 50 also retain at least two of those same partners for a 12 cm raised alternative along the fixed source ray. Of 51 floor-palette goal views, 45 local 3×3 patches are exactly flat; six have small variation, with maximum grayscale standard deviation 0.1322. None approaches the existing texture threshold 3. Wider patches can locate a neighboring checker edge without establishing the height of each uniform interior pixel.

These are local finite-height diagnostics, not a proof that the entire scene is globally indistinguishable. The raw-bilinear probe predicts that local height rejection will not recover the missing footprint, but no new full native-scan veto map was reconstructed; only the analytic anchors exercised the complete veto pipeline. A corrected texture-statistics note is retained alongside the original report; no raw diagnostic result was changed.

A separately frozen, geometrically motivated steeper-camera proposal reduced projected prism extension from about 11.3 cm to 7.65 cm. All 24 predictions were retained. None had full necessary goal-cell source support in at least two of its three retained RGB references; no three-view clique existed. These were planar planning reprojections, not new captures, and no projected image entered a map. No native acquisition was justified or run.

An independent one-micrometre raised-checker example also produced byte-identical images to the empty-floor control. This exposes a limit of the strict “every positive-volume obstruction” claim with finite RGB quantization; it is not a practical robot hazard or permission to ignore obstacles. No minimum obstacle height or floor tolerance was invented to obtain a pass.

## Validation and preservation

The pre-evaluation suite passed 46 meaningful helper tests: 20 height-veto, 7 native-RGB probe, 8 frozen-control and 11 evaluator tests. These test counts are separate from the prior phase's 1,536 production tests. Production source did not change, so that full suite and native regression were not presented as new runs here. Sixteen independent audit contracts also pass, bringing the unique helper-test total to 62. The post-run audit reconstructs every saved FREE grid and checks image/calibration hashes, mask algebra, scalar complete-volume intersections, result denominators and frozen identities; final totals and hashes are in `validation.json`.

The current 260-file source/asset/memory preservation inventory and additional historical evidence inventories are checked independently. Existing environments, installed bundle, model weights, production memory, native captures and failed experiments remain in place. Genesis Studio remains a separate unchanged dependency. No installation, download, commit, push, environment reset or memory reset occurred.

Fresh implementation, plans, controls, logs and audit artifacts are under `work/continuation-20260927-floor-geometry/`. The scientific plot is a diagnostic visualization, not a FREE-map or motion certificate.

## Remaining work and acceptance gate

The same six development navigation requests remain unexecuted. Original start (-.15, .25), arrival (.8, 1.3), occupied (.3, 1.1), .10/.14m radii, .05m projection margin, pixel guard 1 and full-prism three-view rule are unchanged. No candidate bundle was built. Earlier live visual-learning/native ledgers are unchanged and were not rerun.

The next implementation needs evidence of geometry and visibility that can preserve informative floor while rejecting raised surfaces. Another wider-window rule, a best-height score, a finite-sweep “no match” certificate, or a camera batch without a predicted source-support mechanism would repeat already rejected approaches. Any new geometry model or measurement route must first show informative positive and raised-object controls under an explicit observable-data contract, with new cases frozen before tuning; it must not invent a minimum hazard height, fill UNKNOWN, borrow truth, or move the goals. Only then should the original 60-view map be rebuilt, independently scored, and both exact certified routes tested before native requests.

Broader arbitrary visual semantics, cross-room identities, automatic changing-room recovery and hardware remain longer-term gaps. The live learning slice still uses engineered entities and resource motivation; this is not demonstrated personality or physical realtime operation.

## Launch and reproduce

The established visual experiment launches from the existing repository with:

```sh
cd /Users/rrnitish/Documents/Codex/BB8-RL
./scripts/launch-control-room.sh --agency-mode visual --mode 3 --visibility-planning --scene-validity
```

The default memory is `work/visual-agency/bb8.sqlite3`; Stop/Reset revoke motion authority and preserve committed knowledge. Use a separate `--agency-memory` path for a new isolated experiment.

The frozen evaluation is verified with `PYTHONDONTWRITEBYTECODE=1 ./scripts/python.sh work/continuation-20260927-floor-geometry/evaluate.py verify`. Its `run` command intentionally refuses to overwrite existing evidence. A rerun needs a separately named, bound output experiment; do not delete this run to reuse its directory.
