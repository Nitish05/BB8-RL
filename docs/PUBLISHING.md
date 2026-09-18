# Publication and demo assets

The Git repository contains project source, tests, authored primitive scenes,
configuration, selected documentation media and milestone reports. It excludes
environments, datasets, native run logs, training outputs, downloaded upstream
repositories and upstream model weights.

The `v0.1.0` release carries `bb8-demo-assets-v1.zip`. Its checksum is pinned in
`configs/interactive/demo-assets.json`. The archive contains this project's
trained SAC/vision checkpoints, accepted estimated scan memory, three registered
camera configurations, and the synthetic scene. Each file also has an internal
SHA256 entry. The installer validates paths and checksums before extraction is
accepted. Model loading happens only in the explicit simulation worker.

Rebuild locally with `scripts/build-interactive-assets.py --scan-dir <capture-dir>
--output <fresh-directory>`, then independently validate the resulting bundle.
Do not replace the pinned archive silently; publish a new version and update the
checksum when inputs change. The original research reports and failed runs remain
in the prepared workspace; selected results in the README must keep their scope.

Upstream model licenses differ. The portable demo does not redistribute
SuperPoint, LightGlue, MapAnything or Dreamer implementation/model downloads.
Research setup documents retain their upstream links and license notes. BB-8 is
a descriptive project subject; the included robot uses schematic primitives and
is not an official character model or an endorsement. No project software license
has been selected; repository visibility alone does not grant a reuse license.
