# Publication and demo assets

The Git repository contains project source, tests, authored primitive scenes,
configuration, selected documentation media and milestone reports. It excludes
environments, datasets, native run logs, training outputs, downloaded upstream
repositories and upstream model weights.

The current `v0.1.1` release carries `bb8-demo-assets-v2.zip`; the original `v0.1.0`
release and `bb8-demo-assets-v1.zip` remain available. Its checksum is pinned in
`configs/interactive/demo-assets.json`. The archive contains this project's
trained SAC/vision checkpoints, accepted estimated scan memory, three registered
camera configurations, and the synthetic scene. Each file also has an internal
SHA256 entry. The installer validates paths and checksums before extraction is
accepted. Model loading happens only in the explicit simulation worker.

Rebuild locally with `scripts/build-interactive-assets.py --scan-dir <capture-dir>
--memory <audited-memory-directory> --output <fresh-directory>`, then independently
validate the resulting bundle. The demo ZIP contains runtime map artifacts,
models and camera registration, not the complete raw scan/reconstruction history.
Do not replace the pinned archive silently; publish a new version and update the
checksum when inputs change. The original research reports and failed runs remain
in the prepared workspace; selected results in the README must keep their scope.

The continuation adds a portable, checksummed input-manifest route. Capture the
existing developer inputs once, or author a manifest against
`configs/interactive/asset-inputs.schema.json`; future rebuilds use explicit
paths/hashes without depending on the original checkout location:

```bash
./scripts/python.sh scripts/build-interactive-assets.py \
  --inputs /path/to/asset-inputs.json --input-root /relocated/input/tree \
  --validate-inputs-only
./scripts/python.sh scripts/build-interactive-assets.py \
  --inputs /path/to/asset-inputs.json --input-root /relocated/input/tree \
  --output work/fresh-assets
```

The manifest covers the policy, vision checkpoint, recursive scene configuration,
map artifacts, camera metadata, registration and selected protocol. Preflight
finishes before creating output. Archive member ordering and metadata are fixed;
relocated fixture inputs with changed filesystem mtimes produce identical ZIP
hashes on the tested runtime. Compression across different library versions is
not claimed byte-identical. Models are copied/hashed, never loaded by the builder.

Current source package metadata is **0.1.2.dev0**, explicitly unreleased. The
pinned runtime asset release remains **v0.1.1/v2**; no replacement assets or new
release were published by this continuation. A wheel was built and installed into
a fresh ignored target directory, then imported outside the checkout against the
existing prepared dependencies. That is a packaging smoke test, not a clean-machine
native installation. Full recent purpose/camera evidence remains local; its
checksum inventory and compact results accompany the continuation report.

Hosted CI now discovers all public-dependency non-native tests and mutation-tests
independent auditors. The complete non-native job needs a repository Actions
secret `GENESIS_STUDIO_READ_TOKEN` with `contents:read` for the private
`Nitish05/Genesis-Studio` repository. With no secret, it reports **NOT RUN** and
lists the omitted Studio contracts. No local credentials are uploaded or copied.

The optional `bb8-scan-evidence-v1.zip` is separate from the runtime bundle.
It includes frozen scan inputs, masks, all rejected maps, the accepted map,
source snapshots, aggregate native reports, reconstruction instructions and an
internal SHA256 index. The release publishes `SHA256SUMS` for both ZIP files.
Its source commit is recorded in the evidence index. Extract this archive only
at the root of a fresh checkout: replay requires restoring the original v1
baseline at `work/interactive-assets`, so extracting over a working v2 app would
replace its map. Full native capture logs remain local; aggregate reports and
selected actual RGB media are published with their experimental scope.

Upstream model licenses differ. The portable demo does not redistribute
SuperPoint, LightGlue, MapAnything or Dreamer implementation/model downloads.
Research setup documents retain their upstream links and license notes. BB-8 is
a descriptive project subject; the included robot uses schematic primitives and
is not an official character model or an endorsement. No project software license
has been selected; repository visibility alone does not grant a reuse license.
