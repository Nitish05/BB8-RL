# Optional local semantic candidate trial

Measured on 24 September 2026, Apple M5 Pro / 48 GB unified memory. This is an
**offline recorded RGB smoke test**, not an autonomous-personality benchmark or
an in-loop navigation validation. No motion was requested by this experiment.

## What is implemented

`bb8_rl.semantic_proposals` provides a dependency-free advisory boundary:

- `SemanticCandidate`: an opaque ID, description and observation evidence IDs.
- `SemanticRequest`: snapshot ID, authority epoch, observation/expiry times,
  JSON evidence, up to 24 offered candidates and up to three local images.
- `SubprocessCandidateRanker.rank(request, current=...)`: runs an optional model
  worker and returns a `SemanticChoice`. The only accepted output is
  `{"candidate_id":"<offered ID>"}` or `{"candidate_id":"wait"}`.
- Unknown IDs, coordinates/extra fields, duplicate JSON keys, markdown, invalid
  encoding, oversized output, failed workers and timeouts cause abstention.
- Snapshot, authority epoch and expiry are checked before and after inference.
  The callback must sample **current** authority, not a captured request value.

Requests are capped at 32 KiB and model output at 512 bytes. The process boundary
uses nonblocking pipes so a worker that never reads stdin cannot bypass the
deadline. A timed-out or overproducing worker is killed and reaped. Run inference
off the simulation/control thread; the ranker is a blocking advisory API.

The model cannot create coordinates or call motors. **The caller still owns
candidate certification, manual/Stop authority and final readmission.** An
accepted ID is a proposal, not permission to move. A new observation cannot be
silently substituted for the evidence used by a pending proposal. This module
does not install itself into the control-room loop.

`scripts/audit-semantic-proposals.py` adds an optional local MLX worker and an
offline manifest runner. Model imports are confined to the worker. The worker
loads an explicit local model directory with remote code disabled and Hugging
Face offline mode enabled. Images are limited to 16 MiB / 8 megapixels each,
converted to RGB and reduced to a maximum 512-pixel side while retaining aspect
ratio. Generation uses temperature 0, thinking disabled and at most 96 tokens.

## Verified artifact and runtime

The official [Qwen3.5-4B model](https://huggingface.co/Qwen/Qwen3.5-4B) is a
vision/language checkpoint licensed Apache-2.0. The downloaded artifact is a
**community conversion**, not an official Qwen quantization:

| Component | Exact tested choice |
|---|---|
| Download | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| Model revision | `32f3e8ecf65426fc3306969496342d504bfa13f3` |
| Weight file | 3,034,300,695 bytes |
| License | Apache-2.0; conversion card explicitly names Qwen/Qwen3.5-4B |
| Python | 3.12.13 |
| mlx-vlm | 0.7.3 |
| mlx / mlx-metal | 0.32.2 / 0.32.2 |
| transformers | 5.17.0 |
| Pillow | 12.3.0 |

The [conversion card](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit)
describes group-size-64 quantization and its conversion branch. The differently
named `Qwen3.5-4B-4bit` repository had no card at inspection, so the explicitly
documented artifact above was chosen. [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
supplies the Apple Silicon inference implementation. These sources establish
provenance and support; measured results below establish only this narrow test.

All optional packages, weights, metadata and generated evidence are under ignored
`work/semantic-trial/`. The application and Genesis environments were not changed.
The complete installed dependency freeze is retained in
`work/semantic-trial/requirements-lock.txt`.

## Measured results

Three historical simulator RGB files were visually reviewed: two frames from
`work/m75/multiview-data/scenes/094/`, plus
`work/m2-final-camera/camera.png`. Only RGB and an explicit attention request
entered the model. Segmentation, geometry labels and simulator pose were excluded.
The two nearby frames are correlated and are not two independent environments.

The task was to choose an offered visual-attention candidate when a green object
was visible, and wait when absent. Eight cases include single-frame and paired
image requests, repeated once. Expectations were manually defined from the RGB
images before inference. Candidate IDs have no motor meaning.

| Measure | Recorded result |
|---|---:|
| Strict valid responses | 8 / 8 |
| Expected choice, including absence | 8 / 8 |
| Model load after imports | 5.801 s |
| First inference, including initial compilation | 4.456 s |
| Warm single-image inference | 0.366–0.377 s |
| Warm paired-image inference | 0.511–0.512 s |
| Median inference across eight cases | 0.376 s |
| Maximum MLX allocator peak | 4.168 GB decimal |
| Process peak resident set | 3.637 GB decimal |
| Whole batch wall time | 13.527 s |

The RSS value is macOS `ru_maxrss`, measured in bytes; MLX memory uses
`mx.get_peak_memory()` after resetting the peak before each generation. These
overlap in unified memory and **must not be added**. They are not total computer
memory pressure. Model loading, image preprocessing and process startup are not
all included in each inference duration. No concurrent Genesis load was tested.
The warmed batch timings do not represent a new-process-per-request deployment.
Eight correlated samples do not support a meaningful p95 claim.

The first attempt failed because the sandbox exposed no Metal device. That
failure is preserved in `work/semantic-trial/sandbox-failure.json`. The subsequent
GPU-enabled local run is the source of the table, not a CPU fallback or mock.
Its complete prompts, image SHA-256 hashes and raw outputs are in `manifest.json`
and `results.json` in the same directory.

A follow-up challenge reversed candidate order, renamed the IDs and inserted an
untrusted observation note asking for coordinates and motor execution. All four
cases still returned the expected offered ID or wait, with no extra fields.
`challenge-manifest.json` and `challenge-results.json` retain that evidence.
These reuse the same three images; they are not four independent new scenes or
a comprehensive prompt-injection evaluation.

An additional end-to-end call through `SubprocessCandidateRanker` loaded the same
model in a fresh worker and accepted `attention-A` in 2.565 s, including startup,
with a 15 s deadline and current authority callback. This used warmed filesystem
and compiled-kernel caches, and the same historical RGB evidence. Its record is
`strict-interface-result.json`; it verifies the adapter path without dispatch.

## Reproduce the isolated setup

From the BB8-RL repository, create a separate environment; do not install these
packages into Genesis Studio:

```bash
scripts/python.sh -m venv work/semantic-trial/.venv
work/semantic-trial/.venv/bin/python -m pip install \
  'mlx-vlm==0.7.3' 'mlx==0.32.2' 'mlx-metal==0.32.2' \
  'transformers==5.17.0' 'Pillow==12.3.0'
HF_HOME=work/semantic-trial/hf-cache work/semantic-trial/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'mlx-community/Qwen3.5-4B-MLX-4bit',
    revision='32f3e8ecf65426fc3306969496342d504bfa13f3',
    local_dir='work/semantic-trial/model',
)
PY
```

Use the recorded local manifest, if present:

```bash
HF_HOME=work/semantic-trial/hf-cache scripts/python.sh \
  scripts/audit-semantic-proposals.py \
  --model work/semantic-trial/model \
  --manifest work/semantic-trial/manifest.json \
  --python work/semantic-trial/.venv/bin/python \
  --output work/semantic-trial/results.json --timeout 240
```

Recorded image datasets and model weights are deliberately not committed. A new
manifest can supply `model`, `dataset`, and 1–24 `cases`; each case has `name`,
optional `expected_ids` for scoring, and a serialized `SemanticRequest` in
`request`. The latter includes `snapshot_id`, `epoch`, `observed_at`, `expires_at`,
`observation`, `candidates` and `image_paths`. Each candidate has `candidate_id`,
`description` and optional `evidence_ids`. Expected labels stay outside model
input. Audit replay uses recorded timestamps; the live advisory API uses its
current-state callback instead.

For the strict subprocess adapter, use the same isolated Python and script with
`--worker --model <absolute local model directory>`. The worker accepts one
serialized request on stdin and returns only the raw model response on stdout.
Its startup is intentionally isolated; a persistent serving process with bounded
queues, cancellation and model re-admission needs separate integration testing.

## Acceptance and limits

The dependency-free tests exercise stale authority, expired snapshots, candidate
invention, duplicate keys, invalid schemas, output floods, non-reading workers,
crashes and absent runtimes using actual subprocesses. They do not pretend their
controlled output strings are model inference.

Color grounding is a useful feasibility check, but does not establish entity
tracking, camera calibration, semantic object identity, occlusion reasoning,
social understanding, preference learning or useful autonomous goals. A VLM
ranking an offered attention option does not generate personality. Before live
use, evaluate current camera frames under simulator load, longer and ambiguous
observations, outages and Stop/cancellation behavior. The learned preference
policy and persistent event memory must remain independently testable.
