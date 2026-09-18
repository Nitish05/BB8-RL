# MapAnything M7.6 isolated experiment

This experiment performs actual pretrained inference on the existing synthetic RGB scan. It supplies known synthetic metric intrinsics and camera poses, and does not supply depth, segmentation, object geometry, or the held-out final-perch image. It is an offline reconstruction experiment, not a closed-loop navigation success demonstration.

## Official sources and exact pins

- [Official MapAnything code](https://github.com/facebookresearch/map-anything), commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
- [Official Apache checkpoint](https://huggingface.co/facebook/map-anything-apache), revision `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`.
- [Official DINOv2 code](https://github.com/facebookresearch/dinov2), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- `model.safetensors` SHA256: `fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201` (approximately 4.6 GiB).
- `config.json` SHA256: `65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5`.

The runner checks source commits and both checkpoint file hashes. DINOv2 architecture loading is redirected within a narrow `torch.hub.load` context to the pinned local checkout, with encoder pretraining disabled because the full Apache checkpoint already contains those weights. Upstream source and checkpoint weights are unchanged. No unpinned secondary encoder download occurs during inference.

## Runtime isolation and reproduction

All environments, third-party code, model weights and generated outputs live under `work/m76/mapanything`, which is ignored by version control. Existing simulation/training environments are unchanged. Tested Python is 3.12.13 on Apple Silicon macOS, with Torch 2.8.0, torchvision 0.23.0, MapAnything 1.1.4, UniCeption 0.1.7, NumPy 2.5.3, safetensors 0.8.0 and huggingface_hub 1.32.0. A complete actual package freeze is stored in each inference run's `dependencies.txt`, for example `run-8/dependencies.txt`; it includes an absolute editable MapAnything checkout path that must be adjusted when recreating the environment elsewhere.

From the BB8-RL repository, the installation used this isolated structure:

```sh
.venv-dreamer/bin/python -m venv work/m76/mapanything/venv
git clone https://github.com/facebookresearch/map-anything.git work/m76/mapanything/vendor/map-anything
git -C work/m76/mapanything/vendor/map-anything checkout 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
git clone https://github.com/facebookresearch/dinov2.git work/m76/mapanything/vendor/dinov2
git -C work/m76/mapanything/vendor/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
work/m76/mapanything/venv/bin/python -m pip install --upgrade pip
work/m76/mapanything/venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 -e work/m76/mapanything/vendor/map-anything
work/m76/mapanything/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='facebook/map-anything-apache',
    revision='00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a',
    allow_patterns=['config.json', 'model.safetensors', 'README.md'],
    local_dir='work/m76/mapanything/checkpoint',
)
PY
```

The commands above reproduce the tested direct constraints; use the saved `dependencies.txt` for exact transitive versions. Do not install this optional stack into the existing simulator runtime.

For a fresh output path (the runner refuses to overwrite one):

```sh
export BB8_PYTHON="$PWD/work/m76/mapanything/venv/bin/python"
export MPLCONFIGDIR="$PWD/work/m76/mapanything/mpl-cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=2
scripts/python.sh scripts/reconstruct-mapanything.py \
  --scan /path/to/native-scan/view-probe \
  --runtime-root "$PWD/work/m76/mapanything" \
  --output "$PWD/work/m76/mapanything/example-new-run" \
  --indices 0 3 6 9 12 15 18 21
```

The recorded runs use MPS, FP32, official fixed-mapping resize to 518×392 pixels, memory-efficient dense heads with minibatch size 1, non-ambiguous/edge masks, and no confidence-percentile or multiview-consistency filtering. `PYTORCH_ENABLE_MPS_FALLBACK=1` permits an unsupported operation to fall back to CPU; no such fallback warning appeared in the successful logs. The upstream FP32 autocast warning means autocast was disabled, as requested, and is not an inference failure.

## Camera coordinates and honest metric scale

Input JSON intrinsics are pinhole OpenCV intrinsics. `world_to_camera` is inverted to obtain OpenCV right/down/forward camera-to-world matrices; the adapter verifies proper rigid transforms. Only RGB, K, C2W, and `is_metric_scale=True` reach the model. The official preprocessor jointly crops/resizes RGB and K. `scan-023` is reserved for independent registration and is rejected as a reconstruction input by default.

The model encodes input poses and metric baseline as learned priors, but predicts its own depths, rays, camera poses and global metric factor. Those outputs do not exactly satisfy the supplied poses. In the eight-view run, raw model geometry had a scale bias: predicted scan-camera baselines were too small even though the correct synthetic metric prior was supplied.

The adapter now also computes a proper least-squares Sim(3) between **predicted scan-camera centers and the already supplied scan-camera centers**. It accepts no surfaces, objects, floor heights or held-out-camera pose. Run-8's camera-derived scale is `1.1055686794202917`; center residuals are median 0.01939 m, p95 0.08973 m and maximum 0.09957 m. This is explicit post-inference enforcement of known calibration, not metric accuracy recovered from RGB alone. Raw predictions are retained unchanged. Root's independent geometry evaluation remains necessary; the correction does not establish collision-safe geometry.

The initially recorded runs can be postprocessed without a new inference:

```sh
scripts/python.sh scripts/reconstruct-mapanything.py \
  --pose-align-existing work/m76/mapanything/run-8 \
  --output work/m76/mapanything/example-new-pose-alignment
```

Future inference runs automatically export the alignment alongside raw geometry. `run-8-pose-aligned` preserves the initial run and references its manifest and NPZ hash.

## Artifact contract

Every successful run has `reconstruction.npz`, binary PLY point clouds, `manifest.json`, `dependencies.txt` and a preview. NPZ uses non-pickled numeric/string arrays. Masks remain required even though dense point arrays include invalid locations.

| Field | Meaning |
| --- | --- |
| `rgb_uint8[V,H,W,3]` | Actual processed RGB, aligned with each dense array |
| `depth_z_m[V,H,W]` | Raw predicted optical-Z depth with the model's own metric scale |
| `confidence[V,H,W]` | Uncalibrated learned score, not covariance or collision probability |
| `valid_mask[V,H,W]` | Upstream ambiguity/edge mask plus finite positive depth validation |
| `points_world_calibrated[V,H,W,3]` | Raw predicted Z reprojected using exact supplied processed K and C2W |
| `points_world_model_anchored` | Raw model points rigidly anchored by the first input camera only |
| `depth_z_pose_scale_m` | Predicted Z multiplied by the camera-center-derived Sim(3) scale |
| `points_world_calibrated_pose_scale` | Corrected Z reprojected using exact supplied K/C2W; paired with `depth_z_pose_scale_m` |
| `points_world_model_pose_aligned` | Model's own global points transformed by the full camera-derived Sim(3) |
| `camera_to_world_input`, `intrinsics_input_processed` | Supplied synthetic calibration after appropriate preprocessing |
| `camera_to_model_predicted`, `intrinsics_predicted` | Actual estimated camera geometry, preserved for audit |
| `world_from_model_similarity` | Derived Sim(3), including scale; it is not a rigid camera pose |

`view_ids` identify scan frames. The manifest reports array shapes, pins, image/calibration hashes, known-input conditioning, timing and memory, alignment residuals, and explicit absence of free-space claims.

## Resource limits and overlapping batch coverage

Four- and eight-view joint inference succeeded. The initial eight-view inference took 12.11 s, with 30.34 s total including loading/export and 10.31 GB process peak RSS. MPS driver allocation at the end was 23.11 GB, including cached allocator memory; this is a separate measure, not process RSS or an isolated peak GPU benchmark. Other project CPU work may have run concurrently, so these are observed offline elapsed times.

The 23-view FP32 attempt stopped at `Invalid buffer size: 50.87 GiB` in attention; `run-23/manifest.json` and its log preserve that failure. A full FP16 attempt stopped before inference because a high-watermark of 0.9 conflicted with the default low-watermark 1.4; it does not constitute an FP16 quality or performance result. That failure is preserved in `run-23-fp16` and no further full-attention tuning was pursued.

The attempted bounded coverage schedule used independent overlapping eight-view windows 0–7, 5–12, 10–17 and 15–22. Windows 0–7 and 5–12 succeeded. Windows 10–17 and 15–22 produced degenerate estimated camera centers and were rejected by the Sim(3) rank gate. The one diagnostic retry of 10–17 preserved raw arrays and reproduced the failure: all eight camera centers were identical, every depth pixel was 1.4033545 m, every confidence was 1.8873966, and recovered predicted focal length fx was negative. These outputs are not accepted reconstruction evidence. The cause of this collapse is unresolved; it may require a separate runtime/model investigation. No surface-fit rescue, relaxed gate, or further retry was performed.

`batch-10-17-raw-preserved` and `batch-15-22-raw-preserved` contain raw output NPZ files and failed manifests. The first failed 10–17 run predates the raw-save fix and contains its failure manifest/log only. Successful subsequent inference also preserves `reconstruction-raw.npz` before camera-prior enforcement.

The final `merged-valid-16` artifact combines the globally distributed eight-view run first, followed by the two successful consecutive windows, in that fixed order. It contains **16 unique views**: 0–12, 15, 18 and 21. Missing reconstruction indices are **13, 14, 16, 17, 19, 20 and 22**; scan-023 remains deliberately held out. The globally distributed source covers the room's viewing angles, but this does not imply complete surfaces or all 23 scan frames. The merge keeps the first occurrence of overlapping frames and preserves per-view batch provenance; it does not claim a joint 23-view attention pass or a globally optimized fused map.

```sh
scripts/python.sh scripts/reconstruct-mapanything.py \
  --merge-batches work/m76/mapanything/run-8-pose-aligned \
    work/m76/mapanything/batch-00-07 work/m76/mapanything/batch-05-12 \
  --output work/m76/mapanything/example-new-merged-valid-16
```

Merged model-frame arrays remain batch-local. Use exported world-frame arrays, and consult `source_batch_index`, `source_batch_view_index`, per-view similarity matrices and the source manifests when auditing. Never interpret a missing surface as observed free space. Geometry acceptance and held-out registration are separate downstream gates.
