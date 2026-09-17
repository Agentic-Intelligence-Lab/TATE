# ARX EEF Pi0.5 Training

This directory builds and trains the final ARX LeRobot repositories. The
training model consumes 16-D end-effector (EEF) state/action, rather than the
source 14-D joint vectors.

## 1. Data flow

```text
Ego raw LeRobot videos
  -> batch_preprocess (EEF variant -> correction -> IK -> package)
  -> outputs/experiments/<experiment>/datasets/<ego_variant>

ARX real LeRobot data (14-D measured joints + three cameras)
  -> build_arx_cotrain_datasets.py

packaged ego joints + real joints
  -> final 16-D EEF LeRobot train/eval repository
  -> compute_arx_eef_norm_stats.py
  -> train_arx_eef_pytorch.py
```

`batch_preprocess` creates an ego variant. It does **not** make the final
training repository: final packaging is always performed by
`build_arx_cotrain_datasets.py`, even for real-only training. That final step
adds EEF conversion, action/state masks, image masks, source provenance, and
the requested real-data split.

The final 16-D layout is:

```text
[left x y z qx qy qz qw gripper,
 right x y z qx qy qz qw gripper]
```

For `stack_cube`, the left eight dimensions are filled with canonical values
and marked false in `policy.state_mask` and `policy.action_mask`; only the
right arm contributes to loss. `stack_cola` supervises all 16 dimensions.

## 2. FK and gripper conventions

The final exporter performs FK in
`training/build_arx_cotrain_datasets.py`, for **both** real and packaged-ego
14-D joint trajectories:

```text
left joints  = state[0:6]     left gripper  = state[6]
right joints = state[7:13]    right gripper = state[13]
FK output    = ARX TCP pose in each arm's zero-flange frame
```

It loads `assets/mujoco_arx_scene/scene.xml` and
`cfg/preprocess/base/RealSenseD405.yaml`, calls ARX forward kinematics per
frame, and writes `[xyz, qx, qy, qz, qw]`. The batch preprocessing stage
performs the opposite EEF-to-joint IK operation for ego; final packaging
converts the resulting joints back to the common EEF representation.

The final training gripper feature is binary `float32`:

```text
g = 1 if raw >= -2.6 else 0
0 = fully open, 1 = fully closed
```

The cotrain exporter applies the same `raw >= -2.6` threshold used by the ARX
diagnostic adapter to real recordings, so real and packaged-ego examples share
the same 0=open, 1=closed labels. In every final repository,
`action[t]` is the next valid source-frame EEF state, including its gripper.

## 3. Produce an ego correction variant

Run this only when the requested ego variant has not yet been packaged. For
example, position+rotation correction:

```bash
cd /mnt/data/xule/TATE
PY=thirdparty/openpi/.venv/bin/python

$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_correction_ablation.yaml \
  --variants finger_center_hys085__position_rotation \
  --stages retarget,package \
  --force-stage retarget,package
```

For cola, use the active configuration `stack_cola_v2.yaml` instead. Its
`experiment_id` is `stack_cola_v2_50`. Successful output is:

```text
outputs/experiments/<experiment>/datasets/finger_center_hys085__position_rotation/
```

Use `--stages package --force-stage package` instead when the corresponding
IK trajectories already exist and only the LeRobot package is missing.

## 4. Build final training repositories

All commands below are run at the TATE repository root. `--overwrite` replaces
only the named output repository; omit it to fail safely if that repository
already exists.

### Real-only, all 20 episodes in one train repository

```bash
$PY training/build_arx_cotrain_datasets.py \
  --tasks stack_cube stack_cola \
  --sources real \
  --single-train-split \
  --dataset-label real_all \
  --dataset-modes no_dropout \
  --overwrite
```

Outputs:

```text
outputs/lerobot/local/arx_eef_stack_cube_real_all_nodropout_train/
outputs/lerobot/local/arx_eef_stack_cola_real_all_nodropout_train/
```

### Merge existing compatible LeRobot datasets

For already-packaged LeRobot repositories (including three or more real-data
collections), use the generic merger.  It appends episodes, remaps task IDs by
task text, regenerates episode/global frame indices, and hard-links videos by
default.  Source repositories are never modified.

```bash
$PY training/merge_lerobot_datasets.py \
  --source-root /data/real_set_1 \
  --source-root /data/real_set_2 \
  --source-root /data/real_set_3 \
  --output-root outputs/lerobot \
  --repo-id local/stack_cola_real_merged \
  --overwrite
```

Every source must have the same FPS, LeRobot feature metadata, parquet schema,
and video streams.  Pass `--video-mode copy` when the output needs to survive
after source files are removed; otherwise the default hard links avoid copying
the video bytes.

### Co-train all real episodes with position+rotation ego

```bash
$PY training/build_arx_cotrain_datasets.py \
  --tasks stack_cube \
  --sources ego real \
  --ego-experiment stack_cube_correction_ablation_v1 \
  --ego-variant finger_center_hys085__position_rotation \
  --single-train-split \
  --dataset-label cotrain_all_posrot \
  --dataset-modes no_dropout \
  --overwrite

$PY training/build_arx_cotrain_datasets.py \
  --tasks stack_cola \
  --sources ego real \
  --ego-experiment stack_cola_v2_50 \
  --ego-variant finger_center_hys085__position_rotation \
  --single-train-split \
  --dataset-label cotrain_all_posrot \
  --dataset-modes no_dropout \
  --overwrite
```

### Co-train default real 10/10 split with uncorrected ego

The train repository contains all selected `none` ego episodes plus real
episodes 0--9. The eval repository contains only real episodes 10--19.

```bash
$PY training/build_arx_cotrain_datasets.py \
  --tasks stack_cube \
  --ego-variant finger_center_hys085__none \
  --dataset-label cotrain_real10_ego_none \
  --dataset-modes no_dropout \
  --overwrite

$PY training/build_arx_cotrain_datasets.py \
  --tasks stack_cola \
  --ego-variant finger_center_hys085__none \
  --dataset-label cotrain_real10_ego_none \
  --dataset-modes no_dropout \
  --overwrite
```

`policy.image_mask` order is `[head, left, right]`. Ego frames always use
`[1, 0, 0]`; real no-dropout frames use `[1, 1, 1]`. Camera dropout is enabled
with `--dataset-modes camera_dropout`; by default, real-train frames sample
full/head-only/head+left/head+right at `0.5/0.25/0.125/0.125`. Real eval is
always all-camera.

## 5. Compute normalization for every new train repository

The normalization assets must come from the exact training repository, not its
eval companion or another ablation.

```bash
$PY training/compute_arx_eef_norm_stats.py \
  --repo-id local/arx_eef_stack_cube_cotrain_all_posrot_nodropout_train \
  --dataset-root outputs/lerobot/local/arx_eef_stack_cube_cotrain_all_posrot_nodropout_train \
  --assets-base-dir outputs/openpi_assets \
  --model pi05 \
  --full-finetune \
  --batch-size 8 \
  --num-workers 4
```

## 6. Train Pi0.5

Run from the OpenPI checkout. This example trains the all-real cube
repository. Replace `repo-id`, `dataset-root`, and `exp-name` for another
experiment.

```bash
cd /mnt/data/xule/TATE/thirdparty/openpi

./.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  /mnt/data/xule/TATE/training/train_arx_eef_pytorch.py \
  --repo-id local/arx_eef_stack_cube_real_all_nodropout_train \
  --dataset-root /mnt/data/xule/TATE/outputs/lerobot/local/arx_eef_stack_cube_real_all_nodropout_train \
  --model pi05 \
  --pytorch-weight-path /mnt/workspace/sunxiaoquan/models/pi05_base \
  --assets-base-dir /mnt/data/xule/TATE/outputs/openpi_assets \
  --checkpoint-base-dir /mnt/data/xule/TATE/outputs/openpi_checkpoints \
  --batch-size 2 \
  --num-workers 4 \
  --epochs 2 \
  --save-interval 1000 \
  --log-interval 20 \
  --exp-name stack_cube_real_all_pi05
```

`--epochs` defaults to `2`. When `--num-train-steps` is omitted, total
optimizer steps are `floor(total_frames / batch_size) * epochs`. An explicit
`--num-train-steps N` overrides `--epochs`. The current trainer has no
automatic offline eval loop; use its standalone evaluation script with an eval
repository after checkpoint creation.

Use `--wandb` only after `wandb login` has been completed in this same `.venv`.
Without W&B, loss is printed to the terminal at `--log-interval` intervals but
is not saved as a local loss-curve file.

## 7. Offline evaluation

`evaluate_arx_eef_pytorch.py` evaluates the first action predicted from each
eval observation against its recorded action. It reports per-dimension RMSE and
`overall_active_rmse`; the latter excludes dimensions disabled by the eval
dataset's `arx_eef.default_action_mask` (the unused left arm of `stack_cube`).
It is an offline regression check, not a closed-loop robot-success metric.

The eval data and normalization must be deliberately separated: use the eval
repository for `--eval-*`, but use the *training* repository's stats via
`--norm-repo-id`. `--checkpoint` accepts an exact checkpoint directory; use
`--checkpoint-dir` plus `--step` only when selecting a numeric subdirectory.

```bash
cd /mnt/data/xule/TATE/thirdparty/openpi

TATE_OPENPI_ROOT=$PWD ./.venv/bin/python \
  /mnt/data/xule/TATE/training/evaluate_arx_eef_pytorch.py \
  --eval-repo-id local/arx_eef_stack_cube_cotrain_real10_ego_posrot_nodropout_eval \
  --eval-dataset-root /mnt/data/xule/lerobot/local/arx_eef_stack_cube_cotrain_real10_ego_posrot_nodropout_eval \
  --checkpoint /mnt/data/xule/TATE/outputs/openpi_checkpoints/arx_eef/stack_cube_posrot_pi05/5000 \
  --assets-base-dir /mnt/data/xule/TATE/outputs/openpi_assets \
  --norm-repo-id local/arx_eef_stack_cube_cotrain_real10_ego_posrot_nodropout_train \
  --batch-size 2 \
  --num-samples 256 \
  --sample-steps 10 \
  --zero-noise \
  --output-json /mnt/data/xule/TATE/outputs/eval/stack_cube_posrot_step5000.json \
  --output-npz /mnt/data/xule/TATE/outputs/eval/stack_cube_posrot_step5000.npz
```

If you have a copied standalone statistics directory instead, replace
`--norm-repo-id ...` with `--norm-stats-dir /absolute/path/to/stats_dir`.
The directory is the one containing the OpenPI saved normalization files, not
its parent assets directory.
