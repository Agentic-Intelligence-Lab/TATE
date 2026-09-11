# ARX OpenPI Training

All project-specific code stays here; `thirdparty/openpi` is unchanged.

## 1. Preprocess ego videos

```bash
scripts/batch_preprocess_ego.sh \
  DATA/stack_cube_ego/videos/observation.images.head \
  outputs/stack_cube_ego_preprocess \
  --max-files 1
```

The output contains `preprocess/hand_keypoints_eef_vis.mp4` and
`preprocess/eef.json` for each input video.

## 2. Convert the ARX dataset

```bash
thirdparty/openpi/.venv/bin/python training/build_lerobot_arx_dataset.py \
  --source-root DATA/stack_cube_ego \
  --repo-id local/arx_stack_cube_ego \
  --output-root outputs/lerobot \
  --overwrite
```

For real-robot data, use `DATA/stack_cube_arx` and
`local/arx_stack_cube_arx`.

To build one co-training dataset from both ego and real-robot data:

```bash
thirdparty/openpi/.venv/bin/python training/build_lerobot_arx_dataset.py \
  --source-root DATA/stack_cube_ego \
  --source-root DATA/stack_cube_arx \
  --repo-id local/arx_stack_cube_cotrain \
  --output-root outputs/lerobot \
  --overwrite
```

Then compute normalization for the co-training dataset:

```bash
thirdparty/openpi/.venv/bin/python training/compute_arx_norm_stats.py \
  --dataset-root outputs/lerobot/local/arx_stack_cube_cotrain \
  --repo-id local/arx_stack_cube_cotrain
```

## 3. Move to HPC

Copy these paths:

```text
training/
thirdparty/openpi/                 # same revision: 15a9616a00943ada6c20a0f158e3adb39df2ccac
outputs/lerobot/local/arx_stack_cube_ego/
outputs/openpi_assets/arx_joint/local/arx_stack_cube_ego/
```

The OpenPI `.venv` does not need to be copied. Recreate it on HPC with
`uv sync --frozen`, then apply the PyTorch transformers patch described in the
OpenPI README. Raw `DATA/` and preprocessing outputs are not needed for
training.

## 4. Check the loader

```bash
thirdparty/openpi/.venv/bin/python training/check_arx_joint_dataloader.py \
  --dataset-root outputs/lerobot/local/arx_stack_cube_ego \
  --repo-id local/arx_stack_cube_ego \
  --skip-norm-stats
```

## 5. Compute normalization

```bash
thirdparty/openpi/.venv/bin/python training/compute_arx_norm_stats.py \
  --dataset-root outputs/lerobot/local/arx_stack_cube_ego \
  --repo-id local/arx_stack_cube_ego
```

## 6. Train Pi0.5

```bash
cd thirdparty/openpi
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  ../../training/train_arx_joint_pytorch.py \
  --repo-id local/arx_stack_cube_ego \
  --dataset-root ../../outputs/lerobot/local/arx_stack_cube_ego \
  --model pi05 \
  --pytorch-weight-path /mnt/workspace/sunxiaoquan/models/pi05_base \
  --batch-size 8
```

The raw ARX state/action layout is 14D:
`left_joint_1..6,left_gripper,right_joint_1..6,right_gripper`.
The OpenPI model keeps its 32D action head; only the first 14 dimensions are
supervised. For `local/arx_stack_cube_ego`, the head camera is enabled and both
wrist-camera masks are set to `False`; `local/arx_stack_cube_arx` and
`local/arx_stack_cube_cotrain` use all three cameras. Checkpoints and assets are
written below `outputs`.

For co-training, replace both `local/arx_stack_cube_ego` paths above with
`local/arx_stack_cube_cotrain`.

## 7. Evaluate

```bash
thirdparty/openpi/.venv/bin/python training/evaluate_arx_joint_pytorch.py \
  --repo-id local/arx_stack_cube_ego \
  --dataset-root outputs/lerobot/local/arx_stack_cube_ego \
  --exp-name arx_joint_pi05_pytorch \
  --device cuda:0
```
