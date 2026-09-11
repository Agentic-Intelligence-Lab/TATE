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

## 3. Check the loader

```bash
thirdparty/openpi/.venv/bin/python training/check_arx_joint_dataloader.py \
  --dataset-root outputs/lerobot/local/arx_stack_cube_ego \
  --repo-id local/arx_stack_cube_ego \
  --skip-norm-stats
```

For co-training, replace the repo id and dataset root with
`local/arx_stack_cube_cotrain` and
`outputs/lerobot/local/arx_stack_cube_cotrain`.

## 4. Compute normalization

```bash
thirdparty/openpi/.venv/bin/python training/compute_arx_norm_stats.py \
  --dataset-root outputs/lerobot/local/arx_stack_cube_ego \
  --repo-id local/arx_stack_cube_ego
```

## 5. Train Pi0.5

```bash
cd thirdparty/openpi
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  ../../training/train_arx_joint_pytorch.py \
  --repo-id local/arx_stack_cube_ego \
  --dataset-root ../../outputs/lerobot/local/arx_stack_cube_ego \
  --model pi05 \
  --pytorch-weight-path /mnt/data/szeluresearch/models/pi05_base \
  --batch-size 8
```

The raw ARX state/action layout is 14D:
`left_joint_1..6,left_gripper,right_joint_1..6,right_gripper`.
The OpenPI model keeps its 32D action head; only the first 14 dimensions are
supervised. Checkpoints and assets are written below `outputs`.

## 6. Evaluate

```bash
thirdparty/openpi/.venv/bin/python training/evaluate_arx_joint_pytorch.py \
  --repo-id local/arx_stack_cube_ego \
  --dataset-root outputs/lerobot/local/arx_stack_cube_ego \
  --exp-name arx_joint_pi05_pytorch \
  --device cuda:0
```
