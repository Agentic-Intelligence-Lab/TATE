# ARX EEF OpenPI Training

Project-specific training code lives in `training/`; `thirdparty/openpi` is not modified.

## 1. Preprocess Ego Videos

```bash
scripts/batch_preprocess_ego.sh \
  DATA/stack_cube_ego/videos/observation.images.head \
  outputs/stack_cube_ego_preprocess \
  --max-files 1  # omit for all videos
```

Each episode output should contain `preprocess/eef.json`.

## 2. Build LeRobot EEF Datasets

Real robot source data stores 14D joint vectors, but this builder converts them to 16D EEF state/action.

```bash
thirdparty/openpi/.venv/bin/python training/build_arx_lerobot_dataset_from_joint.py \
  --source-root DATA/stack_cube_arx \
  --repo-id local/arx_eef_stack_cube_arx \
  --output-root outputs/lerobot \
  --overwrite
```

Ego data reads videos from `DATA/stack_cube_ego` and EEF labels from `outputs/stack_cube_ego_preprocess`.

```bash
thirdparty/openpi/.venv/bin/python training/build_arx_lerobot_dataset_from_ego.py \
  --source-root DATA/stack_cube_ego \
  --eef-root outputs/stack_cube_ego_preprocess \
  --repo-id local/arx_eef_stack_cube_ego \
  --output-root outputs/lerobot \
  --overwrite
```

For corrected EEF labels:

```bash
thirdparty/openpi/.venv/bin/python training/build_arx_lerobot_dataset_from_ego.py \
  --source-root DATA/stack_cube_ego \
  --eef-root outputs/stack_cube_ego_preprocess \
  --pose-key eef_pose_corrected \
  --repo-id local/arx_eef_stack_cube_ego \
  --output-root outputs/lerobot \
  --overwrite
```

Both generated datasets use:

```text
left:  x y z qx qy qz qw gripper
right: x y z qx qy qz qw gripper
```

The model keeps OpenPI's 32D action head; only the first 16 EEF dimensions are supervised.

For the TATE ego variant plus corrected real data, use the direct cotrain
exporter instead. It creates separate 16D EEF train/eval repositories, keeps
the correction's real episode split (0--9 train, 10--19 eval), adds the
per-element action mask required by the single-arm cube task, and stores one
camera availability mask per frame.

```bash
thirdparty/openpi/.venv/bin/python training/build_arx_cotrain_datasets.py \
  --overwrite
```

It writes the following repositories under `outputs/lerobot/local/`:

```text
arx_eef_stack_cube_cotrain_{nodropout,camdrop}_{train,eval}
arx_eef_stack_cola_cotrain_{nodropout,camdrop}_{train,eval}
```

`nodropout` provides all three real cameras; `camdrop` uses full/head-only/
head+left/head+right probabilities of `0.5/0.25/0.125/0.125` for real-train
frames. Ego is always `[head=1, left=0, right=0]`; eval is always all three
real cameras. To make a new real split deliberately, pass either
`--real-train-ratio R --split-seed S --allow-correction-split-override` or
both `--real-train-ids ...` and `--real-eval-ids ...` with the same override.
The acknowledgement is required because the existing correction was fitted on
the default calibration episodes.

## 3. Check The Loader

```bash
thirdparty/openpi/.venv/bin/python training/check_arx_eef_dataloader.py \
  --dataset-root outputs/lerobot/local/arx_eef_stack_cube_ego \
  --repo-id local/arx_eef_stack_cube_ego \
  --skip-norm-stats
```

For the direct cotrain repositories, `policy.image_mask` is read per frame:
ego uses only the head camera, while real frames follow the selected dropout
mode. Unavailable images are both marked unavailable to OpenPI and zeroed
before entering the model.

## 4. Compute Normalization

```bash
thirdparty/openpi/.venv/bin/python training/compute_arx_eef_norm_stats.py \
  --dataset-root outputs/lerobot/local/arx_eef_stack_cube_ego \
  --repo-id local/arx_eef_stack_cube_ego
```

This writes stats under `outputs/openpi_assets/arx_eef/<repo-id>/`.

## 5. Train Pi0.5

```bash
cd thirdparty/openpi
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  ../../training/train_arx_eef_pytorch.py \
  --repo-id local/arx_eef_stack_cube_cotrain_camdrop_train \
  --dataset-root ../../outputs/lerobot/local/arx_eef_stack_cube_cotrain_camdrop_train \
  --model pi05 \
  --pytorch-weight-path /mnt/workspace/sunxiaoquan/models/pi05_base \
  --batch-size 8
```

The trainer reads `meta/info.json` and masks its 16-element action loss. Thus
cube trains only the right-arm EEF dimensions; cola trains all 16 dimensions.
Use `--action-mask` only to explicitly override that dataset metadata.

Checkpoints are written under `outputs/openpi_checkpoints/arx_eef/`.

## 6. Evaluate

```bash
thirdparty/openpi/.venv/bin/python training/evaluate_arx_eef_pytorch.py \
  --repo-id local/arx_eef_stack_cube_ego \
  --dataset-root outputs/lerobot/local/arx_eef_stack_cube_ego \
  --exp-name arx_eef_pi05_pytorch \
  --device cuda:0
```

## 7. Move To HPC

For ego-only training, copy:

```text
training/
thirdparty/openpi/
outputs/lerobot/local/arx_eef_stack_cube_ego/
outputs/openpi_assets/arx_eef/local/arx_eef_stack_cube_ego/
```

The OpenPI `.venv` does not need to be copied. Recreate it on HPC with the OpenPI environment setup, and use `/mnt/workspace/sunxiaoquan/models/pi05_base` as `--pytorch-weight-path`.


## 8. HPC Debug

``` sh
uv pip install pytest
uv pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
uv pip install 'datasets>=2.16,<3'

# Transformer库报错
cd /mnt/data/yuanmingqi/code/TATE/thirdparty/openpi

uv pip install transformers==4.53.2
TRANSFORMERS_DIR=$(python - <<'PY'
import pathlib, transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)

cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/

python - <<'PY'
from transformers.models.siglip import check
print(check.check_whether_transformers_replace_is_installed_correctly())
PY
```
