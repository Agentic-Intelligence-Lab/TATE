#!/usr/bin/env bash
# Two-stage Pi0.5 full fine-tuning:
#   1. fine-tune the base model on ego-only data;
#   2. initialize a new run from stage 1 and fine-tune on real-only data.
#
# Defaults reproduce the stack_redcube_e1, ego=99, real=20 setup. Ego samples
# keep their head-camera visual input by default. Counts and source dataset
# roots can be selected independently, for example:
#
#   EGO_SOURCE_DATASET=/data/ego_lerobot \
#   REAL_SOURCE_DATASET=/data/real_lerobot \
#   EGO_COUNT=60 REAL_COUNT=12 \
#     bash scripts/train_ego_then_real.sh
#
# Generated datasets and normalization assets are rebuilt by default. To reuse
# existing ones, set BUILD_DATASETS=0 and/or COMPUTE_NORM=0.

set -Eeuo pipefail

TATE_ROOT="${TATE_ROOT:-/mnt/data/xule/TATE}"
OPENPI_ROOT="${OPENPI_ROOT:-${TATE_ROOT}/thirdparty/openpi}"
PY="${PY:-${OPENPI_ROOT}/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-${OPENPI_ROOT}/.venv/bin/torchrun}"

BUILD_DATASET="${TATE_ROOT}/training/build_arx_cotrain_datasets.py"
NORM="${TATE_ROOT}/training/compute_arx_eef_norm_stats.py"
TRAIN="${TATE_ROOT}/training/train_arx_eef_pytorch.py"

# build_arx_cotrain_datasets.py writes under <output-root>/local.
LEROBOT_ROOT="${LEROBOT_ROOT:-${TATE_ROOT}/outputs/lerobot}"
ASSETS="${ASSETS:-${TATE_ROOT}/outputs/openpi_assets/0922}"
CHECKPOINTS="${CHECKPOINTS:-${TATE_ROOT}/outputs/openpi_checkpoints/0922}"
LOG_DIR="${LOG_DIR:-${TATE_ROOT}/outputs/logs}"
BASE_WEIGHTS="${BASE_WEIGHTS:-/mnt/workspace/sunxiaoquan/models/pi05_base}"

TASK="${TASK:-stack_redcube_e1}"
EGO_COUNT="${EGO_COUNT:-99}"
REAL_COUNT="${REAL_COUNT:-20}"
# EGO_SOURCE_DATASET should be a packaged correction variant: its corrected EEF
# has already been retargeted by IK into 14-D ARX joints. REAL_SOURCE_DATASET is
# the measured 14-D real-robot LeRobot dataset. Leave either empty to use TASK's
# built-in source location.
EGO_SOURCE_DATASET="${EGO_SOURCE_DATASET:-}"
REAL_SOURCE_DATASET="${REAL_SOURCE_DATASET:-}"
EGO_IMAGE_MODE="${EGO_IMAGE_MODE:-head}"
EGO_EPOCHS="${EGO_EPOCHS:-2}"
REAL_EPOCHS="${REAL_EPOCHS:-2}"
GLOBAL_BS="${GLOBAL_BS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
BUILD_DATASETS="${BUILD_DATASETS:-1}"
COMPUTE_NORM="${COMPUTE_NORM:-1}"
USE_WANDB="${USE_WANDB:-1}"

# Task aliases e1/e2/e3 use v1/v2/v3 in the generated repository name.
case "$TASK" in
  stack_redcube_e1) OUTPUT_TASK="stack_redcube_v1" ;;
  stack_redcube_e2) OUTPUT_TASK="stack_redcube_v2" ;;
  stack_redcube_e3) OUTPUT_TASK="stack_redcube_v3" ;;
  *) OUTPUT_TASK="$TASK" ;;
esac

EGO_LABEL="${EGO_LABEL:-sequential_ego_e${EGO_COUNT}}"
REAL_LABEL="${REAL_LABEL:-sequential_real_r${REAL_COUNT}}"
EGO_REPO="local/arx_eef_${OUTPUT_TASK}_${EGO_LABEL}_nodropout_train"
REAL_REPO="local/arx_eef_${OUTPUT_TASK}_${REAL_LABEL}_nodropout_train"
EGO_DATASET="${LEROBOT_ROOT}/${EGO_REPO}"
REAL_DATASET="${LEROBOT_ROOT}/${REAL_REPO}"
EGO_EXP_NAME="${EGO_EXP_NAME:-${OUTPUT_TASK}_ego_e${EGO_COUNT}_pi05}"
REAL_EXP_NAME="${REAL_EXP_NAME:-${OUTPUT_TASK}_ego_e${EGO_COUNT}_then_real_r${REAL_COUNT}_pi05}"

if (( NUM_GPUS <= 0 || GLOBAL_BS <= 0 || GLOBAL_BS % NUM_GPUS != 0 )); then
  echo "ERROR: GLOBAL_BS (${GLOBAL_BS}) must be positive and divisible by NUM_GPUS (${NUM_GPUS})." >&2
  exit 1
fi
if (( EGO_COUNT <= 0 || REAL_COUNT <= 0 || EGO_EPOCHS <= 0 || REAL_EPOCHS <= 0 )); then
  echo "ERROR: dataset counts and epoch counts must all be positive." >&2
  exit 1
fi
if [[ "$EGO_IMAGE_MODE" != "head" && "$EGO_IMAGE_MODE" != "none" ]]; then
  echo "ERROR: EGO_IMAGE_MODE must be 'head' or 'none', got '${EGO_IMAGE_MODE}'." >&2
  exit 1
fi
if [[ ! -x "$PY" || ! -x "$TORCHRUN" ]]; then
  echo "ERROR: OpenPI Python or torchrun is missing under ${OPENPI_ROOT}/.venv/bin." >&2
  exit 1
fi
if [[ ! -f "${BASE_WEIGHTS}/model.safetensors" ]]; then
  echo "ERROR: base model is missing: ${BASE_WEIGHTS}/model.safetensors" >&2
  exit 1
fi

mkdir -p "$ASSETS" "$CHECKPOINTS" "$LOG_DIR"
cd "$TATE_ROOT"

build_datasets() {
  local -a ego_source_args=()
  local -a real_source_args=()

  if [[ -n "$EGO_SOURCE_DATASET" ]]; then
    ego_source_args+=(--ego-dataset "$EGO_SOURCE_DATASET")
  fi
  if [[ -n "$REAL_SOURCE_DATASET" ]]; then
    real_source_args+=(--real-dataset "$REAL_SOURCE_DATASET")
  fi

  echo "[data 1/2] Building ego-only dataset: ${EGO_REPO}"
  echo "             source: ${EGO_SOURCE_DATASET:-TASK default}"
  echo "             episodes: ${EGO_COUNT}, image mode: ${EGO_IMAGE_MODE}"
  "$PY" "$BUILD_DATASET" \
    --tasks "$TASK" \
    --sources ego \
    --ego-count "$EGO_COUNT" \
    --single-train-split \
    --dataset-label "$EGO_LABEL" \
    --dataset-modes no_dropout \
    --ego-image-mode "$EGO_IMAGE_MODE" \
    --output-root "$LEROBOT_ROOT" \
    "${ego_source_args[@]}" \
    --overwrite

  echo "[data 2/2] Building real-only dataset: ${REAL_REPO}"
  echo "             source: ${REAL_SOURCE_DATASET:-TASK default}"
  echo "             episodes: ${REAL_COUNT}"
  "$PY" "$BUILD_DATASET" \
    --tasks "$TASK" \
    --sources real \
    --real-count "$REAL_COUNT" \
    --single-train-split \
    --dataset-label "$REAL_LABEL" \
    --dataset-modes no_dropout \
    --output-root "$LEROBOT_ROOT" \
    "${real_source_args[@]}" \
    --overwrite
}

compute_norm() {
  local repo_id="$1"
  local dataset_root="$2"

  CUDA_VISIBLE_DEVICES="" "$PY" "$NORM" \
    --repo-id "$repo_id" \
    --dataset-root "$dataset_root" \
    --assets-base-dir "$ASSETS" \
    --model pi05 \
    --full-finetune \
    --batch-size 8 \
    --num-workers "$NUM_WORKERS"
}

train_four_gpu() {
  local repo_id="$1"
  local dataset_root="$2"
  local exp_name="$3"
  local initial_weights="$4"
  local epochs="$5"
  local -a wandb_args=()

  if [[ "$USE_WANDB" == "1" ]]; then
    wandb_args+=(--wandb)
  fi

  env \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    TATE_OPENPI_ROOT="$OPENPI_ROOT" \
    "$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    "$TRAIN" \
    --repo-id "$repo_id" \
    --dataset-root "$dataset_root" \
    --model pi05 \
    --pytorch-weight-path "$initial_weights" \
    --assets-base-dir "$ASSETS" \
    --checkpoint-base-dir "$CHECKPOINTS" \
    --batch-size "$GLOBAL_BS" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$epochs" \
    --save-interval "$SAVE_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --exp-name "$exp_name" \
    "${wandb_args[@]}"
}

latest_checkpoint() {
  local exp_name="$1"
  local exp_root="${CHECKPOINTS}/arx_eef/${exp_name}"
  local model_file checkpoint_dir step step_number
  local best_step=-1
  local best_dir=""

  if [[ ! -d "$exp_root" ]]; then
    echo "ERROR: experiment checkpoint directory does not exist: ${exp_root}" >&2
    return 1
  fi

  while IFS= read -r -d '' model_file; do
    checkpoint_dir="${model_file%/model.safetensors}"
    step="${checkpoint_dir##*/}"
    if [[ "$step" =~ ^[0-9]+$ ]]; then
      step_number=$((10#$step))
      if (( step_number > best_step )); then
        best_step="$step_number"
        best_dir="$checkpoint_dir"
      fi
    fi
  done < <(find "$exp_root" -type f -name model.safetensors -print0)

  if [[ -z "$best_dir" ]]; then
    echo "ERROR: no numeric checkpoint containing model.safetensors under ${exp_root}" >&2
    return 1
  fi
  printf '%s\n' "$best_dir"
}

if [[ "$BUILD_DATASETS" == "1" ]]; then
  build_datasets
fi

for dataset_root in "$EGO_DATASET" "$REAL_DATASET"; do
  if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
    echo "ERROR: dataset is missing: ${dataset_root}" >&2
    exit 1
  fi
done

if [[ "$COMPUTE_NORM" == "1" ]]; then
  echo "[norm 1/2] Computing ego-only normalization stats"
  compute_norm "$EGO_REPO" "$EGO_DATASET"
  echo "[norm 2/2] Computing real-only normalization stats"
  compute_norm "$REAL_REPO" "$REAL_DATASET"
fi

echo "[train 1/2] Pi0.5 base -> ego-only"
train_four_gpu \
  "$EGO_REPO" \
  "$EGO_DATASET" \
  "$EGO_EXP_NAME" \
  "$BASE_WEIGHTS" \
  "$EGO_EPOCHS" \
  2>&1 | tee "${LOG_DIR}/${EGO_EXP_NAME}.log"

EGO_WEIGHT_PATH="$(latest_checkpoint "$EGO_EXP_NAME")"
echo "Stage-1 weights: ${EGO_WEIGHT_PATH}"

echo "[train 2/2] ego checkpoint -> real-only"
train_four_gpu \
  "$REAL_REPO" \
  "$REAL_DATASET" \
  "$REAL_EXP_NAME" \
  "$EGO_WEIGHT_PATH" \
  "$REAL_EPOCHS" \
  2>&1 | tee "${LOG_DIR}/${REAL_EXP_NAME}.log"

REAL_WEIGHT_PATH="$(latest_checkpoint "$REAL_EXP_NAME")"
echo "Done. Final two-stage checkpoint: ${REAL_WEIGHT_PATH}"
