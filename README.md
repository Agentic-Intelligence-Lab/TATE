# TATE

ARX + HumanEgo/WiLoR preprocessing, EEF export, MuJoCo IK retargeting, and replay.

Use `/home/ymq/miniconda3/envs/lifego/bin/python`, or activate the env first and use `python`.
All generated results should be written under `outputs/`.

## Preprocess

Video preprocessing is split into a reusable WiLoR reconstruction and a cheap
camera-to-arm EEF export.

```bash
python -m preprocess.reconstruct_wilor \
  --video DATA/arx_ego_dataset/test.mp4 \
  --session outputs/test \
  --cfg cfg/preprocess/base/Preprocess.yaml

python -m preprocess.export_eef \
  --hands outputs/test/preprocess/wilor_hands.json \
  --camera-calibration cfg/preprocess/base/RealSenseD405.yaml \
  --eef-config cfg/preprocess/base/EEFExport.yaml \
  --out outputs/test/preprocess/eef.json

# Select a mode without editing the config (finger_center, humanego, qwen):
# add --hand2gripper-mode humanego to the export command.

python -m preprocess.visualize_wilor_cache \
  --video DATA/arx_ego_dataset/test.mp4 \
  --hands outputs/test/preprocess/wilor_hands.json \
  --eef outputs/test/preprocess/eef.json \
  --eef-config cfg/preprocess/base/EEFExport.yaml \
  --out outputs/test/preprocess/wilor_eef_vis.mp4

# Outputs:
# outputs/test/preprocess/wilor_hands.json
# outputs/test/preprocess/eef.json

# Notes:
# EEF JSON uses each arm's zero-position flange frame and already stores ARX TCP orientation.
```

See `preprocess/README.md` for frame-limited validation and the compatibility
`preprocess.Preprocess` entry point.

## Batch preprocessing, correction, and evaluation

The manifest-driven batch runner keeps WiLoR caches, Hand2Gripper variants,
real-anchor correction variants, IK, and derived LeRobot datasets separate and
reproducible. See [preprocess/BATCH_PREPROCESS.md](preprocess/BATCH_PREPROCESS.md).

Real ARX calibration fitting, optional camera-ray/XYZ anchor correction,
task-level SO(3) correction, timestamp-based alignment, DTW, and held-out
evaluation are documented in [evaluation/README.md](evaluation/README.md). The
machine-readable boundary between preprocessing and evaluation is
[ALIGNMENT_EVAL_INTERFACE.md](ALIGNMENT_EVAL_INTERFACE.md).

Evaluation reads trajectory timestamps from EEF/LeRobot data and does not infer
them from MP4 duration or frame count. Video/timestamp repair therefore remains
a preprocessing responsibility.

## Retarget

EEF JSON -> dual-arm IK joint trajectory.

```bash
python real2sim/retarget_arx_with_mink.py \
  --eef outputs/test/preprocess/eef.json \
  --out outputs/test/ik/dual_arm_ik_mink.npz

# outputs:
# outputs/test/ik/dual_arm_ik_mink.npz
```

## Replay
Use `--viewer` for the interactive mode.
```bash
# replay eef data
python real2sim/replay_arx_mujoco.py \
  --mode eef \
  --data outputs/test/preprocess/eef.json \
  --out outputs/test/replay/eef_replay.mp4

# replay ik data
python real2sim/replay_arx_mujoco.py \
  --mode joint \
  --data outputs/test/ik/dual_arm_ik_mink.npz \
  --out outputs/test/replay/dual_arm_ik_replay.mp4

# replay realbot data
python real2sim/replay_arx_mujoco.py \
  --mode joint \
  --data DATA/arx_ego_dataset/data/chunk-000/file-000.parquet \
  --out outputs/test/replay/file-000_joint_replay.mp4

# no --data means static scene
python real2sim/replay_arx_mujoco.py --viewer  # interactive static scene
```

Notes:

- Joint npz replay moves both arms when `left_joint_qpos` and `right_joint_qpos` are present.
- Parquet joint replay reads left arm from dims `0:7` and right arm from dims `7:14`.

## Real Robot Replay

Dry-run validates ranges and interpolated command speed only. Real execution requires both risk flags.

```bash
python real2sim/replay_arx_realbot.py \
  --data outputs/test/ik/dual_arm_ik_mink.npz  # dry-run IK joint replay

python real2sim/replay_arx_realbot.py \
  --data DATA/arx_ego_dataset/data/chunk-000/file-000.parquet  # dry-run real robot log replay

python real2sim/replay_arx_realbot.py \
  --data outputs/test/ik/dual_arm_ik_mink.npz \
  --execute \
  --yes-i-understand-risk  # commands real robot; keep E-stop/power cutoff ready
```

The ARX SDK under `/home/ymq/code/ARX5_beta` must match the Python interpreter used for real execution.
