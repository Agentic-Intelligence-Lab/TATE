# TATE

ARX + HumanEgo/WiLoR preprocessing, EEF export, MuJoCo IK retargeting, and replay.

Use `/home/ymq/miniconda3/envs/lifego/bin/python`, or activate the env first and use `python`.
All generated results should be written under `outputs/`.

## Preprocess

Video -> WiLoR hand keypoints video + EEF JSON.

```bash
python -m preprocess.Preprocess \
  --mps_path outputs/test \
  --video_path DATA/arx_ego_dataset/test.mp4 \
  --cfg_path cfg/preprocess/base/Preprocess.yaml \
  --task serve_bread \
  --no-gif

# Outputs:
# outputs/test/preprocess/hand_keypoints_eef_vis.mp4
# outputs/test/preprocess/eef.json

# Notes:
# `--task` is kept for HumanEgo CLI compatibility and does not affect the current mp4 WiLoR pipeline.
# EEF JSON uses each arm's zero-position flange frame: z-up, +x forward, +y left.
```

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
