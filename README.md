# TATE

TATE is the ARX + ego RGB data processing and evaluation pipeline. It covers:

```text
camera and robot calibration
  ├─ real LeRobot joint data -> FK TCP references
  └─ ego LeRobot RGB -> WiLoR -> robot-frame EEF -> correction -> IK
                                                        ├─ LeRobot training datasets
                                                        ├─ held-out evaluation
                                                        ├─ MuJoCo replay
                                                        └─ real-robot replay
```

This README presents one end-to-end workflow. Algorithm details, configuration
fields, and artifact contracts live in the corresponding subdirectory docs.

## Documentation

| Topic | Documentation |
| --- | --- |
| Camera intrinsics and extrinsics | [calibration/README.md](calibration/README.md) |
| Real LeRobot data to FK/TCP references | [real_data/README.md](real_data/README.md) |
| Single-episode and batch ego preprocessing | [preprocess/README.md](preprocess/README.md) |
| Batch configuration, caching, manifests, and fields | [preprocess/BATCH_PREPROCESS.md](preprocess/BATCH_PREPROCESS.md) |
| Correction, alignment, and evaluation | [evaluation/README.md](evaluation/README.md) |
| Preprocessing-to-evaluation machine interface | [ALIGNMENT_EVAL_INTERFACE.md](ALIGNMENT_EVAL_INTERFACE.md) |
| ARX MuJoCo model | [assets/mujoco_arx_scene/README.md](assets/mujoco_arx_scene/README.md) |

## 0. Environment and paths

Use the TATE environment for data processing, evaluation, and MuJoCo:

```bash
cd /home/xule/le_ws/TATE
PY=/home/xule/miniconda3/envs/lifego/bin/python
```

Raw datasets are stored under `/home/xule/le_ws/Data_TATE`, and generated
artifacts belong under `outputs/`. Hardware execution uses the ARX SDK Python
environment; the current example path is
`/home/qijun/ARX5_beta/.venv/bin/python`.

The commands below use the bimanual `stack_cola` task. For the right-arm task,
replace the configurations with their `stack_cube_*` counterparts and select
the `right` side where required.

## 1. Verify camera and robot calibration

Ego EEF export depends on:

```text
cfg/preprocess/base/RealSenseD405.yaml
```

It must contain intrinsics for the current video resolution, the D405
extrinsics relative to both robot arms, and the side-specific `T_tcp_in_hand`
transforms. Recalibrate after changing the rig, camera resolution, or tool. See
[calibration/README.md](calibration/README.md) for capture, validation, and YAML
update commands.

## 2. Convert real data into evaluation references

Run ARX forward kinematics on the real LeRobot joint dataset and express the
result in the per-arm zero-flange frames:

```bash
$PY -m real_data.arx_lerobot_adapter \
  --input /home/xule/le_ws/Data_TATE/stack_cola_arx \
  --out outputs/arx_real_flange/stack_cola_arx
```

This creates per-episode TCP/flange JSON files and the reference manifest:

```text
outputs/arx_real_flange/stack_cola_arx/manifest.json
```

See [real_data/README.md](real_data/README.md) for gripper calibration, FK frame
definitions, and conversion options.

## 3. Prepare the real split and correction artifacts

Correction fitting and final evaluation must use disjoint real episodes. Check
the task split before fitting:

```text
cfg/evaluation/splits/stack_cola_real_v1.json
```

Fit position anchors using only the calibration cohort:

```bash
$PY -m correction.fit \
  --real-manifest outputs/arx_real_flange/stack_cola_arx/manifest.json \
  --real-split cfg/evaluation/splits/stack_cola_real_v1.json \
  --eval-config cfg/evaluation/stack_cola_arx.yaml \
  --out outputs/corrections/stack_cola/xyz_mean_target_min_bending_all_anchors.json
```

The artifact paths referenced by the batch YAML must match the fitted outputs.
A task-level rotation correction requires a preliminary unrotated ego
experiment and is fitted with `correction.fit_rotation`. See
[evaluation/README.md](evaluation/README.md) for the held-out protocol, and use
the correction commands' `--help` output for fitting options.

## 4. Batch-process ego data

### 4.1 Inspect the plan

Resolve the dataset, episodes, trim windows, stages, and variants without
creating outputs:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --dry-run
```

### 4.2 Validate one episode

Process episode 0 without creating a temporary one-episode LeRobot package:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --episodes 0 \
  --stages wilor,eef,correct,retarget,visualize
```

Inspect the raw/final EEF, grasp events, IK, and diagnostic videos before the
complete run.

### 4.3 Process the complete dataset

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml
```

Omitting `--stages` is equivalent to `--stages all`:

| Stage | Result |
| --- | --- |
| `wilor` | Ego RGB to shared two-hand reconstruction caches |
| `eef` | Each Hand2Gripper mode to robot-frame raw TCP trajectories |
| `correct` | Each H2G/correction combination to final EEF trajectories |
| `retarget` | Final EEF to ARX joint IK trajectories |
| `visualize` | RGB + WiLoR + raw EEF diagnostic videos |
| `package` | Final EEF/IK written into one LeRobot training dataset per run |

Batch processing resumes by default: repeating a command skips complete outputs
with matching signatures. Do not normally add `--fail-fast`, so one bad episode
does not stop the remaining dataset. See
[preprocess/BATCH_PREPROCESS.md](preprocess/BATCH_PREPROCESS.md) for stage-only
runs, forced regeneration, episode/run filters, failure semantics, and output
paths.

The example experiment ID may change as configurations evolve. Read the current
`experiment_id` from the batch YAML and replace
`stack_cola_h2g_ablation_v6` in the commands below when necessary.

## 5. Inspect experiment outputs

```text
outputs/experiments/stack_cola_h2g_ablation_v6/
├── manifest.json
├── manifest.tsv
├── resolved_experiment.yaml
├── logs/
├── artifacts/eef/<h2g-id>/episode_000000/eef_raw.json
├── artifacts/variants/<run-id>/episode_000000/eef.json
├── artifacts/variants/<run-id>/episode_000000/ik.npz
├── artifacts/visualizations/<h2g-id>/episode_000000/wilor_eef_vis.mp4
└── datasets/<run-id>/
```

`manifest.json` is the evaluation entry point and complete experiment
provenance. `datasets/<run-id>` contains the LeRobot datasets used for policy
training.

## 6. Run held-out evaluation

First use a dry run to verify the ego cohort, real split, active arms, variants,
and pair count:

```bash
$PY -m evaluation.run_eval \
  --experiment-manifest outputs/experiments/stack_cola_h2g_ablation_v6/manifest.json \
  --real-manifest outputs/arx_real_flange/stack_cola_arx/manifest.json \
  --real-split cfg/evaluation/splits/stack_cola_real_v1.json \
  --eval-config cfg/evaluation/stack_cola_arx.yaml \
  --out outputs/evaluation/stack_cola_h2g_ablation_v6_lifego \
  --dry-run
```

Then run or resume evaluation:

```bash
$PY -m evaluation.run_eval \
  --experiment-manifest outputs/experiments/stack_cola_h2g_ablation_v6/manifest.json \
  --real-manifest outputs/arx_real_flange/stack_cola_arx/manifest.json \
  --real-split cfg/evaluation/splits/stack_cola_real_v1.json \
  --eval-config cfg/evaluation/stack_cola_arx.yaml \
  --out outputs/evaluation/stack_cola_h2g_ablation_v6_lifego \
  --resume
```

Cross-variant results are written to `comparison.csv` and `comparison.json`.
Each variant directory contains its aggregate summary, episode metrics,
exclusion reasons, and saved alignments. See
[evaluation/README.md](evaluation/README.md) for metrics and filters.

## 7. Replay in MuJoCo

Select a successful final run and episode:

```bash
EXP=outputs/experiments/stack_cola_h2g_ablation_v6
RUN=finger_center_hys085__xyz_mean_target_min_bending_all_anchors
EEF="$EXP/artifacts/variants/$RUN/episode_000000/eef.json"
IK="$EXP/artifacts/variants/$RUN/episode_000000/ik.npz"
```

Use EEF marker replay to inspect TCP positions and orientations:

```bash
$PY real2sim/replay_arx_mujoco.py \
  --mode eef \
  --data "$EEF" \
  --viewer
```

Use joint replay to inspect robot and gripper motion produced by IK:

```bash
$PY real2sim/replay_arx_mujoco.py \
  --mode joint \
  --data "$IK" \
  --viewer
```

Compare real TCP and ego EEF paths before/after correction, with grasp-event
correspondences and pose errors:

```bash
$PY real2sim/replay_traj_compare_mujoco.py \
  --realbot outputs/arx_real_flange/stack_cola_arx/episode_000000.json \
  --eef outputs/experiments/stack_cola_v2_50/artifacts/variants/finger_center_hys085__none/episode_000000/eef.json \
  --corrected-eef outputs/experiments/stack_cola_v2_50/artifacts/variants/finger_center_hys085__position_rotation/episode_000000/eef.json \
  --side both \
  --viewer
```

Remove `--viewer` and add `--out outputs/replay/example.mp4` to render an MP4
instead of opening the interactive viewer.

## 8. Replay on the real robot

Before hardware execution, inspect the IK in MuJoCo and run the real-robot
script without `--execute`:

```bash
$PY real2sim/replay_arx_realbot.py \
  --data "$IK" \
  --sdk-root /home/qijun/ARX5_beta \
  --speed 0.5 \
  --ramp-time 8
```

After verifying joint ranges, velocity limits, side assignment, and gripper
commands, use the ARX SDK environment to connect to hardware:

```bash
/home/qijun/ARX5_beta/.venv/bin/python real2sim/replay_arx_realbot.py \
  --data "$IK" \
  --sdk-root /home/qijun/ARX5_beta \
  --left-can can1 \
  --right-can can3 \
  --speed 0.5 \
  --ramp-time 8 \
  --execute \
  --yes-i-understand-risk
```

Hardware execution sends real arm and gripper commands. Confirm that emergency
stop or power cutoff is available, the workspace is clear, and CAN ports match
the intended arms. Start at reduced speed. On normal completion, the script
holds the final pose until the operator presses Enter and then returns the arms
home; abnormal exits enter protect mode.

To sparsify corrected EEF into hardware waypoints instead of replaying the
batch-generated IK, use `real2sim/replay_arx_eef_waypoints.py`. Run its offline
IK/FK check first, then add `--execute --yes-i-understand-risk`. Refer to
`--help` and the script's safety limits before hardware use.
