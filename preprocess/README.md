# Ego preprocessing

This directory converts ego RGB into dual-arm TCP trajectories expressed in
robot coordinates. It also supports real-data correction, IK, diagnostic
visualization, and packaging derived LeRobot training datasets.

There are two entry points:

- single-episode debugging with separate WiLoR, EEF export, and visualization
  commands;
- manifest-driven batch experiments over all episodes and configured variants.

See [BATCH_PREPROCESS.md](BATCH_PREPROCESS.md) for the batch configuration,
cache fingerprints, manifests, and output contract. See
[../calibration/README.md](../calibration/README.md) for camera calibration.

## Data flow

```text
ego RGB
  └─ WiLoR reconstruction ──> wilor_hands.json (shared cache)
       └─ Hand2Gripper ─────> eef_raw.json (robot-frame TCP)
            ├─ correction ──> eef.json
            ├─ visualization -> wilor_eef_vis.mp4
            └─ retarget ────> ik.npz
                                 └─ package -> LeRobot dataset
```

WiLoR inference is the expensive stage, so it is isolated from downstream
processing. Changing Hand2Gripper, camera extrinsics, TCP definitions,
correction, or IK parameters does not rerun a still-valid WiLoR cache.

## Environment and base configuration

Run all commands from the TATE repository root:

```bash
cd /home/xule/le_ws/TATE
PY=/home/xule/miniconda3/envs/lifego/bin/python
```

The main configuration files are:

- `cfg/preprocess/base/Preprocess.yaml`: WiLoR and preprocessing submodules;
- `cfg/preprocess/base/RealSenseD405.yaml`: camera intrinsics, per-arm
  extrinsics, and TCP transforms;
- `cfg/preprocess/base/EEFExport.yaml`: default Hand2Gripper, grasp hysteresis,
  and kinematic limits;
- `cfg/preprocess/batch/*.yaml`: batch experiments, variants, correction, IK,
  and packaging.

## Single-episode debugging

The following example uses the first `stack_cube` video. This workflow is
intended for inspecting detections, coordinate axes, and gripper state; it does
not create an experiment manifest.

```bash
VIDEO=/home/xule/le_ws/Data_TATE/stack_cube_ego/videos/observation.images.head/chunk-000/file-000.mp4
SESSION=outputs/stack_cube_ego_ep000
```

### 1. Reconstruct the WiLoR cache

```bash
$PY -m preprocess.reconstruct_wilor \
  --video "$VIDEO" \
  --session "$SESSION" \
  --cfg cfg/preprocess/base/Preprocess.yaml \
  --wilor-pretrained-dir /home/xule/le_ws/LifEgo/.cache/wilor_mini
```

Add `--max-frames 40` for a short smoke run.

### 2. Export EEF from the cache

```bash
$PY -m preprocess.export_eef \
  --hands "$SESSION/preprocess/wilor_hands.json" \
  --camera-calibration cfg/preprocess/base/RealSenseD405.yaml \
  --eef-config cfg/preprocess/base/EEFExport.yaml \
  --hand2gripper-mode finger_center \
  --grasp-mode finger_center \
  --arm-mode single_arm \
  --active-sides right \
  --out "$SESSION/preprocess/eef.json"
```

`--hand2gripper-mode` controls TCP position and orientation and accepts
`finger_center`, `humanego`, or `qwen`. `--grasp-mode` independently controls
the grasp ratio and hysteresis state. Pose ablations should use the same
`grasp_mode` so that all pose modes share identical grasp events.

### 3. Visualize WiLoR, TCP, and grasp state

```bash
$PY -m preprocess.visualize_wilor_cache \
  --video "$VIDEO" \
  --hands "$SESSION/preprocess/wilor_hands.json" \
  --eef "$SESSION/preprocess/eef.json" \
  --eef-config cfg/preprocess/base/EEFExport.yaml \
  --hand2gripper-mode finger_center \
  --grasp-mode finger_center \
  --out "$SESSION/preprocess/wilor_eef_vis.mp4"
```

The overlay shows the grasp signal, open/close thresholds, hysteresis band,
and final binary state. It only consumes cached results, so changing display
parameters or previewing grasp thresholds does not rerun WiLoR.

Single-episode outputs are written as:

```text
outputs/stack_cube_ego_ep000/preprocess/
├── wilor_hands.json
├── eef.json
└── wilor_eef_vis.mp4
```

The compatibility entry point `python -m preprocess.Preprocess` remains
available, but new experiments should prefer the split commands above or the
batch runner.

## Batch preprocessing

The batch entry point reads a LeRobot ego dataset and uses an experiment YAML
to produce multiple reproducible Hand2Gripper/correction variants:

```bash
$PY -m preprocess.batch_preprocess --config <batch-config.yaml>
```

The main examples are:

- `cfg/preprocess/batch/stack_cube_h2g_ablation.yaml`: right-arm task;
- `cfg/preprocess/batch/stack_cola_h2g_ablation.yaml`: bimanual task;
- `cfg/preprocess/batch/stack_*_correction_ablation.yaml`: position and
  orientation correction ablations.

### Batch stages

Omitting `--stages` runs `wilor,eef,correct,visualize,package`; it does not
run IK retargeting. Use `--stages all` or explicitly include `retarget` when
IK output is required.

| Stage | Input | Purpose and output |
| --- | --- | --- |
| `wilor` | ego RGB | Reconstruct both hands into shared `wilor_hands.json` caches |
| `eef` | WiLoR cache | Produce `eef_raw.json` for each Hand2Gripper variant |
| `correct` | raw EEF | Produce final `eef.json` for each run; `none` is the identity variant |
| `retarget` | final EEF | Solve ARX joint trajectories with MuJoCo/Mink into `ik.npz` |
| `visualize` | RGB + cache + raw EEF | Render a diagnostic MP4 per H2G and episode |
| `package` | final EEF + optional IK | Create a derived LeRobot dataset per final run |

### Recommended workflow

First validate dataset discovery, episode ranges, and temporal trimming without
creating outputs:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --dry-run
```

Then process one episode without packaging a temporary one-episode dataset:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --episodes 0 \
  --stages wilor,eef,correct,retarget,visualize
```

After checking visualization and EEF, run the complete dataset:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml
```

To reuse existing WiLoR caches and regenerate only downstream trajectories:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --stages eef,correct
```

To render or resume only the visualization stage:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --stages visualize
```

Add `--visualization-max-frames 100` for a short preview.

### Selecting episodes and variants

```bash
# One episode, a list, an inclusive range, or a limited selection
--episodes 0
--episodes 0,3,7
--episodes 10:15
--episodes 10:15 --limit 2

# Select runs[].id, not a hand2gripper_variants key
--variants qwen_hys085__none
```

A combined example is:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --episodes 0:4 \
  --variants finger_center_hys085__none,qwen_hys085__none \
  --stages eef,correct,retarget,visualize
```

### Resume and failure handling

Batch processing resumes by default. A stage is skipped only when its manifest
status, signature, and output files all match. Use the following form to
deliberately regenerate stages:

```bash
--force-stage eef,correct,retarget
```

Do not normally add `--fail-fast`. By default, an episode or variant failure is
recorded in the manifest and processing continues. If an ego trajectory cannot
satisfy the grasp events required by a correction artifact, that combination
is marked `skipped` and is not retargeted. With
`lerobot.require_all_episodes: false`, packaging includes the successful subset
and records exclusions in dataset provenance.

Use a new `experiment_id` after changing source data, processing semantics,
calibration, or correction artifacts. An experiment ID must never mix
incompatible results.

## Coordinate and calibration conventions

Both `eef_raw.json` and `eef.json` already use the ARX TCP orientation; IK does
not apply another orientation fix. Each side's `eef_pose_world` is expressed in
its `left_flange_zero` or `right_flange_zero` frame. Camera intrinsics are
selected by exact video resolution and are not scaled automatically. Recheck
`RealSenseD405.yaml` after changing the camera, resolution, or crop policy.
