# Manifest-driven batch preprocessing

`preprocess.batch_preprocess` converts a LeRobot ego dataset into reusable
WiLoR caches, multiple EEF/correction variants, IK trajectories, diagnostic
videos, and derived LeRobot training datasets. Every stage records its input
fingerprint, status, and outputs, enabling safe resume and experiment
provenance.

See [README.md](README.md) for a concise usage guide. This document describes
the batch configuration, dependency graph, and artifact contract. The
evaluation boundary is defined in
[../ALIGNMENT_EVAL_INTERFACE.md](../ALIGNMENT_EVAL_INTERFACE.md).

## 1. Stages and dependencies

```text
LeRobot ego episode
  │
  ├─ wilor ───────────────> shared wilor_hands.json
  │                            │
  │                            ├─ eef[H2G-A] -> eef_raw.json
  │                            └─ eef[H2G-B] -> eef_raw.json
  │                                      │
  │                                      ├─ correct[none] -> eef.json
  │                                      └─ correct[method] -> eef.json
  │                                                        │
  │                                                        └─ retarget -> ik.npz
  │
  ├─ RGB + WiLoR + raw EEF ──> visualize -> wilor_eef_vis.mp4
  │
  └─ source LeRobot + final EEF + IK -> package -> derived LeRobot dataset
```

| Stage | Required input | Granularity | Output |
| --- | --- | --- | --- |
| `wilor` | ego RGB | episode | Shared 2D/3D hand reconstruction cache |
| `eef` | WiLoR cache and camera calibration | H2G × episode | Uncorrected TCP `eef_raw.json` |
| `correct` | raw EEF and optional correction artifacts | run × episode | Final `eef.json` |
| `retarget` | final EEF and MuJoCo scene | run × episode | ARX joint targets in `ik.npz` |
| `visualize` | RGB, WiLoR, and raw EEF | H2G × episode | Hand/TCP/grasp diagnostic video |
| `package` | source dataset, final EEF, and optional IK | run | Derived LeRobot dataset |

An H2G is an entry in `hand2gripper_variants`. A run combines one H2G with one
correction. Runs using the same H2G share the WiLoR and raw EEF artifacts. They
also share one visualization instead of re-encoding it for every correction.

## 2. Generate the real-data calibration manifest

Any experiment that uses a correction artifact needs a real-data reference
manifest. Generate it once from the matching ARX LeRobot dataset before
running correction fitting or batch preprocessing:

```bash
cd /home/xule/le_ws/TATE
PY=/home/xule/miniconda3/envs/lifego/bin/python

$PY -m real_data.arx_lerobot_adapter \
  --input /home/xule/le_ws/Data_TATE/stack_cola_arx \
  --out outputs/arx_real_flange/stack_cola_arx
```

This writes one FK-derived TCP/flange JSON per real episode and the manifest
referenced by `source.real_manifest`:

```text
outputs/arx_real_flange/stack_cola_arx/manifest.json
```

Set `--input` to the same dataset as `source.real_dataset`, and set
`source.real_manifest` to the resulting `manifest.json`. The adapter uses the
default ARX MuJoCo scene and `cfg/preprocess/base/RealSenseD405.yaml`; pass
`--scene` or `--calibration` when your rig uses different files. It never
modifies the source LeRobot dataset. To intentionally regenerate an existing
output directory, append `--overwrite`.

After the manifest exists, define the held-out real split and fit the position
and/or rotation correction artifacts referenced from `correction_variants`.
See [../evaluation/README.md](../evaluation/README.md) for those commands and
[../real_data/README.md](../real_data/README.md) for FK and gripper options.

## 3. Experiment YAML

The following example shows the complete structure:

```yaml
experiment_id: my_stack_cola_v1

source:
  ego_dataset: /path/to/stack_cola_ego
  video_key: observation.images.head
  real_dataset: /path/to/stack_cola_arx
  real_manifest: outputs/arx_real_flange/stack_cola_arx/manifest.json

trajectory:
  arm_mode: bimanual                 # single_arm | bimanual
  active_sides: [left, right]        # single-arm example: [right]
  trim:
    start_seconds: 1.5
    end_seconds: 1.5

outputs:
  cache_root: outputs/cache/wilor
  experiment_root: outputs/experiments

wilor:
  id: wilor_mini_d405_v1
  preprocess_config: cfg/preprocess/base/Preprocess.yaml
  pretrained_dir: /path/to/wilor_mini

geometry:
  camera_calibration: cfg/preprocess/base/RealSenseD405.yaml

hand2gripper_variants:
  finger_center_hys085:
    mode: finger_center
    grasp_mode: finger_center
    eef_config: cfg/preprocess/base/EEFExport.yaml
    grasp_close_ratio: 0.85
    grasp_open_ratio: 0.90
    grasp_min_frames: 10

  qwen_hys085:
    mode: qwen
    grasp_mode: finger_center
    eef_config: cfg/preprocess/base/EEFExport.yaml
    grasp_close_ratio: 0.85
    grasp_open_ratio: 0.90
    grasp_min_frames: 10

correction_variants:
  none:
    mode: none

  position:
    mode: pose_correction
    artifact: outputs/corrections/stack_cola/position.json
    options:
      target_mode: mean
      space: free_xyz
      propagation: min_bending

runs:
  - id: finger_center_hys085__none
    hand2gripper: finger_center_hys085
    correction: none
  - id: finger_center_hys085__position
    hand2gripper: finger_center_hys085
    correction: position

retarget:
  enabled: true
  scene: assets/mujoco_arx_scene/scene.xml
  options:
    n_iter: 80
    dt: 0.05
    solver: daqp

lerobot:
  enabled: true
  task: "pick two cola cans and place them in the brown box"
  video_mode: hardlink
  replace_state_action: true
  action_alignment: same_frame
  gripper_open_raw: -3.4
  gripper_closed_raw: 0.1
  require_all_episodes: false
```

### Arm mode and temporal trimming

A single-arm experiment must contain exactly one active side. A bimanual
experiment must contain both `left` and `right`. Trim durations are converted
to frames using the dataset FPS and applied consistently to WiLoR, EEF,
correction, IK, all videos, and derived parquet files. The frame interval is
half-open. A trimmed video is encoded once under
`outputs/cache/video_segments` and then reused.

### Pose mode and grasp mode

`mode` controls hand-to-TCP position and orientation. `grasp_mode` controls the
grasp distance, ratio, and hysteresis state; it defaults to `mode` when omitted.
Experiments that compare only pose definitions should use a common
`grasp_mode`. Otherwise, the Qwen virtual-fingertip signal and the thumb-index
signal may yield different event sequences because their scales differ.

### Correction and runs

`correction_variants.none` materializes raw EEF at a stable final EEF path.
Position artifacts, rotation artifacts, or their combination use
`pose_correction`. `runs[].id` is the final variant ID used by `--variants`,
evaluation outputs, and derived LeRobot datasets.

If a configuration references correction artifacts, first convert the real
data, define a held-out-safe real split, and fit the artifacts. See
[../evaluation/README.md](../evaluation/README.md) for the evaluation protocol.

## 4. Running an experiment

```bash
cd /home/xule/le_ws/TATE
PY=/home/xule/miniconda3/envs/lifego/bin/python
```

### Dry run

Read only LeRobot metadata and print the dataset fingerprint, episodes, trim
windows, selected stages, and runs:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --dry-run
```

### Single-episode smoke run

Inspect one episode's EEF, grasp events, IK, and visualization without creating
a temporary one-episode package:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --episodes 0 \
  --stages wilor,eef,correct,retarget,visualize
```

### Complete dataset

The default value of `--stages` is `all`:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml
```

```text
wilor,eef,correct,retarget,visualize,package
```

### Running selected stages

```bash
# Rebuild downstream trajectories from existing WiLoR caches.
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --stages eef,correct,retarget

# Render or resume visualization only.
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --stages visualize

# Repackage successful results.
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cola_h2g_ablation.yaml \
  --stages package \
  --force-stage package
```

When running only downstream stages, their upstream files must already exist.
If an upstream stage is forced and all dependents must be refreshed, include
all relevant downstream stages in the same invocation.

### Selecting episodes, runs, and visualization scope

```bash
# Episode IDs and inclusive ranges.
--episodes 0,3,10:15

# Keep only the first N resolved episodes.
--limit 2

# Comma-separated runs[].id values.
--variants finger_center_hys085__none,qwen_hys085__none

# Render only the first 100 frames.
--visualization-max-frames 100

# Override axis length and grasp-ratio plot range.
--visualization-axis-length 0.06
--visualization-ratio-plot-max 1.5
```

## 5. Resume, force, and failure policy

Each manifest stage record contains a status, signature, timestamp,
dependencies, and output paths. A stage is resume-skipped only if its status is
`complete`, its signature matches, and its outputs still exist. Repeat the same
command after an interruption.

`--force-stage` accepts comma-separated stages:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --episodes 0 \
  --stages eef,correct,retarget \
  --force-stage eef,correct,retarget
```

An experiment ID is immutable by default: a changed configuration or source
dataset fingerprint is rejected to prevent incompatible artifacts from being
mixed. If replacing that experiment is intentional, use:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --override-experiment
```

`--override` is an alias. It deletes only the incompatible
`outputs/experiments/<experiment-id>` directory, then creates a new manifest.
The shared WiLoR cache is outside that directory and is reused when its own
fingerprint still matches. Raw EEF, corrected EEF, IK, visualizations, logs,
and packaged datasets inside the replaced experiment are regenerated.

By default, ordinary failures are recorded and processing continues with the
remaining episodes and variants. Only explicit `--fail-fast` stops at an
ordinary failure. An ego trajectory with missing, ambiguous, or mismatched
grasp events is not applicable to the correction artifact and is recorded as
`skipped`. Such a case does not trigger fail-fast and is not retargeted.

With `lerobot.require_all_episodes: false`, packaging includes only episodes
that have the required final EEF and IK outputs. Excluded source IDs are stored
in `meta/tate_preprocess.json`. If a partial cohort was packaged first, use
`--force-stage package` when replacing it with the full cohort.

## 6. Fingerprints and immutable experiments

The shared WiLoR cache identity includes:

- source dataset fingerprint and video key;
- temporal trim;
- WiLoR configuration and referenced configuration files;
- checkpoint identity.

Different H2G and correction experiments can therefore reuse one WiLoR cache.
The experiment manifest additionally fingerprints camera calibration, EEF
configuration, implementation files, and correction artifacts. One
`experiment_id` must always resolve to the same configuration and execution
dependencies. Use a new ID after changing these semantics; existing experiments
are never overwritten implicitly.

## 7. Output layout

```text
outputs/cache/wilor/<dataset>/<dataset-hash>/<wilor-id>/
└── episode_000000/
    ├── wilor_hands.json
    └── wilor_hands.meta.json

outputs/cache/video_segments/<dataset>/<dataset-hash>/<trim-window>/
└── episode_000000/<video-key>.mp4

outputs/experiments/<experiment-id>/
├── resolved_experiment.yaml
├── manifest.json
├── manifest.tsv
├── logs/
│   ├── wilor/
│   ├── retarget/
│   └── visualize/
├── artifacts/
│   ├── eef/<h2g-id>/episode_000000/eef_raw.json
│   ├── visualizations/<h2g-id>/episode_000000/wilor_eef_vis.mp4
│   └── variants/<run-id>/episode_000000/
│       ├── eef.json
│       └── ik.npz
└── datasets/<run-id>/
    ├── data/
    ├── videos/
    └── meta/
        ├── info.json
        ├── stats.json
        ├── tasks.parquet
        ├── episodes/
        └── tate_preprocess.json
```

## 8. Derived LeRobot contract

Derived datasets retain source episode, frame, and timestamp provenance and add:

```text
tate.eef_raw.<side>.pose
tate.eef.<side>.pose
tate.eef.<side>.gripper
tate.eef.<side>.grasp_ratio
tate.eef.<side>.valid
```

With `replace_state_action: true`, retargeted ARX joint targets replace
`observation.state` and `action`. `action_alignment` accepts `same_frame` or
`next_frame`. Untrimmed videos are hard-linked when possible. Image statistics
are recomputed for trimmed videos. `lerobot.task` is written consistently to
frame data, episode metadata, and `tasks.parquet`.

## 9. Custom correction interface

The built-in `none` and `pose_correction` modes cover the current experiments.
A custom correction can be configured as:

```yaml
correction_variants:
  custom:
    mode: custom_name
    entrypoint: package.module:function
```

The entry point may accept any subset of these keyword arguments:

```text
source_eef
output_eef
config
episode_index
experiment_id
variant_id
```

It must write a complete `tate.dual_arm_eef` v2 JSON to `output_eef` without
modifying `source_eef`.
