# Batch ego preprocessing

The batch runner turns one LeRobot ego dataset into reusable WiLoR caches,
multiple EEF variants, optional corrected EEF and IK artifacts, and one derived
LeRobot v3 dataset per final variant.

## Data reuse

Artifacts are separated by their actual dependency:

```text
source episode -> WiLoR cache
WiLoR cache -> one raw EEF per hand2gripper variant
raw EEF -> one final EEF per correction variant
final EEF -> IK -> derived LeRobot dataset
WiLoR cache + raw EEF + RGB -> diagnostic visualization
```

Two correction variants using the same hand2gripper therefore share both the
WiLoR cache and raw EEF. A configuration fingerprint and source dataset
fingerprint are recorded in `manifest.json`; an existing experiment ID cannot
be reused with a different resolved configuration.

Pose construction and grasp-state classification can be selected independently:

```yaml
hand2gripper_variants:
  qwen_pose_finger_grasp:
    mode: qwen                  # EEF position and orientation
    grasp_mode: finger_center   # grasp ratio and hysteresis state
```

`grasp_mode` defaults to `mode` for backward compatibility. Using one shared
`grasp_mode` keeps grasp events comparable in pose-mode ablations.

## Arm mode and temporal trimming

Arm participation and the episode window are experiment-level properties:

```yaml
trajectory:
  arm_mode: bimanual             # single_arm | bimanual
  active_sides: [left, right]    # single_arm contains exactly one side
  trim:
    start_seconds: 1.0
    end_seconds: 1.5
```

Seconds are converted to frames using the LeRobot dataset FPS. The resulting
half-open source-frame interval is used consistently by WiLoR, EEF export,
correction, retargeting, parquet materialization, and every video stream. A
trimmed video is encoded once under `outputs/cache/video_segments` and then
hard-linked into every EEF variant. Untrimmed datasets continue to hard-link
the original videos without re-encoding.

WiLoR caches retain both detected hands, while EEF export masks inactive sides.
The canonical EEF and LeRobot layouts still contain left and right fields so
single-arm and bimanual data share one schema. `arm_mode` and `active_sides`
are recorded in the experiment manifest, EEF JSON, `meta/info.json`, and
`meta/tate_preprocess.json`.

## Validate the plan

From the TATE repository root:

```bash
PY=/home/xule/miniconda3/envs/lifego/bin/python

$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --dry-run
```

The dry run reads only LeRobot metadata and prints resolved episodes, videos,
frame ranges, stages, and variants. It creates no output.

## Process one episode first

Run the reusable reconstruction and all configured EEF variants:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --episodes 0 \
  --stages wilor,eef,correct
```

Then retarget and materialize one-episode LeRobot datasets:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --episodes 0 \
  --stages retarget,package
```

This creates a deliberately partial dataset for validation. When the same
experiment is later packaged with all episodes, pass `--force-stage package`
to replace that partial materialization; canonical EEF/IK artifacts and shared
WiLoR caches are not removed.

Run all episodes and stages with:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml
```

`all` also renders one diagnostic MP4 for each episode and selected
hand2gripper variant. The visualization reads existing WiLoR and raw EEF
artifacts and never reruns WiLoR. To render or resume only this stage:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --stages visualize
```

For a short preview, add `--visualization-max-frames 100`. Axis length and the
grasp-ratio plot range can be overridden with `--visualization-axis-length`
and `--visualization-ratio-plot-max`. Use `--force-stage visualize` to replace
otherwise valid visualization outputs.

Select one or more final variants with a comma-separated list:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --variants finger_center_hys085__none \
  --stages eef,correct
```

Episode selectors accept IDs and inclusive ranges, for example
`--episodes 0,3,10:15`. `--limit 1` is also available for smoke runs.

## Resume and force behavior

Resume is the default. A stage is skipped only when its status, signature, and
output files all match. To deliberately regenerate a stage:

```bash
$PY -m preprocess.batch_preprocess \
  --config cfg/preprocess/batch/stack_cube_h2g_ablation.yaml \
  --episodes 0 \
  --stages eef,correct \
  --force-stage eef,correct
```

Use a new `experiment_id` for a semantic configuration change. This prevents
mixing results produced with different hand2gripper, correction, calibration,
or training-target definitions.

An ego episode whose grasp events cannot satisfy a fitted correction artifact
is recorded as `skipped`, and processing continues with the remaining variants
and episodes. This expected data mismatch does not trigger `--fail-fast`, and
retargeting is not attempted for that episode/variant. Other implementation or
configuration errors remain `failed` and still obey `--fail-fast`. Set
`lerobot.require_all_episodes: false` to package the successful subset; excluded
source episodes are recorded in the derived dataset provenance.

## Output layout

```text
outputs/cache/wilor/<source>/<dataset-hash>/<wilor-id>/
└── episode_000000/wilor_hands.json

outputs/cache/video_segments/<source>/<dataset-hash>/<trim-window>/
└── episode_000000/<video-key>.mp4

outputs/experiments/<experiment-id>/
├── resolved_experiment.yaml
├── manifest.json
├── manifest.tsv
├── logs/
├── artifacts/
│   ├── eef/<hand2gripper-id>/episode_000000/eef_raw.json
│   ├── visualizations/<hand2gripper-id>/episode_000000/wilor_eef_vis.mp4
│   └── variants/<variant-id>/episode_000000/
│       ├── eef.json
│       └── ik.npz
└── datasets/<variant-id>/
    ├── data/
    ├── videos/
    └── meta/
        ├── info.json
        ├── stats.json
        ├── episodes/
        └── tate_preprocess.json
```

Without temporal trimming, the derived dataset preserves the original RGB
videos without re-encoding. With trimming, exact-frame H.264 clips are created
for all source video keys and image statistics are recomputed from the decoded
clips. `video_mode: hardlink` uses hard links when possible and automatically
falls back to copying. Source locations are retained in
`tate.source_episode_index`, `tate.source_frame_index`, and
`tate.source_timestamp`; derived episode and frame IDs are contiguous from
zero.

The dataset contains raw and final EEF columns:

```text
tate.eef_raw.<side>.pose
tate.eef.<side>.pose
tate.eef.<side>.gripper
tate.eef.<side>.grasp_ratio
tate.eef.<side>.valid
```

When `replace_state_action: true`, retargeted ARX joint targets replace
`observation.state` and `action`. `same_frame` and `next_frame` action alignment
are both supported and recorded in `meta/tate_preprocess.json`.
Set `lerobot.task` explicitly when the source dataset contains a stale task
instruction. The packager then rewrites frame `task_index`, `tasks.parquet`, and
per-episode task metadata together.

## Correction integration

The batch runner does not fit corrections. A correction is fitted separately
from calibration-only real episodes, then configured as a variant with a Python
entrypoint:

```yaml
correction_variants:
  anchor_xyz:
    mode: real_anchor
    artifact: outputs/correction/stack_cube/anchor_xyz.json
    # Optional artifact from `python -m correction.fit_rotation`.
    rotation_artifact: outputs/correction/stack_cube/rotation.json
    options:
      target_mode: mean        # mean | sample
      space: free_xyz          # free_xyz | ray_depth
      propagation: min_bending # linear | min_bending | smooth_spline
```

`real_anchor` directly adapts the migrated `correction.apply` implementation.
Other correction implementations can be attached with
`entrypoint: python.module:function`.
The fitted artifact must exist before the experiment is first executed because
its content fingerprint is part of the immutable experiment identity.

The entrypoint is called with keyword arguments:

```text
source_eef
output_eef
config
episode_index
experiment_id
variant_id
```

The function may accept all fields or only the fields it needs. It must write a
`tate.dual_arm_eef` v2 JSON to `output_eef` without modifying `source_eef`.

See `ALIGNMENT_EVAL_INTERFACE.md` for the evaluation-side manifest and feature
contract.
