# Batch Preprocessing Contract for Alignment and Evaluation

## 1. Purpose and status

This document defines the interface expected between TATE's planned batch ego
preprocessing pipeline and the alignment/evaluation code that is being migrated
from LifEgo.

It is an interface contract, not a description of an already completed batch
implementation. The alignment/evaluation migration should target this contract
instead of depending on temporary output paths such as
`outputs/<session>/preprocess/eef.json`.

The main design goals are:

1. The same raw ego episodes and shared WiLoR caches can produce multiple EEF
   variants without overwriting one another.
2. A variant is defined by both its `hand2gripper` configuration and its
   real-data correction configuration.
3. The processed ego result can be materialized as a LeRobot dataset for policy
   training while remaining directly usable for trajectory evaluation.
4. Alignment and evaluation are dual-arm aware and work for both single-active-
   arm and bimanual tasks.
5. Fitted correction and held-out evaluation are strictly separated to prevent
   data leakage.

## 2. Ownership boundaries

The expected data flow is:

```text
raw ego LeRobot RGB
    -> shared WiLoR cache
    -> raw EEF for one hand2gripper variant
    -> corrected/final EEF for one correction variant
    -> IK retargeting
    -> derived LeRobot dataset

raw ARX LeRobot joints
    -> real FK trajectories
    -> calibration-only correction fitting

final ego EEF + held-out real FK
    -> temporal alignment
    -> metrics and reports
```

The responsibilities are intentionally separated:

- Batch preprocessing owns episode discovery, artifact production, variant
  identity, provenance, manifests, and LeRobot materialization.
- Correction fitting owns learned parameters derived from real calibration
  episodes. Applying a correction creates a new immutable EEF variant.
- Alignment owns temporal/event correspondence only. It must not silently fit
  or apply a spatial correction.
- Evaluation owns metric computation and aggregation. It must not modify input
  trajectories or datasets.
- Policy packaging consumes the same final EEF/IK artifacts but must not become
  a dependency of the evaluator.

An evaluator may read either the EEF sidecar JSON or the EEF columns in the
derived LeRobot dataset. Both representations must originate from the same
canonical final EEF artifact and carry its fingerprint.

## 3. Current TATE inputs that must be supported

### 3.1 Ego EEF sidecar

The current exporter writes schema:

```text
tate.dual_arm_eef, schema_version = 2
```

Relevant top-level fields are:

```text
fps
total_frames
source_video
source_wilor_cache
arm_mode                   # single_arm | bimanual
active_sides               # [right], [left], or [left, right]
temporal_window            # source frame range and applied trim
eef_coordinate_convention
hand2gripper
tcp_local_transform
arm_transforms
frames
```

Each frame contains `idx`, `ts`, `hand_l`, and `hand_r`. A hand entry may be
`null`. When present, the evaluator should use:

```text
tcp_pose_eef_frame       # canonical 4x4 pose for comparison
grasp_state             # 0=open, 1=closed
grasp_ratio
confidence
eef_frame               # e.g. left_flange_zero/right_flange_zero
```

`eef_pose_world` is currently a compatibility alias of
`tcp_pose_eef_frame`. New evaluation code should prefer
`tcp_pose_eef_frame` and validate `pose_semantics == "arx_tcp"`.

### 3.2 Real ARX FK trajectory

The current real-data adapter writes schema:

```text
tate.arx_real_flange_trajectory, schema_version = 1
```

Each frame contains:

```text
idx
frame_index
timestamp_s
arms.left.T_tcp_in_output_frame
arms.right.T_tcp_in_output_frame
arms.<side>.output_frame
arms.<side>.gripper.binary
arms.<side>.gripper.continuous
```

The real dataset manifest uses:

```text
tate.arx_real_flange_dataset_manifest, schema_version = 1
```

and maps `episode_index` to each real trajectory JSON. Alignment/evaluation
should accept this manifest as the normal real-data input, rather than requiring
the caller to enumerate JSON files manually.

### 3.3 Coordinate-frame requirement

Comparisons are valid only when the two poses for a side use the same output
frame:

```text
ego hand_l.eef_frame == real arms.left.output_frame
ego hand_r.eef_frame == real arms.right.output_frame
```

The expected frames are currently `left_flange_zero` and
`right_flange_zero`. Left and right trajectories must never be transformed into
one another's local frame for metric computation. A mismatch must be a hard
error, not a warning.

## 4. Batch preprocessing manifest expected by evaluation

The planned batch runner will provide one authoritative experiment manifest.
The alignment/evaluation migration should consume a manifest with the following
logical content, while allowing additive fields in future schema versions:

```json
{
  "schema": "tate.preprocess_experiment_manifest",
  "schema_version": 1,
  "experiment_id": "stack_cola_ablation_v1",
  "source_ego_dataset": {
    "path": "/abs/path/to/stack_cola_ego",
    "fingerprint": "..."
  },
  "source_real_dataset": {
    "path": "/abs/path/to/stack_cola_arx",
    "fingerprint": "..."
  },
  "variants": {
    "finger_center__none": {
      "hand2gripper_id": "finger_center",
      "correction_id": "none",
      "config_fingerprint": "...",
      "derived_lerobot_dataset": "/abs/path/to/dataset",
      "episodes": {
        "0": {
          "status": "complete",
          "wilor_cache": "/abs/path/to/wilor_hands.json",
          "raw_eef": "/abs/path/to/eef_raw.json",
          "final_eef": "/abs/path/to/eef.json",
          "final_eef_fingerprint": "...",
          "ik": "/abs/path/to/ik.npz",
          "valid_frames": {"left": 557, "right": 540},
          "grasp_events": {"left": 4, "right": 4}
        }
      }
    }
  }
}
```

Evaluation must select episodes from this manifest, not by recursively globbing
arbitrary output directories. It should:

- include only episodes whose required stage is `complete`;
- report excluded and failed episode IDs;
- use a fixed requested episode cohort across variants;
- report when variants do not have identical episode coverage;
- verify the final EEF fingerprint when both JSON and LeRobot forms exist.

The manifest path, manifest fingerprint, variant ID, configuration fingerprint,
and selected episode IDs must be copied into every evaluation result.

## 5. Expected derived LeRobot EEF features

Each processed variant will eventually be materialized as an independent
LeRobot dataset. Derived episode/frame/timestamp values are reindexed from zero;
the original coordinates are retained in `tate.source_episode_index`,
`tate.source_frame_index`, and `tate.source_timestamp`. The following numeric
features are expected for evaluation:

```text
tate.eef_raw.left.pose       float32[7]  x,y,z,qx,qy,qz,qw
tate.eef_raw.right.pose      float32[7]
tate.eef.left.pose           float32[7]  final/post-correction EEF
tate.eef.right.pose          float32[7]
tate.eef.left.gripper        int64[1] or bool[1]
tate.eef.right.gripper       int64[1] or bool[1]
tate.eef.left.grasp_ratio    float32[1]
tate.eef.right.grasp_ratio   float32[1]
tate.eef.left.valid          bool[1]
tate.eef.right.valid         bool[1]
tate.source_episode_index    int64[1]
tate.source_frame_index      int64[1]
tate.source_timestamp        float64[1]
```

Quaternion order is explicitly `xyzw`. The pose is the ARX TCP pose expressed
in the corresponding side's zero-flange frame. The dataset metadata must record
the frame name for each side.

Missing detections must not be encoded as a valid zero pose. They should have a
fixed numeric placeholder plus `valid=false`; consumers must always use the
valid mask. The source frame must not be removed merely because one hand is
missing, since that would break RGB/timestamp correspondence.

For policy training, the same dataset may additionally contain retargeted
`observation.state` and `action`. These fields are not the source of truth for
EEF evaluation; evaluation should read the `tate.eef.*` fields.

The LeRobot loader is a required adapter, but EEF JSON support should be kept
for debugging, visualization, replay, and compatibility during migration.

## 6. Common internal trajectory model

All input adapters should normalize data into one representation before any
alignment or metric code runs. A suggested logical type is:

```python
DualArmTrajectory(
    episode_id: int | str,
    timestamps_s: ndarray[N],
    sides: {
        "left": ArmTrajectory(
            pose_xyzw: ndarray[N, 7],
            valid: ndarray[N, bool],
            gripper_binary: ndarray[N],
            gripper_continuous: ndarray[N] | None,
            grasp_ratio: ndarray[N] | None,
            frame_name: str,
        ),
        "right": ArmTrajectory(...),
    },
    source: dict,
)
```

Required adapters are:

1. `tate.dual_arm_eef` JSON loader.
2. `tate.arx_real_flange_trajectory` JSON loader.
3. Derived LeRobot dataset/episode loader.

Schema-specific field handling must remain inside these loaders. Alignment and
metric functions should operate only on the normalized representation.

Loaders must validate:

- monotonically increasing timestamps;
- finite values on valid frames;
- normalized quaternions;
- positive FPS or usable timestamps;
- pose/frame convention compatibility;
- gripper convention `0=open, 1=closed`;
- equal lengths for all arrays belonging to one episode.

## 7. Alignment expectations

### 7.1 Spatial alignment is not temporal alignment

The EEF exporter and real FK adapter are responsible for expressing poses in
matching ARX frames. The alignment stage must not silently estimate an SE(3)
transform to compensate for mismatched frames.

Any fitted position/orientation correction is an explicit preprocessing
variant, with its own artifact, real calibration split, and variant ID. The
evaluator compares the resulting variant as-is.

### 7.2 Event extraction

Event extraction must work independently for left and right arms. Events are
transitions in the binary gripper sequence:

```text
0 -> 1: close
1 -> 0: open
```

Requirements:

- Preserve the transition type and source frame/time.
- Do not count the initial state as a transition.
- Respect validity gaps; do not create a transition across an unbounded missing
  interval without recording that ambiguity.
- Task-specific expected event sequences and critical segments belong in
  configuration, not hard-coded conditionals scattered across the evaluator.
- A side with no expected activity is inactive, not failed. Activity policy
  must come from the task/evaluation config.

The alignment report must retain detected events even when the episode is
excluded because its sequence is unhealthy.

### 7.3 Resampling and pose interpolation

Ego and real episodes may differ in duration and sample count. Resampling must
use physical timestamps:

- linear interpolation for translation and continuous gripper values;
- quaternion SLERP for orientation;
- nearest/step interpolation for binary gripper state;
- no interpolation across invalid gaps larger than a configurable threshold.

The target sample count/rate and all gap thresholds must be recorded in the
evaluation configuration and output.

### 7.4 Alignment modes

The migration should support at least:

1. Full-trajectory alignment.
2. Event-to-event segment alignment, such as pickup-close to place-open.
3. Position-DTW within the selected full trajectory or event segment.

DTW cost and constraints must be configurable. The initial compatible default
may use translation distance, but the saved output must identify the exact cost
definition. The alignment path should be retained per evaluated pair so metric
results are reproducible and debuggable.

Do not assume ego episode `N` corresponds to real episode `N`. Ego and real
sets currently have different sizes. Pairing is an explicit evaluation policy,
for example:

```text
all_pairs
fixed_pairs_from_manifest
```

The default comparison for distribution-level evaluation should be `all_pairs`
over the selected ego cohort and held-out real cohort.

## 8. Evaluation expectations

### 8.1 Split safety

Real episodes used to fit a correction must never enter the held-out reference
set used to score that correction.

A split manifest should provide at least:

```json
{
  "task": "stack_cola",
  "calibration": [0, 1, 2],
  "eval": [10, 11, 12],
  "reserve": [],
  "seed": 0
}
```

Evaluation must fail if:

- calibration and eval episode IDs overlap;
- a correction artifact says it used any selected eval episode;
- the split task/dataset fingerprint does not match the requested real dataset;
- the split references missing episodes.

The evaluator must never create or refit a correction automatically. Fitting
and evaluation are separate commands/stages.

### 8.2 Minimum metric layers

Metrics should be computed per side and then explicitly aggregated. At minimum
the migrated evaluator should expose:

- translation error over the alignment path: mean, median, RMSE, p90;
- orientation geodesic error in degrees: mean, median, p90;
- trajectory shape/relative-motion error after subtracting the segment start;
- duration and path-length statistics;
- grasp event count and event-sequence validity;
- event-anchor position and orientation errors;
- valid-frame/detection coverage;
- alignment diagnostics such as DTW normalized cost and path length.

Metrics that intentionally remove a global offset or fit a transform must be
named separately from absolute metrics. They must not replace the absolute
score.

For bimanual tasks, output should contain:

```text
left metrics
right metrics
bimanual aggregate over active sides
optional inter-arm relative-pose metrics
```

The inter-arm metric is meaningful only after both side poses are expressed in
a documented common scene frame. The current per-side zero-flange poses cannot
be subtracted directly. If the evaluator uses
`T_eef_frame_in_scene` to construct scene-frame poses, it must record the exact
calibration fingerprint used.

### 8.3 Aggregation

Aggregation should occur in layers:

1. aligned ego-real pair;
2. ego episode across selected real references;
3. variant across ego episodes;
4. optional task-level summary across active sides.

This prevents long episodes or a larger number of valid frames from silently
dominating the result. Reports must include sample counts, excluded counts, and
dispersion, not only one mean value.

Variant comparison must use the same ego cohort, same held-out real set, same
event definition, and same alignment configuration. A comparison report should
refuse or prominently mark non-comparable runs.

## 9. Expected code structure

The alignment/evaluation migration is expected to produce a structure similar
to:

```text
evaluation/
├── __init__.py
├── types.py                 # normalized trajectory/result types
├── schemas.py               # schema/version and convention validation
├── loaders/
│   ├── eef_json.py
│   ├── real_fk_json.py
│   └── lerobot.py
├── events.py                # per-side gripper event extraction
├── resample.py              # timestamp interpolation and validity gaps
├── align.py                 # segment selection and DTW
├── metrics.py               # pure per-pair/per-side metrics
├── aggregate.py             # episode/variant aggregation
├── run_eval.py              # manifest-driven CLI
└── report.py                # JSON/CSV/Markdown reporting
```

Correction code should be kept outside the evaluator, for example:

```text
correction/
├── fit.py
├── apply.py
└── methods/
```

Pure algorithms migrated from LifEgo may be reused, but LifEgo-specific path
discovery, Nero assumptions, single-hand assumptions, and CSV/JSONL coupling
should not be carried into the new core modules.

## 10. CLI expectations

Exact option names may change, but the normal batch evaluation interface should
be manifest-driven:

```bash
python -m evaluation.run_eval \
  --experiment-manifest outputs/experiments/stack_cola_ablation_v1/manifest.json \
  --variants finger_center__none finger_center__anchor_xyz \
  --real-manifest outputs/cache/real_fk/stack_cola_arx/<hash>/manifest.json \
  --real-split cfg/evaluation/splits/stack_cola_real_v1.json \
  --eval-config cfg/evaluation/stack_cola.yaml \
  --out outputs/evaluation/stack_cola_ablation_v1
```

Useful selection/debug options should include:

```text
--episodes
--sides
--segment
--pairing
--limit
--dry-run
--resume
--force
```

`--dry-run` should resolve and print variants, ego episode IDs, real eval IDs,
active sides, pair count, and output paths without loading all trajectory data.

## 11. Evaluation output contract

Suggested output layout:

```text
outputs/evaluation/<experiment_id>/
├── eval_config_resolved.yaml
├── comparison.json
├── comparison.csv
└── <variant_id>/
    ├── summary.json
    ├── episodes.csv
    ├── exclusions.json
    └── alignments/
        └── <ego_id>__<real_id>__<side>.npz
```

Every `summary.json` should record:

- experiment and variant identity;
- source manifest paths and fingerprints;
- code commit and dirty state;
- selected ego and real episode IDs;
- real split identity;
- active-side policy;
- complete resolved alignment/metric configuration;
- per-side and aggregate metric statistics;
- failures and exclusions with reasons.

Evaluation outputs are derived artifacts and must never be written into the
source or derived LeRobot dataset directories.

## 12. Required compatibility and acceptance checks

The migrated alignment/evaluation implementation should satisfy the following
before the batch pipeline depends on it:

1. Load the existing single-episode `tate.dual_arm_eef` v2 output.
2. Load the existing `tate.arx_real_flange_trajectory` v1 output and manifest.
3. Produce identical normalized trajectories from an EEF sidecar and from the
   same episode's derived LeRobot EEF columns.
4. Correctly evaluate a right-arm-only task without treating the absent left
   arm as a zero trajectory or a failed episode.
5. Correctly evaluate left and right arms independently for a bimanual task.
6. Reject mismatched coordinate frames and quaternion conventions.
7. Reject a real split with calibration/eval leakage.
8. Preserve and report invalid WiLoR intervals.
9. Resolve the same ego episode cohort for every compared variant.
10. Reproduce a saved pair result from its recorded alignment path and config.

## 13. Integration rule for both migrations

To allow batch preprocessing and alignment/evaluation to be developed in
parallel, both sides should depend only on these stable boundaries:

- `tate.dual_arm_eef` v2 as the current canonical EEF sidecar;
- `tate.arx_real_flange_trajectory` v1 and its dataset manifest for real data;
- the planned experiment manifest described in Section 4;
- the planned `tate.eef.*` LeRobot columns described in Section 5;
- explicit schema versions, coordinate-frame names, variant IDs, episode IDs,
  and artifact fingerprints.

If alignment/evaluation needs information not present in these boundaries, it
should add a documented manifest/schema field rather than infer it from a file
or directory name.
