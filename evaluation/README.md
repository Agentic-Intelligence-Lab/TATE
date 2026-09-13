# TATE alignment, real-anchor correction, and evaluation

This implementation targets `ALIGNMENT_EVAL_INTERFACE.md`. It normalizes EEF
sidecars, real ARX FK JSON, and derived LeRobot EEF columns into one dual-arm
trajectory model before event extraction, timestamp resampling, alignment, or
metrics.

The migrated LifEgo logic retained here includes:

- gripper-transition anchors derived from signal direction rather than event parity;
- evaluation strictly between each trajectory's first and last gripper events;
- translation-based DTW with orientation evaluated along the position path;
- held-out real-to-real leave-one-out noise floors;
- the LifEgo metrics `rho_se3`, `D_pos_mm`, `rho_pos`, `D_rot_deg`, `rho_rot`,
  `rho_offset`, `rho_shape`, and `eta_disp`.

LifEgo-specific directory globbing, Nero TCP reconstruction, single-arm CSV,
and implicit spatial fitting are intentionally not retained.

## 1. Fit a correction from calibration-only real episodes

```bash
python -m correction.fit \
  --real-manifest /path/to/real_fk/manifest.json \
  --real-split cfg/evaluation/splits/stack_cube_real_v1.json \
  --eval-config cfg/evaluation/stack_cube_arx.yaml \
  --out outputs/corrections/stack_cube_anchor_xyz.json
```

The artifact records every real calibration episode and the unused held-out
episode IDs. It never modifies an EEF trajectory.

`correction.anchors` accepts gripper-event ordinals together with `start` and
`end`. Explicit endpoint anchors are fitted from the first/last valid real pose
and override the propagation layer's legacy zero-displacement endpoint pins.

Batch correction variants with `mode: pose_correction` may specify a position
`artifact`, a `rotation_artifact`, or both. This supports clean none / position /
rotation / position+rotation ablations without applying a dummy position fit.

An optional task-level local rotation bias can be fitted from one uncorrected
experiment variant against the same real calibration split:

```bash
python -m correction.fit_rotation \
  --experiment-manifest /path/to/experiment/manifest.json \
  --variant finger_center__none \
  --real-manifest /path/to/real_fk/manifest.json \
  --real-split cfg/evaluation/splits/stack_cube_real_v1.json \
  --eval-config cfg/evaluation/stack_cube_arx.yaml \
  --out outputs/corrections/stack_cube_rotation.json
```

The held-out real IDs are not loaded during fitting. Use at least two healthy
ego episodes; the artifact records their IDs and the per-ego fit spread.

## 2. Apply the fitted artifact as a new immutable EEF variant

```bash
python -m correction.apply \
  --input /path/to/eef_raw.json \
  --correction outputs/corrections/stack_cube_anchor_xyz.json \
  --rotation-correction outputs/corrections/stack_cube_rotation.json \
  --out /path/to/variant/eef.json \
  --space free_xyz \
  --propagation min_bending
```

`linear`, `min_bending`, and `smooth_spline` propagation are available. Batch
preprocessing owns the final variant path, fingerprint, and experiment manifest.
`free_xyz` reproduces unconstrained anchor displacement; `ray_depth` moves each
valid EEF point only along its calibrated camera ray. With `--target-mode
sample`, targets use a stable per-episode/per-side/per-anchor seed;
`--max-mahalanobis` optionally truncates outlier samples.

## 3. Resolve an evaluation without loading trajectories

```bash
python -m evaluation.run_eval \
  --experiment-manifest outputs/experiments/stack_cube_ablation_v1/manifest.json \
  --variants finger_center__none finger_center__anchor_xyz \
  --real-manifest outputs/cache/real_fk/stack_cube_arx/HASH/manifest.json \
  --real-split cfg/evaluation/splits/stack_cube_real_v1.json \
  --eval-config cfg/evaluation/stack_cube_arx.yaml \
  --out outputs/evaluation/stack_cube_ablation_v1 \
  --dry-run
```

The dry run prints the fixed common ego cohort, held-out real IDs, active arms,
pair count, and output path.

## 4. Run evaluation

Remove `--dry-run` from the command above. Outputs follow the contract:

```text
eval_config_resolved.yaml
comparison.json
comparison.csv
real_to_real_noise_floor.json
<variant>/summary.json
<variant>/episodes.csv
<variant>/exclusions.json
<variant>/alignments/<ego>__<real>__<side>.npz
```

The task configs default to `--segment first:last`. Use `--episodes ...` for an
explicit fixed ego cohort, and `--sides left right` for a bimanual task.
`eta_disp` is a within-ego-set dispersion ratio, so it is reported as unavailable
when fewer than two ego episodes are selected. Comparison tables use compact IDs
such as `finger_center` and `finger_center+xyz`; `comparison.json` records their
mapping to full manifest IDs.
