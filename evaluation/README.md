# TATE trajectory evaluation

The evaluator compares preprocessed EEF variants with held-out ARX trajectories.
It does not modify experiment artifacts. By default, every selected variant is
evaluated on the same set of complete ego episodes.

## Evaluation protocol

- Validate each arm's configured gripper-event sequence.
- Evaluate only the trajectory between its first and last gripper events.
- Align translation with position DTW and evaluate orientation on that path.
- Compute held-out real-to-real leave-one-out noise floors.
- Report the LifEgo metrics `rho_se3`, `D_pos_mm`, `rho_pos`, `D_rot_deg`,
  `rho_rot`, `rho_offset`, `rho_shape`, and `eta_disp`.

Task-specific arms, event sequences, alignment settings, and real-data pairing
are defined in `cfg/evaluation/*.yaml`. The real split must keep calibration and
evaluation episode IDs disjoint.

## Example: Stack Cola v2 (50 ego episodes)

Run from the repository root:

```bash
cd /home/xule/le_ws/TATE

PYTHONPATH=. /home/xule/miniconda3/envs/lifego/bin/python \
  -m evaluation.run_eval \
  --experiment-manifest outputs/experiments/stack_cola_v2_50/manifest.json \
  --real-manifest outputs/arx_real_flange/stack_cola_arx/manifest.json \
  --real-split cfg/evaluation/splits/stack_cola_real_v1.json \
  --eval-config cfg/evaluation/stack_cola_arx.yaml \
  --out outputs/evaluation/stack_cola_v2_50_lifego \
  --resume
```

This evaluates both manifest variants on both arms. With 50 ego episodes and
10 held-out real episodes under `all_pairs`, each variant resolves 500 ego-real
episode pairs. `--resume` is safe for a new output directory and skips compatible
variant results already completed in an interrupted or previous run.

Append `--dry-run` to resolve and print the variants, common ego cohort, real
split, active arms, and pair count without loading trajectories. Useful optional
filters are:

```text
--variants VARIANT_ID ...   evaluate only named variants
--episodes EPISODE_ID ...   use an explicit ego cohort
--limit N                   use the first N resolved ego episodes for debugging
--sides left right          override the active arms
--segment first:last        override the evaluated event-bounded segment
```

If no `--variants` or `--episodes` are supplied, all manifest variants and their
common complete episode cohort are used. Episodes failing loading, event-health,
or alignment checks are recorded rather than silently discarded.

## Outputs

```text
eval_config_resolved.yaml
comparison.csv
comparison.json
real_to_real_noise_floor.json
<variant>/summary.json
<variant>/episodes.csv
<variant>/exclusions.json
<variant>/alignments/<ego>__<real>__<side>.npz
```

Use `comparison.csv` for the compact cross-variant table, `summary.json` for
per-arm and bimanual aggregates, and `exclusions.json` to diagnose missing
episodes or invalid event sequences. `eta_disp` requires at least two healthy
ego episodes; otherwise it is reported as unavailable.
