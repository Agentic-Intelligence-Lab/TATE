# Real-data splits

Real calibration and held-out evaluation episodes must be selected explicitly.
Do not generate or change a split inside the evaluator.

Example:

```json
{
  "schema": "tate.real_evaluation_split",
  "schema_version": 1,
  "task": "stack_cube",
  "dataset_fingerprint": "<source_real_dataset fingerprint from the experiment manifest>",
  "real_manifest_fingerprint": "<sha256 of the real FK manifest>",
  "calibration": [0, 1, 2],
  "eval": [10, 11, 12],
  "reserve": [],
  "seed": 0
}
```

Both fingerprint fields are optional for legacy data, but should be included in
new splits. The evaluator rejects fingerprint mismatch, overlap among all three
cohorts, duplicate/missing IDs, and any correction fitted with a selected
evaluation episode.
