# WiLoR Hand / EEF Preprocessing

This directory now keeps only the WiLoR hand reconstruction path. It reconstructs
per-frame hand keypoints and an end-effector-style midpoint pose from an existing
RGB + camera-metadata sequence.

## Inputs

The runner expects frames laid out as:

```text
<mps_path>/preprocess/all_data/<idx>/
├── rgb.png
└── aria_cam_rgb.json
```

`aria_cam_rgb.json` must provide camera intrinsics `k`, camera pose `c2w`, timestamp
`ts`, and optionally `fps`.

## Run

```bash
python -m preprocess.Preprocess \
  --mps_path <mps_path> \
  --cfg_path cfg/preprocess/base/AriaHands.yaml
```

Batch mode still scans `mps_*_<index>_vrs` folders:

```bash
python -m preprocess.Preprocess \
  --mps_path <parent_dir> \
  --range 0 10 \
  --cfg_path cfg/preprocess/base/AriaHands.yaml
```

## Outputs

For each frame, WiLoR writes:

```text
<mps_path>/preprocess/all_data/<idx>/wilor_hands.json
```

The EEF-style pose is stored per hand as:

```json
{
  "hand_r": {
    "midpoint_pose_opt_world": [[...], [...], [...], [...]],
    "midpoint_translation_opt_world": [...],
    "midpoint_orientation_opt_world": [[...], [...], [...]],
    "grasp_state": 0
  }
}
```

If `--no-video` is not used, a visualization video is written to:

```text
<mps_path>/preprocess/vis/wilor_hands_vis.mp4
```

## Remaining Files

- `Preprocess.py`: thin CLI/batch wrapper around WiLoR.
- `WiLoRHands.py`: WiLoR reconstruction and per-frame JSON export.
- `AriaCamTypes.py`: lightweight camera/frame data containers used by WiLoR.
- `AriaHandsTypes.py`: hand data containers and JSON serialization.
- `AriaHandsOptimizer.py`: temporal smoothing and velocity estimation.
- `AriaHandsOps.py`: optional visualization and analysis helpers.
- `cfg/preprocess/base/AriaHands.yaml`: WiLoR hand-processing configuration.
