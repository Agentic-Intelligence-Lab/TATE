# WiLoR hand and dual-arm EEF preprocessing

The RGB pipeline has three independent stages:

1. ego RGB -> camera-frame WiLoR hand cache;
2. hand cache -> poses in the left/right ARX zero-flange frames;
3. video + cache (+ optional EEF) -> diagnostic visualization video.

The split is intentional: WiLoR inference is expensive, while camera
extrinsics and downstream correction may change repeatedly. Stage 2 can be
rerun without invoking WiLoR.

## Single-episode validation

From the repository root:

```bash
PY=/home/xule/miniconda3/envs/lifego/bin/python
VIDEO=/home/xule/le_ws/Data_TATE/stack_cube_ego/videos/observation.images.head/chunk-000/file-000.mp4
SESSION=outputs/stack_cube_ego_ep000

$PY -m preprocess.reconstruct_wilor \
  --video "$VIDEO" \
  --session "$SESSION" \
  --cfg cfg/preprocess/base/Preprocess.yaml \
  --wilor-pretrained-dir /home/xule/le_ws/LifEgo/.cache/wilor_mini

$PY -m preprocess.export_eef \
  --hands "$SESSION/preprocess/wilor_hands.json" \
  --camera-calibration cfg/preprocess/base/RealSenseD405.yaml \
  --eef-config cfg/preprocess/base/EEFExport.yaml \
  --out "$SESSION/preprocess/eef.json"

$PY -m preprocess.visualize_wilor_cache \
  --video "$VIDEO" \
  --hands "$SESSION/preprocess/wilor_hands.json" \
  --eef "$SESSION/preprocess/eef.json" \
  --eef-config cfg/preprocess/base/EEFExport.yaml \
  --out "$SESSION/preprocess/wilor_eef_vis.mp4"
```

Add `--max-frames 40` to the first command for a short reconstruction run.
Visualization is an independent cache consumer, so it can be rerun without
WiLoR. Its panels show thumb-index distance, wrist-middle-MCP palm size,
`grasp_ratio`, both hysteresis thresholds, the current band, and final state.
Use `--grasp-close-ratio` and `--grasp-open-ratio` to preview candidate values.

The compatibility orchestrator remains available:

```bash
$PY -m preprocess.Preprocess \
  --mps_path "$SESSION" --video_path "$VIDEO" --stage all --no-video
```

`--stage wilor` and `--stage eef --hands PATH` run either half separately.

## Outputs

```text
<session>/preprocess/
├── wilor_hands.json              # reusable, camera-frame reconstruction
├── eef.json                      # dual-arm zero-flange-frame targets
└── hand_keypoints_eef_vis.mp4    # optional overlay
```

`wilor_hands.json` records the exact intrinsic profile, WiLoR focal length,
handedness policy, and depth source. `eef.json` records both calibrated
`T_cam_in_*_arm_base` matrices and the transforms actually applied.

The EEF stage reconstructs the selected LifEgo hand-to-gripper frame directly
from cached 3D keypoints. Set `hand2gripper.mode` in `EEFExport.yaml` (or pass
`--hand2gripper-mode`) to `finger_center`, `humanego`, or `qwen`.
`finger_center` uses the thumb-to-index MCP jaw axis plus a wrist-to-four-finger
MCP-centroid forward seed; `humanego` uses the original wrist-to-thumb/index-MCP
midpoint seed; `qwen` uses Qwen-RobotManip's 0.7 index + 0.3 middle virtual tip.
It computes grasp state with the configured close/open hysteresis band, then
applies the side-specific `T_tcp_in_hand` from the camera calibration. A side
without an explicit matrix falls back to `T_hand_to_ee @ T_ee_axis_correct`.
The current right-hand matrix maps TCP axes to hand
`[forward, -jaw, up]`, so `tcp_pose_eef_frame` already has the ARX real-robot
TCP direction convention. The compatibility alias
`eef_pose_world` contains the same pose in the per-hand `right_flange_zero` or
`left_flange_zero` frame. Retargeting requires this TCP-ready representation
and applies no additional orientation correction.

## Calibration safety

Intrinsics are selected by the decoded video resolution. Automatic scaling is
not used: a 16:9-to-4:3 change may crop the sensor and cannot safely be inferred
from width and height alone. The current D405 YAML contains exact 1280x720 and
640x480 profiles.
