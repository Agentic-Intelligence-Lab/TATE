# ARX head-camera extrinsic calibration

This directory calibrates the fixed RealSense D405 camera against either ARX
arm base using AprilTags. The known tag-corner coordinates and `--arm` must
refer to the same arm-base frame.

The result is written to one of these independent YAML fields:

- `arm_extrinsics.left.T_cam_in_left_arm_base`
- `arm_extrinsics.right.T_cam_in_right_arm_base`

The shared legacy `extrinsics.c2w` field is intentionally left unchanged: one
arm-base PnP solve does not determine TATE's shared scene frame.

## Dependencies

Run the scripts in the TATE environment. Calibration needs `numpy`, `scipy`,
and an OpenCV build with `cv2.aruco` (normally `opencv-contrib-python`). Direct
RealSense capture additionally needs `pyrealsense2`. Tag family `41h12` needs
`pupil-apriltags`; the default `36h11` family does not.

## Calibrate from the camera

Left arm:

```bash
python calibration/calibrate_realsense_extrinsics.py \
  --arm left \
  --capture --capture-count 3 \
  --tag-corners-base path/to/tag_corners_base_arx_left.json \
  --out outputs/camera_extrinsics/left
```

Right arm:

```bash
python calibration/calibrate_realsense_extrinsics.py \
  --arm right \
  --capture --capture-count 3 \
  --tag-corners-base path/to/tag_corners_base_arx_right.json \
  --out outputs/camera_extrinsics/right
```

By default, a successful run updates
`cfg/preprocess/base/RealSenseD405.yaml`, including the image resolution,
intrinsics, distortion, selected arm's extrinsic, and result provenance. Pass
`--no-update-yaml` to produce diagnostics without changing the YAML. Existing
images can be supplied with `--images image1.png image2.png` instead of
`--capture`.

With the RealSense capture backend, the script reads intrinsics directly from
the active color stream profile. The default camera serial and the saved
factory-intrinsics fallback in `configs/` correspond to the current ARX head
D405 at 1280x720. Supply `--camera-serial` for another RealSense, or
`--intrinsics-json` when processing existing images from another camera.

## Import an existing calibration result

```bash
python calibration/update_yaml_from_result.py \
  --arm left \
  --result /path/to/camera_extrinsics_arx_left.json

python calibration/update_yaml_from_result.py \
  --arm right \
  --result /path/to/camera_extrinsics_arx_right.json
```

Add `--dry-run` to validate and print the proposed YAML without writing it.
The stored SHA-256 digest lets a result file be checked later even though only
its portable basename is recorded in the repository.

## Tag-corner convention

Each tag entry is the 3D position, in metres, of its `top_left` corner as seen
by the camera when the printed tag is upright. The solver estimates the other
corners from `tag_size_m`, including an unknown tag yaw and common chirality.
The detector currently applies `ARUCO_CORNER_REORDER = [2, 3, 0, 1]`; verify
this ordering if the tag artwork or printing pipeline changes.
