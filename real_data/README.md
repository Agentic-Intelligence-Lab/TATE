# ARX real-data adapter

`arx_lerobot_adapter.py` converts ARX LeRobot joint data into TCP and flange
poses expressed in each arm's zero-flange coordinate frame:

- `left_flange_zero`
- `right_flange_zero`

It reads joint values from `observation.state` by default, runs FK against the
ARX MuJoCo scene, and does not modify the source dataset.

Each arm/frame stores:

- the six raw joint positions;
- `T_flange_in_output_frame`;
- `T_tcp_in_output_frame`;
- the original gripper scalar as `gripper.raw`;
- `gripper.continuous` in `[0, 1]`, with fully open `0` and fully closed `1`;
- `gripper.binary`, with open `0` and closed `1`;
- the corresponding raw/continuous/binary `action` values when available.

The default ARX raw calibration is open `-3.4`, closed `0.1`, with a binary
threshold of `-2.6`. Values greater than or equal to `-2.6` are classified as
closed. The threshold is configurable from the CLI and is recorded in every
output file. With the current recordings, the expected state sequence begins
with closed, followed by open, closed, and open.

## Convert one episode

```bash
PY=/home/xule/miniconda3/envs/lifego/bin/python

$PY -m real_data.arx_lerobot_adapter \
  --input /home/xule/le_ws/Data_TATE/stack_cube_arx/data/chunk-000/file-000.parquet \
  --out outputs/arx_real_flange/stack_cube_arx_ep000
```

The episode is written as `episode_000000.json`, accompanied by
`manifest.json`.

## Convert the complete dataset

```bash
PY=/home/xule/miniconda3/envs/lifego/bin/python

$PY -m real_data.arx_lerobot_adapter \
  --input /home/xule/le_ws/Data_TATE/stack_cube_arx \
  --out outputs/arx_real_flange/stack_cube_arx
```

To test another threshold without overwriting the first result, use a new
output directory:

```bash
$PY -m real_data.arx_lerobot_adapter \
  --input /home/xule/le_ws/Data_TATE/stack_cube_arx \
  --out outputs/arx_real_flange/stack_cube_arx_threshold_m2p7 \
  --gripper-binary-threshold-raw -2.7
```
