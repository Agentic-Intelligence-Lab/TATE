# TATE stack-cube ARX deployment

The control panel is a separate user service on the ARX host, listening at
`127.0.0.1:8089`. The existing Fold Box console remains on 8088. Forward the
loopback port over SSH and open `http://127.0.0.1:8089`:

```bash
ssh -L 8089:127.0.0.1:8089 qijun@192.168.2.136
```

Enter the checkpoint **directory on the ARX host** in the page's Checkpoint field.
The default is:

```text
/home/qijun/models/TATE/openpi_checkpoints/arx_eef/stack_cube_cotrain_camdrop_pi05_bs16/2857
```

Place a **completed** `model.safetensors` and `norm_stats.json` directly in that
directory. The previous `assets/local/arx_eef_stack_cube_cotrain_camdrop_train/`
layout remains supported when no flat `norm_stats.json` is present. Other paths
can be selected for checkpoints trained with the same ARX EEF model and asset id.
The page checks the safetensors header against the file length;
OpenPI performs the actual model load. For the default checkpoint, the known
size is 7,473,091,464 bytes and SHA-256 is
`0085595d0582d397737a813da06be505dca8040034397cb578d218c803fb1bc5`.
The PaliGemma tokenizer under `/home/qijun/.cache/openpi/big_vision/` is already
on the ARX host.
An incomplete `model.safetensors.partial` does not count as the model.

The deployment scripts use the repository containing them as `TATE_APP_ROOT` by
default. After cloning TATE to `/home/qijun/TATE`, start the page directly with:

```bash
cd /home/qijun/TATE
./deployment/run_ui.sh
```

The existing OpenPI checkout remains at `/home/qijun/models/TATE/openpi`. It is
separate from the TATE repository and can be reused after cloning. To install or
refresh its isolated inference environment with Python 3.11, matching training,
without changing the ARX SDK environment:

```bash
cd /home/qijun/TATE
TATE_OPENPI_ROOT=/home/qijun/models/TATE/openpi \
  ./deployment/install_openpi_env.sh
```

For a persistent user service, copy `deployment/tate-arx-ui.service` to
`~/.config/systemd/user/`. Its default checkout location is `%h/TATE`, namely
`/home/qijun/TATE` for user `qijun`.

The model-load and dummy-image inference buttons are optional. Preparing a real
session loads the model, observes the live cameras and right-arm state, and
shows the first guarded target. Send `RUN` after reviewing that target.
The preparation phase does not send
joint or gripper motion commands. Hardware execution is right-arm-only, defaults
to an action pacing limit of 5 Hz and 50 actions per inference, and stops after
five minutes. On the training L20Y, dummy-image inference without Torch
compilation took 1.09 seconds, so the actual inference rate is lower than the selected limit.
The console's gripper threshold defaults to `0.5`: a predicted continuous grasp
value greater than this value commands closure; otherwise it commands opening.
The console persists runner and policy output in a timestamped file under
`/home/qijun/TATE/logs/` by default. Set `TATE_LOG_DIR` before starting the
service to store these logs elsewhere.
Pause, stop, and the existing ARX reset wrapper are available on the page. The hardware
wrapper takes the shared arm lock and restores the original ARX data and button
services when the session exits.

The controls remain disabled while the model or OpenPI environment is missing.
The page and camera previews can be inspected before transferring the model.
