#!/usr/bin/env bash
set -euo pipefail
APP_ROOT=${TATE_APP_ROOT:-/home/qijun/models/TATE/app}
ROBOT_PYTHON=${TATE_ROBOT_PYTHON:-/home/qijun/ARX5_beta/.venv/bin/python}
exec "$ROBOT_PYTHON" "$APP_ROOT/deployment/control_ui.py"
