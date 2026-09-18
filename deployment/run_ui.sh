#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_APP_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
APP_ROOT=${TATE_APP_ROOT:-$DEFAULT_APP_ROOT}
ROBOT_PYTHON=${TATE_ROBOT_PYTHON:-/home/qijun/ARX5_beta/.venv/bin/python}
exec "$ROBOT_PYTHON" "$APP_ROOT/deployment/control_ui.py"
