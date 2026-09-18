#!/usr/bin/env bash
# Launch one guarded ARX right-arm session against the already-loaded policy.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_APP_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
APP_ROOT=${TATE_APP_ROOT:-$DEFAULT_APP_ROOT}
ROBOT_PYTHON=${TATE_ROBOT_PYTHON:-/home/qijun/ARX5_beta/.venv/bin/python}
ROBOT_SCRIPT="$APP_ROOT/deployment/robot_runner.py"
POLICY_PORT=8019
mode=${1:-}
shift || true

case "$mode" in
  --execute)
    ;;
  *)
    echo "usage: $0 --execute [robot-runner options]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$ROBOT_PYTHON" ]]; then
  echo "ARX robot Python environment missing: $ROBOT_PYTHON" >&2
  exit 2
fi

if pgrep -af 'fold_box_policy_robot.py|run_fold_box_policy.sh|run_reset_arx_home.sh' >/dev/null; then
  echo "fold-box policy or reset task is active; ARX hardware is unavailable" >&2
  exit 3
fi

data_was_active=false
button_was_active=false
systemctl --user is-active --quiet arx-data-station.service && data_was_active=true
systemctl --user is-active --quiet arx-button-control.service && button_was_active=true
cleanup() {
  echo "Restoring original ARX services"
  if [[ "$data_was_active" == true ]]; then
    systemctl --user start arx-data-station.service || true
  fi
  if [[ "$button_was_active" == true ]]; then
    systemctl --user start arx-button-control.service || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ "$button_was_active" == true ]]; then
  echo "Pausing arx-button-control.service to release the ARX arm lock"
  systemctl --user stop arx-button-control.service
  echo "arx-button-control.service stopped"
fi

lock_path=${XDG_RUNTIME_DIR:-/run/user/1000}/arx-arm-control.lock
mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
locked=false
for _ in $(seq 1 25); do
  if flock -n 9; then
    locked=true
    break
  fi
  sleep 0.2
done
if [[ "$locked" != true ]]; then
  echo "another ARX controller still owns the hardware lock" >&2
  fuser -v "$lock_path" >&2 || true
  exit 3
fi
export TATE_ARM_LOCK_FD=9
echo "ARX arm lock acquired"

if pgrep -af 'fold_box_policy_robot.py|run_fold_box_policy.sh|run_reset_arx_home.sh' >/dev/null; then
  echo "fold-box policy or reset task started while acquiring the ARX lock" >&2
  exit 3
fi

systemctl --user stop arx-data-station.service
sleep 2

if ! curl -fsS --max-time 1 "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null; then
  echo "OpenPI policy server is not ready; confirm and load the checkpoint in the 8089 console first" >&2
  exit 4
fi
echo "Reusing OpenPI policy server on 127.0.0.1:$POLICY_PORT"

if "$ROBOT_PYTHON" "$ROBOT_SCRIPT" --execute --policy-url "http://127.0.0.1:$POLICY_PORT" "$@"; then
  exit 0
else
  exit $?
fi
