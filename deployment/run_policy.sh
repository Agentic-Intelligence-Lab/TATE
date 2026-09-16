#!/usr/bin/env bash
# Launch OpenPI offline checks or one guarded ARX right-arm session.
set -euo pipefail

APP_ROOT=${TATE_APP_ROOT:-/home/qijun/models/TATE/app}
POLICY_PYTHON=${TATE_POLICY_PYTHON:-/home/qijun/models/TATE/openpi/.venv/bin/python}
ROBOT_PYTHON=${TATE_ROBOT_PYTHON:-/home/qijun/ARX5_beta/.venv/bin/python}
POLICY_SCRIPT="$APP_ROOT/deployment/serve_policy.py"
ROBOT_SCRIPT="$APP_ROOT/deployment/robot_runner.py"
POLICY_PORT=8019
mode=${1:-}
shift || true

if [[ ! -x "$POLICY_PYTHON" ]]; then
  echo "OpenPI Python environment missing: $POLICY_PYTHON" >&2
  exit 2
fi

case "$mode" in
  --check)
    exec "$POLICY_PYTHON" "$POLICY_SCRIPT" --mode load-only
    ;;
  --smoke)
    exec "$POLICY_PYTHON" "$POLICY_SCRIPT" --mode smoke
    ;;
  --execute)
    ;;
  *)
    echo "usage: $0 --check | --smoke | --execute [robot-runner options]" >&2
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
policy_pid=
policy_log=$(mktemp /tmp/tate_arx_policy.XXXXXX.log)

cleanup() {
  if [[ -n "$policy_pid" ]]; then
    kill "$policy_pid" 2>/dev/null || true
    wait "$policy_pid" 2>/dev/null || true
  fi
  echo "Restoring original ARX services"
  if [[ "$data_was_active" == true ]]; then
    systemctl --user start arx-data-station.service || true
  fi
  if [[ "$button_was_active" == true ]]; then
    systemctl --user start arx-button-control.service || true
  fi
  rm -f "$policy_log"
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

echo "Loading OpenPI checkpoint; this can take several minutes"
"$POLICY_PYTHON" "$POLICY_SCRIPT" --mode serve --port "$POLICY_PORT" >"$policy_log" 2>&1 &
policy_pid=$!
ready=false
for attempt in $(seq 1 240); do
  if curl -fsS --max-time 1 "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$policy_pid" 2>/dev/null; then
    break
  fi
  if (( attempt % 15 == 0 )); then
    echo "Still loading OpenPI checkpoint (${attempt}s)"
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "OpenPI policy server failed to start" >&2
  tail -30 "$policy_log" >&2
  exit 4
fi
echo "OpenPI policy server ready on 127.0.0.1:$POLICY_PORT"

"$ROBOT_PYTHON" "$ROBOT_SCRIPT" --execute --policy-url "http://127.0.0.1:$POLICY_PORT" "$@"
