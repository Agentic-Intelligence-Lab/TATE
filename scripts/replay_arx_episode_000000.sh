#!/usr/bin/env bash
# Replay /home/qijun/TATE/replay/episode_000000.json on the ARX bimanual robot.
#
# Usage (from a machine that can SSH to the robot host):
#   bash scripts/replay_arx_episode_000000.sh --dry-run
#   bash scripts/replay_arx_episode_000000.sh --execute
#
# The execute path requires an interactive confirmation here and the Python
# program's explicit --yes-i-understand-risk flag on the robot host.
set -euo pipefail

REMOTE_HOST="${TATE_ARX_HOST:-qijun@192.168.2.136}"
REMOTE_ROOT="${TATE_ARX_ROOT:-/home/qijun/TATE}"
ROBOT_PYTHON="${TATE_ROBOT_PYTHON:-/home/qijun/ARX5_beta/.venv/bin/python}"
SDK_ROOT="${TATE_ARX_SDK_ROOT:-/home/qijun/ARX5_beta}"
EPISODE="${TATE_ARX_EPISODE:-/home/qijun/TATE/replay/episode_000000.json}"

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

ARGS=(
  "--data" "$EPISODE"
  "--sdk-root" "$SDK_ROOT"
  "--left-can" "${TATE_LEFT_CAN:-can1}"
  "--right-can" "${TATE_RIGHT_CAN:-can3}"
  "--rate" "${TATE_REPLAY_RATE:-15}"
  "--speed" "${TATE_REPLAY_SPEED:-0.5}"
  "--max-joint-speed" "${TATE_MAX_JOINT_SPEED:-0.4}"
  "--max-gripper-speed" "${TATE_MAX_GRIPPER_SPEED:-0.8}"
  "--ramp-time" "${TATE_RAMP_TIME:-8}"
)

if [[ "$MODE" == "--execute" ]]; then
  read -r -p "确认急停可用、工作区清空且 can1=左臂/can3=右臂；输入 REPLAY 开始: " CONFIRM
  if [[ "$CONFIRM" != "REPLAY" ]]; then
    echo "未确认，未向机械臂发送任何命令。"
    exit 1
  fi
  ARGS+=("--execute" "--yes-i-understand-risk")
fi

printf -v REMOTE_COMMAND '%q ' \
  "$ROBOT_PYTHON" "$REMOTE_ROOT/real2sim/replay_arx_realbot.py" "${ARGS[@]}"
ssh -tt "$REMOTE_HOST" "cd $(printf '%q' "$REMOTE_ROOT") && $REMOTE_COMMAND"
