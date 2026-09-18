#!/usr/bin/env python3
"""Guarded, right-arm-only ARX execution for the TATE stack-cube policy."""

from __future__ import annotations

import argparse
import base64
import json
import os
import select
import signal
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from deployment.constants import ACTION_HORIZON, CAMERA_SERIALS, PREVIEW_DIR, PROMPT, TCP_OFFSET_M  # noqa: E402
from deployment.eef_math import (  # noqa: E402
    GuardLimits,
    JOINT_LOWER,
    JOINT_UPPER,
    flange_to_tcp_state,
    model_state,
    tcp_action_to_flange,
    validate_ik_solution,
)


SDK_ROOT = Path(os.environ.get("TATE_ARX_SDK_ROOT", "/home/qijun/ARX5_beta"))
PAYLOAD_URDF = Path(
    os.environ.get(
        "TATE_ARX_URDF",
        "/home/qijun/le_ws/LifEgo/datacollection/arx/models/"
        "X5-2025-gripper-handle-0p65kg.urdf",
    )
)
START_POSE_FILE = APP_ROOT / "deployment" / "start_pose.json"


def start_pose_trajectory(
    sdk, current: np.ndarray, limits: GuardLimits, tcp_offset_m: np.ndarray
) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """Preflight a slow joint ramp to the recorded frame before any command."""
    record = json.loads(START_POSE_FILE.read_text(encoding="utf-8"))
    target = np.asarray(record["joint_position_rad"], dtype=np.float64)
    grip = float(record["gripper_raw"])
    if target.shape != (6,) or not np.isfinite(target).all() or not np.isfinite(grip):
        raise ValueError("recorded start pose is malformed")
    if np.any(target < JOINT_LOWER) or np.any(target > JOINT_UPPER) or not -3.4 <= grip <= 0.1:
        raise ValueError("recorded start pose is outside joint or gripper limits")
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("current robot feedback is malformed")
    # Smoothstep limits peak velocity to about 0.12 rad/s at 5 Hz.
    count = max(15, int(np.ceil(np.max(np.abs(target - current[:6])) / 0.016)))
    frames = []
    for i in range(1, count + 1):
        u = i / count
        alpha = u * u * (3.0 - 2.0 * u)
        q = current[:6] + alpha * (target - current[:6])
        g = float(current[6] + alpha * (grip - current[6]))
        flange = np.asarray(sdk.forward_kinematics(q, type=2), dtype=np.float64)
        tcp = flange_to_tcp_state(flange, g, tcp_offset_m=tcp_offset_m)
        for value, (lower, upper), axis in zip(tcp[:3], (limits.x_range, limits.y_range, limits.z_range), "xyz"):
            if not lower <= value <= upper:
                raise ValueError(f"start-pose ramp TCP {axis} outside workspace: {value:.4f}")
        frames.append((q, g))
    return frames, record


def move_to_start_pose(arm, frames: list[tuple[np.ndarray, float]], should_stop) -> None:
    consecutive_lag = 0
    for i, (q, grip) in enumerate(frames, 1):
        if should_stop():
            raise RuntimeError("start-pose movement stopped by operator")
        tick = time.monotonic()
        send_target(arm, q, grip)
        time.sleep(max(0.0, 0.2 - (time.monotonic() - tick)))
        feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        if feedback.shape != (7,) or not np.isfinite(feedback).all():
            raise RuntimeError("invalid feedback during start-pose movement")
        lag = float(np.max(np.abs(feedback[:6] - q)))
        consecutive_lag = consecutive_lag + 1 if lag > 0.15 else 0
        if consecutive_lag >= 8:
            raise RuntimeError(f"start-pose tracking lag exceeded 0.15 rad: {lag:.3f}")
        if i % 10 == 0 or i == len(frames):
            print(f"START_POSE_PROGRESS {i}/{len(frames)} max_joint_lag_rad={lag:.3f}", flush=True)
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline and not should_stop():
        feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        # An object can stop the gripper before its requested endpoint.  Joint
        # convergence is required here; endpoint gripper feedback is not.
        if np.max(np.abs(feedback[:6] - frames[-1][0])) <= 0.05:
            print(f"START_POSE_REACHED {np.array2string(feedback, precision=5)}", flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError("recorded start pose was not reached within tolerance")


def read_observation(arm, rig, sdk, tcp_offset_m: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    frames = {role: rig.frame(role) for role in CAMERA_SERIALS}
    now_ns = time.monotonic_ns()
    if any(frame is None for frame in frames.values()):
        raise RuntimeError("one or more cameras are unavailable")
    timestamps = [frame.received_ns for frame in frames.values()]
    if any(now_ns - stamp > 250_000_000 for stamp in timestamps):
        raise RuntimeError("camera frame older than 250 ms")
    if max(timestamps) - min(timestamps) > 100_000_000:
        raise RuntimeError("camera synchronization skew exceeds 100 ms")

    feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    if feedback.shape != (7,) or not np.isfinite(feedback).all():
        raise RuntimeError(f"invalid right-arm feedback: {feedback.shape}")
    flange = np.asarray(sdk.forward_kinematics(feedback[:6], type=2), dtype=np.float64)
    tcp = flange_to_tcp_state(flange, float(feedback[6]), tcp_offset_m=tcp_offset_m)
    encoded: dict[str, str] = {}
    for role, frame in frames.items():
        ok, jpg = cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError(f"cannot JPEG-encode {role} camera frame")
        data = jpg.tobytes()
        encoded[role] = base64.b64encode(data).decode("ascii")
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        temporary = PREVIEW_DIR / f".{role}.{os.getpid()}.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, PREVIEW_DIR / f"{role}.jpg")
    observation = {
        "images": encoded,
        "state": model_state(tcp).tolist(),
        "image_mask": [True, True, True],
        "prompt": PROMPT,
    }
    return observation, feedback, tcp


def infer(policy_url: str, observation: dict, timeout: float) -> tuple[np.ndarray, float]:
    body = json.dumps(observation, separators=(",", ":")).encode("utf-8")
    request = Request(policy_url.rstrip("/") + "/infer", data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"policy request failed: {exc.read(1000).decode(errors='replace')}") from exc
    latency = time.monotonic() - started
    actions = np.asarray(result.get("actions"), dtype=np.float64)
    if actions.shape != (ACTION_HORIZON, 16) or not np.isfinite(actions).all():
        raise RuntimeError(f"policy returned invalid actions: {actions.shape}")
    return actions, latency


def guarded_target(
    sdk,
    action: np.ndarray,
    feedback: np.ndarray,
    tcp: np.ndarray,
    limits: GuardLimits,
    tcp_offset_m: np.ndarray,
    gripper_threshold: float,
):
    predicted_gripper = float(action[15])
    if not 0.0 <= predicted_gripper <= 1.0:
        print(
            "GRIPPER_ACTION_CLIPPED "
            f"raw={predicted_gripper:.6f} clipped={np.clip(predicted_gripper, 0.0, 1.0):.6f}",
            flush=True,
        )
    flange, gripper_raw = tcp_action_to_flange(
        action[8:16], tcp, limits, tcp_offset_m=tcp_offset_m, gripper_threshold=gripper_threshold
    )
    joints = np.asarray(sdk.inverse_kinematics(flange, q_init=feedback[:6], type=2), dtype=np.float64)
    actual = np.asarray(sdk.forward_kinematics(joints, type=2), dtype=np.float64)
    validate_ik_solution(joints, feedback[:6], flange, actual, limits)
    return joints, gripper_raw, flange


def send_target(arm, joints: np.ndarray, gripper_raw: float) -> None:
    if arm.fault is not None:
        raise RuntimeError(f"right arm fault: {arm.fault}")
    started = time.monotonic()
    if not arm.set_joint_positions(joints, duration=0.0):
        raise RuntimeError("right-arm set_joint_positions failed")
    if not arm.set_gripper_pos(float(gripper_raw), duration=0.0):
        raise RuntimeError("right-arm set_gripper_pos failed")
    if time.monotonic() - started > 0.25:
        raise RuntimeError("robot command watchdog exceeded 250 ms")


def wait_for_run(should_stop) -> bool:
    print("READY_FOR_RUN: type RUN after reviewing the first guarded target", flush=True)
    while not should_stop():
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if readable:
            return sys.stdin.readline().strip() == "RUN"
    return False


def wait_for_start_pose(should_stop) -> bool:
    print("READY_FOR_START_POSE: review the recorded target, then confirm MOVE", flush=True)
    while not should_stop():
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if readable:
            return sys.stdin.readline().strip() == "MOVE"
    return False


def parse_resume_config(line: str) -> tuple[float, int, float, np.ndarray]:
    """Validate runtime parameters sent by the console while the arm is held."""
    if not line.startswith("RESUME "):
        raise ValueError("expected RESUME command")
    try:
        value = json.loads(line.removeprefix("RESUME "))
        fps = float(value["fps"])
        n_action_steps = int(value["n_action_steps"])
        gripper_threshold = float(value["gripper_threshold"])
        tcp_offset_m = np.asarray(value["tcp_offset_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid resume configuration") from exc
    if not np.isfinite(fps) or not 1 <= fps <= 10:
        raise ValueError("resume fps must be in [1, 10]")
    if not 1 <= n_action_steps <= ACTION_HORIZON:
        raise ValueError(f"resume n_action_steps must be in [1, {ACTION_HORIZON}]")
    if not np.isfinite(gripper_threshold) or not 0 <= gripper_threshold <= 1:
        raise ValueError("resume gripper_threshold must be in [0, 1]")
    if tcp_offset_m.shape != (3,) or not np.isfinite(tcp_offset_m).all() or np.linalg.norm(tcp_offset_m) > 0.30:
        raise ValueError("resume tcp_offset_m is invalid")
    return fps, n_action_steps, gripper_threshold, tcp_offset_m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-pose", action="store_true", help="move to recorded episode 0 frame 60 before inference")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8019")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="close the gripper when the predicted continuous value is greater than this threshold",
    )
    parser.add_argument("--max-policy-latency", type=float, default=5.0)
    parser.add_argument("--max-runtime-seconds", type=int, default=300)
    parser.add_argument(
        "--tcp-offset-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
        default=None,
        help="flange-to-TCP translation for both policy observation and action decoding; use 0.105 0.0018 -0.0063 for an old checkpoint",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("hardware runner requires --execute through the guarded wrapper")
    if not 1 <= args.fps <= 10 or not 1 <= args.n_action_steps <= ACTION_HORIZON:
        parser.error(
            f"fps must be [1, 10] and n-action-steps must be [1, {ACTION_HORIZON}]"
        )
    if not np.isfinite(args.gripper_threshold) or not 0.0 <= args.gripper_threshold <= 1.0:
        parser.error("--gripper-threshold must be in [0, 1]")
    tcp_offset_m = np.asarray(TCP_OFFSET_M if args.tcp_offset_m is None else args.tcp_offset_m, dtype=np.float64)
    if not np.isfinite(tcp_offset_m).all() or np.linalg.norm(tcp_offset_m) > 0.30:
        parser.error("--tcp-offset-m must be finite and have magnitude at most 0.30 m")
    lock_fd = int(os.environ.get("TATE_ARM_LOCK_FD", "-1"))
    if lock_fd < 0:
        raise RuntimeError("hardware lock not inherited from deployment wrapper")
    os.fstat(lock_fd)
    if not PAYLOAD_URDF.is_file():
        raise FileNotFoundError(PAYLOAD_URDF)

    stop = False
    pause_toggle = False
    hold_requested = False
    reset_requested = False
    resume_requested = False
    normal_completion = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    def request_pause(_signal, _frame):
        nonlocal pause_toggle
        pause_toggle = True

    def request_hold(_signal, _frame):
        nonlocal stop, hold_requested
        hold_requested = True
        stop = True

    def request_reset(_signal, _frame):
        nonlocal reset_requested
        reset_requested = True

    def request_resume(_signal, _frame):
        nonlocal resume_requested
        resume_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_pause)
    signal.signal(signal.SIGUSR2, request_hold)
    signal.signal(signal.SIGHUP, request_reset)
    signal.signal(signal.SIGCONT, request_resume)

    sys.path.insert(0, str(SDK_ROOT))
    import bimanual as sdk
    from arx_data_station.cameras import RealSenseRig
    from bimanual import SingleArm

    rig = None
    arm = None
    limits = GuardLimits()
    print(f"TCP_OFFSET_M {np.array2string(tcp_offset_m, precision=6)}", flush=True)
    print(f"GRIPPER_THRESHOLD {args.gripper_threshold:.4f}", flush=True)
    try:
        rig = RealSenseRig(width=640, height=480, fps=30)
        rig.start(CAMERA_SERIALS)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if all(rig.status()[role]["online"] for role in CAMERA_SERIALS):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"cameras did not become ready: {rig.status()}")

        arm = SingleArm({"can_port": "can3", "type": 2, "urdf_path": str(PAYLOAD_URDF)})
        if not arm.protect_mode():
            raise RuntimeError("right arm did not enter protect mode")
        if args.start_pose:
            initial = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            frames, record = start_pose_trajectory(sdk, initial, limits, tcp_offset_m)
            print(f"START_POSE_SOURCE episode={record['episode_index']} frame={record['frame_index']} side={record['side']}", flush=True)
            print(f"START_POSE_CURRENT_JOINTS {np.array2string(initial, precision=5)}", flush=True)
            print(f"START_POSE_TARGET_JOINTS {np.array2string(frames[-1][0], precision=5)} gripper_raw={frames[-1][1]:.5f}", flush=True)
            print(f"START_POSE_RAMP frames={len(frames)} rate_hz=5", flush=True)
            if not wait_for_start_pose(lambda: stop):
                print("MOVE not confirmed; no motion action sent", flush=True)
                return 2
            print("MOVE accepted; moving right arm to recorded start pose", flush=True)
            move_to_start_pose(arm, frames, lambda: stop)
            # Images and state must be read again after the arm has moved.
        observation, feedback, tcp = read_observation(arm, rig, sdk, tcp_offset_m)
        actions, latency = infer(args.policy_url, observation, args.max_policy_latency + 1)
        if latency > args.max_policy_latency:
            raise RuntimeError(f"preflight policy latency {latency:.2f}s exceeds watchdog")
        first_target = actions[0, 8:16]
        delta_xyz = first_target[:3] - tcp[:3]
        print(f"FIRST_CURRENT_TCP {np.array2string(tcp, precision=5)}", flush=True)
        print(f"FIRST_CURRENT_JOINTS {np.array2string(feedback, precision=5)}", flush=True)
        print(f"FIRST_TARGET_TCP {np.array2string(first_target, precision=5)}", flush=True)
        print(f"FIRST_TCP_DELTA_XYZ {np.array2string(delta_xyz, precision=5)} norm_m={np.linalg.norm(delta_xyz):.5f}", flush=True)
        print(f"FIRST_POLICY_LATENCY {latency:.3f}s", flush=True)
        joints, gripper_raw, flange = guarded_target(
            sdk, actions[0], feedback, tcp, limits, tcp_offset_m, args.gripper_threshold
        )
        print(f"FIRST_TARGET_FLANGE {np.array2string(flange, precision=5)}", flush=True)
        print(f"FIRST_TARGET_JOINTS {np.array2string(joints, precision=5)} gripper_raw={gripper_raw:.4f}", flush=True)
        if not wait_for_run(lambda: stop):
            print("RUN not confirmed; no motion action sent", flush=True)
            return 2

        print("RUN accepted; using a fresh observation before the first command", flush=True)
        period = 1.0 / args.fps
        paused = False
        steps = 0
        while True:
            started = time.monotonic()
            while not stop and time.monotonic() - started < args.max_runtime_seconds:
                if pause_toggle:
                    pause_toggle = False
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED", flush=True)
                if paused:
                    feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                    send_target(arm, feedback[:6], float(np.clip(feedback[6], -3.4, 0.1)))
                    time.sleep(period)
                    continue
                observation, feedback, tcp = read_observation(arm, rig, sdk, tcp_offset_m)
                actions, latency = infer(args.policy_url, observation, args.max_policy_latency + 1)
                if latency > args.max_policy_latency:
                    raise RuntimeError(f"policy latency {latency:.2f}s exceeds watchdog")
                if stop or pause_toggle:
                    continue
                for action in actions[: args.n_action_steps]:
                    if stop or pause_toggle:
                        break
                    tick = time.monotonic()
                    feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                    flange = np.asarray(sdk.forward_kinematics(feedback[:6], type=2), dtype=np.float64)
                    tcp = flange_to_tcp_state(flange, float(feedback[6]), tcp_offset_m=tcp_offset_m)
                    joints, gripper_raw, _ = guarded_target(
                        sdk, action, feedback, tcp, limits, tcp_offset_m, args.gripper_threshold
                    )
                    send_target(arm, joints, gripper_raw)
                    steps += 1
                    if steps % max(1, round(args.fps)) == 0:
                        print(f"step={steps} policy_latency={latency:.3f}s right_tcp={np.array2string(tcp, precision=4)}", flush=True)
                    time.sleep(max(0.0, period - (time.monotonic() - tick)))
            if not hold_requested:
                print(f"TEST_FINISHED steps={steps}", flush=True)
                return 0

            feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            if feedback.shape != (7,) or not np.isfinite(feedback).all():
                raise RuntimeError("invalid feedback while entering position hold")
            hold_joints, hold_gripper = feedback[:6].copy(), float(np.clip(feedback[6], -3.4, 0.1))
            print(f"HOLDING_POSITION joints={np.array2string(hold_joints, precision=5)} gripper_raw={hold_gripper:.5f}", flush=True)
            while not reset_requested and not resume_requested:
                readable, _, _ = select.select([sys.stdin], [], [], 0)
                if readable:
                    line = sys.stdin.readline().strip()
                    try:
                        fps, n_action_steps, gripper_threshold, tcp_offset_m = parse_resume_config(line)
                        args.fps = fps
                        args.n_action_steps = n_action_steps
                        args.gripper_threshold = gripper_threshold
                        print(
                            "RESUME_CONFIG "
                            f"fps={args.fps:.3f} n_action_steps={args.n_action_steps} "
                            f"gripper_threshold={args.gripper_threshold:.4f} "
                            f"tcp_offset_m={np.array2string(tcp_offset_m, precision=6)}",
                            flush=True,
                        )
                        resume_requested = True
                        continue
                    except ValueError as exc:
                        print(f"RESUME_CONFIG_REJECTED {exc}", flush=True)
                send_target(arm, hold_joints, hold_gripper)
                time.sleep(0.2)
            if reset_requested:
                print("RESET_ACCEPTED: returning right arm to home", flush=True)
                if not arm.go_home(5.0, wait=True):
                    raise RuntimeError("right-arm go_home failed")
                reset_requested = False
                feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                if feedback.shape != (7,) or not np.isfinite(feedback).all():
                    raise RuntimeError("invalid feedback after homing")
                hold_joints, hold_gripper = feedback[:6].copy(), float(np.clip(feedback[6], -3.4, 0.1))
                print(
                    f"HOME_REACHED: holding joints={np.array2string(hold_joints, precision=5)} "
                    f"gripper_raw={hold_gripper:.5f}",
                    flush=True,
                )
                hold_requested = True
                stop = True
                continue
            resume_requested = False
            hold_requested = False
            stop = False
            paused = False
            period = 1.0 / args.fps
            print("RESUMED_FROM_HOLD", flush=True)
    finally:
        if arm is not None:
            if not normal_completion:
                try:
                    arm.protect_mode()
                except Exception as exc:
                    print(f"protect_mode failed: {exc}", file=sys.stderr, flush=True)
            try:
                arm.close()
            except Exception:
                pass
        if rig is not None:
            rig.stop()


if __name__ == "__main__":
    raise SystemExit(main())
