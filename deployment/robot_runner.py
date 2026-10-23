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
from dataclasses import replace
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
    GRIPPER_OPEN_RAW,
    GRIPPER_CLOSED_RAW,
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
BIMANUAL_START_TRAJECTORY = Path(
    os.environ.get("TATE_BIMANUAL_START_TRAJECTORY", str(APP_ROOT / "replay" / "episode_000000.json"))
)
BIMANUAL_START_FRAME = int(os.environ.get("TATE_BIMANUAL_START_FRAME", "120"))


def start_pose_trajectory(
    sdk, current: np.ndarray, limits: GuardLimits, tcp_offset_m: np.ndarray
) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """Return the recorded start pose as one direct joint command."""
    record = json.loads(START_POSE_FILE.read_text(encoding="utf-8"))
    target = np.asarray(record["joint_position_rad"], dtype=np.float64)
    grip = float(record["gripper_raw"])
    if target.shape != (6,) or not np.isfinite(target).all() or not np.isfinite(grip):
        raise ValueError("recorded start pose is malformed")
    if np.any(target < JOINT_LOWER) or np.any(target > JOINT_UPPER) or not -3.4 <= grip <= 0.1:
        raise ValueError("recorded start pose is outside joint or gripper limits")
    return [(target, grip)], record


def move_to_start_pose(arm, frames: list[tuple[np.ndarray, float]], should_stop) -> None:
    q, grip = frames[-1]
    if should_stop():
        raise RuntimeError("start-pose movement stopped by operator")
    # The ARX controller executes this target at its configured joint speed.
    # Do not interpolate or repeatedly issue intermediate commands.
    send_target(arm, q, grip)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if should_stop():
            # A direct target remains active in the ARX controller until a new
            # command arrives. Cancel it before this process exits or enters
            # protect mode, so a UI-service restart cannot complete the old
            # move-to-frame-60 command in the background.
            feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            if feedback.shape == (7,) and np.isfinite(feedback).all():
                send_target(arm, feedback[:6], float(np.clip(feedback[6], -3.4, 0.1)))
                print("START_POSE_CANCELLED_HOLDING_CURRENT_POSITION", flush=True)
            raise RuntimeError("start-pose movement stopped by operator")
        feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        if feedback.shape != (7,) or not np.isfinite(feedback).all():
            raise RuntimeError("invalid feedback during start-pose movement")
        if np.max(np.abs(feedback[:6] - q)) <= 0.05:
            print(f"START_POSE_REACHED {np.array2string(feedback, precision=5)}", flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError("recorded start pose was not reached within tolerance")


def bimanual_start_pose() -> tuple[dict[str, tuple[np.ndarray, float]], dict]:
    """Load and validate a paired joint target from the real demonstration."""
    record = json.loads(BIMANUAL_START_TRAJECTORY.read_text(encoding="utf-8"))
    if record.get("schema") != "tate.arx_real_flange_trajectory":
        raise ValueError("bimanual start trajectory has an unsupported schema")
    if record.get("robot") != "arx5_2025_bimanual":
        raise ValueError("bimanual start trajectory is not for arx5_2025_bimanual")
    frames = record.get("frames")
    if not isinstance(frames, list):
        raise ValueError("bimanual start trajectory has no frames")
    frame = next((value for value in frames if value.get("frame_index") == BIMANUAL_START_FRAME), None)
    if frame is None:
        raise ValueError(f"bimanual start trajectory has no frame {BIMANUAL_START_FRAME}")
    targets = {}
    for side in ("left", "right"):
        try:
            arm = frame["arms"][side]
            joints = np.asarray(arm["joint_position_rad"], dtype=np.float64)
            gripper = float(arm["gripper"]["raw"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"frame {BIMANUAL_START_FRAME} has invalid {side}-arm data") from exc
        if joints.shape != (6,) or not np.isfinite(joints).all() or not np.isfinite(gripper):
            raise ValueError(f"frame {BIMANUAL_START_FRAME} has malformed {side}-arm target")
        if np.any(joints < JOINT_LOWER) or np.any(joints > JOINT_UPPER) or not -3.4 <= gripper <= 0.1:
            raise ValueError(f"frame {BIMANUAL_START_FRAME} {side}-arm target is outside limits")
        targets[side] = (joints, gripper)
    return targets, frame


def move_bimanual_to_start_pose(arms, targets, should_stop) -> None:
    if should_stop():
        raise RuntimeError("bimanual start-pose movement stopped by operator")
    send_bimanual_targets(arms, targets)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if should_stop():
            hold_bimanual_position(arms)
            raise RuntimeError("bimanual start-pose movement stopped by operator")
        reached = True
        for side in ("left", "right"):
            feedback = np.asarray(arms[side].get_joint_positions(), dtype=np.float64)
            if feedback.shape != (7,) or not np.isfinite(feedback).all():
                raise RuntimeError(f"invalid {side}-arm feedback during start-pose movement")
            if np.max(np.abs(feedback[:6] - targets[side][0])) > 0.05:
                reached = False
        if reached:
            print("BIMANUAL_START_POSE_REACHED", flush=True)
            return
        time.sleep(0.1)
    hold_bimanual_position(arms)
    raise RuntimeError("bimanual start pose was not reached within tolerance")


def read_observation(arm, rig, sdk, tcp_offset_m: np.ndarray, prompt: str) -> tuple[dict, np.ndarray, np.ndarray]:
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
        "prompt": prompt,
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


class IKUnsolved(ValueError):
    """SDK reports no IK solution; not a generic safety or hardware failure."""


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
    try:
        solution = sdk.inverse_kinematics(flange, q_init=feedback[:6], type=2)
    except ValueError as exc:
        if not str(exc).startswith("IK 无解:"):
            raise
        detail = json.dumps({
            "side": "right", "target_tcp": np.asarray(action[8:16]).tolist(),
            "target_flange": flange.tolist(), "q_init": feedback[:6].tolist(),
            "reason": str(exc),
        }, ensure_ascii=False)
        raise IKUnsolved(detail) from exc
    joints = np.asarray(solution, dtype=np.float64)
    actual = np.asarray(sdk.forward_kinematics(joints, type=2), dtype=np.float64)
    validate_ik_solution(joints, feedback[:6], flange, actual, limits)
    return joints, gripper_raw, flange


def execute_single_step(arm, sdk, action, feedback, tcp, limits, tcp_offset_m,
                        gripper_threshold, step_index, should_cancel=lambda: False) -> bool:
    try:
        joints, gripper, _ = guarded_target(
            sdk, action, feedback, tcp, limits, tcp_offset_m, gripper_threshold
        )
    except IKUnsolved as exc:
        print(f"SINGLE_STEP_SKIPPED step={step_index} {exc}", flush=True)
        return False
    if should_cancel():
        return False
    send_target(arm, joints, gripper)
    return True


def preflight_target(arm, rig, sdk, limits: GuardLimits, tcp_offset_m: np.ndarray, args) -> None:
    """Inspect one fresh policy target before accepting a RUN command."""
    observation, feedback, tcp = read_observation(arm, rig, sdk, tcp_offset_m, args.prompt)
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
    print("READY_FOR_RUN: type RUN to authorize guarded execution", flush=True)
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


def read_bimanual_observation(arms, rig, sdk, tcp_offset_m: np.ndarray, prompt: str) -> tuple[dict, dict, dict]:
    """Read a synchronized dual-arm observation in the training layout.

    The action/state contract is always [left(8), right(8)].  Keeping that
    order here avoids the very dangerous failure mode of sending a left target
    to the right controller.
    """
    frames = {role: rig.frame(role) for role in CAMERA_SERIALS}
    now_ns = time.monotonic_ns()
    if any(frame is None for frame in frames.values()):
        raise RuntimeError("one or more cameras are unavailable")
    stamps = [frame.received_ns for frame in frames.values()]
    if any(now_ns - stamp > 250_000_000 for stamp in stamps) or max(stamps) - min(stamps) > 100_000_000:
        raise RuntimeError("camera frames are stale or not synchronized")
    feedback, tcp = {}, {}
    for side in ("left", "right"):
        value = np.asarray(arms[side].get_joint_positions(), dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {side}-arm feedback: {value.shape}")
        feedback[side] = value
        flange = np.asarray(sdk.forward_kinematics(value[:6], type=2), dtype=np.float64)
        tcp[side] = flange_to_tcp_state(flange, float(value[6]), tcp_offset_m=tcp_offset_m)
    encoded = {}
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
    return {"images": encoded, "state": np.concatenate((tcp["left"], tcp["right"])).tolist(),
            "image_mask": [True, True, True], "prompt": prompt}, feedback, tcp


class BimanualIKUnsolved(IKUnsolved):
    """An SDK no-solution result, distinct from faults and safety rejections."""


def guarded_bimanual_targets(sdk, action, feedback, tcp, limits, tcp_offset_m, gripper_threshold):
    targets = {}
    for side, offset in (("left", 0), ("right", 8)):
        # The dual-arm demonstration extends farther along left-arm negative Y.
        side_limits = replace(limits, y_range=(-0.30, limits.y_range[1])) if side == "left" else limits
        flange, gripper = tcp_action_to_flange(action[offset:offset + 8], tcp[side], side_limits,
                                                tcp_offset_m=tcp_offset_m, gripper_threshold=gripper_threshold)
        try:
            solution = sdk.inverse_kinematics(flange, q_init=feedback[side][:6], type=2)
        except ValueError as exc:
            # This SDK exposes no dedicated exception for non-convergence.
            # Do not swallow unrelated input errors or hardware exceptions.
            if not str(exc).startswith("IK 无解:"):
                raise
            detail = json.dumps({
                "side": side, "target_tcp": np.asarray(action[offset:offset + 8]).tolist(),
                "target_flange": flange.tolist(), "q_init": feedback[side][:6].tolist(),
                "reason": str(exc),
            }, ensure_ascii=False)
            raise BimanualIKUnsolved(detail) from exc
        joints = np.asarray(solution, dtype=np.float64)
        actual = np.asarray(sdk.forward_kinematics(joints, type=2), dtype=np.float64)
        validate_ik_solution(joints, feedback[side][:6], flange, actual, limits)
        targets[side] = (joints, gripper)
    return targets


def execute_bimanual_step(arms, sdk, action, feedback, tcp, limits, tcp_offset_m,
                          gripper_threshold, step_index, should_cancel=lambda: False) -> bool:
    """Skip only an unsolved IK pair; never partially send a rejected pair."""
    try:
        targets = guarded_bimanual_targets(
            sdk, action, feedback, tcp, limits, tcp_offset_m, gripper_threshold
        )
    except BimanualIKUnsolved as exc:
        print(f"BIMANUAL_STEP_SKIPPED step={step_index} {exc}", flush=True)
        return False
    if should_cancel():
        return False
    send_bimanual_targets(arms, targets)
    return True


def send_bimanual_targets(arms, targets) -> None:
    """Issue only validated targets, retaining protect-mode on either failure."""
    for side in ("left", "right"):
        arm = arms[side]
        joints, gripper = targets[side]
        if arm.fault is not None or not arm.set_joint_positions(joints, duration=0.0) or not arm.set_gripper_pos(float(gripper), duration=0.0):
            raise RuntimeError(f"{side}-arm command failed")


class BimanualControl:
    """Separate operator hold/reset/pause from actual process shutdown."""
    def __init__(self):
        self.shutdown = False
        self.hold = False
        self.reset = False
        self.pause = False

    def on_signal(self, signum, _frame=None):
        if signum in (signal.SIGINT, signal.SIGTERM):
            self.shutdown = True
        elif signum == signal.SIGUSR2:
            self.hold = True
        elif signum == signal.SIGHUP:
            self.reset = True
            self.hold = True
        elif signum == signal.SIGUSR1:
            self.pause = not self.pause

    def interrupted(self):
        return self.shutdown or self.hold


def capture_bimanual_hold(arms):
    targets = {}
    for side in ("left", "right"):
        feedback = np.asarray(arms[side].get_joint_positions(), dtype=np.float64)
        if feedback.shape != (7,) or not np.isfinite(feedback).all():
            raise RuntimeError(f"invalid {side}-arm feedback while holding")
        targets[side] = (feedback[:6].copy(), float(np.clip(feedback[6], -3.4, 0.1)))
    return targets


def hold_bimanual_position(arms, targets=None) -> None:
    # Try both sides even if one controller is faulty.
    targets = capture_bimanual_hold(arms) if targets is None else targets
    errors = []
    for side in ("left", "right"):
        try:
            arm = arms[side]
            joints, gripper = targets[side]
            if arm.fault is not None:
                raise RuntimeError(f"{side}-arm fault: {arm.fault}")
            if not arm.set_joint_positions(joints, duration=0.0):
                raise RuntimeError(f"{side}-arm hold command failed")
            if not arm.set_gripper_pos(gripper, duration=0.0):
                raise RuntimeError(f"{side}-arm gripper hold command failed")
        except Exception as exc:
            errors.append(f"{side}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def wait_bimanual_grippers(arms, target, control):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if control.interrupted():
            raise RuntimeError("bimanual gripper movement interrupted")
        feedback = capture_bimanual_hold(arms)
        if any(arm.fault is not None for arm in arms.values()):
            raise RuntimeError("arm fault during gripper movement")
        if all(abs(grip - target) <= 0.2 for _, grip in feedback.values()):
            return
        time.sleep(0.1)
    raise RuntimeError("bimanual gripper feedback timeout")


def reset_bimanual_home(arms, control):
    print("RESET_ACCEPTED: returning both arms to home", flush=True)
    if control.interrupted():
        raise RuntimeError("bimanual reset interrupted")
    targets = capture_bimanual_hold(arms)
    opened = {side: (q, GRIPPER_OPEN_RAW) for side, (q, _) in targets.items()}
    hold_bimanual_position(arms, opened)
    wait_bimanual_grippers(arms, GRIPPER_OPEN_RAW, control)
    print("BIMANUAL_RESET_GRIPPERS_OPEN", flush=True)
    for side in ("left", "right"):
        if control.interrupted():
            raise RuntimeError("bimanual reset interrupted")
        if arms[side].fault is not None or not arms[side].go_home(5.0, wait=False, gripper=False):
            raise RuntimeError(f"{side}-arm go_home failed")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if control.interrupted():
            raise RuntimeError("bimanual reset interrupted")
        targets = capture_bimanual_hold(arms)
        if any(arm.fault is not None for arm in arms.values()):
            raise RuntimeError("arm fault during bimanual reset")
        if all(np.max(np.abs(q)) <= 0.05 for q, _ in targets.values()):
            if control.interrupted():
                raise RuntimeError("bimanual reset interrupted before gripper close")
            targets = {side: (q, GRIPPER_CLOSED_RAW) for side, (q, _) in targets.items()}
            hold_bimanual_position(arms, targets)
            wait_bimanual_grippers(arms, GRIPPER_CLOSED_RAW, control)
            print("HOME_REACHED: both arms holding at home", flush=True)
            return targets
        time.sleep(0.1)
    raise RuntimeError("bimanual home feedback timeout")


def wait_bimanual_resume(arms, args, control):
    """Remain alive on stop/error; only explicit shutdown closes controllers."""
    control.hold = False
    control.pause = False
    targets = None
    healthy = False
    last_error = None
    while not control.shutdown:
        if control.reset:
            control.reset = False
            control.hold = False
            try:
                targets = reset_bimanual_home(arms, control)
            except Exception as exc:
                print(f"BIMANUAL_RESET_REJECTED {exc}", flush=True)
                targets = None  # Capture the stopped pose, not the pre-reset pose.
            healthy = False
        control.hold = False
        try:
            if targets is None:
                targets = capture_bimanual_hold(arms)
            hold_bimanual_position(arms, targets)
            if not healthy:
                print("HOLDING_POSITION bimanual=true; fixed targets; Prepare or Reset available", flush=True)
            healthy = True
            last_error = None
        except Exception as exc:
            healthy = False
            if str(exc) != last_error:
                print(f"BIMANUAL_HOLD_FAILED cannot guarantee position hold: {exc}", flush=True)
                last_error = str(exc)
        readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not readable:
            continue
        line = sys.stdin.readline().strip()
        if not line:
            # Do not release controllers on a closed UI input stream.
            time.sleep(0.1)
            continue
        try:
            fps, steps, threshold, offset, start_pose, prompt = parse_resume_config(line)
            if not healthy:
                raise ValueError("position hold is unhealthy; inspect hardware before resuming")
            args.fps, args.n_action_steps = fps, steps
            args.gripper_threshold, args.prompt, args.start_pose = threshold, prompt, start_pose
            args.tcp_offset_m = offset
            control.pause = False
            print("BIMANUAL_PREPARING: applying RESUME config and rechecking targets", flush=True)
            return True
        except ValueError as exc:
            print(f"RESUME_CONFIG_REJECTED {exc}", flush=True)
    return False


def prepare_bimanual(arms, rig, sdk, args, limits, offset, control):
    if control.interrupted():
        return False
    if args.start_pose:
        targets, frame = bimanual_start_pose()
        print(f"BIMANUAL_START_POSE_SOURCE path={BIMANUAL_START_TRAJECTORY} frame={frame['frame_index']}", flush=True)
        for side, (joints, gripper) in targets.items():
            print(f"BIMANUAL_START_POSE_TARGET side={side} joints={joints.tolist()} gripper_raw={gripper}", flush=True)
        if not wait_for_start_pose(control.interrupted):
            return False
        if control.interrupted():
            return False
        print(f"MOVE accepted; moving both arms to stack-cola demonstration frame {BIMANUAL_START_FRAME}", flush=True)
        move_bimanual_to_start_pose(arms, targets, control.interrupted)
        if not wait_for_run(control.interrupted):
            return False
    # Prepare itself authorizes inference when no recorded start move is selected.
    # Every action, including index zero, is validated/skipped in the execution loop.
    if control.interrupted():
        return False
    print("RUN accepted; dual-arm guarded execution started", flush=True)
    return True


def run_bimanual(args, sdk, SingleArm, RealSenseRig, control) -> int:
    rig, arms = None, {}
    limits = GuardLimits()
    try:
        rig = RealSenseRig(width=640, height=480, fps=30)
        rig.start(CAMERA_SERIALS)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if control.shutdown:
                return 0
            if all(rig.status()[role]["online"] for role in CAMERA_SERIALS):
                break
            time.sleep(.1)
        else:
            raise RuntimeError(f"cameras did not become ready: {rig.status()}")
        for side, can_port in (("left", args.left_can), ("right", args.right_can)):
            arms[side] = SingleArm({"can_port": can_port, "type": 2})
            if not arms[side].protect_mode():
                raise RuntimeError(f"{side} arm did not enter protect mode")
        while not control.shutdown:
            try:
                offset = np.asarray(TCP_OFFSET_M if args.tcp_offset_m is None else args.tcp_offset_m, dtype=np.float64)
                ready = prepare_bimanual(arms, rig, sdk, args, limits, offset, control)
                deadline = time.monotonic() + args.max_runtime_seconds
                period, steps = 1.0 / args.fps, 0
                paused_targets = None
                while ready and not control.interrupted() and time.monotonic() < deadline:
                    if control.pause:
                        if paused_targets is None:
                            paused_targets = capture_bimanual_hold(arms)
                            print("PAUSED", flush=True)
                        hold_bimanual_position(arms, paused_targets)
                        time.sleep(period)
                        continue
                    if paused_targets is not None:
                        paused_targets = None
                        print("RESUMED", flush=True)
                    observation, feedback, tcp = read_bimanual_observation(arms, rig, sdk, offset, args.prompt)
                    actions, latency = infer(args.policy_url, observation, args.max_policy_latency + 1)
                    if latency > args.max_policy_latency:
                        raise RuntimeError(f"policy latency {latency:.2f}s exceeds watchdog")
                    skipped = 0
                    for step_index, action in enumerate(actions[:args.n_action_steps]):
                        if control.interrupted() or control.pause or time.monotonic() >= deadline:
                            break
                        tick = time.monotonic()
                        feedback = {s: np.asarray(arms[s].get_joint_positions(), dtype=np.float64) for s in ("left", "right")}
                        tcp = {s: flange_to_tcp_state(np.asarray(sdk.forward_kinematics(feedback[s][:6], type=2)), float(feedback[s][6]), tcp_offset_m=offset) for s in ("left", "right")}
                        if execute_bimanual_step(arms, sdk, action, feedback, tcp, limits, offset,
                                                 args.gripper_threshold, step_index,
                                                 should_cancel=lambda: control.interrupted() or control.pause):
                            steps += 1
                        else:
                            skipped += 1
                        time.sleep(max(0.0, period - (time.monotonic() - tick)))
                    if skipped:
                        print(f"BIMANUAL_CHUNK_SKIPPED count={skipped}", flush=True)
                print(f"BIMANUAL_TEST_ENDED steps={steps}; retaining controllers", flush=True)
            except Exception as exc:
                # Includes network/camera/SDK failures, not just IK ValueError.
                print(f"BIMANUAL_CYCLE_REJECTED {type(exc).__name__}: {exc}", flush=True)
            if control.shutdown or not wait_bimanual_resume(arms, args, control):
                break
        return 0
    finally:
        # Reached only for real shutdown or failure to initialize controllers.
        for arm in arms.values():
            try: arm.protect_mode()
            except Exception: pass
            try: arm.close()
            except Exception: pass
        if rig is not None:
            rig.stop()


def parse_resume_config(line: str) -> tuple[float, int, float, np.ndarray, bool, str]:
    """Validate runtime parameters sent by the console while the arm is held."""
    if not line.startswith("RESUME "):
        raise ValueError("expected RESUME command")
    try:
        value = json.loads(line.removeprefix("RESUME "))
        fps = float(value["fps"])
        n_action_steps = int(value["n_action_steps"])
        gripper_threshold = float(value["gripper_threshold"])
        tcp_offset_m = np.asarray(value["tcp_offset_m"], dtype=np.float64)
        start_pose = value["start_pose"]
        prompt = str(value["prompt"]).strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid resume configuration") from exc
    if not np.isfinite(fps) or not 1 <= fps <= 50:
        raise ValueError("resume fps must be in [1, 50]")
    if not 1 <= n_action_steps <= ACTION_HORIZON:
        raise ValueError(f"resume n_action_steps must be in [1, {ACTION_HORIZON}]")
    if not np.isfinite(gripper_threshold) or not 0 <= gripper_threshold <= 1:
        raise ValueError("resume gripper_threshold must be in [0, 1]")
    if tcp_offset_m.shape != (3,) or not np.isfinite(tcp_offset_m).all() or np.linalg.norm(tcp_offset_m) > 0.30:
        raise ValueError("resume tcp_offset_m is invalid")
    if not isinstance(start_pose, bool):
        raise ValueError("resume start_pose must be boolean")
    if not prompt or len(prompt) > 1000:
        raise ValueError("resume prompt must contain 1 to 1000 characters")
    return fps, n_action_steps, gripper_threshold, tcp_offset_m, start_pose, prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--arm-mode", choices=("single", "bimanual"), default="single")
    parser.add_argument("--left-can", default=os.environ.get("TATE_LEFT_CAN", "can1"))
    parser.add_argument("--right-can", default=os.environ.get("TATE_RIGHT_CAN", "can3"))
    parser.add_argument("--start-pose", action="store_true", help="move to recorded start pose before inference (single-arm: frame 60; bimanual default: frame 120)")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8019")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--prompt", default=PROMPT, help="language instruction passed with every policy observation")
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
    if not 1 <= args.fps <= 50 or not 1 <= args.n_action_steps <= ACTION_HORIZON:
        parser.error(
            f"fps must be [1, 50] and n-action-steps must be [1, {ACTION_HORIZON}]"
        )
    if not np.isfinite(args.gripper_threshold) or not 0.0 <= args.gripper_threshold <= 1.0:
        parser.error("--gripper-threshold must be in [0, 1]")
    args.prompt = args.prompt.strip()
    if not args.prompt or len(args.prompt) > 1000:
        parser.error("--prompt must contain 1 to 1000 characters")
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
    resume_start_pose = False
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

    if args.arm_mode == "bimanual":
        control = BimanualControl()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2, signal.SIGHUP):
            signal.signal(sig, control.on_signal)
        return run_bimanual(args, sdk, SingleArm, RealSenseRig, control)

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
            print("START_POSE_DIRECT_COMMAND", flush=True)
            if not wait_for_start_pose(lambda: stop):
                print("MOVE not confirmed; no motion action sent", flush=True)
                return 2
            print("MOVE accepted; moving right arm to recorded start pose", flush=True)
            move_to_start_pose(arm, frames, lambda: stop)
            # Images and state must be read again after the arm has moved.
        if args.start_pose and not wait_for_run(lambda: stop):
            print("RUN not confirmed; no motion action sent", flush=True)
            return 2

        print("RUN accepted; using a fresh observation before the first command", flush=True)
        period = 1.0 / args.fps
        paused = False
        steps = 0
        while True:
            started = time.monotonic()
            try:
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
                    observation, feedback, tcp = read_observation(arm, rig, sdk, tcp_offset_m, args.prompt)
                    actions, latency = infer(args.policy_url, observation, args.max_policy_latency + 1)
                    if latency > args.max_policy_latency:
                        raise RuntimeError(f"policy latency {latency:.2f}s exceeds watchdog")
                    if stop or pause_toggle:
                        continue
                    for step_index, action in enumerate(actions[: args.n_action_steps]):
                        if stop or pause_toggle:
                            break
                        tick = time.monotonic()
                        feedback = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                        flange = np.asarray(sdk.forward_kinematics(feedback[:6], type=2), dtype=np.float64)
                        tcp = flange_to_tcp_state(flange, float(feedback[6]), tcp_offset_m=tcp_offset_m)
                        sent = execute_single_step(
                            arm, sdk, action, feedback, tcp, limits, tcp_offset_m,
                            args.gripper_threshold, step_index, should_cancel=lambda: stop or pause_toggle,
                        )
                        steps += int(sent)
                        if steps % max(1, round(args.fps)) == 0:
                            print(f"step={steps} policy_latency={latency:.3f}s right_tcp={np.array2string(tcp, precision=4)}", flush=True)
                        time.sleep(max(0.0, period - (time.monotonic() - tick)))
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"SINGLE_CYCLE_REJECTED {type(exc).__name__}: {exc}", flush=True)
                hold_requested = True
                stop = True
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
                        fps, n_action_steps, gripper_threshold, tcp_offset_m, resume_start_pose, prompt = parse_resume_config(line)
                        args.fps = fps
                        args.n_action_steps = n_action_steps
                        args.gripper_threshold = gripper_threshold
                        args.prompt = prompt
                        print(
                            "RESUME_CONFIG "
                            f"fps={args.fps:.3f} n_action_steps={args.n_action_steps} "
                            f"gripper_threshold={args.gripper_threshold:.4f} "
                            f"tcp_offset_m={np.array2string(tcp_offset_m, precision=6)} "
                            f"start_pose={resume_start_pose}",
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
            if resume_start_pose:
                initial = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                frames, record = start_pose_trajectory(sdk, initial, limits, tcp_offset_m)
                print(f"START_POSE_SOURCE episode={record['episode_index']} frame={record['frame_index']} side={record['side']}", flush=True)
                print(f"START_POSE_CURRENT_JOINTS {np.array2string(initial, precision=5)}", flush=True)
                print(f"START_POSE_TARGET_JOINTS {np.array2string(frames[-1][0], precision=5)} gripper_raw={frames[-1][1]:.5f}", flush=True)
                print("START_POSE_DIRECT_COMMAND automatic_resume=true", flush=True)
                move_to_start_pose(arm, frames, lambda: stop)
                if not wait_for_run(lambda: stop):
                    print("RUN not confirmed after start pose; no motion action sent", flush=True)
                    return 2
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
