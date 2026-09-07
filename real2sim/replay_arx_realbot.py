#!/usr/bin/env python3
"""Replay ARX joint trajectories on the real robot.

Default mode is dry-run validation only. Real hardware commands require both
--execute and --yes-i-understand-risk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "outputs" / "test" / "ik" / "dual_arm_ik_mink.npz"
DEFAULT_SDK_ROOT = Path("../ARX5_beta")

FPS = 30.0
SIM_GRIPPER_MAX_M = 0.088
SDK_GRIPPER_CLOSED = -3.4
SDK_GRIPPER_OPEN = 0.1

LEFT_SLICE = slice(0, 7)
RIGHT_SLICE = slice(7, 14)
SIDE_ORDER = ("left", "right")
JOINT_LOWER = np.deg2rad([-150.0, 0.0, 0.0, -90.0, -90.0, -120.0])
JOINT_UPPER = np.asarray([np.pi, np.deg2rad(210.0), np.pi, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(120.0)])
RANGE_TOL = np.deg2rad(0.25)


@dataclass
class SideTrajectory:
    arm_qpos: np.ndarray
    gripper_pos: np.ndarray


@dataclass
class JointTrajectory:
    time_s: np.ndarray
    sides: dict[str, SideTrajectory]
    source: str


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def sim_width_to_sdk_gripper(width_m: np.ndarray) -> np.ndarray:
    width = np.clip(np.asarray(width_m, dtype=np.float64), 0.0, SIM_GRIPPER_MAX_M)
    alpha = width / SIM_GRIPPER_MAX_M
    return SDK_GRIPPER_CLOSED + alpha * (SDK_GRIPPER_OPEN - SDK_GRIPPER_CLOSED)


def infer_gripper_from_npz_columns(columns: np.ndarray) -> np.ndarray:
    values = np.asarray(columns, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] >= 2:
        return sim_width_to_sdk_gripper(values[:, 0] + values[:, 1])
    values = values.reshape(-1)
    if np.nanmin(values) < -0.2 or np.nanmax(values) > 0.2:
        return np.clip(values, SDK_GRIPPER_CLOSED, SDK_GRIPPER_OPEN)
    return sim_width_to_sdk_gripper(values)


def side_from_npz(data: dict[str, np.ndarray], side: str) -> SideTrajectory | None:
    q = data.get(f"{side}_joint_qpos")
    if q is None:
        q = data.get(f"{side}_arm_qpos")
    if q is None and side == "right":
        q = data.get("joint_qpos")
    if q is None:
        return None

    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 6:
        raise RuntimeError(f"{side} joint data must be Nx6 or wider, got {q.shape}")

    if q.shape[1] >= 7:
        gripper = infer_gripper_from_npz_columns(q[:, 6:])
    elif f"{side}_gripper_pos" in data:
        gripper = np.asarray(data[f"{side}_gripper_pos"], dtype=np.float64).reshape(-1)
    elif f"{side}_grasp" in data:
        grasp = np.asarray(data[f"{side}_grasp"], dtype=np.int32).reshape(-1)
        gripper = np.where(grasp > 0, SDK_GRIPPER_CLOSED, SDK_GRIPPER_OPEN)
    else:
        gripper = np.full(len(q), SDK_GRIPPER_OPEN, dtype=np.float64)

    return SideTrajectory(arm_qpos=q[:, :6], gripper_pos=gripper)


def load_npz(path: Path) -> JointTrajectory:
    raw = np.load(path, allow_pickle=False)
    data = {name: raw[name] for name in raw.files}
    right = side_from_npz(data, "right")
    if right is None:
        raise RuntimeError(f"NPZ must contain at least right_joint_qpos/right_arm_qpos/joint_qpos: {path}")
    sides = {"right": right}
    left = side_from_npz(data, "left")
    if left is not None:
        sides["left"] = left

    n = len(right.arm_qpos)
    time_s = np.asarray(data.get("time_s", np.arange(n, dtype=np.float64) / FPS), dtype=np.float64).reshape(-1)
    return JointTrajectory(time_s=time_s, sides=sides, source="npz")


def load_parquet(path: Path) -> JointTrajectory:
    import pandas as pd

    df = pd.read_parquet(path)
    source_col = "observation.state" if "observation.state" in df.columns else "action"
    if source_col not in df.columns:
        raise RuntimeError(f"Parquet requires observation.state or action: {path}")
    state = np.stack(df[source_col].to_numpy()).astype(np.float64)
    if state.ndim != 2 or state.shape[1] < RIGHT_SLICE.stop:
        raise RuntimeError(f"{source_col} must contain at least 14 values, got {state.shape}")

    if "timestamp" in df.columns:
        time_s = df["timestamp"].to_numpy(dtype=np.float64)
        time_s -= time_s[0]
    else:
        time_s = np.arange(len(state), dtype=np.float64) / FPS

    left = state[:, LEFT_SLICE]
    right = state[:, RIGHT_SLICE]
    return JointTrajectory(
        time_s=time_s,
        sides={
            "left": SideTrajectory(left[:, :6], np.clip(left[:, 6], SDK_GRIPPER_CLOSED, SDK_GRIPPER_OPEN)),
            "right": SideTrajectory(right[:, :6], np.clip(right[:, 6], SDK_GRIPPER_CLOSED, SDK_GRIPPER_OPEN)),
        },
        source="parquet",
    )


def load_trajectory(path: Path) -> JointTrajectory:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    if suffix == ".parquet":
        return load_parquet(path)
    raise RuntimeError(f"Unsupported data format {suffix}; expected .npz or .parquet")


def validate_trajectory(traj: JointTrajectory) -> None:
    if "right" not in traj.sides:
        raise RuntimeError("Trajectory must contain at least right-arm data")
    n = len(traj.time_s)
    if n == 0:
        raise RuntimeError("Trajectory is empty")
    if not np.isfinite(traj.time_s).all():
        raise RuntimeError("time_s contains NaN/Inf")
    if len(traj.time_s) != n:
        raise RuntimeError("Invalid time_s")
    if n > 1 and np.any(np.diff(traj.time_s) < 0.0):
        raise RuntimeError("time_s must be monotonic")

    for side, data in traj.sides.items():
        if data.arm_qpos.shape != (n, 6):
            raise RuntimeError(f"{side} arm_qpos shape must be {(n, 6)}, got {data.arm_qpos.shape}")
        if data.gripper_pos.shape != (n,):
            raise RuntimeError(f"{side} gripper_pos shape must be {(n,)}, got {data.gripper_pos.shape}")
        if not np.isfinite(data.arm_qpos).all() or not np.isfinite(data.gripper_pos).all():
            raise RuntimeError(f"{side} trajectory contains NaN/Inf")
        below = data.arm_qpos < (JOINT_LOWER - RANGE_TOL)
        above = data.arm_qpos > (JOINT_UPPER + RANGE_TOL)
        if below.any() or above.any():
            bad = np.argwhere(below | above)[0]
            value = data.arm_qpos[bad[0], bad[1]]
            raise RuntimeError(f"{side} joint {bad[1] + 1} frame {bad[0]} out of range: {value:.4f} rad")
        np.clip(data.arm_qpos, JOINT_LOWER, JOINT_UPPER, out=data.arm_qpos)
        if np.any(data.gripper_pos < SDK_GRIPPER_CLOSED - RANGE_TOL) or np.any(data.gripper_pos > SDK_GRIPPER_OPEN + RANGE_TOL):
            raise RuntimeError(f"{side} gripper command outside [{SDK_GRIPPER_CLOSED}, {SDK_GRIPPER_OPEN}]")
        np.clip(data.gripper_pos, SDK_GRIPPER_CLOSED, SDK_GRIPPER_OPEN, out=data.gripper_pos)


def duration_between(time_s: np.ndarray, i0: int, i1: int, speed: float) -> float:
    if len(time_s) <= 1:
        return 1.0 / FPS
    dt = float(time_s[i1] - time_s[i0])
    if dt <= 0.0:
        dt = 1.0 / FPS
    return dt / max(float(speed), 1e-6)


def max_segment_duration(a: dict[str, tuple[np.ndarray, float]], b: dict[str, tuple[np.ndarray, float]], max_joint_speed: float, max_gripper_speed: float) -> float:
    needed = 0.0
    for side in b:
        if side not in a:
            continue
        dq = np.max(np.abs(b[side][0] - a[side][0]))
        dg = abs(float(b[side][1] - a[side][1]))
        needed = max(needed, float(dq) / max_joint_speed, dg / max_gripper_speed)
    return needed


def frame_at(traj: JointTrajectory, i: int) -> dict[str, tuple[np.ndarray, float]]:
    return {side: (data.arm_qpos[i].copy(), float(data.gripper_pos[i])) for side, data in traj.sides.items()}


def build_command_frames(traj: JointTrajectory, args: argparse.Namespace) -> list[dict[str, tuple[np.ndarray, float]]]:
    frames = [frame_at(traj, 0)]
    for i in range(1, len(traj.time_s)):
        prev = frame_at(traj, i - 1)
        curr = frame_at(traj, i)
        nominal_dt = duration_between(traj.time_s, i - 1, i, args.speed)
        safe_dt = max(nominal_dt, max_segment_duration(prev, curr, args.max_joint_speed, args.max_gripper_speed))
        steps = max(1, int(np.ceil(safe_dt * args.rate)))
        for step in range(1, steps + 1):
            alpha = step / steps
            interp = {}
            for side in curr:
                if side in prev:
                    q = (1.0 - alpha) * prev[side][0] + alpha * curr[side][0]
                    g = (1.0 - alpha) * prev[side][1] + alpha * curr[side][1]
                else:
                    q, g = curr[side]
                interp[side] = (q, float(g))
            frames.append(interp)
    return frames


def summarize(traj: JointTrajectory, command_frames: list[dict[str, tuple[np.ndarray, float]]], args: argparse.Namespace) -> None:
    print(f"source={traj.source} frames={len(traj.time_s)} command_frames={len(command_frames)} rate={args.rate:.1f}Hz")
    print(f"sides={','.join(side for side in SIDE_ORDER if side in traj.sides)}")
    for side in SIDE_ORDER:
        data = traj.sides.get(side)
        if data is None:
            continue
        dq = np.diff(data.arm_qpos, axis=0)
        dt = np.diff(traj.time_s)
        dt[dt <= 0.0] = 1.0 / FPS
        vel = np.abs(dq / dt[:, None]) if len(dq) else np.zeros((0, 6))
        print(
            f"{side}: q_min={np.array2string(data.arm_qpos.min(axis=0), precision=3)} "
            f"q_max={np.array2string(data.arm_qpos.max(axis=0), precision=3)} "
            f"max_input_vel={float(vel.max()) if vel.size else 0.0:.3f}rad/s "
            f"max_command_vel={command_max_velocity(command_frames, side, args.rate):.3f}rad/s "
            f"gripper=[{float(data.gripper_pos.min()):.3f},{float(data.gripper_pos.max()):.3f}]"
        )
    print("dry-run only; add --execute --yes-i-understand-risk to command real hardware")


def command_max_velocity(command_frames: list[dict[str, tuple[np.ndarray, float]]], side: str, rate: float) -> float:
    q = [frame[side][0] for frame in command_frames if side in frame]
    if len(q) < 2:
        return 0.0
    dq = np.diff(np.asarray(q, dtype=np.float64), axis=0)
    return float(np.abs(dq * rate).max())


def import_sdk(sdk_root: Path):
    sdk_root = as_abs(sdk_root)
    if not sdk_root.is_dir():
        raise FileNotFoundError(f"SDK root not found: {sdk_root}")
    sys.path.insert(0, str(sdk_root))
    try:
        from bimanual import SingleArm, Loading
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ARX SDK from "
            f"{sdk_root}. The shipped bimanual extension may need the SDK's "
            "Python version/dependencies, or it must be rebuilt for this interpreter."
        ) from exc

    return SingleArm, Loading


def build_real_arms(args: argparse.Namespace):
    SingleArm, Loading = import_sdk(Path(args.sdk_root))
    configs = {
        "left": {"can_port": args.left_can, "type": args.arm_type},
        "right": {"can_port": args.right_can, "type": args.arm_type},
    }
    arms = {}
    built = []
    try:
        for side in SIDE_ORDER:
            if side not in args.active_sides:
                continue
            with Loading(f"connect {side} arm on {configs[side]['can_port']}"):
                arm = SingleArm(configs[side])
            arms[side] = arm
            built.append(arm)
    except Exception:
        for arm in built:
            try:
                arm.close()
            except Exception:
                pass
        raise
    return SingleArm, arms


def current_frame_from_arms(arms: dict[str, Any], first: dict[str, tuple[np.ndarray, float]]) -> dict[str, tuple[np.ndarray, float]]:
    frame = {}
    for side, arm in arms.items():
        pos = np.asarray(arm.get_joint_positions(), dtype=np.float64).reshape(-1)
        if len(pos) < 6:
            raise RuntimeError(f"{side} arm returned fewer than 6 joint positions: {pos}")
        gripper = float(pos[6]) if len(pos) >= 7 else first[side][1]
        frame[side] = (pos[:6], gripper)
    return frame


def interpolate_frames(start: dict[str, tuple[np.ndarray, float]], end: dict[str, tuple[np.ndarray, float]], duration: float, rate: float) -> list[dict[str, tuple[np.ndarray, float]]]:
    steps = max(1, int(np.ceil(duration * rate)))
    frames = []
    for step in range(1, steps + 1):
        alpha = step / steps
        frame = {}
        for side in end:
            q = (1.0 - alpha) * start[side][0] + alpha * end[side][0]
            g = (1.0 - alpha) * start[side][1] + alpha * end[side][1]
            frame[side] = (q, float(g))
        frames.append(frame)
    return frames


def protect_and_close(SingleArm, arms: dict[str, Any]) -> None:
    for arm in arms.values():
        try:
            arm.protect_mode()
        except Exception:
            pass
    try:
        SingleArm.hold(1.0)
    except Exception:
        pass
    for arm in arms.values():
        try:
            arm.close()
        except Exception:
            pass


def send_frame(arms: dict[str, Any], frame: dict[str, tuple[np.ndarray, float]], duration: float) -> None:
    for side, arm in arms.items():
        q, gripper = frame[side]
        arm.set_joint_positions(positions=q.tolist(), duration=float(duration))
        arm.set_gripper_pos(float(gripper))


def check_faults(arms: dict[str, Any]) -> None:
    for side, arm in arms.items():
        fault = getattr(arm, "fault", None)
        if fault:
            raise RuntimeError(f"{side} arm fault: {fault}")


def execute_trajectory(traj: JointTrajectory, command_frames: list[dict[str, tuple[np.ndarray, float]]], args: argparse.Namespace) -> None:
    if not args.execute:
        summarize(traj, command_frames, args)
        return
    if not args.yes_i_understand_risk:
        raise RuntimeError("Real hardware replay requires --yes-i-understand-risk")

    args.active_sides = tuple(side for side in SIDE_ORDER if side in traj.sides)
    SingleArm, arms = build_real_arms(args)
    try:
        first = command_frames[0]
        current = current_frame_from_arms(arms, first)
        ramp_duration = max(
            args.ramp_time,
            max_segment_duration(current, first, args.max_joint_speed, args.max_gripper_speed),
        )
        ramp_frames = interpolate_frames(current, first, ramp_duration, args.rate)
        print(f"ramp_to_start duration={ramp_duration:.2f}s frames={len(ramp_frames)}")

        period = 1.0 / args.rate
        for frame in ramp_frames:
            send_frame(arms, frame, period)
            check_faults(arms)
            time.sleep(period)

        print(f"execute replay frames={len(command_frames)} rate={args.rate:.1f}Hz")
        for i, frame in enumerate(command_frames):
            t0 = time.monotonic()
            send_frame(arms, frame, period)
            check_faults(arms)
            if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or i == len(command_frames) - 1):
                print(f"sent {i + 1}/{len(command_frames)}")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: entering protect mode")
        raise
    except Exception:
        traceback.print_exc()
        raise
    finally:
        protect_and_close(SingleArm, arms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="IK npz or real robot parquet")
    parser.add_argument("--sdk-root", default=str(DEFAULT_SDK_ROOT))
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can3")
    parser.add_argument("--arm-type", type=int, default=2, help="X5-2025 on AC one is type 2")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--max-joint-speed", type=float, default=0.6, help="rad/s safety limit")
    parser.add_argument("--max-gripper-speed", type=float, default=1.0, help="SDK gripper units/s safety limit")
    parser.add_argument("--ramp-time", type=float, default=3.0, help="Minimum seconds to move from current pose to first frame")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Send commands to the real robot")
    parser.add_argument("--yes-i-understand-risk", action="store_true", help="Required together with --execute")
    args = parser.parse_args()

    if args.rate <= 0.0:
        raise RuntimeError("--rate must be positive")
    if args.max_joint_speed <= 0.0 or args.max_gripper_speed <= 0.0:
        raise RuntimeError("speed limits must be positive")

    traj = load_trajectory(as_abs(args.data))
    validate_trajectory(traj)
    command_frames = build_command_frames(traj, args)
    execute_trajectory(traj, command_frames, args)


if __name__ == "__main__":
    main()
