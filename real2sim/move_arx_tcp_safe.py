#!/usr/bin/env python3
"""Guarded one-shot TCP motion test for an ARX X5 arm.

The default mode is offline-only: it imports the SDK and checks IK/FK without
opening CAN or constructing ``SingleArm``. Hardware motion requires both
``--execute`` and ``--yes-i-understand-risk``.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDK_ROOT = REPO_ROOT.parent / "ARX5_beta"
SIDE_TO_CAN = {"left": "can1", "right": "can3"}
JOINT_LOWER = np.deg2rad([-150.0, 0.0, 0.0, -90.0, -90.0, -120.0])
JOINT_UPPER = np.deg2rad([180.0, 210.0, 180.0, 90.0, 90.0, 120.0])
BLOCKING_PROCESS_MARKERS = (
    "arx_button_control",
    "fold_box_policy_robot",
    "safe_dual_gravity.py",
    "test_gravity_compensation.py",
    "single_arm_demo.py",
    "dual_arm_demo.py",
    "test_remote_arm.py",
    "test_remote_encoder.py",
)


def acquire_arm_control_lock():
    runtime_directory = Path(
        os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    )
    if not runtime_directory.is_dir():
        runtime_directory = Path("/tmp")
    lock_path = runtime_directory / "arx-arm-control.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if runtime_directory == Path("/tmp") or exc.errno not in (
            errno.EACCES,
            errno.EROFS,
        ):
            raise
        lock_path = Path("/tmp/arx-arm-control.lock")
        descriptor = os.open(lock_path, flags, 0o600)
    stream = os.fdopen(descriptor, "r+", encoding="ascii")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError("another ARX arm-control process holds the control lock") from exc
    return stream


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_distance_deg(a_rpy: np.ndarray, b_rpy: np.ndarray) -> float:
    relative = rotation_from_rpy(a_rpy).T @ rotation_from_rpy(b_rpy)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(6)
    target = np.asarray(target, dtype=np.float64).reshape(6)
    return (
        float(np.linalg.norm(actual[:3] - target[:3])),
        rotation_distance_deg(actual[3:], target[3:]),
    )


def resolve_target(args: argparse.Namespace, current_pose: np.ndarray | None) -> np.ndarray:
    if args.target is not None:
        return np.asarray(args.target, dtype=np.float64)
    if current_pose is None:
        raise RuntimeError("--delta requires --current-pose in offline mode")
    return np.asarray(current_pose, dtype=np.float64) + np.asarray(args.delta, dtype=np.float64)


def validate_target_change(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_deg: float,
) -> tuple[float, float]:
    translation, rotation = pose_error(current_pose, target_pose)
    if translation > max_translation_m + 1e-12:
        raise RuntimeError(
            f"TCP translation {translation:.6f}m exceeds limit {max_translation_m:.6f}m"
        )
    if rotation > max_rotation_deg + 1e-9:
        raise RuntimeError(
            f"TCP rotation {rotation:.3f}deg exceeds limit {max_rotation_deg:.3f}deg"
        )
    return translation, rotation


def validate_ik(
    q: np.ndarray,
    q_seed: np.ndarray,
    fk_pose: np.ndarray,
    target_pose: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError(f"IK returned invalid joint vector: shape={q.shape}, q={q}")
    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        raise RuntimeError(f"IK solution is outside X5 joint limits: {q}")
    max_joint_step = float(np.max(np.abs(q - np.asarray(q_seed, dtype=np.float64))))
    if max_joint_step > args.max_joint_step_rad + 1e-12:
        raise RuntimeError(
            f"IK joint step {max_joint_step:.4f}rad exceeds limit "
            f"{args.max_joint_step_rad:.4f}rad"
        )
    pos_error, rot_error = pose_error(fk_pose, target_pose)
    if pos_error > args.ik_position_tolerance_m:
        raise RuntimeError(
            f"IK/FK position residual {pos_error:.6f}m exceeds "
            f"{args.ik_position_tolerance_m:.6f}m"
        )
    if rot_error > args.ik_rotation_tolerance_deg:
        raise RuntimeError(
            f"IK/FK rotation residual {rot_error:.3f}deg exceeds "
            f"{args.ik_rotation_tolerance_deg:.3f}deg"
        )
    return max_joint_step, pos_error, rot_error


def controller_processes() -> list[str]:
    matches: list[str] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in BLOCKING_PROCESS_MARKERS):
            matches.append(f"pid={entry.name} {command.strip()}")
    return sorted(matches)


def import_sdk(sdk_root: Path) -> Any:
    root = sdk_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ARX SDK root not found: {root}")
    sys.path.insert(0, str(root))
    try:
        import bimanual
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"failed to import ARX SDK from {root}; use its CPython 3.12 environment"
        ) from exc
    return bimanual


def offline_check(args: argparse.Namespace, sdk: Any) -> None:
    current = None if args.current_pose is None else np.asarray(args.current_pose, dtype=np.float64)
    target = resolve_target(args, current)
    seed = np.asarray(args.seed_q, dtype=np.float64)
    if current is not None:
        translation, rotation = validate_target_change(
            current,
            target,
            max_translation_m=args.max_translation_m,
            max_rotation_deg=args.max_rotation_deg,
        )
        print(f"requested_change translation={translation:.6f}m rotation={rotation:.3f}deg")
    else:
        print("warning: no current pose supplied; hardware displacement guard was not evaluated")

    q = np.asarray(
        sdk.inverse_kinematics(target, q_init=seed, type=args.arm_type), dtype=np.float64
    )
    fk_pose = np.asarray(sdk.forward_kinematics(q, type=args.arm_type), dtype=np.float64)
    max_step, pos_error, rot_error = validate_ik(q, seed, fk_pose, target, args)
    print(f"target_xyzrpy={np.array2string(target, precision=6)}")
    print(f"ik_q={np.array2string(q, precision=6)}")
    print(
        f"offline_check=PASS max_seed_step={max_step:.4f}rad "
        f"fk_position_error={pos_error:.6f}m fk_rotation_error={rot_error:.3f}deg"
    )
    print("offline only; no CAN connection was opened and no motion command was sent")


def protect_and_close(arm: Any) -> None:
    try:
        arm.protect_mode()
    except Exception:  # noqa: BLE001
        pass
    try:
        arm.close()
    except Exception:  # noqa: BLE001
        pass


def execute_once(args: argparse.Namespace, sdk: Any) -> None:
    if not args.yes_i_understand_risk:
        raise RuntimeError("hardware motion also requires --yes-i-understand-risk")
    lock_handle = acquire_arm_control_lock()
    try:
        _execute_once_locked(args, sdk)
    finally:
        lock_handle.close()


def _execute_once_locked(args: argparse.Namespace, sdk: Any) -> None:
    blockers = controller_processes()
    if blockers and not args.ignore_controller_process_check:
        details = "\n  ".join(blockers)
        raise RuntimeError(
            "possible ARX controller processes are already running; stop them first:\n  " + details
        )
    if args.duration < args.min_duration:
        raise RuntimeError(
            f"duration {args.duration:.3f}s is below enforced minimum {args.min_duration:.3f}s"
        )

    can_port = args.can_port or SIDE_TO_CAN[args.side]
    config = {"can_port": can_port, "type": args.arm_type}
    arm = None
    sent = False
    try:
        with sdk.Loading(f"connect {args.side} arm on {can_port}"):
            arm = sdk.SingleArm(config)
        current_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
        current_q = np.asarray(arm.get_joint_positions(), dtype=np.float64).reshape(-1)[:6]
        target = resolve_target(args, current_pose)
        translation, rotation = validate_target_change(
            current_pose,
            target,
            max_translation_m=args.max_translation_m,
            max_rotation_deg=args.max_rotation_deg,
        )
        q = np.asarray(arm.inverse_kinematics(target, q_init=current_q), dtype=np.float64)
        fk_pose = np.asarray(arm.forward_kinematics(q), dtype=np.float64)
        max_step, pos_error, rot_error = validate_ik(q, current_q, fk_pose, target, args)

        print(f"side={args.side} can={can_port} arm_type={args.arm_type}")
        print(f"current_xyzrpy={np.array2string(current_pose, precision=6)}")
        print(f"target_xyzrpy={np.array2string(target, precision=6)}")
        print(f"current_q={np.array2string(current_q, precision=6)}")
        print(f"target_q={np.array2string(q, precision=6)}")
        print(
            f"guards=PASS translation={translation:.6f}m rotation={rotation:.3f}deg "
            f"max_joint_step={max_step:.4f}rad ik_pos={pos_error:.6f}m "
            f"ik_rot={rot_error:.3f}deg"
        )
        print(f"one motion command will be sent in {args.countdown:.1f}s; Ctrl+C to abort")
        time.sleep(args.countdown)
        result = arm.set_ee_pose_xyzrpy(target, duration=args.duration)
        sent = True
        if result is False:
            raise RuntimeError("SDK rejected set_ee_pose_xyzrpy")

        deadline = time.monotonic() + args.duration + args.timeout_padding
        final_pose = current_pose
        while time.monotonic() < deadline:
            fault = getattr(arm, "fault", None)
            if fault:
                raise RuntimeError(f"ARX arm fault: {fault}")
            final_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
            feedback_pos, feedback_rot = pose_error(final_pose, target)
            if (
                feedback_pos <= args.feedback_position_tolerance_m
                and feedback_rot <= args.feedback_rotation_tolerance_deg
            ):
                print(
                    f"motion=PASS feedback_position_error={feedback_pos:.6f}m "
                    f"feedback_rotation_error={feedback_rot:.3f}deg"
                )
                return
            time.sleep(args.poll_period)
        feedback_pos, feedback_rot = pose_error(final_pose, target)
        raise RuntimeError(
            f"motion did not converge: position_error={feedback_pos:.6f}m "
            f"rotation_error={feedback_rot:.3f}deg"
        )
    finally:
        if arm is not None:
            if sent:
                print("entering protect mode and closing arm")
            protect_and_close(arm)


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target", type=float, nargs=6, metavar=("X", "Y", "Z", "R", "P", "YAW")
    )
    target_group.add_argument(
        "--delta", type=float, nargs=6, metavar=("DX", "DY", "DZ", "DR", "DP", "DYAW")
    )
    parser.add_argument(
        "--current-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "R", "P", "YAW"),
        help="required with --delta during offline checking; ignored during execution",
    )
    parser.add_argument("--seed-q", type=float, nargs=6, default=[0.0, 1.2, 1.5, 0.0, 0.0, 0.0])
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--sdk-root", default=str(DEFAULT_SDK_ROOT))
    parser.add_argument("--can-port", default=None)
    parser.add_argument("--arm-type", type=int, default=2)
    parser.add_argument("--duration", type=positive, default=5.0)
    parser.add_argument("--min-duration", type=positive, default=3.0)
    parser.add_argument("--max-translation-m", type=positive, default=0.02)
    parser.add_argument("--max-rotation-deg", type=positive, default=5.0)
    parser.add_argument("--max-joint-step-rad", type=positive, default=0.35)
    parser.add_argument("--ik-position-tolerance-m", type=positive, default=0.002)
    parser.add_argument("--ik-rotation-tolerance-deg", type=positive, default=1.0)
    parser.add_argument("--feedback-position-tolerance-m", type=positive, default=0.005)
    parser.add_argument("--feedback-rotation-tolerance-deg", type=positive, default=2.0)
    parser.add_argument("--countdown", type=float, default=5.0)
    parser.add_argument("--timeout-padding", type=positive, default=3.0)
    parser.add_argument("--poll-period", type=positive, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes-i-understand-risk", action="store_true")
    parser.add_argument(
        "--ignore-controller-process-check",
        action="store_true",
        help="dangerous: allow execution while a known ARX control process is running",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.countdown < 0.0 or not math.isfinite(args.countdown):
        raise RuntimeError("--countdown must be finite and non-negative")
    sdk = import_sdk(Path(args.sdk_root))
    if args.execute:
        execute_once(args, sdk)
    else:
        offline_check(args, sdk)


if __name__ == "__main__":
    main()
