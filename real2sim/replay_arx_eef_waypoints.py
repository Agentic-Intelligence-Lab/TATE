#!/usr/bin/env python3
"""Replay sampled corrected-EEF waypoints on one or both ARX arms.

Normal hardware flow deliberately remains in position control between every
waypoint and while waiting at the final waypoint. After the operator presses
Enter, the arm returns home and the SDK connection closes without an explicit
protect-mode transition. Faults, exceptions, EOF, and Ctrl+C use protect mode
as an emergency fallback.

Without ``--execute`` this program performs SDK IK/FK checks only and never
constructs ``SingleArm`` or opens CAN.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .move_arx_tcp_safe import (
        JOINT_LOWER,
        JOINT_UPPER,
        SIDE_TO_CAN,
        acquire_arm_control_lock,
        controller_processes,
        import_sdk,
        pose_error,
    )
except ImportError:  # Direct execution: python real2sim/replay_arx_eef_waypoints.py
    from move_arx_tcp_safe import (
        JOINT_LOWER,
        JOINT_UPPER,
        SIDE_TO_CAN,
        acquire_arm_control_lock,
        controller_processes,
        import_sdk,
        pose_error,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDK_ROOT = REPO_ROOT.parent / "ARX5_beta"
HAND_KEYS = {"left": "hand_l", "right": "hand_r"}
FRAME_NAMES = {"left": "left_flange_zero", "right": "right_flange_zero"}
# This must match the TCP site used to generate/correct the current EEF
# artifacts. It decodes the artifact; it is not the latest physical fingertip
# measurement.
DEFAULT_TCP_OFFSET_M = (0.105, 0.0018, -0.0063)


@dataclass(frozen=True)
class Waypoint:
    frame_index: int
    timestamp_s: float
    matrix: np.ndarray
    xyzrpy: np.ndarray
    phase: str = "trajectory"
    grasp_state: int = 0
    tcp_matrix: np.ndarray | None = None


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to SDK-style XYZ fixed-axis roll/pitch/yaw."""
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def validate_pose_matrix(value: Any, *, frame_index: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError(f"frame {frame_index} has an invalid 4x4 TCP pose")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError(f"frame {frame_index} TCP pose has an invalid last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise RuntimeError(f"frame {frame_index} TCP rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise RuntimeError(f"frame {frame_index} TCP rotation determinant is not +1")
    return matrix


def tcp_target_to_sdk_flange(
    tcp_matrix: np.ndarray, tcp_offset_flange_m: np.ndarray
) -> np.ndarray:
    """Convert an artifact TCP pose to the SDK's moving-flange pose."""
    tcp = np.asarray(tcp_matrix, dtype=np.float64).reshape(4, 4)
    offset = np.asarray(tcp_offset_flange_m, dtype=np.float64).reshape(3)
    flange = tcp.copy()
    flange[:3, 3] = tcp[:3, 3] - tcp[:3, :3] @ offset
    return flange


def load_corrected_eef(
    path: Path,
    side: str,
    allow_uncorrected: bool,
    tcp_offset_flange_m: np.ndarray | tuple[float, float, float] = DEFAULT_TCP_OFFSET_M,
) -> list[Waypoint]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != "tate.dual_arm_eef" or payload.get("schema_version") != 2:
        raise RuntimeError(f"expected tate.dual_arm_eef v2: {path}")
    convention = payload.get("eef_coordinate_convention") or {}
    if convention.get("pose_semantics") != "arx_tcp" or not convention.get(
        "tcp_orientation_applied", False
    ):
        raise RuntimeError("EEF does not contain an orientation-corrected ARX TCP pose")
    frame_name = (convention.get("per_side_frames") or {}).get(side)
    if frame_name != FRAME_NAMES[side]:
        raise RuntimeError(
            f"{side} EEF frame must be {FRAME_NAMES[side]!r}, got {frame_name!r}"
        )
    correction = (payload.get("metadata") or {}).get("real_anchor_correction")
    if not correction and not allow_uncorrected:
        raise RuntimeError(
            "EEF has no metadata.real_anchor_correction; pass --allow-uncorrected-eef "
            "only when this is intentional"
        )

    hand_key = HAND_KEYS[side]
    waypoints: list[Waypoint] = []
    for ordinal, frame in enumerate(payload.get("frames") or []):
        hand = frame.get(hand_key)
        if not hand or hand.get("tcp_pose_eef_frame") is None:
            continue
        index = int(frame.get("idx", ordinal))
        stamp = frame.get("ts")
        timestamp = (
            float(stamp) * 1e-9
            if stamp is not None
            else ordinal / float(payload.get("fps") or 30.0)
        )
        tcp_matrix = validate_pose_matrix(hand["tcp_pose_eef_frame"], frame_index=index)
        matrix = tcp_target_to_sdk_flange(tcp_matrix, tcp_offset_flange_m)
        xyzrpy = np.concatenate([matrix[:3, 3], matrix_to_rpy(matrix[:3, :3])])
        grasp_state = int(hand.get("grasp_state", -1))
        if grasp_state not in (0, 1):
            raise RuntimeError(
                f"frame {index} {side} grasp_state must be 0=open or 1=closed"
            )
        waypoints.append(
            Waypoint(
                index,
                timestamp,
                matrix,
                xyzrpy,
                grasp_state=grasp_state,
                tcp_matrix=tcp_matrix,
            )
        )
    if not waypoints:
        raise RuntimeError(f"no valid {side} TCP poses in {path}")
    return waypoints


def select_waypoints(
    available: list[Waypoint],
    count: int,
    start_frame: int | None,
    end_frame: int | None,
) -> list[Waypoint]:
    selected = [
        waypoint
        for waypoint in available
        if (start_frame is None or waypoint.frame_index >= start_frame)
        and (end_frame is None or waypoint.frame_index <= end_frame)
    ]
    if not selected:
        raise RuntimeError("no valid EEF poses remain in the requested frame range")
    if count < 2:
        raise RuntimeError("--num-waypoints must be at least 2")
    if len(selected) <= count:
        return selected
    offsets = np.rint(np.linspace(0, len(selected) - 1, count)).astype(np.int64)
    unique_offsets = list(dict.fromkeys(int(value) for value in offsets))
    return [selected[offset] for offset in unique_offsets]


def select_synchronized_waypoints(
    available_by_side: dict[str, list[Waypoint]],
    count: int,
    start_frame: int | None,
    end_frame: int | None,
) -> dict[str, list[Waypoint]]:
    """Select a shared frame timeline and retain both sides of every grasp edge."""
    if count < 2:
        raise RuntimeError("--num-waypoints must be at least 2")
    maps = {
        side: {waypoint.frame_index: waypoint for waypoint in available}
        for side, available in available_by_side.items()
    }
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    frame_indices = sorted(
        frame
        for frame in common
        if (start_frame is None or frame >= start_frame)
        and (end_frame is None or frame <= end_frame)
    )
    if not frame_indices:
        raise RuntimeError("no common valid EEF frames remain for the selected arms")

    if len(frame_indices) <= count:
        selected_indices = set(frame_indices)
    else:
        offsets = np.rint(np.linspace(0, len(frame_indices) - 1, count)).astype(np.int64)
        selected_indices = {frame_indices[int(offset)] for offset in offsets}

    for side, mapping in maps.items():
        previous_frame = frame_indices[0]
        previous_state = mapping[previous_frame].grasp_state
        for frame in frame_indices[1:]:
            state = mapping[frame].grasp_state
            if state != previous_state:
                selected_indices.add(previous_frame)
                selected_indices.add(frame)
            previous_frame = frame
            previous_state = state

    ordered = sorted(selected_indices)
    return {side: [mapping[frame] for frame in ordered] for side, mapping in maps.items()}


def select_from_first_grasp_events(
    available_by_side: dict[str, list[Waypoint]],
    count: int,
    start_frame: int | None,
    end_frame: int | None,
) -> dict[str, list[Waypoint]]:
    """Use each arm's first grasp-change frame as its independent approach target.

    After those (possibly different) entry frames, all arms resume on one shared
    timeline strictly after the latest entry frame.
    """
    if count < 2:
        raise RuntimeError("--num-waypoints must be at least 2")
    maps = {
        side: {waypoint.frame_index: waypoint for waypoint in available}
        for side, available in available_by_side.items()
    }
    entries: dict[str, Waypoint] = {}
    for side, available in available_by_side.items():
        for previous, current in zip(available, available[1:]):
            if current.grasp_state == previous.grasp_state:
                continue
            if start_frame is not None and current.frame_index < start_frame:
                continue
            if end_frame is not None and current.frame_index > end_frame:
                continue
            entries[side] = current
            break
        if side not in entries:
            raise RuntimeError(
                f"no {side} grasp-state change exists in the requested frame range"
            )

    latest_entry = max(point.frame_index for point in entries.values())
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    tail_frames = sorted(
        frame
        for frame in common
        if frame > latest_entry and (end_frame is None or frame <= end_frame)
    )
    if not tail_frames:
        raise RuntimeError("no common trajectory frames remain after the grasp-event entry points")

    tail_count = count - 1
    if len(tail_frames) <= tail_count:
        selected_tail = set(tail_frames)
    else:
        offsets = np.rint(np.linspace(0, len(tail_frames) - 1, tail_count)).astype(np.int64)
        selected_tail = {tail_frames[int(offset)] for offset in offsets}

    # Preserve every later grasp edge and its immediately preceding frame.
    for side, mapping in maps.items():
        previous_frame = entries[side].frame_index
        previous_state = entries[side].grasp_state
        for frame in tail_frames:
            state = mapping[frame].grasp_state
            if state != previous_state:
                if previous_frame > latest_entry:
                    selected_tail.add(previous_frame)
                selected_tail.add(frame)
            previous_frame = frame
            previous_state = state

    ordered_tail = sorted(selected_tail)
    return {
        side: [entries[side], *(maps[side][frame] for frame in ordered_tail)]
        for side in maps
    }


def grasp_events(waypoints: list[Waypoint]) -> list[tuple[int, int, int]]:
    return [
        (current.frame_index, previous.grasp_state, current.grasp_state)
        for previous, current in zip(waypoints, waypoints[1:])
        if previous.grasp_state != current.grasp_state
    ]


def build_joint_approach(
    current_q: np.ndarray,
    target_q: np.ndarray,
    current_pose: np.ndarray,
    fk_ik: Any,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build a reachable joint-space ramp whose FK increments remain bounded."""
    start_q = np.asarray(current_q, dtype=np.float64).reshape(6)
    end_q = np.asarray(target_q, dtype=np.float64).reshape(6)
    largest_change = float(np.max(np.abs(end_q - start_q)))
    if largest_change <= 1e-8:
        return []
    if (
        not args.allow_long_approach
        and largest_change > args.max_approach_total_joint_rad
    ):
        raise RuntimeError(
            f"approach total joint change {largest_change:.4f}rad exceeds "
            f"{args.max_approach_total_joint_rad:.4f}rad"
        )

    steps = max(1, int(math.ceil(largest_change / args.approach_joint_step_rad)))
    while steps <= args.max_approach_steps:
        result: list[tuple[np.ndarray, np.ndarray]] = []
        previous_q = start_q
        previous_pose = np.asarray(current_pose, dtype=np.float64).reshape(6)
        valid = True
        for step in range(1, steps + 1):
            alpha = step / steps
            q = (1.0 - alpha) * start_q + alpha * end_q
            pose = np.asarray(fk_ik.forward_kinematics(q), dtype=np.float64).reshape(6)
            try:
                validate_joint_solution(
                    q,
                    previous_q,
                    pose,
                    pose,
                    max_joint_step_rad=args.approach_joint_step_rad,
                    position_tolerance_m=args.ik_position_tolerance_m,
                    rotation_tolerance_deg=args.ik_rotation_tolerance_deg,
                )
                validate_segment_pose(
                    previous_pose,
                    pose,
                    max_translation_m=args.approach_translation_step_m,
                    max_rotation_deg=args.approach_rotation_step_deg,
                )
            except RuntimeError:
                valid = False
                break
            result.append((q, pose))
            previous_q = q
            previous_pose = pose
        if valid:
            return result
        steps *= 2
    raise RuntimeError(
        "could not make the joint-space approach satisfy the configured FK step limits "
        f"within {args.max_approach_steps} steps"
    )


def validate_joint_solution(
    q: np.ndarray,
    previous_q: np.ndarray,
    fk_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    max_joint_step_rad: float | None,
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> tuple[float, float, float]:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError(f"IK returned invalid q: shape={q.shape}, q={q}")
    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        raise RuntimeError(f"IK solution is outside X5 joint limits: {q}")
    step = float(np.max(np.abs(q - np.asarray(previous_q, dtype=np.float64))))
    if max_joint_step_rad is not None and step > max_joint_step_rad + 1e-12:
        raise RuntimeError(
            f"IK joint step {step:.4f}rad exceeds limit {max_joint_step_rad:.4f}rad"
        )
    pos_error, rot_error = pose_error(fk_pose, target_pose)
    if pos_error > position_tolerance_m:
        raise RuntimeError(
            f"IK/FK position residual {pos_error:.6f}m exceeds {position_tolerance_m:.6f}m"
        )
    if rot_error > rotation_tolerance_deg:
        raise RuntimeError(
            f"IK/FK rotation residual {rot_error:.3f}deg exceeds {rotation_tolerance_deg:.3f}deg"
        )
    return step, pos_error, rot_error


def validate_segment_pose(
    previous_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_deg: float,
) -> tuple[float, float]:
    translation, rotation = pose_error(previous_pose, target_pose)
    if translation > max_translation_m + 1e-12:
        raise RuntimeError(
            f"waypoint translation {translation:.4f}m exceeds {max_translation_m:.4f}m"
        )
    if rotation > max_rotation_deg + 1e-9:
        raise RuntimeError(
            f"waypoint rotation {rotation:.2f}deg exceeds {max_rotation_deg:.2f}deg"
        )
    return translation, rotation


def solve_chain(
    waypoints: list[Waypoint],
    q_seed: np.ndarray,
    fk_ik: Any,
    args: argparse.Namespace,
    *,
    current_pose: np.ndarray | None,
) -> list[np.ndarray]:
    solutions: list[np.ndarray] = []
    previous_q = np.asarray(q_seed, dtype=np.float64)
    previous_pose = current_pose
    for ordinal, waypoint in enumerate(waypoints):
        if previous_pose is not None:
            validate_segment_pose(
                previous_pose,
                waypoint.xyzrpy,
                max_translation_m=(
                    args.max_first_translation_m if ordinal == 0 else args.max_segment_translation_m
                ),
                max_rotation_deg=(
                    args.max_first_rotation_deg if ordinal == 0 else args.max_segment_rotation_deg
                ),
            )
        try:
            q = np.asarray(
                fk_ik.inverse_kinematics(waypoint.xyzrpy, q_init=previous_q),
                dtype=np.float64,
            )
        except Exception as exc:
            tcp_xyz = (
                "unknown"
                if waypoint.tcp_matrix is None
                else np.array2string(waypoint.tcp_matrix[:3, 3], precision=6)
            )
            raise RuntimeError(
                f"IK failed at waypoint {ordinal + 1}/{len(waypoints)}: "
                f"phase={waypoint.phase}, frame={waypoint.frame_index}, "
                f"sdk_flange_xyzrpy={np.array2string(waypoint.xyzrpy, precision=6)}, "
                f"eef_tcp_xyz={tcp_xyz}: {exc}"
            ) from exc
        fk_pose = np.asarray(fk_ik.forward_kinematics(q), dtype=np.float64)
        validate_joint_solution(
            q,
            previous_q,
            fk_pose,
            waypoint.xyzrpy,
            # Offline mode has no measured robot state.  Its first q is only an
            # IK seed, so seed-to-solution distance is not a physical movement
            # and must not be treated as one.  Hardware mode supplies
            # current_pose and continues to enforce the strict first-step cap.
            max_joint_step_rad=(
                None
                if ordinal == 0 and current_pose is None
                else (
                    args.max_first_joint_step_rad
                    if ordinal == 0
                    else args.max_segment_joint_step_rad
                )
            ),
            position_tolerance_m=args.ik_position_tolerance_m,
            rotation_tolerance_deg=args.ik_rotation_tolerance_deg,
        )
        solutions.append(q)
        previous_q = q
        previous_pose = waypoint.xyzrpy
    return solutions


class ModuleKinematics:
    def __init__(self, sdk: Any, arm_type: int):
        self.sdk = sdk
        self.arm_type = arm_type

    def inverse_kinematics(self, pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        return self.sdk.inverse_kinematics(pose, q_init=q_init, type=self.arm_type)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        return self.sdk.forward_kinematics(q, type=self.arm_type)


def print_plan(waypoints: list[Waypoint], solutions: list[np.ndarray]) -> None:
    print(f"waypoints={len(waypoints)}")
    for ordinal, (waypoint, q) in enumerate(zip(waypoints, solutions), start=1):
        print(
            f"  {ordinal:02d}: phase={waypoint.phase} frame={waypoint.frame_index} "
            f"ts={waypoint.timestamp_s:.3f} "
            f"grasp={waypoint.grasp_state} "
            f"sdk_flange_xyzrpy={np.array2string(waypoint.xyzrpy, precision=5)} "
            + (
                ""
                if waypoint.tcp_matrix is None
                else f"eef_tcp_xyz={np.array2string(waypoint.tcp_matrix[:3, 3], precision=5)} "
            )
            + f"q={np.array2string(q, precision=5)}"
        )


def wait_for_target(
    arm: Any, target: np.ndarray, duration: float, args: argparse.Namespace
) -> None:
    deadline = time.monotonic() + duration + args.timeout_padding
    last_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
    while time.monotonic() < deadline:
        fault = getattr(arm, "fault", None)
        if fault:
            raise RuntimeError(f"ARX arm fault: {fault}")
        last_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
        pos_error, rot_error = pose_error(last_pose, target)
        if (
            pos_error <= args.feedback_position_tolerance_m
            and rot_error <= args.feedback_rotation_tolerance_deg
        ):
            return
        time.sleep(args.poll_period)
    pos_error, rot_error = pose_error(last_pose, target)
    raise RuntimeError(
        f"waypoint did not converge: position_error={pos_error:.6f}m "
        f"rotation_error={rot_error:.3f}deg"
    )


def wait_for_approach_target(
    arm: Any,
    target_q: np.ndarray,
    target_pose: np.ndarray,
    duration: float,
    args: argparse.Namespace,
) -> None:
    deadline = time.monotonic() + duration + args.timeout_padding
    last_q = np.asarray(arm.get_joint_positions(), dtype=np.float64).reshape(-1)[:6]
    last_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
    while time.monotonic() < deadline:
        fault = getattr(arm, "fault", None)
        if fault:
            raise RuntimeError(f"ARX arm fault: {fault}")
        last_q = np.asarray(arm.get_joint_positions(), dtype=np.float64).reshape(-1)[:6]
        last_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
        position_error, rotation_error = pose_error(last_pose, target_pose)
        if (
            position_error <= args.approach_feedback_position_tolerance_m
            and rotation_error <= args.approach_feedback_rotation_tolerance_deg
        ):
            joint_error = float(np.max(np.abs(last_q - target_q)))
            if joint_error > args.feedback_joint_tolerance_rad:
                print(
                    "approach EEF pose converged; joint residual is informational: "
                    f"max_joint_error={joint_error:.5f}rad"
                )
            return
        time.sleep(args.poll_period)
    joint_error = float(np.max(np.abs(last_q - target_q)))
    joint_residual = np.asarray(target_q, dtype=np.float64) - last_q
    position_error, rotation_error = pose_error(last_pose, target_pose)
    raise RuntimeError(
        "approach FK target did not converge: "
        f"position_error={position_error:.6f}m, "
        f"rotation_error={rotation_error:.3f}deg, "
        f"max_joint_error={joint_error:.5f}rad, "
        f"joint_residual_rad={np.array2string(joint_residual, precision=5)}, "
        f"actual_pose={np.array2string(last_pose, precision=5)}, "
        f"fk_target_pose={np.array2string(target_pose, precision=5)}"
    )


def command_approach_joint_target(
    arm: Any,
    target_q: np.ndarray,
    target_pose: np.ndarray,
    duration: float,
    args: argparse.Namespace,
    *,
    label: str,
) -> None:
    """Command one bounded approach step, retrying only convergence timeouts."""
    attempts = args.approach_retries + 1
    for attempt in range(1, attempts + 1):
        result = arm.set_joint_positions(
            positions=np.asarray(target_q, dtype=np.float64).tolist(),
            duration=float(duration),
        )
        if result is False:
            raise RuntimeError(f"SDK rejected {label} (attempt {attempt}/{attempts})")
        try:
            wait_for_approach_target(arm, target_q, target_pose, duration, args)
            return
        except RuntimeError as exc:
            if getattr(arm, "fault", None) or attempt == attempts:
                raise RuntimeError(
                    f"{label} failed after {attempt}/{attempts} attempt(s): {exc}"
                ) from exc
            current_q = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            ).reshape(-1)[:6]
            residual = np.asarray(target_q, dtype=np.float64) - current_q
            print(
                f"{label} did not settle on attempt {attempt}/{attempts}; "
                f"retrying the same bounded target, joint_residual_rad="
                f"{np.array2string(residual, precision=5)}"
            )


def offline_check(args: argparse.Namespace, sdk: Any, waypoints: list[Waypoint]) -> None:
    kinematics = ModuleKinematics(sdk, args.arm_type)
    seed = np.asarray(args.seed_q, dtype=np.float64)
    solutions = solve_chain(waypoints, seed, kinematics, args, current_pose=None)
    print(
        "offline note: the seed-to-first-IK joint distance is informational only; "
        "hardware execution checks the measured current state against the first waypoint"
    )
    print_plan(waypoints, solutions)
    print("offline_check=PASS")
    print("no SingleArm was constructed, no CAN connection opened, and no command sent")


def offline_check_multi(
    args: argparse.Namespace,
    sdk: Any,
    waypoints_by_side: dict[str, list[Waypoint]],
) -> None:
    print(f"selected_sides={','.join(waypoints_by_side)}")
    print(f"waypoints_per_side={len(next(iter(waypoints_by_side.values())))}")
    for side, waypoints in waypoints_by_side.items():
        print(
            f"\n[{side}] entry_frame={waypoints[0].frame_index} "
            f"entry_grasp_state={waypoints[0].grasp_state}"
        )
        print(f"[{side}] later_grasp_events={grasp_events(waypoints)}")
        offline_check(args, sdk, waypoints)


def wait_for_all_targets(
    arms: dict[str, Any],
    targets: dict[str, np.ndarray],
    duration: float,
    args: argparse.Namespace,
) -> None:
    deadline = time.monotonic() + duration + args.timeout_padding
    errors: dict[str, tuple[float, float]] = {}
    while time.monotonic() < deadline:
        converged = True
        for side, arm in arms.items():
            fault = getattr(arm, "fault", None)
            if fault:
                raise RuntimeError(f"{side} ARX arm fault: {fault}")
            pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
            errors[side] = pose_error(pose, targets[side])
            if (
                errors[side][0] > args.feedback_position_tolerance_m
                or errors[side][1] > args.feedback_rotation_tolerance_deg
            ):
                converged = False
        if converged:
            return
        time.sleep(args.poll_period)
    details = ", ".join(
        f"{side}: position={error[0]:.6f}m rotation={error[1]:.3f}deg"
        for side, error in errors.items()
    )
    raise RuntimeError(f"dual-arm waypoint did not converge ({details})")


def gripper_command(args: argparse.Namespace, grasp_state: int) -> float:
    return args.gripper_closed if grasp_state == 1 else args.gripper_open


def set_grasp_states(
    arms: dict[str, Any],
    states: dict[str, int],
    args: argparse.Namespace,
    *,
    label: str,
) -> None:
    for side, state in states.items():
        value = gripper_command(args, state)
        result = arms[side].set_gripper_pos(float(value))
        if result is False:
            raise RuntimeError(f"SDK rejected {side} gripper command at {label}")
        print(
            f"gripper {side} {label}: state={state} "
            f"({'closed' if state else 'open'}) sdk_value={value:.3f}"
        )
    time.sleep(args.gripper_settle_s)


def execute_multi(
    args: argparse.Namespace,
    sdk: Any,
    waypoints_by_side: dict[str, list[Waypoint]],
) -> None:
    if not args.yes_i_understand_risk:
        raise RuntimeError("hardware execution also requires --yes-i-understand-risk")
    lock_handle = acquire_arm_control_lock()
    try:
        _execute_multi_locked(args, sdk, waypoints_by_side)
    finally:
        lock_handle.close()


def _execute_multi_locked(
    args: argparse.Namespace,
    sdk: Any,
    waypoints_by_side: dict[str, list[Waypoint]],
) -> None:
    blockers = controller_processes()
    if blockers:
        raise RuntimeError("competing ARX control process detected:\n  " + "\n  ".join(blockers))
    can_ports = {
        "left": args.left_can,
        "right": args.right_can,
    }
    if len(waypoints_by_side) == 1 and args.can_port is not None:
        can_ports[next(iter(waypoints_by_side))] = args.can_port

    arms: dict[str, Any] = {}
    normal_completion = False
    try:
        for side in ("left", "right"):
            if side not in waypoints_by_side:
                continue
            with sdk.Loading(f"connect {side} arm on {can_ports[side]}"):
                arms[side] = sdk.SingleArm(
                    {"can_port": can_ports[side], "type": args.arm_type}
                )

        current_pose: dict[str, np.ndarray] = {}
        current_q: dict[str, np.ndarray] = {}
        solutions: dict[str, list[np.ndarray]] = {}
        approaches: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for side, arm in arms.items():
            waypoints = waypoints_by_side[side]
            current_pose[side] = np.asarray(
                arm.get_ee_pose_xyzrpy(), dtype=np.float64
            ).reshape(6)
            current_q[side] = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            ).reshape(-1)[:6]
            if not args.allow_long_approach:
                validate_segment_pose(
                    current_pose[side],
                    waypoints[0].xyzrpy,
                    max_translation_m=args.max_approach_total_translation_m,
                    max_rotation_deg=args.max_approach_total_rotation_deg,
                )
            try:
                solutions[side] = solve_chain(
                    waypoints, current_q[side], arm, args, current_pose=None
                )
                seed_name = "measured_current_q"
            except (RuntimeError, ValueError) as current_seed_error:
                print(
                    f"{side} current-state IK seed failed ({current_seed_error}); "
                    "retrying configured seed"
                )
                solutions[side] = solve_chain(
                    waypoints,
                    np.asarray(args.seed_q, dtype=np.float64),
                    arm,
                    args,
                    current_pose=None,
                )
                seed_name = "configured_seed_q"
            approaches[side] = build_joint_approach(
                current_q[side],
                solutions[side][0],
                current_pose[side],
                arm,
                args,
            )
            print(
                f"\n[{side}] can={can_ports[side]} approach_mode=joint_position "
                f"ik_seed={seed_name} steps={len(approaches[side])}"
            )
            print(f"current_pose={np.array2string(current_pose[side], precision=5)}")
            print(f"current_q={np.array2string(current_q[side], precision=5)}")
            print(f"first_target_q={np.array2string(solutions[side][0], precision=5)}")
            print(f"grasp_events={grasp_events(waypoints)}")
            print_plan(waypoints, solutions[side])

        input(
            "All-arm preflight passed. Inspect both paths; press Enter to approach "
            "the first poses, Ctrl+C to abort > "
        )
        # Approach one arm at a time so the operator can supervise each move.
        for side in ("left", "right"):
            if side not in arms:
                continue
            if args.approach_control_mode == "direct-joint":
                if not approaches[side]:
                    continue
                target_q, target_pose = approaches[side][-1]
                print(
                    f"{side} official direct joint move to entry; "
                    f"validated_path_steps={len(approaches[side])} "
                    f"duration={args.approach_duration:.2f}s"
                )
                command_approach_joint_target(
                    arms[side],
                    target_q,
                    target_pose,
                    args.approach_duration,
                    args,
                    label=f"{side} direct joint approach",
                )
                continue
            for ordinal, (target_q, target_pose) in enumerate(
                approaches[side], start=1
            ):
                print(
                    f"{side} approaching start {ordinal}/{len(approaches[side])} "
                    f"duration={args.approach_duration:.2f}s"
                )
                command_approach_joint_target(
                    arms[side],
                    target_q,
                    target_pose,
                    args.approach_duration,
                    args,
                    label=f"{side} approach step {ordinal}/{len(approaches[side])}",
                )

        last_grasp: dict[str, int] = {}
        if args.replay_grasp:
            initial = {
                side: waypoints[0].grasp_state
                for side, waypoints in waypoints_by_side.items()
            }
            description = ", ".join(
                f"{side}={'closed' if state else 'open'}" for side, state in initial.items()
            )
            input(
                f"First poses reached. Initial grippers will be {description}; "
                "press Enter to apply and start trajectory, Ctrl+C to abort > "
            )
            set_grasp_states(arms, initial, args, label="initial")
            last_grasp.update(initial)
        else:
            input(
                "First poses reached; gripper replay is disabled. Press Enter to start "
                "trajectory, Ctrl+C to abort > "
            )

        waypoint_count = len(next(iter(waypoints_by_side.values())))
        for ordinal in range(1, waypoint_count):
            targets = {
                side: waypoints[ordinal].xyzrpy
                for side, waypoints in waypoints_by_side.items()
            }
            frame = next(iter(waypoints_by_side.values()))[ordinal].frame_index
            print(
                f"moving synchronized trajectory {ordinal + 1}/{waypoint_count} "
                f"frame={frame} duration={args.segment_duration:.2f}s"
            )
            for side, arm in arms.items():
                result = arm.set_ee_pose_xyzrpy(
                    targets[side], duration=float(args.segment_duration)
                )
                if result is False:
                    raise RuntimeError(
                        f"SDK rejected {side} trajectory waypoint {ordinal + 1}"
                    )
            wait_for_all_targets(arms, targets, args.segment_duration, args)

            if args.replay_grasp:
                changed = {
                    side: waypoints[ordinal].grasp_state
                    for side, waypoints in waypoints_by_side.items()
                    if waypoints[ordinal].grasp_state != last_grasp[side]
                }
                if changed:
                    set_grasp_states(arms, changed, args, label=f"frame_{frame}")
                    last_grasp.update(changed)

        input(
            "Final synchronized waypoint reached; position control remains active. "
            "Press Enter to return both arms home > "
        )
        for side in ("left", "right"):
            if side not in arms:
                continue
            if getattr(arms[side], "fault", None):
                raise RuntimeError(f"{side} ARX arm fault before homing: {arms[side].fault}")
            result = arms[side].go_home(float(args.home_duration), wait=True)
            if result is False:
                raise RuntimeError(f"SDK rejected {side} go_home")
        normal_completion = True
        print("home=PASS for all selected arms; closing without calling protect_mode")
    finally:
        if not normal_completion:
            for side, arm in arms.items():
                print(f"abnormal exit: entering protect mode on {side} before close")
                try:
                    arm.protect_mode()
                except Exception as exc:  # noqa: BLE001
                    print(f"{side} protect_mode failed: {exc}")
        for arm in arms.values():
            try:
                arm.close()
            except Exception as exc:  # noqa: BLE001
                print(f"arm close failed: {exc}")


def execute(args: argparse.Namespace, sdk: Any, waypoints: list[Waypoint]) -> None:
    if not args.yes_i_understand_risk:
        raise RuntimeError("hardware execution also requires --yes-i-understand-risk")
    lock_handle = acquire_arm_control_lock()
    try:
        _execute_locked(args, sdk, waypoints)
    finally:
        lock_handle.close()


def _execute_locked(args: argparse.Namespace, sdk: Any, waypoints: list[Waypoint]) -> None:
    blockers = controller_processes()
    if blockers:
        raise RuntimeError("competing ARX control process detected:\n  " + "\n  ".join(blockers))
    can_port = args.can_port or SIDE_TO_CAN[args.side]
    arm = None
    normal_completion = False
    try:
        with sdk.Loading(f"connect {args.side} arm on {can_port}"):
            arm = sdk.SingleArm({"can_port": can_port, "type": args.arm_type})
        current_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(6)
        current_q = np.asarray(arm.get_joint_positions(), dtype=np.float64).reshape(-1)[:6]
        if not args.allow_long_approach:
            validate_segment_pose(
                current_pose,
                waypoints[0].xyzrpy,
                max_translation_m=args.max_approach_total_translation_m,
                max_rotation_deg=args.max_approach_total_rotation_deg,
            )
        try:
            solutions = solve_chain(
                waypoints, current_q, arm, args, current_pose=None
            )
            ik_seed_name = "measured_current_q"
        except (RuntimeError, ValueError) as current_seed_error:
            print(f"current-state IK seed failed ({current_seed_error}); retrying offline seed")
            solutions = solve_chain(
                waypoints,
                np.asarray(args.seed_q, dtype=np.float64),
                arm,
                args,
                current_pose=None,
            )
            ik_seed_name = "configured_seed_q"
        approach = build_joint_approach(
            current_q, solutions[0], current_pose, arm, args
        )
        print(f"side={args.side} can={can_port} current_pose={current_pose}")
        print(
            f"approach_mode=joint_position ik_seed={ik_seed_name} steps={len(approach)} "
            f"joint_step<={args.approach_joint_step_rad:.4f}rad "
            f"fk_translation_step<={args.approach_translation_step_m:.4f}m "
            f"fk_rotation_step<={args.approach_rotation_step_deg:.2f}deg"
        )
        print(f"current_q={np.array2string(current_q, precision=5)}")
        print(f"first_target_q={np.array2string(solutions[0], precision=5)}")
        print_plan(waypoints, solutions)
        input("Preflight passed. Keep the workspace clear; press Enter to start, Ctrl+C to abort > ")

        approach_commands = (
            [approach[-1]]
            if args.approach_control_mode == "direct-joint" and approach
            else approach
        )
        for ordinal, (target_q, target_pose) in enumerate(approach_commands, start=1):
            print(
                f"approaching start {ordinal}/{len(approach_commands)} "
                f"duration={args.approach_duration:.2f}s"
            )
            command_approach_joint_target(
                arm,
                target_q,
                target_pose,
                args.approach_duration,
                args,
                label=f"approach step {ordinal}/{len(approach_commands)}",
            )
            print(f"reached approach {ordinal}/{len(approach)}")

        trajectory_start = 1 if approach else 0
        for ordinal in range(trajectory_start, len(waypoints)):
            waypoint = waypoints[ordinal]
            print(
                f"moving trajectory {ordinal + 1}/{len(waypoints)} "
                f"frame={waypoint.frame_index} duration={args.segment_duration:.2f}s"
            )
            result = arm.set_ee_pose_xyzrpy(
                waypoint.xyzrpy, duration=float(args.segment_duration)
            )
            if result is False:
                raise RuntimeError(f"SDK rejected trajectory waypoint {ordinal + 1}")
            wait_for_target(arm, waypoint.xyzrpy, args.segment_duration, args)
            print(f"reached trajectory {ordinal + 1}/{len(waypoints)}")

        input(
            "Final waypoint reached and position control remains active. "
            "Press Enter to return home > "
        )
        if getattr(arm, "fault", None):
            raise RuntimeError(f"ARX arm fault before homing: {arm.fault}")
        result = arm.go_home(float(args.home_duration), wait=True)
        if result is False:
            raise RuntimeError("SDK rejected go_home")
        if getattr(arm, "fault", None):
            raise RuntimeError(f"ARX arm fault after homing: {arm.fault}")
        normal_completion = True
        print("home=PASS; closing without calling protect_mode")
    finally:
        if arm is not None:
            if not normal_completion:
                print("abnormal exit: entering protect mode before close")
                try:
                    arm.protect_mode()
                except Exception as exc:  # noqa: BLE001
                    print(f"protect_mode failed: {exc}")
            arm.close()


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def at_least_two(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eef", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), required=True)
    parser.add_argument(
        "--num-waypoints",
        type=at_least_two,
        default=96,
        help="uniform samples including both endpoints (default: 96)",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--start-at-first-grasp-event",
        action="store_true",
        help=(
            "approach each selected arm's first grasp-state-change frame, then "
            "continue on a shared later timeline"
        ),
    )
    parser.add_argument("--allow-uncorrected-eef", action="store_true")
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    parser.add_argument("--can-port", default=None)
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can3")
    parser.add_argument("--arm-type", type=int, default=2)
    parser.add_argument(
        "--tcp-offset-m",
        type=finite_number,
        nargs=3,
        default=list(DEFAULT_TCP_OFFSET_M),
        metavar=("X", "Y", "Z"),
        help=(
            "flange-to-TCP translation used by the input EEF artifact, in the "
            "moving flange frame (default: 0.105 0.0018 -0.0063)"
        ),
    )
    parser.add_argument("--seed-q", type=float, nargs=6, default=[0.0, 1.2, 1.5, 0.0, 0.0, 0.0])
    parser.add_argument("--segment-duration", type=positive, default=3.0)
    parser.add_argument("--approach-duration", type=positive, default=1.5)
    parser.add_argument(
        "--approach-control-mode",
        choices=("segmented", "direct-joint"),
        default="segmented",
        help=(
            "segmented sends every validated ramp point; direct-joint validates "
            "the ramp but sends one official SDK joint target (default: segmented)"
        ),
    )
    parser.add_argument(
        "--approach-retries",
        type=int,
        default=2,
        help="number of times to resend an unsettled approach step (default: 2)",
    )
    parser.add_argument("--approach-joint-step-rad", type=positive, default=0.20)
    parser.add_argument("--approach-translation-step-m", type=positive, default=0.02)
    parser.add_argument("--approach-rotation-step-deg", type=positive, default=5.0)
    parser.add_argument("--max-approach-total-joint-rad", type=positive, default=2.0)
    parser.add_argument("--max-approach-total-translation-m", type=positive, default=0.30)
    parser.add_argument("--max-approach-total-rotation-deg", type=positive, default=90.0)
    parser.add_argument(
        "--allow-long-approach",
        action="store_true",
        help=(
            "do not reject an approach solely for total translation, rotation, or "
            "joint span; hard joint limits and all per-step/FK checks remain enabled"
        ),
    )
    parser.add_argument("--max-approach-steps", type=at_least_two, default=256)
    parser.add_argument("--home-duration", type=positive, default=5.0)
    parser.add_argument("--max-first-translation-m", type=positive, default=0.10)
    parser.add_argument("--max-first-rotation-deg", type=positive, default=30.0)
    parser.add_argument("--max-segment-translation-m", type=positive, default=0.10)
    parser.add_argument("--max-segment-rotation-deg", type=positive, default=35.0)
    parser.add_argument("--max-first-joint-step-rad", type=positive, default=0.70)
    parser.add_argument("--max-segment-joint-step-rad", type=positive, default=0.50)
    parser.add_argument("--ik-position-tolerance-m", type=positive, default=0.002)
    parser.add_argument("--ik-rotation-tolerance-deg", type=positive, default=1.0)
    parser.add_argument("--feedback-position-tolerance-m", type=positive, default=0.006)
    parser.add_argument("--feedback-rotation-tolerance-deg", type=positive, default=3.0)
    parser.add_argument(
        "--approach-feedback-position-tolerance-m", type=positive, default=0.006
    )
    parser.add_argument(
        "--approach-feedback-rotation-tolerance-deg", type=positive, default=5.0
    )
    parser.add_argument("--feedback-joint-tolerance-rad", type=positive, default=0.04)
    parser.add_argument("--timeout-padding", type=positive, default=3.0)
    parser.add_argument("--poll-period", type=positive, default=0.05)
    parser.add_argument(
        "--replay-grasp",
        action="store_true",
        help="replay EEF grasp_state transitions; otherwise never command grippers",
    )
    parser.add_argument("--gripper-open", type=finite_number, default=-3.0)
    parser.add_argument("--gripper-closed", type=finite_number, default=0.0)
    parser.add_argument("--gripper-settle-s", type=positive, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes-i-understand-risk", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.approach_retries < 0 or args.approach_retries > 5:
        raise RuntimeError("--approach-retries must be in [0, 5]")
    if args.side == "both" and args.can_port is not None:
        raise RuntimeError("--can-port is only valid for a single side; use --left-can/--right-can")
    if not (-3.14 <= args.gripper_open <= 0.1):
        raise RuntimeError("--gripper-open must stay within the X5-2025 SDK range [-3.14, 0.1]")
    if not (-3.14 <= args.gripper_closed <= 0.1):
        raise RuntimeError("--gripper-closed must stay within the X5-2025 SDK range [-3.14, 0.1]")
    if args.gripper_open >= args.gripper_closed:
        raise RuntimeError("--gripper-open must be less than --gripper-closed")
    tcp_offset = np.asarray(args.tcp_offset_m, dtype=np.float64)
    if float(np.linalg.norm(tcp_offset)) > 0.30:
        raise RuntimeError("--tcp-offset-m magnitude exceeds the 0.30m safety bound")
    print(
        "target_semantics: EEF artifact TCP -> SDK moving flange; "
        f"T_flange_to_tcp_translation_m={np.array2string(tcp_offset, precision=6)}"
    )
    eef_path = args.eef.expanduser().resolve()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    available_by_side = {
        side: load_corrected_eef(
            eef_path, side, args.allow_uncorrected_eef, tcp_offset
        )
        for side in sides
    }
    selector = (
        select_from_first_grasp_events
        if args.start_at_first_grasp_event
        else select_synchronized_waypoints
    )
    waypoints_by_side = selector(
        available_by_side, args.num_waypoints, args.start_frame, args.end_frame
    )
    sdk = import_sdk(args.sdk_root)
    if args.execute:
        execute_multi(args, sdk, waypoints_by_side)
    else:
        offline_check_multi(args, sdk, waypoints_by_side)


if __name__ == "__main__":
    main()
