"""Convert TATE's right-arm TCP actions into guarded ARX flange/joint targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.constants import TCP_OFFSET_M


CANONICAL_LEFT = np.asarray([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
JOINT_LOWER = np.deg2rad([-150, 0, 0, -90, -90, -120])
JOINT_UPPER = np.deg2rad([180, 210, 180, 90, 90, 120])
GRIPPER_OPEN_RAW = -3.4
GRIPPER_CLOSED_RAW = 0.1
GRIPPER_BINARY_THRESHOLD_RAW = -2.6


@dataclass(frozen=True)
class GuardLimits:
    ik_position_tolerance_m: float = 0.01
    ik_rotation_tolerance_deg: float = 5.0
    x_range: tuple[float, float] = (-0.02, 0.45)
    y_range: tuple[float, float] = (-0.20, 0.40)
    z_range: tuple[float, float] = (-0.25, 0.18)


def rotation_error_deg(a: Rotation, b: Rotation) -> float:
    return float(np.rad2deg((a.inv() * b).magnitude()))


def flange_to_tcp_state(
    flange_xyzrpy: np.ndarray,
    gripper_raw: float,
    *,
    tcp_offset_m: np.ndarray | tuple[float, float, float] = TCP_OFFSET_M,
) -> np.ndarray:
    flange = np.asarray(flange_xyzrpy, dtype=np.float64)
    if flange.shape != (6,) or not np.isfinite(flange).all() or not np.isfinite(gripper_raw):
        raise ValueError("invalid flange pose or gripper feedback")
    orientation = Rotation.from_euler("xyz", flange[3:])
    offset = np.asarray(tcp_offset_m, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("TCP offset must be three finite values")
    tcp_xyz = flange[:3] + orientation.apply(offset)
    # Match the binary labels used by the cotrain exporter.  Feedback need not
    # reach either endpoint while an object is held.
    gripper = float(gripper_raw >= GRIPPER_BINARY_THRESHOLD_RAW)
    return np.asarray([*tcp_xyz, *orientation.as_quat(), gripper], dtype=np.float32)


def model_state(right_tcp: np.ndarray) -> np.ndarray:
    right = np.asarray(right_tcp, dtype=np.float32)
    if right.shape != (8,) or not np.isfinite(right).all():
        raise ValueError("right TCP state must have eight finite values")
    return np.concatenate((CANONICAL_LEFT, right))


def tcp_action_to_flange(
    action: np.ndarray,
    current_tcp: np.ndarray,
    limits: GuardLimits,
    *,
    tcp_offset_m: np.ndarray | tuple[float, float, float] = TCP_OFFSET_M,
    gripper_threshold: float = 0.5,
) -> tuple[np.ndarray, float]:
    target = np.asarray(action, dtype=np.float64)
    current = np.asarray(current_tcp, dtype=np.float64)
    if target.shape != (8,) or current.shape != (8,) or not np.isfinite(target).all() or not np.isfinite(current).all():
        raise ValueError("target and current TCP must have eight finite values")
    if not 0.0 <= target[7] <= 1.0:
        raise ValueError("model gripper action must be in [0, 1]")
    if not np.isfinite(gripper_threshold) or not 0.0 <= gripper_threshold <= 1.0:
        raise ValueError("gripper threshold must be in [0, 1]")
    for value, (lower, upper), name in zip(target[:3], (limits.x_range, limits.y_range, limits.z_range), "xyz"):
        if not lower <= value <= upper:
            raise ValueError(f"target TCP {name} outside workspace: {value:.4f}")
    quat = target[3:7]
    norm = float(np.linalg.norm(quat))
    # A regressed quaternion is only approximately unit length. Normalize it
    # before converting to a rotation, while rejecting badly formed outputs.
    if not 0.9 <= norm <= 1.1:
        raise ValueError(f"target quaternion norm invalid: {norm:.4f}")
    target_rotation = Rotation.from_quat(quat / norm)
    current_quat_norm = float(np.linalg.norm(current[3:7]))
    if not 0.99 <= current_quat_norm <= 1.01:
        raise ValueError("current TCP quaternion is invalid")
    offset = np.asarray(tcp_offset_m, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("TCP offset must be three finite values")
    flange_xyz = target[:3] - target_rotation.apply(offset)
    flange_xyzrpy = np.asarray([*flange_xyz, *target_rotation.as_euler("xyz")], dtype=np.float64)
    # The policy is trained on binary grasp labels.  Command an endpoint so
    # the gripper supplies holding force even when an object prevents closure.
    gripper_raw = GRIPPER_CLOSED_RAW if target[7] > gripper_threshold else GRIPPER_OPEN_RAW
    return flange_xyzrpy, gripper_raw


def validate_ik_solution(
    joints: np.ndarray,
    current_joints: np.ndarray,
    target_flange: np.ndarray,
    fk_flange: np.ndarray,
    limits: GuardLimits,
) -> None:
    q = np.asarray(joints, dtype=np.float64)
    current = np.asarray(current_joints, dtype=np.float64)
    target = np.asarray(target_flange, dtype=np.float64)
    actual = np.asarray(fk_flange, dtype=np.float64)
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in (q, current, target, actual)):
        raise ValueError("IK/FK produced an invalid six-element vector")
    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        raise ValueError("IK target outside joint limits")
    if float(np.linalg.norm(actual[:3] - target[:3])) > limits.ik_position_tolerance_m:
        raise ValueError("IK/FK position residual exceeds tolerance")
    if rotation_error_deg(Rotation.from_euler("xyz", actual[3:]), Rotation.from_euler("xyz", target[3:])) > limits.ik_rotation_tolerance_deg:
        raise ValueError("IK/FK rotation residual exceeds tolerance")
