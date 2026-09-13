"""Timestamp-based pose and gripper interpolation with invalid-gap protection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .types import ArmTrajectory


@dataclass
class ResampledArm:
    timestamps_s: np.ndarray
    pose_xyzw: np.ndarray
    valid: np.ndarray
    gripper_binary: np.ndarray
    gripper_continuous: np.ndarray | None


def target_timestamps(
    source_timestamps_s: np.ndarray,
    *,
    sample_count: int | None,
    sample_rate_hz: float | None,
) -> np.ndarray:
    start = float(source_timestamps_s[0])
    end = float(source_timestamps_s[-1])
    if end <= start:
        raise ValueError("cannot resample a zero-duration trajectory")
    if sample_rate_hz is not None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        count = max(2, int(round((end - start) * sample_rate_hz)) + 1)
    else:
        count = int(sample_count or 200)
        if count < 2:
            raise ValueError("sample_count must be at least 2")
    return np.linspace(start, end, count, dtype=np.float64)


def resample_arm(
    timestamps_s: np.ndarray,
    arm: ArmTrajectory,
    target_s: np.ndarray,
    *,
    max_invalid_gap_s: float,
) -> ResampledArm:
    source_t = np.asarray(timestamps_s, dtype=np.float64)
    target_s = np.asarray(target_s, dtype=np.float64)
    poses = np.full((len(target_s), 7), np.nan, dtype=np.float64)
    valid = np.zeros(len(target_s), dtype=bool)
    binary = np.zeros(len(target_s), dtype=np.int8)
    continuous = None
    if arm.gripper_continuous is not None:
        continuous = np.full(len(target_s), np.nan, dtype=np.float64)

    source_valid = np.asarray(arm.valid, dtype=bool)
    for out_index, timestamp in enumerate(target_s):
        right = int(np.searchsorted(source_t, timestamp, side="left"))
        if right < len(source_t) and np.isclose(source_t[right], timestamp, atol=1e-10):
            left = right
        else:
            left = right - 1
        if left < 0 or right >= len(source_t):
            continue
        if left == right:
            if not source_valid[left]:
                continue
            poses[out_index] = arm.pose_xyzw[left]
            binary[out_index] = int(arm.gripper_binary[left])
            if continuous is not None:
                continuous[out_index] = float(arm.gripper_continuous[left])
            valid[out_index] = True
            continue
        gap = float(source_t[right] - source_t[left])
        if gap <= 0.0 or gap > max_invalid_gap_s or not (source_valid[left] and source_valid[right]):
            continue
        alpha = float((timestamp - source_t[left]) / gap)
        poses[out_index, :3] = (1.0 - alpha) * arm.pose_xyzw[left, :3] + alpha * arm.pose_xyzw[right, :3]
        rotations = Rotation.from_quat(arm.pose_xyzw[[left, right], 3:])
        poses[out_index, 3:] = Slerp([source_t[left], source_t[right]], rotations)([timestamp]).as_quat()[0]
        nearest = left if alpha < 0.5 else right
        binary[out_index] = int(arm.gripper_binary[nearest])
        if continuous is not None:
            continuous[out_index] = (
                (1.0 - alpha) * arm.gripper_continuous[left]
                + alpha * arm.gripper_continuous[right]
            )
        valid[out_index] = True
    return ResampledArm(target_s, poses, valid, binary, continuous)
