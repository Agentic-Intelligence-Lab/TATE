"""Loader for canonical ``tate.dual_arm_eef`` v2 sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import SchemaError, matrix_to_pose_xyzw, read_json, require_schema, sha256_file, validate_trajectory
from ..types import ArmTrajectory, DualArmTrajectory, SIDES


def _timestamps(data: dict[str, Any], frames: list[dict[str, Any]], path: Path) -> np.ndarray:
    fps = float(data.get("fps", 0.0))
    if fps <= 0.0:
        raise SchemaError(f"{path}: fps must be positive")
    if all("timestamp_s" in frame for frame in frames):
        return np.asarray([frame["timestamp_s"] for frame in frames], dtype=np.float64)
    if all("ts" in frame for frame in frames):
        raw = np.asarray([frame["ts"] for frame in frames], dtype=np.float64)
        # Current TATE sidecars use integer nanoseconds. Small decimal values are seconds.
        scale = 1e-9 if len(raw) > 1 and np.nanmedian(np.diff(raw)) > 1e3 else 1.0
        return (raw - raw[0]) * scale
    return np.arange(len(frames), dtype=np.float64) / fps


def load_eef_json(path: str | Path, episode_id: int | str | None = None) -> DualArmTrajectory:
    path = Path(path).resolve()
    data = read_json(path)
    require_schema(data, "tate.dual_arm_eef", 2, path)
    convention = data.get("eef_coordinate_convention") or {}
    if convention.get("pose_semantics") != "arx_tcp":
        raise SchemaError(f"{path}: EEF pose_semantics must be 'arx_tcp'")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SchemaError(f"{path}: frames must be a non-empty list")
    if data.get("total_frames") not in (None, len(frames)):
        raise SchemaError(f"{path}: total_frames does not match frames")
    timestamps = _timestamps(data, frames, path)
    configured_frames = convention.get("per_side_frames") or {}
    sides: dict[str, ArmTrajectory] = {}
    for side in SIDES:
        key = "hand_l" if side == "left" else "hand_r"
        poses = np.full((len(frames), 7), np.nan, dtype=np.float64)
        valid = np.zeros(len(frames), dtype=bool)
        binary = np.zeros(len(frames), dtype=np.int8)
        ratio = np.full(len(frames), np.nan, dtype=np.float64)
        frame_names: set[str] = set()
        for index, frame in enumerate(frames):
            hand = frame.get(key)
            if hand is None:
                continue
            if hand.get("pose_semantics", "arx_tcp") != "arx_tcp":
                raise SchemaError(f"{path}: {key} frame {index} is not an ARX TCP pose")
            pose = hand.get("tcp_pose_eef_frame")
            if pose is None:
                pose = hand.get("eef_pose_world")
            poses[index] = matrix_to_pose_xyzw(pose, f"{path}:{key}[{index}]")
            state = int(hand.get("grasp_state"))
            if state not in (0, 1):
                raise SchemaError(f"{path}: {key} frame {index} grasp_state must be 0 or 1")
            binary[index] = state
            if hand.get("grasp_ratio") is not None:
                ratio[index] = float(hand["grasp_ratio"])
            valid[index] = bool(hand.get("valid", True))
            frame_names.add(str(hand.get("eef_frame") or configured_frames.get(side) or ""))
        frame_names.discard("")
        if len(frame_names) > 1:
            raise SchemaError(f"{path}: inconsistent {side} EEF frames: {sorted(frame_names)}")
        frame_name = next(iter(frame_names), str(configured_frames.get(side) or ""))
        if not frame_name:
            raise SchemaError(f"{path}: no frame name for {side}")
        sides[side] = ArmTrajectory(
            pose_xyzw=poses,
            valid=valid,
            gripper_binary=binary,
            grasp_ratio=ratio,
            frame_name=frame_name,
        )
    metadata = data.get("metadata") or {}
    correction = metadata.get("real_anchor_correction") or data.get("correction")
    return validate_trajectory(
        DualArmTrajectory(
            episode_id=episode_id if episode_id is not None else data.get("episode_id", path.stem),
            timestamps_s=timestamps,
            sides=sides,
            source={
                "kind": "ego_eef_json",
                "path": str(path),
                "fingerprint": sha256_file(path),
                "correction": correction,
            },
        )
    )
