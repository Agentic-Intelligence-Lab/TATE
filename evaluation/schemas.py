"""Shared schema, convention, fingerprint, and trajectory validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .types import DualArmTrajectory, SIDES


class SchemaError(ValueError):
    pass


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"expected a JSON object in {path}")
    return value


def require_schema(data: dict[str, Any], name: str, version: int, path: Path) -> None:
    if data.get("schema") != name or data.get("schema_version") != version:
        raise SchemaError(
            f"{path}: expected {name!r} schema_version={version}, got "
            f"{data.get('schema')!r} schema_version={data.get('schema_version')!r}"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path, relative_to: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(relative_to) / path
    return path.resolve()


def matrix_to_pose_xyzw(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SchemaError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise SchemaError(f"{label} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise SchemaError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise SchemaError(f"{label} rotation determinant is not +1")
    return np.concatenate([matrix[:3, 3], Rotation.from_matrix(rotation).as_quat()])


def validate_trajectory(trajectory: DualArmTrajectory) -> DualArmTrajectory:
    timestamps = np.asarray(trajectory.timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise SchemaError("trajectory timestamps must be a non-empty vector")
    if not np.all(np.isfinite(timestamps)) or (
        len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0)
    ):
        raise SchemaError("trajectory timestamps must be finite and strictly increasing")
    if not trajectory.sides:
        raise SchemaError("trajectory has no arm entries")
    for side, arm in trajectory.sides.items():
        if side not in SIDES:
            raise SchemaError(f"unsupported arm side {side!r}")
        if not arm.frame_name:
            raise SchemaError(f"{side} trajectory has no frame name")
        if np.asarray(arm.pose_xyzw).shape != (len(timestamps), 7):
            raise SchemaError(f"{side} pose must have shape {(len(timestamps), 7)}")
        if np.asarray(arm.valid).shape != (len(timestamps),):
            raise SchemaError(f"{side} valid mask has the wrong length")
        if np.asarray(arm.gripper_binary).shape != (len(timestamps),):
            raise SchemaError(f"{side} binary gripper has the wrong length")
        if np.any(~np.isin(arm.gripper_binary, [0, 1])):
            raise SchemaError(f"{side} binary gripper must use 0=open, 1=closed")
        valid = np.asarray(arm.valid, dtype=bool)
        if valid.any():
            poses = np.asarray(arm.pose_xyzw, dtype=np.float64)[valid]
            if not np.all(np.isfinite(poses)):
                raise SchemaError(f"{side} contains non-finite values on valid frames")
            norms = np.linalg.norm(poses[:, 3:], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise SchemaError(f"{side} contains non-normalized xyzw quaternions")
        for name in ("gripper_continuous", "grasp_ratio"):
            value = getattr(arm, name)
            if value is not None and np.asarray(value).shape != (len(timestamps),):
                raise SchemaError(f"{side} {name} has the wrong length")
    trajectory.timestamps_s = timestamps
    return trajectory


def require_matching_frame(ego: DualArmTrajectory, real: DualArmTrajectory, side: str) -> None:
    ego_frame = ego.sides[side].frame_name
    real_frame = real.sides[side].frame_name
    if ego_frame != real_frame:
        raise SchemaError(
            f"coordinate-frame mismatch for {side}: ego={ego_frame!r}, real={real_frame!r}"
        )


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
