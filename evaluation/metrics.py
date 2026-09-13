"""LifEgo trajectory distances used by the TATE evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .align import AlignmentResult, position_dtw
from .schemas import require_matching_frame
from .types import DualArmTrajectory


def _rotation_errors_deg(ego_quat: np.ndarray, real_quat: np.ndarray) -> np.ndarray:
    delta = Rotation.from_quat(ego_quat).inv() * Rotation.from_quat(real_quat)
    return np.degrees(delta.magnitude())


def evaluate_pair_side(
    ego: DualArmTrajectory,
    real: DualArmTrajectory,
    side: str,
    alignment: AlignmentResult,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return pair distances and centroids needed by the LifEgo metrics.

    Absolute position DTW determines the correspondence used for rotation.
    Shape is a second DTW after subtracting each trajectory's own centroid.
    """
    require_matching_frame(ego, real, side)
    ego_valid = alignment.ego.pose_xyzw[alignment.ego.valid]
    real_valid = alignment.real.pose_xyzw[alignment.real.valid]
    ego_centroid = ego_valid[:, :3].mean(axis=0)
    real_centroid = real_valid[:, :3].mean(axis=0)

    ego_pose = alignment.ego.pose_xyzw[alignment.ego_indices]
    real_pose = alignment.real.pose_xyzw[alignment.real_indices]
    rotation_deg = float(
        np.mean(_rotation_errors_deg(ego_pose[:, 3:], real_pose[:, 3:]))
    )
    _, _, shape_m = position_dtw(
        ego_valid[:, :3] - ego_centroid,
        real_valid[:, :3] - real_centroid,
        config.get("dtw_window_ratio"),
    )
    return {
        "D_pos_m": float(alignment.normalized_cost),
        "D_rot_deg": rotation_deg,
        "D_shape_m": shape_m,
        "ego_centroid_m": ego_centroid.tolist(),
        "real_centroid_m": real_centroid.tolist(),
    }
