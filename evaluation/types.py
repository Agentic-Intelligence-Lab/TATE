"""Schema-independent trajectory containers used by the evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SIDES = ("left", "right")


@dataclass
class ArmTrajectory:
    pose_xyzw: np.ndarray
    valid: np.ndarray
    gripper_binary: np.ndarray
    frame_name: str
    gripper_continuous: np.ndarray | None = None
    grasp_ratio: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.pose_xyzw)

    @property
    def position(self) -> np.ndarray:
        return self.pose_xyzw[:, :3]

    @property
    def quaternion_xyzw(self) -> np.ndarray:
        return self.pose_xyzw[:, 3:]


@dataclass
class DualArmTrajectory:
    episode_id: int | str
    timestamps_s: np.ndarray
    sides: dict[str, ArmTrajectory]
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.timestamps_s)


@dataclass(frozen=True)
class GripperEvent:
    ordinal: int
    frame_index: int
    timestamp_s: float
    from_state: int
    to_state: int
    transition: str
    ambiguous: bool = False
    invalid_gap_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "transition": self.transition,
            "ambiguous": self.ambiguous,
            "invalid_gap_s": self.invalid_gap_s,
        }
