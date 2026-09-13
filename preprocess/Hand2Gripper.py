"""LifEgo hand-to-gripper conversions for independent EEF export.

Inputs use TATE's cached ``aria_humanego_21`` keypoint order.  This is already
the order used internally by LifEgo's HumanEgo/Qwen conversions, so unlike the
original LifEgo WiLoR entry point no keypoint remapping is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


THUMB_TIP = 0
INDEX_TIP = 1
MIDDLE_TIP = 2
WRIST = 5
THUMB_MCP = 6
INDEX_MCP = 8
MIDDLE_MCP = 11
RING_MCP = 14
PINKY_MCP = 17
PALM_CENTER = 20


def normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return None if norm < eps else vector / norm


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


@dataclass
class FingerCenterTarget:
    T_hand_in_cam: np.ndarray
    grasp_state: int
    grasp_ratio: float | None
    confidence: float | None
    is_right: bool
    mode: str = ""


class FingerCenter:
    """LifEgo's finger-center hand-frame definition.

    Position is the thumb/index fingertip midpoint. The jaw axis (column 0) is
    kept primary from thumb MCP to index MCP. The forward seed points from the
    wrist to the centroid of the index/middle/ring/pinky MCPs and is projected
    off the jaw axis to form forward (column 1).
    """

    MODE_NAME = "finger_center"
    DEFAULT_GRASP_CLOSE_RATIO = 0.45
    DEFAULT_GRASP_OPEN_RATIO = 0.55
    DEFAULT_GRASP_MIN_FRAMES = 1

    def __init__(
        self,
        grasp_close_ratio: float | None = None,
        grasp_open_ratio: float | None = None,
        grasp_min_frames: int | None = None,
    ) -> None:
        self.grasp_close_ratio = float(
            self.DEFAULT_GRASP_CLOSE_RATIO
            if grasp_close_ratio is None
            else grasp_close_ratio
        )
        self.grasp_open_ratio = float(
            self.DEFAULT_GRASP_OPEN_RATIO
            if grasp_open_ratio is None
            else grasp_open_ratio
        )
        if self.grasp_open_ratio < self.grasp_close_ratio:
            raise ValueError(
                "grasp_open_ratio must be greater than or equal to grasp_close_ratio"
            )
        self.grasp_min_frames = int(
            self.DEFAULT_GRASP_MIN_FRAMES
            if grasp_min_frames is None
            else grasp_min_frames
        )
        if self.grasp_min_frames < 1:
            raise ValueError("grasp_min_frames must be >= 1")
        self._previous_rotation: np.ndarray | None = None
        self._grasp_state: int | None = None

    def _wrist_rotation(self, keypoints: np.ndarray) -> np.ndarray | None:
        forward = normalize(keypoints[PALM_CENTER] - keypoints[WRIST])
        if forward is None:
            return None
        lateral = keypoints[INDEX_MCP] - keypoints[MIDDLE_MCP]
        jaw = normalize(np.cross(forward, lateral))
        if jaw is None:
            return None
        up = normalize(np.cross(jaw, forward))
        if up is None:
            return None
        forward = normalize(np.cross(up, jaw))
        return None if forward is None else np.column_stack([jaw, forward, up])

    def _position(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        return 0.5 * (keypoints[THUMB_TIP] + keypoints[INDEX_TIP])

    def _jaw_axis_raw(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        return keypoints[INDEX_MCP] - keypoints[THUMB_MCP]

    def _forward_seed(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        return (
            (
                keypoints[INDEX_MCP]
                + keypoints[MIDDLE_MCP]
                + keypoints[RING_MCP]
                + keypoints[PINKY_MCP]
            )
            / 4.0
            - keypoints[WRIST]
        )

    def _hand_pose(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray | None:
        position = self._position(keypoints, is_right=is_right)
        jaw = normalize(self._jaw_axis_raw(keypoints, is_right=is_right))
        forward_seed = self._forward_seed(keypoints, is_right=is_right)

        rotation = None
        if jaw is not None and float(np.linalg.norm(forward_seed)) >= 1e-5:
            forward = normalize(
                forward_seed - float(np.dot(forward_seed, jaw)) * jaw
            )
            if forward is not None:
                up = normalize(np.cross(jaw, forward))
                if up is not None:
                    forward = normalize(np.cross(up, jaw))
                    if forward is not None:
                        if self._previous_rotation is not None and float(
                            np.dot(self._previous_rotation[:, 0], jaw)
                        ) < 0.0:
                            jaw = -jaw
                            forward = -forward
                            up = np.cross(jaw, forward)
                        rotation = np.column_stack([jaw, forward, up])

        if rotation is None:
            rotation = self._previous_rotation
        if rotation is None:
            rotation = self._wrist_rotation(keypoints)
        if rotation is None:
            return None
        self._previous_rotation = rotation
        return make_pose(rotation, position)

    def grasp_ratio(self, keypoints: np.ndarray) -> float | None:
        tip_distance = float(np.linalg.norm(keypoints[THUMB_TIP] - keypoints[INDEX_TIP]))
        palm_size = float(np.linalg.norm(keypoints[MIDDLE_MCP] - keypoints[WRIST]))
        return tip_distance / palm_size if palm_size > 0.01 else None

    def _classify_grasp(self, keypoints: np.ndarray) -> tuple[int, float | None]:
        ratio = self.grasp_ratio(keypoints)
        if ratio is None:
            state = int(
                np.linalg.norm(keypoints[THUMB_TIP] - keypoints[INDEX_TIP]) < 0.105
            )
            return state, None

        if self._grasp_state is None:
            self._grasp_state = 1 if ratio < self.grasp_close_ratio else 0
        elif self._grasp_state == 0 and ratio < self.grasp_close_ratio:
            self._grasp_state = 1
        elif self._grasp_state == 1 and ratio > self.grasp_open_ratio:
            self._grasp_state = 0
        return self._grasp_state, ratio

    def from_hand_record(self, hand: dict[str, Any], is_right: bool) -> FingerCenterTarget | None:
        keypoints = hand.get("keypoints_3d_cam")
        if keypoints is None:
            return None
        keypoints = np.asarray(keypoints, dtype=np.float64)
        if keypoints.shape != (21, 3) or not np.all(np.isfinite(keypoints)):
            return None
        pose = self._hand_pose(keypoints, is_right=is_right)
        if pose is None:
            return None
        grasp_state, grasp_ratio = self._classify_grasp(keypoints)
        return FingerCenterTarget(
            T_hand_in_cam=pose,
            grasp_state=grasp_state,
            grasp_ratio=grasp_ratio,
            confidence=hand.get("confidence"),
            is_right=bool(is_right),
            mode=self.MODE_NAME,
        )


class HumanEgo(FingerCenter):
    """Original HumanEgo frame from LifEgo.

    It shares the fingertip-midpoint position and MCP jaw axis with
    ``finger_center``, but uses the HumanEgo forward seed: wrist to the
    thumb/index MCP midpoint.
    """

    MODE_NAME = "humanego"

    def _forward_seed(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        return 0.5 * (keypoints[THUMB_MCP] + keypoints[INDEX_MCP]) - keypoints[WRIST]


class Qwen(HumanEgo):
    """Qwen-RobotManip hand-to-gripper frame, adapted from LifEgo.

    The virtual fingertip is ``0.7 * index_tip + 0.3 * middle_tip``.  The
    handedness-dependent jaw sign matches LifEgo's calibrated convention.
    """

    MODE_NAME = "qwen"
    KVF_INDEX_WEIGHT = 0.7
    KVF_MIDDLE_WEIGHT = 0.3

    def _virtual_fingertip(self, keypoints: np.ndarray) -> np.ndarray:
        return (
            self.KVF_INDEX_WEIGHT * keypoints[INDEX_TIP]
            + self.KVF_MIDDLE_WEIGHT * keypoints[MIDDLE_TIP]
        )

    def _position(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        return 0.5 * (keypoints[THUMB_TIP] + self._virtual_fingertip(keypoints))

    def _jaw_axis_raw(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        sign = 1.0 if is_right else -1.0
        return sign * (self._virtual_fingertip(keypoints) - keypoints[THUMB_TIP])

    def _forward_seed(self, keypoints: np.ndarray, *, is_right: bool) -> np.ndarray:
        return self._virtual_fingertip(keypoints) - keypoints[WRIST]

    def grasp_ratio(self, keypoints: np.ndarray) -> float | None:
        tip_distance = float(
            np.linalg.norm(keypoints[THUMB_TIP] - self._virtual_fingertip(keypoints))
        )
        palm_size = float(np.linalg.norm(keypoints[MIDDLE_MCP] - keypoints[WRIST]))
        return tip_distance / palm_size if palm_size > 0.01 else None


MODES = {
    FingerCenter.MODE_NAME: FingerCenter,
    HumanEgo.MODE_NAME: HumanEgo,
    Qwen.MODE_NAME: Qwen,
}


def make_hand2gripper(
    mode: str = FingerCenter.MODE_NAME,
    *,
    grasp_close_ratio: float | None = None,
    grasp_open_ratio: float | None = None,
    grasp_min_frames: int | None = None,
) -> FingerCenter:
    """Create a conversion mode by its stable configuration name."""
    try:
        converter_cls = MODES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown hand2gripper mode {mode!r}; choose from {sorted(MODES)}"
        ) from None
    return converter_cls(grasp_close_ratio, grasp_open_ratio, grasp_min_frames)


def debounce_grasp(
    states: list[int | None], min_frames: int
) -> tuple[list[int | None], int]:
    """Absorb non-boundary state runs shorter than ``min_frames``."""
    if min_frames <= 1:
        return states, 0
    valid_indices = [index for index, state in enumerate(states) if state is not None]
    if len(valid_indices) < 3:
        return states, 0

    sequence = [states[index] for index in valid_indices]
    changed = 0
    while True:
        bounds = [0]
        for index in range(1, len(sequence)):
            if sequence[index] != sequence[index - 1]:
                bounds.append(index)
        bounds.append(len(sequence))
        for run in range(1, len(bounds) - 2):
            start, end = bounds[run], bounds[run + 1]
            if end - start < min_frames:
                for index in range(start, end):
                    sequence[index] = sequence[start - 1]
                changed += end - start
                break
        else:
            break

    output = list(states)
    for sequence_index, frame_index in enumerate(valid_indices):
        output[frame_index] = sequence[sequence_index]
    return output, changed
