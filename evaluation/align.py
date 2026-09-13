"""Event-segment selection and position-based trajectory correspondence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .events import extract_events
from .resample import ResampledArm, resample_arm, target_timestamps
from .schemas import require_matching_frame
from .types import DualArmTrajectory


class AlignmentError(ValueError):
    pass


@dataclass
class AlignmentResult:
    side: str
    method: str
    cost_definition: str
    normalized_cost: float
    ego_indices: np.ndarray
    real_indices: np.ndarray
    ego: ResampledArm
    real: ResampledArm
    segment: dict[str, int | str | None]


def _segment_slice(
    trajectory: DualArmTrajectory,
    side: str,
    start_event: int | str | None,
    end_event: int | str | None,
    max_invalid_gap_s: float,
) -> slice:
    events = extract_events(trajectory, side, max_invalid_gap_s=max_invalid_gap_s)
    start = 0 if start_event is None else _event_frame(events, start_event, "start")
    end = trajectory.n - 1 if end_event is None else _event_frame(events, end_event, "end")
    if end <= start:
        raise AlignmentError(f"invalid {side} segment boundaries: {start}..{end}")
    return slice(start, end + 1)


def _event_frame(events: list[Any], event_index: int | str, label: str) -> int:
    if event_index == "first":
        event_index = 0
    elif event_index == "last":
        event_index = len(events) - 1
    elif isinstance(event_index, str):
        raise AlignmentError(
            f"{label} event must be an integer, 'first', or 'last'; got {event_index!r}"
        )
    if event_index < 0 or event_index >= len(events):
        raise AlignmentError(f"{label} event {event_index} is unavailable; detected {len(events)} events")
    if events[event_index].ambiguous:
        raise AlignmentError(f"{label} event {event_index} crosses an excessive invalid gap")
    return int(events[event_index].frame_index)


def _slice_trajectory(trajectory: DualArmTrajectory, side: str, selected: slice) -> tuple[np.ndarray, Any]:
    timestamps = trajectory.timestamps_s[selected]
    arm = trajectory.sides[side]
    sliced = type(arm)(
        pose_xyzw=arm.pose_xyzw[selected],
        valid=arm.valid[selected],
        gripper_binary=arm.gripper_binary[selected],
        gripper_continuous=None if arm.gripper_continuous is None else arm.gripper_continuous[selected],
        grasp_ratio=None if arm.grasp_ratio is None else arm.grasp_ratio[selected],
        frame_name=arm.frame_name,
    )
    return timestamps, sliced


def _dtw_path(cost: np.ndarray, window: int | None) -> tuple[np.ndarray, np.ndarray, float]:
    n, m = cost.shape
    accumulated = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    for i in range(1, n + 1):
        if window is None:
            start, stop = 1, m + 1
        else:
            center = int(round(i * m / max(n, 1)))
            start, stop = max(1, center - window), min(m + 1, center + window + 1)
        for j in range(start, stop):
            accumulated[i, j] = cost[i - 1, j - 1] + min(
                accumulated[i - 1, j], accumulated[i, j - 1], accumulated[i - 1, j - 1]
            )
    if not np.isfinite(accumulated[n, m]):
        raise AlignmentError("DTW constraint leaves no valid endpoint path")
    path_i: list[int] = []
    path_j: list[int] = []
    i, j = n, m
    while i > 0 and j > 0:
        path_i.append(i - 1)
        path_j.append(j - 1)
        candidates = (accumulated[i - 1, j - 1], accumulated[i - 1, j], accumulated[i, j - 1])
        move = int(np.argmin(candidates))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return (
        np.asarray(path_i[::-1], dtype=np.int64),
        np.asarray(path_j[::-1], dtype=np.int64),
        float(accumulated[n, m]),
    )


def position_dtw(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    window_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a position-DTW path and LifEgo-normalized distance."""
    positions_a = np.asarray(positions_a, dtype=np.float64)
    positions_b = np.asarray(positions_b, dtype=np.float64)
    if len(positions_a) < 2 or len(positions_b) < 2:
        raise AlignmentError("position DTW needs at least two samples per trajectory")
    cost = np.linalg.norm(positions_a[:, None, :] - positions_b[None, :, :], axis=2)
    window = None
    if window_ratio is not None:
        window = max(
            abs(len(positions_a) - len(positions_b)),
            int(np.ceil(float(window_ratio) * max(len(positions_a), len(positions_b)))),
        )
    indices_a, indices_b, total_cost = _dtw_path(cost, window)
    normalized = total_cost / ((len(positions_a) + len(positions_b)) / 2.0)
    return indices_a, indices_b, float(normalized)


def align_pair(
    ego: DualArmTrajectory,
    real: DualArmTrajectory,
    side: str,
    config: dict[str, Any],
) -> AlignmentResult:
    require_matching_frame(ego, real, side)
    max_gap = float(config.get("max_invalid_gap_s", 0.25))
    segment = config.get("segment") or {}
    start_event = segment.get("start_event")
    end_event = segment.get("end_event")
    ego_slice = _segment_slice(ego, side, start_event, end_event, max_gap)
    real_slice = _segment_slice(real, side, start_event, end_event, max_gap)
    ego_t, ego_arm = _slice_trajectory(ego, side, ego_slice)
    real_t, real_arm = _slice_trajectory(real, side, real_slice)
    sample_count = config.get("sample_count", 200)
    sample_rate = config.get("sample_rate_hz")
    ego_target = target_timestamps(ego_t, sample_count=sample_count, sample_rate_hz=sample_rate)
    real_target = target_timestamps(real_t, sample_count=sample_count, sample_rate_hz=sample_rate)
    ego_resampled = resample_arm(ego_t, ego_arm, ego_target, max_invalid_gap_s=max_gap)
    real_resampled = resample_arm(real_t, real_arm, real_target, max_invalid_gap_s=max_gap)
    ego_keep = np.flatnonzero(ego_resampled.valid)
    real_keep = np.flatnonzero(real_resampled.valid)
    if len(ego_keep) < 2 or len(real_keep) < 2:
        raise AlignmentError(f"too few valid {side} samples after resampling")
    method = str(config.get("method", "position_dtw"))
    if method == "normalized_time":
        count = min(len(ego_keep), len(real_keep))
        ego_indices = ego_keep[np.round(np.linspace(0, len(ego_keep) - 1, count)).astype(int)]
        real_indices = real_keep[np.round(np.linspace(0, len(real_keep) - 1, count)).astype(int)]
        costs = np.linalg.norm(
            ego_resampled.pose_xyzw[ego_indices, :3] - real_resampled.pose_xyzw[real_indices, :3], axis=1
        )
        total_cost = float(costs.sum())
    elif method == "position_dtw":
        positions_ego = ego_resampled.pose_xyzw[ego_keep, :3]
        positions_real = real_resampled.pose_xyzw[real_keep, :3]
        ratio = config.get("dtw_window_ratio")
        local_i, local_j, normalized_cost = position_dtw(positions_ego, positions_real, ratio)
        ego_indices, real_indices = ego_keep[local_i], real_keep[local_j]
    else:
        raise AlignmentError(f"unsupported alignment method {method!r}")
    return AlignmentResult(
        side=side,
        method=method,
        cost_definition="euclidean_translation_m",
        normalized_cost=(
            float(np.mean(costs)) if method == "normalized_time" else normalized_cost
        ),
        ego_indices=ego_indices,
        real_indices=real_indices,
        ego=ego_resampled,
        real=real_resampled,
        segment={"start_event": start_event, "end_event": end_event},
    )
