"""Per-side gripper transition extraction and health checks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .types import DualArmTrajectory, GripperEvent


def extract_events(
    trajectory: DualArmTrajectory,
    side: str,
    *,
    max_invalid_gap_s: float = 0.25,
) -> list[GripperEvent]:
    arm = trajectory.sides[side]
    valid_indices = np.flatnonzero(arm.valid)
    events: list[GripperEvent] = []
    if len(valid_indices) < 2:
        return events
    previous_index = int(valid_indices[0])
    previous_state = int(arm.gripper_binary[previous_index])
    for raw_index in valid_indices[1:]:
        index = int(raw_index)
        state = int(arm.gripper_binary[index])
        if state != previous_state:
            gap_s = float(trajectory.timestamps_s[index] - trajectory.timestamps_s[previous_index])
            has_missing_frames = index != previous_index + 1
            events.append(
                GripperEvent(
                    ordinal=len(events),
                    frame_index=index,
                    timestamp_s=float(trajectory.timestamps_s[index]),
                    from_state=previous_state,
                    to_state=state,
                    transition="close" if state == 1 else "open",
                    ambiguous=has_missing_frames and gap_s > max_invalid_gap_s,
                    invalid_gap_s=gap_s if has_missing_frames else 0.0,
                )
            )
        previous_index = index
        previous_state = state
    return events


def event_sequence(events: Iterable[GripperEvent]) -> list[str]:
    return [event.transition for event in events]


def event_health(
    events: list[GripperEvent],
    expected: list[str] | tuple[str, ...] | None,
    *,
    allow_ambiguous: bool = False,
) -> dict[str, object]:
    detected = event_sequence(events)
    ambiguous = [event.ordinal for event in events if event.ambiguous]
    sequence_ok = expected is None or detected == list(expected)
    ambiguity_ok = allow_ambiguous or not ambiguous
    return {
        "valid": bool(sequence_ok and ambiguity_ok),
        "detected": detected,
        "expected": None if expected is None else list(expected),
        "ambiguous_event_indices": ambiguous,
        "events": [event.to_dict() for event in events],
    }


def invalid_intervals(trajectory: DualArmTrajectory, side: str) -> list[dict[str, object]]:
    """Return every contiguous invalid interval without deleting source frames."""
    invalid = ~np.asarray(trajectory.sides[side].valid, dtype=bool)
    indices = np.flatnonzero(invalid)
    if not len(indices):
        return []
    groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    output = []
    for group in groups:
        start, end = int(group[0]), int(group[-1])
        start_s = float(trajectory.timestamps_s[start])
        end_s = float(trajectory.timestamps_s[end])
        output.append(
            {
                "start_frame": start,
                "end_frame": end,
                "frame_count": int(len(group)),
                "start_s": start_s,
                "end_s": end_s,
                "span_s": end_s - start_s,
            }
        )
    return output
