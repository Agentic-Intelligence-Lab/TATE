"""Canonical correction-anchor identifiers and frame resolution."""

from __future__ import annotations

from typing import Any

import numpy as np


ENDPOINT_ANCHORS = ("start", "end")


def normalize_anchor_id(value: Any) -> str:
    """Normalize an event ordinal or a trajectory endpoint to an artifact key."""
    if isinstance(value, bool):
        raise ValueError(f"invalid correction anchor {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"correction event anchor must be non-negative, got {value}")
        return str(value)
    raw = str(value).strip().lower()
    if raw in ENDPOINT_ANCHORS:
        return raw
    try:
        ordinal = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"correction anchor must be 'start', 'end', or a non-negative event ordinal; got {value!r}"
        ) from exc
    if ordinal < 0:
        raise ValueError(f"correction event anchor must be non-negative, got {ordinal}")
    return str(ordinal)


def anchor_sort_key(anchor_id: str) -> tuple[int, int]:
    anchor_id = normalize_anchor_id(anchor_id)
    if anchor_id == "start":
        return (0, 0)
    if anchor_id == "end":
        return (2, 0)
    return (1, int(anchor_id))


def resolve_anchor_frame(trajectory: Any, side: str, events: list[Any], anchor_id: str) -> int:
    """Resolve start/end to valid trajectory endpoints and integers to events."""
    anchor_id = normalize_anchor_id(anchor_id)
    if anchor_id in ENDPOINT_ANCHORS:
        valid = np.flatnonzero(trajectory.sides[side].valid)
        if not len(valid):
            raise ValueError(f"{side} trajectory has no valid frames")
        return int(valid[0] if anchor_id == "start" else valid[-1])
    ordinal = int(anchor_id)
    if ordinal >= len(events):
        raise ValueError(
            f"{side} trajectory has {len(events)} events but correction needs event {ordinal}"
        )
    return int(events[ordinal].frame_index)
