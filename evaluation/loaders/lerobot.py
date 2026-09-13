"""Loader for planned derived LeRobot datasets containing ``tate.eef.*``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import SchemaError, validate_trajectory
from ..types import ArmTrajectory, DualArmTrajectory, SIDES


def _stack_column(frame: Any, name: str, width: int | None = None) -> np.ndarray:
    if name not in frame.columns:
        raise SchemaError(f"derived LeRobot data is missing {name!r}")
    values = frame[name].to_numpy()
    array = np.stack(values) if len(values) and np.asarray(values[0]).ndim else np.asarray(values)
    if width is not None:
        array = np.asarray(array).reshape(len(frame), width)
    return array


def _frame_name(info: dict[str, Any], provenance: dict[str, Any], side: str) -> str:
    feature = (info.get("features") or {}).get(f"tate.eef.{side}.pose") or {}
    candidates = [
        feature.get("frame"),
        ((info.get("tate") or {}).get("eef_frames") or {}).get(side),
        ((info.get("eef_coordinate_convention") or {}).get("per_side_frames") or {}).get(side),
        ((provenance.get("eef_coordinate_convention") or {}).get("per_side_frames") or {}).get(side),
    ]
    name = next((str(value) for value in candidates if value), "")
    if not name:
        raise SchemaError(f"derived LeRobot metadata has no EEF frame for {side}")
    return name


def load_lerobot_episode(dataset_root: str | Path, episode_id: int | str) -> DualArmTrajectory:
    dataset_root = Path(dataset_root).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    provenance_path = dataset_root / "meta" / "tate_preprocess.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    target = int(episode_id)
    chunks = []
    import pandas as pd

    for parquet in sorted((dataset_root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(parquet)
        identity_column = (
            "tate.source_episode_index"
            if "tate.source_episode_index" in frame.columns
            else "episode_index"
        )
        if identity_column not in frame.columns:
            if target == 0:
                chunks.append(frame)
            continue
        selected = frame[frame[identity_column] == target]
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise FileNotFoundError(f"episode {episode_id} not found in {dataset_root}")
    frame = pd.concat(chunks, ignore_index=True)
    if "frame_index" in frame.columns:
        frame = frame.sort_values("frame_index", kind="stable").reset_index(drop=True)
    fps = float(info.get("fps", 0.0))
    if "timestamp" in frame.columns:
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        timestamps = timestamps - timestamps[0]
    elif fps > 0.0:
        timestamps = np.arange(len(frame), dtype=np.float64) / fps
    else:
        raise SchemaError("derived LeRobot episode has neither timestamps nor positive fps")
    sides = {}
    for side in SIDES:
        pose = _stack_column(frame, f"tate.eef.{side}.pose", 7).astype(np.float64)
        valid = _stack_column(frame, f"tate.eef.{side}.valid").reshape(-1).astype(bool)
        gripper = _stack_column(frame, f"tate.eef.{side}.gripper").reshape(-1).astype(np.int8)
        ratio_name = f"tate.eef.{side}.grasp_ratio"
        ratio = (
            _stack_column(frame, ratio_name).reshape(-1).astype(np.float64)
            if ratio_name in frame.columns
            else None
        )
        sides[side] = ArmTrajectory(
            pose_xyzw=pose,
            valid=valid,
            gripper_binary=gripper,
            grasp_ratio=ratio,
            frame_name=_frame_name(info, provenance, side),
        )
    return validate_trajectory(
        DualArmTrajectory(
            episode_id=episode_id,
            timestamps_s=timestamps,
            sides=sides,
            source={
                "kind": "derived_lerobot",
                "dataset_root": str(dataset_root),
                "provenance": provenance,
            },
        )
    )
