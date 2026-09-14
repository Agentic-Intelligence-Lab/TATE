#!/usr/bin/env python3
"""Build an OpenPI-ready dual-arm ARX EEF dataset from ego videos and EEF JSON.

The source ego root provides the episode/video indexing metadata and camera
videos.  EEF labels are loaded separately so they can be replaced by a
corrected export without changing the video dataset.

Expected EEF layout for the current batch preprocessing script:

    eef-root/
    `-- chunk-000/
        `-- file-000/
            `-- preprocess/eef.json

The EEF JSON uses the current TATE ``frames`` format.  For each frame:

    hand_l.eef_pose_world
    hand_r.eef_pose_world
    hand_l.grasp_state
    hand_r.grasp_state

In this project ``eef_pose_world`` is already expressed in the corresponding
arm's zero-position flange frame, as recorded by ``Preprocess.py``.  The
builder therefore does not apply a second common-base transform.

Output ``observation.state`` and ``action`` are both 16D:

    left:  x y z qx qy qz qw gripper
    right: x y z qx qy qz qw gripper

For ego data, ``observation.state`` is the current EEF frame and ``action`` is
the next EEF frame, with the final action repeated.  Gripper values are
normalized as ``0=closed, 1=open``; HumanEgo's ``grasp_state`` uses the
opposite convention and is inverted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from training.build_arx_lerobot_dataset_from_joint import (
        EEF_DIM,
        PreparedEpisode,
        _continuous_quaternions,
        _episode_source_stats,
        _task_map,
        _video_keys,
        as_abs,
        load_dataset_metadata,
        source_video_path,
        write_lerobot_dataset,
    )
except ImportError:
    from build_arx_lerobot_dataset_from_joint import (
        EEF_DIM,
        PreparedEpisode,
        _continuous_quaternions,
        _episode_source_stats,
        _task_map,
        _video_keys,
        as_abs,
        load_dataset_metadata,
        source_video_path,
        write_lerobot_dataset,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "DATA" / "stack_cube_ego"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "lerobot"
DEFAULT_EEF_ROOT = REPO_ROOT / "outputs" / "stack_cube_ego_preprocess"
DEFAULT_REPO_ID = "local/arx_eef_stack_cube_ego"
DEFAULT_TASK = "fold the paper boxes."


def _as_pose(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("T", "matrix", "pose"):
            if key in value:
                value = value[key]
                break
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose


def _hand_pose(hand: dict[str, Any] | None, pose_key: str) -> np.ndarray | None:
    if not isinstance(hand, dict):
        return None
    candidates = [pose_key, "eef_pose_world", "eef_pose_corrected", "T_ee_in_base"]
    for key in candidates:
        pose = _as_pose(hand.get(key))
        if pose is not None:
            return pose
    return None


def _hand_gripper(hand: dict[str, Any] | None) -> float:
    if not isinstance(hand, dict):
        return 1.0
    if "gripper" in hand:
        return float(np.clip(hand["gripper"], 0.0, 1.0))
    if "gripper_open" in hand:
        return float(np.clip(hand["gripper_open"], 0.0, 1.0))
    if "grasp_state" in hand:
        # HumanEgo: 0=open, 1=closed. ARX EEF action: 0=closed, 1=open.
        return float(1.0 - np.clip(float(hand["grasp_state"]), 0.0, 1.0))
    if "grasp" in hand:
        return float(1.0 - np.clip(float(hand["grasp"]), 0.0, 1.0))
    return 1.0


def _frame_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if isinstance(frames, list):
        return frames
    records = payload.get("records")
    if isinstance(records, list):
        return records
    raise ValueError("EEF JSON must contain a list field named frames or records")


def _record_side(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    keys = ("hand_l", "left", "hand_left") if side == "left" else ("hand_r", "right", "hand_right")
    hand = next((record.get(key) for key in keys if isinstance(record.get(key), dict)), None)
    if isinstance(hand, dict):
        return hand

    # Support a corrected single-arm records file when it explicitly carries
    # a side field. This is useful for replacing one arm without rewriting the
    # other arm's original JSON.
    record_side = str(record.get("side", record.get("hand_side", ""))).lower()
    if record_side in {side, side[0], f"hand_{side[0]}"}:
        return record
    return None


def _fill_missing(values: list[np.ndarray | None], *, side: str, allow_missing_side: bool) -> list[np.ndarray]:
    valid = [index for index, value in enumerate(values) if value is not None]
    if not valid:
        if not allow_missing_side:
            raise ValueError(f"EEF JSON contains no valid {side}-hand pose")
        identity = np.eye(4, dtype=np.float64)
        return [identity.copy() for _ in values]

    filled: list[np.ndarray] = []
    first = valid[0]
    last_pose = np.asarray(values[first], dtype=np.float64)
    for index, value in enumerate(values):
        if value is not None:
            last_pose = np.asarray(value, dtype=np.float64)
        elif index < first:
            last_pose = np.asarray(values[first], dtype=np.float64)
        filled.append(last_pose.copy())
    return filled


def _fill_grippers(values: list[float | None], *, default_open: float = 1.0) -> list[float]:
    valid = [index for index, value in enumerate(values) if value is not None]
    if not valid:
        return [default_open for _ in values]

    filled: list[float] = []
    first = valid[0]
    last_value = float(values[first])
    for index, value in enumerate(values):
        if value is not None:
            last_value = float(value)
        elif index < first:
            last_value = float(values[first])
        filled.append(float(np.clip(last_value, 0.0, 1.0)))
    return filled


def load_eef_episode(
    eef_path: Path,
    *,
    pose_key: str,
    allow_missing_side: bool,
    expected_length: int | None,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(eef_path.read_text(encoding="utf-8"))
    records = _frame_records(payload)
    if expected_length is not None and len(records) != expected_length:
        raise ValueError(
            f"{eef_path} has {len(records)} EEF frames but source episode has {expected_length} frames"
        )
    if not records:
        raise ValueError(f"{eef_path} has no EEF frames")

    poses_by_side: dict[str, list[np.ndarray | None]] = {"left": [], "right": []}
    grippers_by_side: dict[str, list[float | None]] = {"left": [], "right": []}
    timestamps = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"frame {index} in {eef_path} is not an object")
        stamp = record.get("ts", record.get("timestamp"))
        if stamp is None:
            timestamps.append(index / float(fps))
        else:
            value = float(stamp)
            timestamps.append(value * 1e-9 if value > 1e6 else value)
        for side in ("left", "right"):
            hand = _record_side(record, side)
            poses_by_side[side].append(_hand_pose(hand, pose_key))
            grippers_by_side[side].append(None if hand is None else _hand_gripper(hand))

    filled_poses = {
        side: _fill_missing(
            poses_by_side[side],
            side=side,
            allow_missing_side=allow_missing_side,
        )
        for side in ("left", "right")
    }
    filled_grippers = {
        side: _fill_grippers(grippers_by_side[side])
        for side in ("left", "right")
    }
    states = np.empty((len(records), EEF_DIM), dtype=np.float64)
    for index in range(len(records)):
        values = []
        for side in ("left", "right"):
            pose = filled_poses[side][index]
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            values.extend([*pose[:3, 3], *quat, filled_grippers[side][index]])
        states[index] = np.asarray(values, dtype=np.float64)
    for offset in (3, 11):
        states[:, offset : offset + 4] = _continuous_quaternions(states[:, offset : offset + 4])
    if states.shape != (len(records), EEF_DIM) or not np.all(np.isfinite(states)):
        raise ValueError(f"invalid EEF values in {eef_path}")

    timestamps = np.asarray(timestamps, dtype=np.float32)
    timestamps -= timestamps[0]
    actions = np.concatenate([states[1:], states[-1:]], axis=0)
    return states.astype(np.float32), actions.astype(np.float32), timestamps


def _eef_candidates(eef_root: Path, item: dict[str, Any]) -> list[Path]:
    chunk = int(item.get("data/chunk_index", int(item["episode_index"]) // 1000))
    file_index = int(item.get("data/file_index", int(item["episode_index"])))
    stem = f"file-{file_index:03d}"
    chunk_name = f"chunk-{chunk:03d}"
    return [
        eef_root / chunk_name / stem / "preprocess" / "eef.json",
        eef_root / chunk_name / stem / "eef.json",
        eef_root / chunk_name / f"{stem}.json",
        eef_root / stem / "preprocess" / "eef.json",
        eef_root / stem / "eef.json",
        eef_root / "eef.json",
    ]


def resolve_eef_json(eef_root: Path, item: dict[str, Any], explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"missing explicit EEF JSON: {explicit}")
        return explicit
    for candidate in _eef_candidates(eef_root, item):
        if candidate.is_file():
            return candidate
    tried = "\n".join(str(path) for path in _eef_candidates(eef_root, item))
    raise FileNotFoundError(f"could not find EEF JSON for episode {item.get('episode_index')}; tried:\n{tried}")


def build_dataset(
    source_root: Path,
    eef_root: Path,
    output_root: Path,
    repo_id: str,
    overwrite: bool,
    *,
    task: str | None = DEFAULT_TASK,
    pose_key: str = "eef_pose_world",
    allow_missing_side: bool = False,
    explicit_eef: Path | None = None,
) -> Path:
    source_root = source_root.expanduser().resolve()
    eef_root = eef_root.expanduser().resolve()
    source_info, source_tasks, episode_meta = load_dataset_metadata(source_root)
    video_keys = _video_keys(source_info)
    if not video_keys:
        raise ValueError("source ego dataset has no video features")
    task_by_index = _task_map(source_tasks)
    fps = float(source_info.get("fps", 30))
    source_feature_keys = set(source_info.get("features", {}))

    episodes: list[PreparedEpisode] = []
    for item in episode_meta:
        eef_path = resolve_eef_json(
            eef_root,
            item,
            explicit=explicit_eef if len(episode_meta) == 1 else None,
        )
        expected_length = int(item["length"]) if item.get("length") is not None else None
        state, action, timestamp = load_eef_episode(
            eef_path,
            pose_key=pose_key,
            allow_missing_side=allow_missing_side,
            expected_length=expected_length,
            fps=fps,
        )
        episodes.append(
            PreparedEpisode(
                tasks=(
                    [
                        *(
                            episode_tasks_from_source(item, task_by_index)
                        )
                    ]
                    or ([task] if task else [])
                ),
                state=state,
                action=action,
                timestamp=timestamp,
                source_videos={
                    key: source_video_path(source_root, item, key)
                    for key in video_keys
                },
                source_stats=_episode_source_stats(item, source_feature_keys),
            )
        )
        if not episodes[-1].tasks:
            raise ValueError(
                f"episode {item.get('episode_index')} has no task; pass --task or provide task metadata"
            )

    convention = {
        "origin": "each arm's flange position at zero joint position",
        "axes": "right-handed, z-up, +x forward from the flange, +y left",
        "pose_point": "EEF pose from the supplied corrected/preprocessed JSON",
        "quaternion_order": "xyzw",
        "gripper": "0=closed, 1=open",
        "per_side_base": {
            "left": "left_flange_zero",
            "right": "right_flange_zero",
        },
        "source": "Preprocess.py frames[*].hand_{l,r}.eef_pose_world",
        "pose_key": pose_key,
    }
    return write_lerobot_dataset(
        source_info=source_info,
        episodes=episodes,
        output_root=output_root,
        repo_id=repo_id,
        overwrite=overwrite,
        eef_convention=convention,
    )


def episode_tasks_from_source(item: dict[str, Any], task_by_index: dict[int, str]) -> list[str]:
    values = item.get("tasks", [])
    names = []
    for value in values:
        name = task_by_index.get(int(value), str(value)) if isinstance(value, (int, np.integer)) else str(value)
        if name not in names:
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--eef-root", default=str(DEFAULT_EEF_ROOT))
    parser.add_argument("--eef-json", default=None, help="Explicit EEF JSON for a single-episode source.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--pose-key",
        default="eef_pose_world",
        help="Preferred hand pose key; falls back to eef_pose_world/eef_pose_corrected/T_ee_in_base.",
    )
    parser.add_argument(
        "--allow-missing-side",
        action="store_true",
        help="Allow an entirely missing left/right hand and fill it with identity pose.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(
        as_abs(args.source_root),
        as_abs(args.eef_root),
        as_abs(args.output_root),
        args.repo_id,
        args.overwrite,
        task=args.task,
        pose_key=args.pose_key,
        allow_missing_side=args.allow_missing_side,
        explicit_eef=None if args.eef_json is None else as_abs(args.eef_json),
    )


if __name__ == "__main__":
    main()
