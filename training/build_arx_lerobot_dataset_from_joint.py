#!/usr/bin/env python3
"""Build an OpenPI-ready dual-arm ARX EEF dataset from joint-labelled data.

The source dataset is a LeRobot-style dataset with 14D ARX joint vectors:

    left_joint_1..left_joint_6, left_gripper,
    right_joint_1..right_joint_6, right_gripper

The output dataset contains 16D EEF vectors:

    left:  x y z qx qy qz qw gripper
    right: x y z qx qy qz qw gripper

Both output ``observation.state`` and ``action`` are EEF vectors.  Each arm
uses its own zero-position flange frame:

    T_eef_in_arm_zero = inv(T_flange_zero_in_scene) @ T_tcp_in_scene

The left and right branches are therefore never expressed in the other arm's
base frame or in a single common arm frame.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import serialize_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "DATA" / "stack_cube_arx"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "lerobot"
DEFAULT_REPO_ID = "local/arx_eef_stack_cube_arx"
DEFAULT_SCENE = REPO_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"

JOINT_DIM = 14
EEF_DIM = 16
MODEL_ACTION_DIM = 32
DEFAULT_GRIPPER_CLOSED = -3.4
DEFAULT_GRIPPER_OPEN = 0.1
EEF_NAMES = (
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_eef_qx",
    "left_eef_qy",
    "left_eef_qz",
    "left_eef_qw",
    "left_gripper",
    "right_eef_x",
    "right_eef_y",
    "right_eef_z",
    "right_eef_qx",
    "right_eef_qy",
    "right_eef_qz",
    "right_eef_qw",
    "right_gripper",
)
STATE_NAMES = EEF_NAMES
VIDEO_PREFIX = "observation.images."
STANDARD_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def as_abs(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (REPO_ROOT / value).resolve()


def _read_parquet_records(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_tasks(root: Path) -> list[dict[str, Any]]:
    jsonl = _read_jsonl(root / "meta" / "tasks.jsonl")
    if jsonl:
        return jsonl
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        return _read_parquet_records(parquet)
    raise FileNotFoundError(f"missing tasks.jsonl or tasks.parquet under {root / 'meta'}")


def _load_episode_metadata(root: Path) -> list[dict[str, Any]]:
    jsonl = _read_jsonl(root / "meta" / "episodes.jsonl")
    if jsonl:
        episodes = jsonl
    else:
        episodes_root = root / "meta" / "episodes"
        paths = sorted(episodes_root.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"missing episode metadata under {episodes_root}")
        episodes = []
        for path in paths:
            episodes.extend(_read_parquet_records(path))
    episodes.sort(key=lambda item: int(item["episode_index"]))
    return episodes


def load_dataset_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks = _load_tasks(root)
    episodes = _load_episode_metadata(root)
    expected = int(info.get("total_episodes", len(episodes)))
    if expected != len(episodes):
        raise ValueError(f"{info_path} says {expected} episodes, found {len(episodes)}")
    return info, tasks, episodes


def _task_map(tasks: list[dict[str, Any]]) -> dict[int, str]:
    return {int(item["task_index"]): str(item["task"]) for item in tasks}


def episode_tasks(item: dict[str, Any], task_by_index: dict[int, str], fallback: str | None = None) -> list[str]:
    values = item.get("tasks", [])
    names = []
    for value in values:
        name = task_by_index.get(int(value), str(value)) if isinstance(value, (int, np.integer)) else str(value)
        if name not in names:
            names.append(name)
    if not names and fallback:
        names.append(fallback)
    if not names:
        raise ValueError(f"episode {item.get('episode_index')} has no task")
    return names


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _source_data_path(root: Path, item: dict[str, Any]) -> Path:
    if "data/chunk_index" in item and "data/file_index" in item:
        return (
            root
            / "data"
            / f"chunk-{int(item['data/chunk_index']):03d}"
            / f"file-{int(item['data/file_index']):03d}.parquet"
        )
    episode_index = int(item["episode_index"])
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"file-{episode_index:03d}.parquet"


def source_video_path(root: Path, item: dict[str, Any], video_key: str) -> Path:
    prefix = f"videos/{video_key}/"
    if f"{prefix}chunk_index" in item and f"{prefix}file_index" in item:
        return (
            root
            / "videos"
            / video_key
            / f"chunk-{int(item[f'{prefix}chunk_index']):03d}"
            / f"file-{int(item[f'{prefix}file_index']):03d}.mp4"
        )
    episode_index = int(item["episode_index"])
    return (
        root
        / "videos"
        / video_key
        / f"chunk-{episode_index // 1000:03d}"
        / f"file-{episode_index:03d}.mp4"
    )


def _copy_or_fail(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected non-empty matrix, got {values.shape}")
    return {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.asarray([len(values)]),
    }


def _scalar_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("cannot compute scalar stats for an empty vector")
    return {
        "min": np.asarray([np.min(values)]),
        "max": np.asarray([np.max(values)]),
        "mean": np.asarray([np.mean(values)]),
        "std": np.asarray([np.std(values)]),
        "count": np.asarray([len(values)]),
    }


def _constant_stats(value: int, length: int) -> dict[str, np.ndarray]:
    return {
        "min": np.asarray([value]),
        "max": np.asarray([value]),
        "mean": np.asarray([value], dtype=np.float64),
        "std": np.zeros(1, dtype=np.float64),
        "count": np.asarray([length]),
    }


def _range_stats(start: int, length: int) -> dict[str, np.ndarray]:
    return _scalar_stats(np.arange(start, start + length, dtype=np.float64))


def _episode_source_stats(item: dict[str, Any], feature_keys: set[str]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, value in item.items():
        if not key.startswith("stats/"):
            continue
        feature, stat = key[len("stats/") :].rsplit("/", 1)
        if feature not in feature_keys or stat not in {"min", "max", "mean", "std", "count"}:
            continue
        result.setdefault(feature, {})[stat] = np.asarray(value)
    return {
        feature: stats
        for feature, stats in result.items()
        if set(stats) == {"min", "max", "mean", "std", "count"}
    }


def _continuous_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(values), 4):
        raise ValueError(f"invalid quaternion matrix shape: {values.shape}")
    result = values.copy()
    for index in range(len(result)):
        norm = float(np.linalg.norm(result[index]))
        if norm < 1e-8:
            raise ValueError("encountered a zero quaternion")
        result[index] /= norm
        if index and float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def normalize_gripper(values: np.ndarray, closed: float, opened: float) -> np.ndarray:
    if opened <= closed:
        raise ValueError(f"gripper open value must exceed closed value, got {closed}, {opened}")
    values = np.asarray(values, dtype=np.float64)
    return np.clip((values - closed) / (opened - closed), 0.0, 1.0)


class ArxJointToEef:
    """MuJoCo FK converter using independent left/right zero-flange frames."""

    def __init__(self, scene_path: Path):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("MuJoCo is required for joint-to-EEF conversion.") from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.arm_joint_names = {
            "left": tuple(f"left_joint{i}" for i in range(1, 7)),
            "right": tuple(f"right_joint{i}" for i in range(11, 17)),
        }
        self.gripper_joint_names = {
            "left": ("left_joint7", "left_joint8"),
            "right": ("right_joint17", "right_joint18"),
        }
        self.site_names = {"left": "left_tcp", "right": "right_tcp"}
        self.flange_names = {"left": "left_flange", "right": "right_flange"}
        self.arm_qpos_addresses = {
            side: np.asarray([self._joint_qpos_address(name) for name in names], dtype=np.int32)
            for side, names in self.arm_joint_names.items()
        }
        self.site_ids = {side: self._site_id(name) for side, name in self.site_names.items()}
        self.flange_ids = {side: self._site_id(name) for side, name in self.flange_names.items()}

        self.data.qpos[:] = self.model.qpos0
        for side, names in self.gripper_joint_names.items():
            for name in names:
                address = self._joint_qpos_address(name)
                self.data.qpos[address] = self._joint_upper_limit(name)
        mujoco.mj_forward(self.model, self.data)
        self.flange_zero_world = {
            side: self._site_pose(self.flange_ids[side]) for side in ("left", "right")
        }
        self.zero_flange_in_world = {
            side: pose.copy() for side, pose in self.flange_zero_world.items()
        }
        self.eef_convention = {
            "origin": "each arm's flange position at zero joint position",
            "axes": "right-handed, z-up, +x forward from the flange, +y left",
            "pose_point": "TCP site midpoint between gripper fingers",
            "quaternion_order": "xyzw",
            "gripper": "0=closed, 1=open",
            "per_side_base": {
                "left": "left_flange_zero",
                "right": "right_flange_zero",
            },
        }

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"scene is missing joint {name!r}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_upper_limit(self, name: str) -> float:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"scene is missing joint {name!r}")
        return float(self.model.jnt_range[joint_id, 1])

    def _site_id(self, name: str) -> int:
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise RuntimeError(f"scene is missing site {name!r}")
        return int(site_id)

    def _site_pose(self, site_id: int) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[site_id]
        return pose

    def convert(self, joints: np.ndarray, *, gripper_closed: float, gripper_open: float) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != JOINT_DIM:
            raise ValueError(f"expected Nx{JOINT_DIM} joint vectors, got {joints.shape}")

        output = np.empty((len(joints), EEF_DIM), dtype=np.float64)
        for index, row in enumerate(joints):
            self.data.qpos[:] = self.model.qpos0
            for side, joint_slice in (("left", slice(0, 6)), ("right", slice(7, 13))):
                self.data.qpos[self.arm_qpos_addresses[side]] = row[joint_slice]
            self.mujoco.mj_forward(self.model, self.data)

            values = []
            for side in ("left", "right"):
                tcp_world = self._site_pose(self.site_ids[side])
                tcp_in_zero = np.linalg.inv(self.zero_flange_in_world[side]) @ tcp_world
                quat_xyzw = Rotation.from_matrix(tcp_in_zero[:3, :3]).as_quat()
                gripper_index = 6 if side == "left" else 13
                gripper = normalize_gripper(
                    np.asarray([row[gripper_index]]),
                    gripper_closed,
                    gripper_open,
                )[0]
                values.extend([*tcp_in_zero[:3, 3], *quat_xyzw, gripper])
            output[index] = np.asarray(values, dtype=np.float64)

        for offset in (3, 11):
            output[:, offset : offset + 4] = _continuous_quaternions(output[:, offset : offset + 4])
        if not np.all(np.isfinite(output)):
            raise ValueError("joint-to-EEF conversion produced non-finite values")
        return output.astype(np.float32)


@dataclass
class PreparedEpisode:
    tasks: list[str]
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    source_videos: dict[str, Path]
    source_stats: dict[str, dict[str, np.ndarray]]


def _output_features(source_info: dict[str, Any], video_keys: list[str]) -> dict[str, dict[str, Any]]:
    source_features = source_info.get("features", {})
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [EEF_DIM],
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [EEF_DIM],
            "names": list(EEF_NAMES),
        },
    }
    for key in video_keys:
        if key not in source_features:
            raise KeyError(f"source info is missing video feature {key}")
        features[key] = copy.deepcopy(source_features[key])
    for key, feature in STANDARD_FEATURES.items():
        features[key] = copy.deepcopy(feature)
    return features


def _build_output_info(
    source_info: dict[str, Any],
    *,
    video_keys: list[str],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    eef_convention: dict[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(source_info)
    output["codebase_version"] = "v2.1"
    output["features"] = _output_features(source_info, video_keys)
    output.update(
        {
            "fps": int(source_info.get("fps", 30)),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": len(video_keys),
            "total_chunks": (total_episodes + 999) // 1000,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4",
            "robot_type": "arx5_2025_bimanual_eef",
            "splits": {"train": f"0:{total_episodes}"},
            "arx_eef": eef_convention,
        }
    )
    return output


def _episode_table(episode_index: int, global_index: int, episode: PreparedEpisode, task_index: int) -> pa.Table:
    length = len(episode.state)
    if episode.action.shape != episode.state.shape or episode.action.shape != (length, EEF_DIM):
        raise ValueError(f"invalid EEF episode shapes: {episode.state.shape}, {episode.action.shape}")
    if len(episode.timestamp) != length:
        raise ValueError("timestamp length does not match episode length")
    return pa.table(
        {
            "observation.state": pa.array(episode.state.tolist(), type=pa.list_(pa.float32(), EEF_DIM)),
            "action": pa.array(episode.action.tolist(), type=pa.list_(pa.float32(), EEF_DIM)),
            "timestamp": pa.array(np.asarray(episode.timestamp, dtype=np.float32)),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_index, global_index + length, dtype=np.int64)),
            "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
        }
    ).replace_schema_metadata(None)


def write_lerobot_dataset(
    *,
    source_info: dict[str, Any],
    episodes: list[PreparedEpisode],
    output_root: Path,
    repo_id: str,
    overwrite: bool,
    eef_convention: dict[str, Any],
) -> Path:
    if not episodes:
        raise ValueError("no episodes to write")
    video_keys = _video_keys(source_info)
    if not video_keys:
        raise ValueError("source dataset has no video features")

    task_names: list[str] = []
    task_indices: dict[str, int] = {}
    for episode in episodes:
        for task in episode.tasks:
            if task not in task_indices:
                task_indices[task] = len(task_names)
                task_names.append(task)

    total_frames = sum(len(episode.state) for episode in episodes)
    dataset_root = (output_root.expanduser().resolve() / repo_id).resolve()
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"dataset already exists: {dataset_root}; pass --overwrite")
        shutil.rmtree(dataset_root)
    (dataset_root / "data").mkdir(parents=True)
    (dataset_root / "videos").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)

    output_info = _build_output_info(
        source_info,
        video_keys=video_keys,
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(task_names),
        eef_convention=eef_convention,
    )
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(output_info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(
        dataset_root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(task_names)],
    )

    episode_records = []
    episode_stats = []
    global_index = 0
    for episode_index, episode in enumerate(episodes):
        task_index = task_indices[episode.tasks[0]]
        length = len(episode.state)
        data_path = (
            dataset_root
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"file-{episode_index:03d}.parquet"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_episode_table(episode_index, global_index, episode, task_index), data_path)

        for video_key in video_keys:
            source_video = episode.source_videos.get(video_key)
            if source_video is None:
                raise FileNotFoundError(f"episode {episode_index} is missing video {video_key}")
            destination_video = (
                dataset_root
                / "videos"
                / video_key
                / f"chunk-{episode_index // 1000:03d}"
                / f"file-{episode_index:03d}.mp4"
            )
            _copy_or_fail(source_video, destination_video)

        stats = dict(episode.source_stats)
        stats["observation.state"] = _vector_stats(episode.state)
        stats["action"] = _vector_stats(episode.action)
        stats["timestamp"] = _scalar_stats(episode.timestamp)
        stats["frame_index"] = _range_stats(0, length)
        stats["episode_index"] = _constant_stats(episode_index, length)
        stats["index"] = _range_stats(global_index, length)
        stats["task_index"] = _constant_stats(task_index, length)
        episode_records.append(
            {"episode_index": episode_index, "tasks": episode.tasks, "length": length}
        )
        episode_stats.append(
            {"episode_index": episode_index, "stats": serialize_dict(stats)}
        )
        global_index += length

    _write_jsonl(dataset_root / "meta" / "episodes.jsonl", episode_records)
    _write_jsonl(dataset_root / "meta" / "episodes_stats.jsonl", episode_stats)
    aggregate_input = [
        {
            feature: {
                stat: np.asarray(value)
                for stat, value in feature_stats.items()
            }
            for feature, feature_stats in record["stats"].items()
        }
        for record in episode_stats
    ]
    aggregate = aggregate_stats(aggregate_input)
    (dataset_root / "meta" / "stats.json").write_text(
        json.dumps(serialize_dict(aggregate), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Created: {dataset_root}")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {total_frames}")
    print(f"Action/state: {EEF_DIM}D EEF")
    print(f"Cameras: {', '.join(video_keys)}")
    return dataset_root


def build_dataset(
    source_roots: list[Path],
    output_root: Path,
    repo_id: str,
    overwrite: bool,
    *,
    scene_path: Path = DEFAULT_SCENE,
    gripper_closed: float = DEFAULT_GRIPPER_CLOSED,
    gripper_open: float = DEFAULT_GRIPPER_OPEN,
) -> Path:
    if not source_roots:
        raise ValueError("at least one --source-root is required")
    loaded = [load_dataset_metadata(root) for root in source_roots]
    first_info = loaded[0][0]
    first_video_keys = _video_keys(first_info)
    if not first_video_keys:
        raise ValueError("source dataset has no video features")
    for root, (info, _, _) in zip(source_roots, loaded, strict=True):
        if _video_keys(info) != first_video_keys:
            raise ValueError(f"video features differ between source datasets: {root}")
        if int(info.get("fps", 30)) != int(first_info.get("fps", 30)):
            raise ValueError(f"FPS differs between source datasets: {root}")

    converter = ArxJointToEef(scene_path.expanduser().resolve())
    episodes: list[PreparedEpisode] = []
    for root, (info, tasks, episode_meta) in zip(source_roots, loaded, strict=True):
        task_by_index = _task_map(tasks)
        feature_keys = set(info.get("features", {}))
        for item in episode_meta:
            data_path = _source_data_path(root, item)
            if not data_path.is_file():
                raise FileNotFoundError(f"missing source episode data: {data_path}")
            table = pq.read_table(data_path)
            for required in ("observation.state", "action"):
                if required not in table.column_names:
                    raise ValueError(f"{data_path} is missing {required!r}")
            state_joint = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
            action_joint = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            if state_joint.shape != action_joint.shape or state_joint.shape[1] != JOINT_DIM:
                raise ValueError(
                    f"{data_path} must contain matching {JOINT_DIM}D state/action, "
                    f"got {state_joint.shape} and {action_joint.shape}"
                )
            state = converter.convert(
                state_joint,
                gripper_closed=gripper_closed,
                gripper_open=gripper_open,
            )
            action = converter.convert(
                action_joint,
                gripper_closed=gripper_closed,
                gripper_open=gripper_open,
            )
            if "timestamp" in table.column_names:
                timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
            else:
                timestamp = np.arange(len(state), dtype=np.float32) / float(info.get("fps", 30))
            episodes.append(
                PreparedEpisode(
                    tasks=episode_tasks(item, task_by_index),
                    state=state,
                    action=action,
                    timestamp=timestamp,
                    source_videos={
                        key: source_video_path(root, item, key)
                        for key in first_video_keys
                    },
                    source_stats=_episode_source_stats(item, feature_keys),
                )
            )

    return write_lerobot_dataset(
        source_info=first_info,
        episodes=episodes,
        output_root=output_root,
        repo_id=repo_id,
        overwrite=overwrite,
        eef_convention=converter.eef_convention,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        default=None,
        help="Source LeRobot dataset root; repeat to concatenate datasets.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--gripper-closed", type=float, default=DEFAULT_GRIPPER_CLOSED)
    parser.add_argument("--gripper-open", type=float, default=DEFAULT_GRIPPER_OPEN)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_roots = [as_abs(path) for path in (args.source_roots or [str(DEFAULT_SOURCE_ROOT)])]
    build_dataset(
        source_roots,
        as_abs(args.output_root),
        args.repo_id,
        args.overwrite,
        scene_path=as_abs(args.scene),
        gripper_closed=args.gripper_closed,
        gripper_open=args.gripper_open,
    )


if __name__ == "__main__":
    main()
