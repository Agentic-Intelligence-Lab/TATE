#!/usr/bin/env python3
"""Build directly trainable 16-D EEF ARX ego/real cotrain datasets.

The exporter reads immutable 14-D joint datasets and writes independent
LeRobot v2.1 ``train`` and ``eval`` repositories.  State/action are:
``[left xyz, xyzw, gripper, right xyz, xyzw, gripper]``.  Cube's absent left
arm uses a canonical identity pose and false per-element state/action masks.
Ego is head-only; real train can use reproducible camera dropout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from real_data.arx_lerobot_adapter import (
    ArxForwardKinematics,
    DEFAULT_GRIPPER_BINARY_THRESHOLD_RAW,
)


OUTPUT_ROOT = REPO_ROOT / "outputs" / "lerobot"
DEFAULT_SCENE = REPO_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
DEFAULT_CALIBRATION = REPO_ROOT / "cfg" / "preprocess" / "base" / "RealSenseD405.yaml"
VIDEO_KEYS = ("observation.images.head", "observation.images.left", "observation.images.right")
IMAGE_MASK_NAMES = ["head", "left", "right"]
JOINT_DIM, EEF_DIM = 14, 16
EEF_NAMES = [
    "left_eef_x", "left_eef_y", "left_eef_z", "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_eef_qw", "left_gripper",
    "right_eef_x", "right_eef_y", "right_eef_z", "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_eef_qw", "right_gripper",
]
CANONICAL_ARM = np.asarray([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    experiment: str
    variant: str
    prompt: str
    active_sides: tuple[str, ...]
    real_root_override: Path | None = None

    @property
    def ego_root(self) -> Path:
        return REPO_ROOT / "outputs" / "experiments" / self.experiment / "datasets" / self.variant

    @property
    def real_root(self) -> Path:
        if self.real_root_override is not None:
            return self.real_root_override
        return REPO_ROOT.parent / "Data_TATE" / f"{self.name}_arx"

    @property
    def correction_path(self) -> Path:
        return REPO_ROOT / "outputs" / "corrections" / self.name / "xyz_mean_target_min_bending.json"


TASKS = {
    # Keep this synchronized with cfg/preprocess/batch/stack_cube_correction_ablation.yaml.
    "stack_cube": TaskSpec("stack_cube", "stack_cube_correction_ablation_v1", "finger_center_hys085__position_rotation", "pick the cube and stack it on the blue plate", ("right",)),
    # Keep this synchronized with cfg/preprocess/batch/stack_cola_v2.yaml,
    # whose experiment_id is stack_cola_v2_50 and corrected default is pos+rot.
    "stack_cola": TaskSpec("stack_cola", "stack_cola_v2_50", "finger_center_hys085__position_rotation", "pick two colas and stack them on the brown box", ("left", "right")),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataset_single_task_prompt(root: Path) -> str:
    """Read the sole task text from a raw LeRobot dataset's metadata."""
    path = root / "meta" / "tasks.jsonl"
    if path.is_file():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        path = root / "meta" / "tasks.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"custom task dataset needs {root / 'meta/tasks.jsonl'} or {path}")
        records = pq.read_table(path).to_pylist()
    prompts = {str(record["task"]) for record in records if "task" in record}
    if len(prompts) != 1:
        raise ValueError(f"custom task dataset must have exactly one task in {path}; found {sorted(prompts)}")
    return prompts.pop()


def episode_files(root: Path) -> list[Path]:
    paths = sorted((root / "data").rglob("file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no source episode parquets under {root / 'data'}")
    return paths


def source_episode_index(table: pa.Table, path: Path) -> int:
    values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if len(values) == 0 or np.any(values != values[0]):
        raise ValueError(f"{path} must contain one non-empty episode")
    return int(values[0])


def source_video(root: Path, key: str, episode: int) -> Path:
    path = root / "videos" / key / "chunk-000" / f"file-{episode:03d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def eef_mask(active_sides: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(EEF_DIM, dtype=bool)
    if "left" in active_sides:
        mask[:8] = True
    if "right" in active_sides:
        mask[8:] = True
    return mask


def valid_rows(table: pa.Table, source: str, active_sides: tuple[str, ...]) -> np.ndarray:
    """Return t whose state at both t and t+1 is valid; always drops final t."""
    keep = np.ones(len(table) - 1, dtype=bool)
    if source == "ego":
        for side in active_sides:
            key = f"tate.eef.{side}.valid"
            if key not in table.column_names:
                raise ValueError(f"ego source is missing {key}")
            valid = np.asarray(table[key].to_pylist(), dtype=bool)
            keep &= valid[:-1] & valid[1:]
    return keep


def continuous_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    for index in range(len(values)):
        norm = float(np.linalg.norm(values[index]))
        if norm < 1e-8:
            raise ValueError("FK produced zero quaternion")
        values[index] /= norm
        if index and float(np.dot(values[index - 1], values[index])) < 0:
            values[index] *= -1
    return values


def joints_to_eef(joints: np.ndarray, fk: ArxForwardKinematics) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != JOINT_DIM:
        raise ValueError(f"expected N x {JOINT_DIM} joints, got {joints.shape}")
    result = np.empty((len(joints), EEF_DIM), dtype=np.float32)
    for index, row in enumerate(joints):
        poses = fk.forward({"left": row[:6], "right": row[7:13]})
        values: list[float] = []
        for side, gripper_index in (("left", 6), ("right", 13)):
            pose = poses[side]["tcp"]
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            # Keep the real and ego labels in the same binary convention as
            # the diagnostic adapter: raw >= -2.6 means closed (1).
            gripper = float(row[gripper_index] >= DEFAULT_GRIPPER_BINARY_THRESHOLD_RAW)
            values.extend([*pose[:3, 3], *quat, gripper])
        result[index] = values
    for offset in (3, 11):
        result[:, offset : offset + 4] = continuous_quaternions(result[:, offset : offset + 4])
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite result during FK")
    return result


def apply_arm_mask(values: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    if not active_mask[:8].any():
        result[:, :8] = CANONICAL_ARM
    if not active_mask[8:].any():
        result[:, 8:] = CANONICAL_ARM
    return result


def choose_real_image_masks(count: int, dropout: bool, probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not dropout:
        return np.ones((count, 3), dtype=bool)
    patterns = np.asarray(((1, 1, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1)), dtype=bool)
    return patterns[rng.choice(4, size=count, p=probabilities)]


def resolve_real_split(spec: TaskSpec, real_ids: set[int], args: argparse.Namespace) -> tuple[set[int], set[int], str]:
    if args.real_train_ids is not None or args.real_eval_ids is not None:
        if args.real_train_ids is None or args.real_eval_ids is None:
            raise ValueError("provide both --real-train-ids and --real-eval-ids")
        if not args.allow_correction_split_override:
            raise ValueError("custom IDs require --allow-correction-split-override because calibration used a fixed real split")
        train, eval_, description = set(args.real_train_ids), set(args.real_eval_ids), "manual episode IDs"
    elif args.real_train_ratio is not None:
        if not args.allow_correction_split_override:
            raise ValueError("--real-train-ratio requires --allow-correction-split-override because calibration used a fixed real split")
        if not 0 < args.real_train_ratio < 1:
            raise ValueError("--real-train-ratio must be between zero and one")
        values = np.asarray(sorted(real_ids), dtype=np.int64)
        rng = np.random.default_rng(args.split_seed + sum(map(ord, spec.name)))
        rng.shuffle(values)
        count = min(max(1, round(len(values) * args.real_train_ratio)), len(values) - 1)
        train, eval_ = set(map(int, values[:count])), set(map(int, values[count:]))
        description = f"random ratio {args.real_train_ratio:g}, seed {args.split_seed}"
    else:
        correction = read_json(spec.correction_path)
        train = set(map(int, correction["real_calibration_episode_ids"]))
        eval_ = set(map(int, correction["real_eval_episode_ids_not_used"]))
        description = f"correction artifact {spec.correction_path}"
    if train & eval_ or train | eval_ != real_ids:
        raise ValueError(f"split must partition real IDs {sorted(real_ids)}; got train={sorted(train)}, eval={sorted(eval_)}")
    return train, eval_, description


class DatasetWriter:
    def __init__(self, root: Path, source_info: dict[str, Any], spec: TaskSpec, split: str, mode: str, active_mask: np.ndarray, split_source: str, overwrite: bool) -> None:
        if root.exists():
            if not overwrite:
                raise FileExistsError(f"{root} exists; pass --overwrite")
            shutil.rmtree(root)
        self.root, self.source_info, self.spec = root, source_info, spec
        self.split, self.mode, self.active_mask, self.split_source = split, mode, active_mask, split_source
        (root / "data").mkdir(parents=True)
        (root / "videos").mkdir(parents=True)
        (root / "meta").mkdir(parents=True)
        self.global_index = 0
        self.episodes: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.image_counts = {"head_left_right": 0, "head_only": 0, "head_left": 0, "head_right": 0}
        self.counts = {"ego": {"episodes": 0, "frames": 0}, "real": {"episodes": 0, "frames": 0}}

    def add(self, *, source: str, source_root: Path, source_episode: int, selected: np.ndarray, state: np.ndarray, action: np.ndarray, timestamp: np.ndarray, frame_index: np.ndarray, image_mask: np.ndarray) -> None:
        episode, length = len(self.episodes), len(state)
        mask = np.broadcast_to(self.active_mask, (length, EEF_DIM))
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), EEF_DIM)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), EEF_DIM)),
            "timestamp": pa.array(timestamp, type=pa.float32()), "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
            "index": pa.array(np.arange(self.global_index, self.global_index + length, dtype=np.int64)),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
            "policy.state_mask": pa.array(mask.tolist(), type=pa.list_(pa.bool_(), EEF_DIM)),
            "policy.action_mask": pa.array(mask.tolist(), type=pa.list_(pa.bool_(), EEF_DIM)),
            "policy.image_mask": pa.array(image_mask.tolist(), type=pa.list_(pa.bool_(), 3)),
            "policy.domain": pa.array(np.full(length, 0 if source == "ego" else 1, dtype=np.int8)),
            "policy.source_episode_index": pa.array(np.full(length, source_episode, dtype=np.int64)),
            "policy.source_frame_index": pa.array(frame_index, type=pa.int64()),
        }).replace_schema_metadata(None)
        path = self.root / "data" / f"chunk-{episode // 1000:03d}" / f"file-{episode:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        for key in VIDEO_KEYS:
            link_or_copy(source_video(source_root, key, source_episode), self.root / "videos" / key / f"chunk-{episode // 1000:03d}" / f"file-{episode:03d}.mp4")
        self.episodes.append({"episode_index": episode, "tasks": [0], "length": length})
        self.provenance.append({"output_episode_index": episode, "source_domain": source, "source_episode_index": source_episode, "source_frame_range": [int(selected[0]), int(selected[-1])], "frames": length})
        self.counts[source]["episodes"] += 1
        self.counts[source]["frames"] += length
        self.states.append(state); self.actions.append(action); self.global_index += length
        for pattern, name in (((1, 1, 1), "head_left_right"), ((1, 0, 0), "head_only"), ((1, 1, 0), "head_left"), ((1, 0, 1), "head_right")):
            self.image_counts[name] += int(np.all(image_mask == pattern, axis=1).sum())

    def finish(self) -> dict[str, Any]:
        if not self.episodes:
            raise RuntimeError(f"no data written to {self.root}")
        source_features = self.source_info["features"]
        features = {
            "observation.state": {"dtype": "float32", "shape": [EEF_DIM], "names": EEF_NAMES},
            "action": {"dtype": "float32", "shape": [EEF_DIM], "names": EEF_NAMES},
            **{key: source_features[key] for key in VIDEO_KEYS},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None}, "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None}, "index": {"dtype": "int64", "shape": [1], "names": None}, "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "policy.state_mask": {"dtype": "bool", "shape": [EEF_DIM], "names": EEF_NAMES}, "policy.action_mask": {"dtype": "bool", "shape": [EEF_DIM], "names": EEF_NAMES}, "policy.image_mask": {"dtype": "bool", "shape": [3], "names": IMAGE_MASK_NAMES},
            "policy.domain": {"dtype": "int8", "shape": [1], "names": ["0=ego,1=real"]}, "policy.source_episode_index": {"dtype": "int64", "shape": [1], "names": None}, "policy.source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        info = {"codebase_version": "v2.1", "robot_type": "arx5_2025_bimanual_eef", "fps": int(self.source_info["fps"]), "total_episodes": len(self.episodes), "total_frames": self.global_index, "total_tasks": 1, "total_videos": len(VIDEO_KEYS), "total_chunks": (len(self.episodes) + 999) // 1000, "chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet", "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4", "splits": {self.split: f"0:{len(self.episodes)}"}, "features": features,
            "arx_eef": {"state_action_layout": EEF_NAMES, "action_alignment": "next_source_frame_state", "active_sides": list(self.spec.active_sides), "default_state_mask": self.active_mask.tolist(), "default_action_mask": self.active_mask.tolist(), "image_mask_order": IMAGE_MASK_NAMES, "coordinate_frame": "per-arm zero-flange frame", "gripper": "binary: 0=open, 1=closed; real raw >= -2.6 is closed"}}
        state, action = np.concatenate(self.states), np.concatenate(self.actions)
        stat = lambda x: {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [len(x)]}
        write_json(self.root / "meta" / "info.json", info)
        write_json(self.root / "meta" / "stats.json", {"observation.state": stat(state), "action": stat(action)})
        write_jsonl(self.root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": self.spec.prompt}])
        write_jsonl(self.root / "meta" / "episodes.jsonl", self.episodes)
        write_jsonl(self.root / "meta" / "episodes_stats.jsonl", [])
        write_json(self.root / "cotrain_provenance.json", {"schema": "tate.arx_eef_cotrain_dataset", "schema_version": 3, "task": self.spec.name, "split": self.split, "mode": self.mode, "ego_variant": str(self.spec.ego_root), "real_dataset": str(self.spec.real_root), "correction": str(self.spec.correction_path), "real_split_source": self.split_source, "gripper_binary_threshold_raw": DEFAULT_GRIPPER_BINARY_THRESHOLD_RAW, "inactive_arm_policy": "canonical_identity_pose_with_per_element_loss_mask", "image_mask_counts": self.image_counts, "counts": self.counts, "episodes": self.provenance})
        return {"dataset": str(self.root), "episodes": len(self.episodes), "frames": self.global_index, "counts": self.counts}


def output_dataset_root(base: Path, spec: TaskSpec, mode: str, split: str, label: str | None) -> Path:
    if label is None:
        name = f"arx_eef_{spec.name}_cotrain_{mode}_{split}"
    else:
        name = f"arx_eef_{spec.name}_{label}_{mode}_{split}"
    return base / name


def materialize(spec: TaskSpec, output_root: Path, args: argparse.Namespace, *, camera_dropout: bool, probabilities: np.ndarray) -> list[dict[str, Any]]:
    roots = {"ego": spec.ego_root, "real": spec.real_root}
    infos = {source: read_json(roots[source] / "meta" / "info.json") for source in args.sources}
    fps_values = {int(info["fps"]) for info in infos.values()}
    if len(fps_values) != 1:
        raise ValueError(f"FPS mismatch between selected sources for {spec.name}")
    for info in infos.values():
        for key in ("observation.state", "action"):
            if info["features"][key]["shape"] != [JOINT_DIM]:
                raise ValueError(f"{key} source must be {JOINT_DIM}D")
        if any(key not in info["features"] for key in VIDEO_KEYS):
            raise ValueError("source dataset must contain all three camera streams")

    real_tables: list[tuple[Path, pa.Table]] = []
    real_ids: set[int] = set()
    if "real" in args.sources:
        real_tables = [(path, pq.read_table(path)) for path in episode_files(spec.real_root)]
        all_real_ids = {source_episode_index(table, path) for path, table in real_tables}
        if args.real_ids is not None:
            requested = set(args.real_ids)
            unknown = requested.difference(all_real_ids)
            if unknown:
                raise ValueError(f"unknown --real-ids for {spec.name}: {sorted(unknown)}")
            real_tables = [(path, table) for path, table in real_tables if source_episode_index(table, path) in requested]
        real_ids = {source_episode_index(table, path) for path, table in real_tables}
        if not real_ids:
            raise ValueError(f"no real episodes selected for {spec.name}")

    if args.single_train_split:
        train_ids, split_source = real_ids, "all selected sources placed in train"
        splits = ("train",)
    else:
        if "real" not in args.sources:
            raise ValueError("--sources ego requires --single-train-split; otherwise eval would be empty")
        train_ids, _, split_source = resolve_real_split(spec, real_ids, args)
        splits = ("train", "eval")
    mode, active_mask = ("camdrop" if camera_dropout else "nodropout"), eef_mask(spec.active_sides)
    base = output_root / "local"
    source_info = next(iter(infos.values()))
    writers = {
        split: DatasetWriter(
            output_dataset_root(base, spec, mode, split, args.dataset_label),
            source_info,
            spec,
            split,
            mode,
            active_mask,
            split_source,
            args.overwrite,
        )
        for split in splits
    }
    fk, rng = ArxForwardKinematics(args.scene, args.calibration), np.random.default_rng(args.seed + sum(map(ord, spec.name)))
    sources = []
    if "ego" in args.sources:
        sources.append(("ego", spec.ego_root, [(path, pq.read_table(path)) for path in episode_files(spec.ego_root)]))
    if "real" in args.sources:
        sources.append(("real", spec.real_root, real_tables))
    for source, root, tables in sources:
        for path, table in tables:
            source_ep = source_episode_index(table, path)
            selected = np.flatnonzero(valid_rows(table, source, spec.active_sides))
            if not len(selected):
                continue
            joints = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            # Convert each original episode once so that state[t + 1] and
            # action[t] share exactly the same quaternion hemisphere.
            eef = apply_arm_mask(joints_to_eef(joints, fk), active_mask)
            state, action = eef[selected], eef[selected + 1]
            timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)[selected]
            frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)[selected]
            if source == "ego":
                split, image_mask = "train", np.tile(np.asarray([1, 0, 0], dtype=bool), (len(selected), 1))
            else:
                split = "train" if source_ep in train_ids else "eval"
                image_mask = choose_real_image_masks(len(selected), camera_dropout and split == "train", probabilities, rng)
            writers[split].add(source=source, source_root=root, source_episode=source_ep, selected=selected, state=state, action=action, timestamp=timestamp, frame_index=frames, image_mask=image_mask)
    return [writers[split].finish() for split in splits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument("--task-name", default=None, help="Custom real-only task ID used in the output dataset name.")
    parser.add_argument("--real-dataset", type=Path, default=None, help="Custom raw LeRobot root. Use with --task-name for a new real-only task.")
    parser.add_argument("--ego-experiment", default=None, help="Override the ego experiment directory for every selected task.")
    parser.add_argument("--ego-variant", default=None, help="Override the ego dataset variant directory for every selected task.")
    parser.add_argument("--sources", nargs="+", choices=("ego", "real"), default=("ego", "real"), help="Sources to include (default: ego real).")
    parser.add_argument("--real-ids", nargs="*", type=int, default=None, help="Optional real episode IDs to include; useful with --single-train-split.")
    parser.add_argument("--single-train-split", action="store_true", help="Write only a train repository and place every selected episode in it.")
    parser.add_argument("--dataset-label", default=None, help="Name inserted after task, e.g. real_all; avoids overwriting the default cotrain names.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-modes", nargs="+", choices=("no_dropout", "camera_dropout"), default=("no_dropout", "camera_dropout"))
    parser.add_argument("--real-camera-mask-probs", nargs=4, type=float, default=(0.50, 0.25, 0.125, 0.125), metavar=("FULL", "HEAD", "HEAD_LEFT", "HEAD_RIGHT"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-train-ids", nargs="*", type=int, default=None)
    parser.add_argument("--real-eval-ids", nargs="*", type=int, default=None)
    parser.add_argument("--real-train-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--allow-correction-split-override", action="store_true")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    if len(set(args.sources)) != len(args.sources):
        parser.error("--sources cannot contain duplicates")
    if args.dataset_label is not None and (not args.dataset_label or "/" in args.dataset_label):
        parser.error("--dataset-label must be a non-empty name without '/'")
    if (args.real_train_ids is not None or args.real_eval_ids is not None) and args.real_train_ratio is not None:
        parser.error("choose explicit episode IDs or --real-train-ratio, not both")
    probabilities = np.asarray(args.real_camera_mask_probs, dtype=np.float64)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        parser.error("--real-camera-mask-probs must be non-negative and sum to one")
    args.output_root, args.scene, args.calibration = args.output_root.resolve(), args.scene.resolve(), args.calibration.resolve()
    custom_values = (args.task_name, args.real_dataset)
    if any(value is not None for value in custom_values):
        if not all(value is not None for value in custom_values):
            parser.error("provide both --task-name and --real-dataset for a custom task")
        if args.tasks is not None:
            parser.error("--task-name/--real-dataset cannot be combined with --tasks")
        if tuple(args.sources) != ("real",):
            parser.error("a custom task currently supports only --sources real")
        if not args.single_train_split:
            parser.error("a custom task requires --single-train-split")
        if not args.task_name or "/" in args.task_name:
            parser.error("--task-name must be a non-empty name without '/'")
        real_root = args.real_dataset.expanduser().resolve()
        prompt = dataset_single_task_prompt(real_root)
        specs = [TaskSpec(args.task_name, "custom", "unused", prompt, ("right",), real_root)]
    else:
        specs = [TASKS[name] for name in (args.tasks or sorted(TASKS))]
    result = []
    for mode in args.dataset_modes:
        for original_spec in specs:
            spec = original_spec
            if args.ego_experiment is not None or args.ego_variant is not None:
                spec = replace(
                    spec,
                    experiment=args.ego_experiment or spec.experiment,
                    variant=args.ego_variant or spec.variant,
                )
            result.extend(materialize(spec, args.output_root, args, camera_dropout=mode == "camera_dropout", probabilities=probabilities))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
