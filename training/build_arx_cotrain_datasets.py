#!/usr/bin/env python3
"""Materialize task-specific ARX ego/real cotrain datasets.

The source ego variant remains immutable.  This exporter creates a compact
LeRobot-v3-style dataset per task with a common 14-D ARX joint space:

    [left joints(6), left gripper, right joints(6), right gripper]

``action`` is always the next source-frame state.  Single-arm episodes retain the
14-D representation but zero inactive-arm state/action entries and expose
per-dimension masks.  A policy trainer must apply ``policy.action_mask`` to
its loss; zeros are placeholders, not supervision targets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# This file lives at <workspace>/TATE/training/.
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "cotrain"
VIDEO_KEYS = (
    "observation.images.head",
    "observation.images.left",
    "observation.images.right",
)
IMAGE_MASK_NAMES = ["head", "left", "right"]
JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(1, 7)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(1, 7)),
    "right_gripper",
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    experiment: str
    variant: str
    prompt: str
    active_sides: tuple[str, ...]

    @property
    def ego_root(self) -> Path:
        return REPO_ROOT / "outputs" / "experiments" / self.experiment / "datasets" / self.variant

    @property
    def real_root(self) -> Path:
        return REPO_ROOT.parent / "Data_TATE" / f"{self.name}_arx"

    @property
    def correction_path(self) -> Path:
        return REPO_ROOT / "outputs" / "corrections" / self.name / "xyz_mean_target_min_bending.json"


TASKS = {
    "stack_cube": TaskSpec(
        name="stack_cube",
        experiment="stack_cube_h2g_ablation_v4",
        variant="finger_center_hys085__xyz_mean_target_min_bending",
        prompt="pick the cube and stack it on the blue plate",
        active_sides=("right",),
    ),
    "stack_cola": TaskSpec(
        name="stack_cola",
        experiment="stack_cola_h2g_ablation_v4",
        variant="finger_center_hys085__xyz_mean_target_min_bending",
        prompt="pick two colas and stack them on the brown box",
        active_sides=("left", "right"),
    ),
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def vector_mask(active_sides: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(14, dtype=bool)
    if "left" in active_sides:
        mask[:7] = True
    if "right" in active_sides:
        mask[7:] = True
    return mask


def episode_files(dataset_root: Path) -> Iterable[Path]:
    return sorted((dataset_root / "data" / "chunk-000").glob("file-*.parquet"))


def source_episode_index(table: pa.Table, path: Path) -> int:
    ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if len(ids) == 0 or np.any(ids != ids[0]):
        raise ValueError(f"{path} must contain exactly one non-empty episode")
    return int(ids[0])


def valid_rows(table: pa.Table, *, source: str, active_sides: tuple[str, ...]) -> np.ndarray:
    """Return t indices whose current and next state are usable."""
    n = len(table)
    keep = np.ones(n - 1, dtype=bool)  # drop final frame: no true next state
    if source != "ego":
        return keep
    for side in active_sides:
        key = f"tate.eef.{side}.valid"
        if key not in table.column_names:
            raise ValueError(f"ego table is missing required validity column {key!r}")
        valid = np.asarray(table[key].to_pylist(), dtype=bool)
        keep &= valid[:-1] & valid[1:]
    return keep


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def source_video(dataset_root: Path, key: str, source_episode: int) -> Path:
    path = dataset_root / "videos" / key / "chunk-000" / f"file-{source_episode:03d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_real_image_masks(
    count: int,
    *,
    camera_dropout: bool,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return [head, left, right] availability masks for real frames.

    The four probability entries correspond to full, head-only, head+left,
    and head+right views.  Ego always has the fixed head-only mask.
    """
    if not camera_dropout:
        return np.ones((count, 3), dtype=bool)
    choices = rng.choice(4, size=count, p=probabilities)
    patterns = np.asarray(((1, 1, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1)), dtype=bool)
    return patterns[choices]


def materialize(
    spec: TaskSpec,
    output_root: Path,
    overwrite: bool,
    *,
    camera_dropout: bool,
    camera_mask_probabilities: np.ndarray,
    seed: int,
) -> dict:
    ego_info = read_json(spec.ego_root / "meta" / "info.json")
    real_info = read_json(spec.real_root / "meta" / "info.json")
    if ego_info["fps"] != real_info["fps"]:
        raise ValueError(f"FPS mismatch: ego={ego_info['fps']} real={real_info['fps']}")
    for key in ("observation.state", "action"):
        if ego_info["features"][key]["shape"] != [14] or real_info["features"][key]["shape"] != [14]:
            raise ValueError(f"{key} must be 14-D in both source datasets")
    for key in VIDEO_KEYS:
        if ego_info["features"][key]["shape"] != real_info["features"][key]["shape"]:
            raise ValueError(f"video shape mismatch for {key}")

    dataset_mode = "camdrop" if camera_dropout else "nodropout"
    target = output_root / f"{spec.name}_ego_real_fc085_xyz_{dataset_mode}_v1"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; use --overwrite to replace it")
        shutil.rmtree(target)

    correction = read_json(spec.correction_path)
    real_train_ids = {int(x) for x in correction["real_calibration_episode_ids"]}
    real_eval_ids = {int(x) for x in correction["real_eval_episode_ids_not_used"]}
    inactive = ~vector_mask(spec.active_sides)
    action_mask = (~inactive).astype(bool)
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in spec.name))

    output_index = 0
    output_episode = 0
    episode_rows: list[dict] = []
    provenance: list[dict] = []
    counts = {"ego": {"episodes": 0, "frames": 0}, "real_train": {"episodes": 0, "frames": 0}, "real_eval": {"episodes": 0, "frames": 0}}
    state_sum = np.zeros(14, dtype=np.float64)
    state_sumsq = np.zeros(14, dtype=np.float64)
    action_sum = np.zeros(14, dtype=np.float64)
    action_sumsq = np.zeros(14, dtype=np.float64)
    state_min = np.full(14, np.inf)
    state_max = np.full(14, -np.inf)
    action_min = np.full(14, np.inf)
    action_max = np.full(14, -np.inf)
    image_mask_counts = {"head_left_right": 0, "head_only": 0, "head_left": 0, "head_right": 0}

    for source, source_root in (("ego", spec.ego_root), ("real", spec.real_root)):
        for parquet_path in episode_files(source_root):
            table = pq.read_table(parquet_path)
            source_ep = source_episode_index(table, parquet_path)
            keep = valid_rows(table, source=source, active_sides=spec.active_sides)
            source_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            source_ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
            source_frame = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
            selected = np.flatnonzero(keep)
            if len(selected) == 0:
                print(f"[skip] {spec.name} {source} episode {source_ep}: no valid transitions")
                continue

            state = source_state[selected].copy()
            action = source_state[selected + 1].copy()
            # Inactive values are model-input placeholders and must never be supervised.
            state[:, inactive] = 0.0
            action[:, inactive] = 0.0
            n = len(selected)
            split = "train" if source == "ego" or source_ep in real_train_ids else "eval"
            if source == "real" and source_ep not in real_train_ids | real_eval_ids:
                raise ValueError(f"real episode {source_ep} is absent from correction split")
            if source == "ego":
                image_mask = np.zeros((n, 3), dtype=bool)
                image_mask[:, 0] = True
            else:
                image_mask = choose_real_image_masks(
                    n,
                    camera_dropout=camera_dropout and split == "train",
                    probabilities=camera_mask_probabilities,
                    rng=rng,
                )
            for pattern, key in (([1, 1, 1], "head_left_right"), ([1, 0, 0], "head_only"), ([1, 1, 0], "head_left"), ([1, 0, 1], "head_right")):
                image_mask_counts[key] += int(np.all(image_mask == pattern, axis=1).sum())

            columns = {
                "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), type=pa.float32()), 14),
                "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1), type=pa.float32()), 14),
                "timestamp": pa.array(source_ts[selected], type=pa.float32()),
                "frame_index": pa.array(source_frame[selected], type=pa.int64()),
                "episode_index": pa.array(np.full(n, output_episode, dtype=np.int64)),
                "index": pa.array(np.arange(output_index, output_index + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
                "policy.state_mask": pa.FixedSizeListArray.from_arrays(pa.array(np.tile(action_mask, n).reshape(-1), type=pa.bool_()), 14),
                "policy.action_mask": pa.FixedSizeListArray.from_arrays(pa.array(np.tile(action_mask, n).reshape(-1), type=pa.bool_()), 14),
                "policy.image_mask": pa.FixedSizeListArray.from_arrays(pa.array(image_mask.reshape(-1), type=pa.bool_()), 3),
                "policy.domain": pa.array(np.full(n, 0 if source == "ego" else 1, dtype=np.int8)),
                "policy.split": pa.array([split] * n, type=pa.string()),
                "policy.source_episode_index": pa.array(np.full(n, source_ep, dtype=np.int64)),
                "policy.source_frame_index": pa.array(source_frame[selected], type=pa.int64()),
            }
            output_path = target / "data" / "chunk-000" / f"file-{output_episode:03d}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(columns), output_path, compression="zstd")

            for key in VIDEO_KEYS:
                link_or_copy(source_video(source_root, key, source_ep), target / "videos" / key / "chunk-000" / f"file-{output_episode:03d}.mp4")
            duration = float(source_ts[selected[-1]] + 1.0 / ego_info["fps"])
            episode_rows.append({
                "episode_index": output_episode,
                "tasks": [spec.prompt],
                "length": n,
                "data/chunk_index": 0,
                "data/file_index": output_episode,
                "dataset_from_index": output_index,
                "dataset_to_index": output_index + n,
                **{f"videos/{key}/chunk_index": 0 for key in VIDEO_KEYS},
                **{f"videos/{key}/file_index": output_episode for key in VIDEO_KEYS},
                **{f"videos/{key}/from_timestamp": float(source_ts[selected[0]]) for key in VIDEO_KEYS},
                **{f"videos/{key}/to_timestamp": duration for key in VIDEO_KEYS},
            })
            provenance.append({"output_episode_index": output_episode, "source_domain": source, "source_episode_index": source_ep, "split": split, "kept_source_frame_indices": [int(selected[0]), int(selected[-1])], "frames": n})
            bucket = "ego" if source == "ego" else f"real_{split}"
            counts[bucket]["episodes"] += 1
            counts[bucket]["frames"] += n
            for values, total, total_sq, lower, upper in ((state, state_sum, state_sumsq, state_min, state_max), (action, action_sum, action_sumsq, action_min, action_max)):
                total += values.sum(axis=0)
                total_sq += np.square(values).sum(axis=0)
                np.minimum(lower, values.min(axis=0), out=lower)
                np.maximum(upper, values.max(axis=0), out=upper)
            output_index += n
            output_episode += 1

    if output_index == 0:
        raise RuntimeError(f"{spec.name}: no transitions were written")
    meta = target / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    episodes_dir = meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episodes_dir / "file-000.parquet", compression="zstd")
    pq.write_table(pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([spec.prompt])}), meta / "tasks.parquet")
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": [14], "names": JOINT_NAMES},
        **{key: ego_info["features"][key] for key in VIDEO_KEYS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "policy.state_mask": {"dtype": "bool", "shape": [14], "names": JOINT_NAMES},
        "policy.action_mask": {"dtype": "bool", "shape": [14], "names": JOINT_NAMES},
        "policy.image_mask": {"dtype": "bool", "shape": [3], "names": IMAGE_MASK_NAMES},
        "policy.domain": {"dtype": "int8", "shape": [1], "names": ["0=ego,1=real"]},
        "policy.split": {"dtype": "string", "shape": [1], "names": None},
        "policy.source_episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "policy.source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    write_json(meta / "info.json", {"codebase_version": "v3.0", "robot_type": "arx_dual_arm_policy", "fps": ego_info["fps"], "total_episodes": output_episode, "total_frames": output_index, "features": features})
    def summary(total, total_sq, lower, upper):
        mean = total / output_index
        return {"min": lower.tolist(), "max": upper.tolist(), "mean": mean.tolist(), "std": np.sqrt(np.maximum(total_sq / output_index - np.square(mean), 0.0)).tolist(), "count": [output_index] * 14}
    write_json(meta / "stats.json", {"observation.state": summary(state_sum, state_sumsq, state_min, state_max), "action": summary(action_sum, action_sumsq, action_min, action_max)})
    write_json(
        target / "cotrain_provenance.json",
        {
            "schema": "tate.arx_cotrain_dataset",
            "schema_version": 1,
            "task": spec.name,
            "ego_variant": str(spec.ego_root),
            "real_dataset": str(spec.real_root),
            "correction": str(spec.correction_path),
            "fps": ego_info["fps"],
            "state_action_layout": JOINT_NAMES,
            "active_sides": list(spec.active_sides),
            "action_alignment": "next_source_frame_state",
            "final_frame_policy": "dropped",
            "inactive_arm_policy": "zero_input_and_target_with_loss_mask",
            "image_mask_order": IMAGE_MASK_NAMES,
            "ego_image_mask": [True, False, False],
            "real_camera_dropout": {
                "enabled": camera_dropout,
                "applied_to": "real train split only",
                "probabilities": {
                    "head_left_right": float(camera_mask_probabilities[0]),
                    "head_only": float(camera_mask_probabilities[1]),
                    "head_left": float(camera_mask_probabilities[2]),
                    "head_right": float(camera_mask_probabilities[3]),
                },
            },
            "image_mask_counts": image_mask_counts,
            "counts": counts,
            "episodes": provenance,
        },
    )
    return {"dataset": str(target), "counts": counts, "episodes": output_episode, "frames": output_index}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dataset-modes",
        nargs="+",
        choices=("no_dropout", "camera_dropout"),
        default=("no_dropout", "camera_dropout"),
        help="Materialize one or both camera-availability variants.",
    )
    parser.add_argument(
        "--real-camera-mask-probs",
        nargs=4,
        type=float,
        default=(0.50, 0.25, 0.125, 0.125),
        metavar=("FULL", "HEAD", "HEAD_LEFT", "HEAD_RIGHT"),
        help="Real-train camera-dropout probabilities; ignored by no_dropout.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic camera-dropout seed.")
    args = parser.parse_args()
    probabilities = np.asarray(args.real_camera_mask_probs, dtype=np.float64)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        parser.error("--real-camera-mask-probs must be non-negative and sum to 1")
    results = []
    for mode in args.dataset_modes:
        for name in args.tasks:
            results.append(materialize(TASKS[name], args.output_root, args.overwrite, camera_dropout=mode == "camera_dropout", camera_mask_probabilities=probabilities, seed=args.seed))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
