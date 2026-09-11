#!/usr/bin/env python3
"""Convert the repository's ARX LeRobot-v3 export to OpenPI-compatible metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import serialize_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "DATA" / "stack_cube_ego"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "lerobot"
DEFAULT_REPO_ID = "local/arx_stack_cube_ego"


def _read_records(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _episode_stats(record: dict, feature_keys: set[str]) -> dict:
    stats = {}
    prefix = "stats/"
    for key, value in record.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        feature_name, stat_name = remainder.rsplit("/", 1)
        if feature_name not in feature_keys or stat_name not in {"min", "max", "mean", "std", "count"}:
            continue
        stats.setdefault(feature_name, {})[stat_name] = value

    for feature_name, feature_stats in stats.items():
        for stat_name, value in feature_stats.items():
            feature_stats[stat_name] = np.asarray(value)
        if set(feature_stats) != {"min", "max", "mean", "std", "count"}:
            raise ValueError(f"incomplete episode stats for {feature_name}: {sorted(feature_stats)}")
    return stats


def _copy_or_fail(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".parquet":
        # The source exporter stores a newer Hugging Face ``List`` feature
        # descriptor. The OpenPI environment uses a datasets version that can
        # infer the same fixed-size vectors directly from the Arrow schema.
        table = pq.read_table(source).replace_schema_metadata(None)
        pq.write_table(table, destination)
    else:
        shutil.copy2(source, destination)


def _constant_stats(value: int, length: int) -> dict[str, np.ndarray]:
    scalar = np.asarray([value])
    return {
        "min": scalar,
        "max": scalar,
        "mean": scalar.astype(np.float64),
        "std": np.zeros(1, dtype=np.float64),
        "count": np.asarray([length]),
    }


def _range_stats(start: int, length: int) -> dict[str, np.ndarray]:
    end = start + length - 1
    values = np.arange(start, start + length, dtype=np.float64)
    return {
        "min": np.asarray([start]),
        "max": np.asarray([end]),
        "mean": np.asarray([values.mean()]),
        "std": np.asarray([values.std()]),
        "count": np.asarray([length]),
    }


def _rewrite_episode_table(
    source: Path,
    destination: Path,
    *,
    episode_index: int,
    global_index: int,
    task_index: int,
) -> int:
    table = pq.read_table(source)
    length = table.num_rows
    required_columns = {"episode_index", "index", "task_index"}
    missing = required_columns - set(table.column_names)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    updates = {
        "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
        "index": pa.array(np.arange(global_index, global_index + length, dtype=np.int64)),
        "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
    }
    for name, values in updates.items():
        table = table.set_column(table.schema.get_field_index(name), name, values)
    table = table.replace_schema_metadata(None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    return length


def _load_source_metadata(source_root: Path) -> tuple[dict, list[dict], list[dict]]:
    source_root = source_root.expanduser().resolve()
    info_path = source_root / "meta" / "info.json"
    tasks_path = source_root / "meta" / "tasks.parquet"
    stats_path = source_root / "meta" / "stats.json"
    episodes_root = source_root / "meta" / "episodes"

    for path in (info_path, tasks_path, stats_path, episodes_root):
        if not path.exists():
            raise FileNotFoundError(f"missing source metadata: {path}")

    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    source_tasks = _read_records(tasks_path)
    episode_meta_paths = sorted(episodes_root.rglob("*.parquet"))
    episode_meta = []
    for path in episode_meta_paths:
        episode_meta.extend(_read_records(path))
    episode_meta.sort(key=lambda record: int(record["episode_index"]))
    if not episode_meta:
        raise ValueError(f"no episode metadata found under {episodes_root}")

    expected_episodes = int(source_info["total_episodes"])
    if len(episode_meta) != expected_episodes:
        raise ValueError(f"info.json says {expected_episodes} episodes, found {len(episode_meta)}")
    return source_info, source_tasks, episode_meta


def build_dataset(
    source_roots: list[Path],
    output_root: Path,
    repo_id: str,
    overwrite: bool,
) -> Path:
    if not source_roots:
        raise ValueError("at least one source dataset is required")
    sources = [
        (source_root.expanduser().resolve(), *_load_source_metadata(source_root))
        for source_root in source_roots
    ]
    dataset_root = (output_root.expanduser().resolve() / repo_id).resolve()
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"dataset already exists: {dataset_root}; pass --overwrite")
        shutil.rmtree(dataset_root)

    first_info = sources[0][1]
    first_feature_keys = set(first_info["features"])
    first_video_keys = [
        key for key, feature in first_info["features"].items() if feature.get("dtype") == "video"
    ]
    first_fps = first_info.get("fps")
    for source_root, source_info, _, _ in sources[1:]:
        if set(source_info["features"]) != first_feature_keys:
            raise ValueError(f"feature keys differ between source datasets: {source_root}")
        if source_info["features"] != first_info["features"]:
            raise ValueError(f"feature definitions differ between source datasets: {source_root}")
        video_keys = [
            key for key, feature in source_info["features"].items() if feature.get("dtype") == "video"
        ]
        if video_keys != first_video_keys:
            raise ValueError(f"video keys differ between source datasets: {source_root}")
        if source_info.get("fps") != first_fps:
            raise ValueError(f"FPS differs between source datasets: {source_root}")

    task_names: list[str] = []
    task_indices: dict[str, int] = {}
    source_task_maps: list[dict[int, str]] = []
    for _, _, source_tasks, _ in sources:
        task_map = {}
        for item in source_tasks:
            task_name = str(item["task"])
            task_map[int(item["task_index"])] = task_name
            if task_name not in task_indices:
                task_indices[task_name] = len(task_names)
                task_names.append(task_name)
        source_task_maps.append(task_map)

    total_episodes = sum(len(episode_meta) for _, _, _, episode_meta in sources)
    total_frames = sum(
        int(item["length"])
        for _, _, _, episode_meta in sources
        for item in episode_meta
    )
    output_info = dict(first_info)
    output_info.update(
        {
            "codebase_version": "v2.1",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(task_names),
            "total_videos": len(first_video_keys),
            "total_chunks": (total_episodes + 999) // 1000,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4",
            "robot_type": "arx5_2025_bimanual",
            "splits": {"train": f"0:{total_episodes}"},
        }
    )

    dataset_root.mkdir(parents=True)
    (dataset_root / "meta").mkdir()
    (dataset_root / "data").mkdir()
    (dataset_root / "videos").mkdir()
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
    next_episode_index = 0
    global_index = 0
    for source_number, (source_root, source_info, _, episode_meta) in enumerate(sources):
        source_task_map = source_task_maps[source_number]
        for item in episode_meta:
            source_episode_index = int(item["episode_index"])
            tasks = []
            for task in item["tasks"]:
                task_name = source_task_map[int(task)] if isinstance(task, int) else str(task)
                if task_name not in tasks:
                    tasks.append(task_name)
            if not tasks:
                raise ValueError(f"episode {source_episode_index} in {source_root} has no task")
            task_index = task_indices[tasks[0]]
            length = int(item["length"])
            episode_records.append(
                {
                    "episode_index": next_episode_index,
                    "tasks": tasks,
                    "length": length,
                }
            )
            stats = _episode_stats(item, set(source_info["features"]))
            stats["episode_index"] = _constant_stats(next_episode_index, length)
            stats["index"] = _range_stats(global_index, length)
            stats["task_index"] = _constant_stats(task_index, length)
            episode_stats.append(
                {
                    "episode_index": next_episode_index,
                    "stats": serialize_dict(stats),
                }
            )
            next_episode_index += 1
            global_index += length
    _write_jsonl(dataset_root / "meta" / "episodes.jsonl", episode_records)
    _write_jsonl(dataset_root / "meta" / "episodes_stats.jsonl", episode_stats)
    aggregate = aggregate_stats(
        [
            {key: {stat: np.asarray(value) for stat, value in values.items()} for key, values in item["stats"].items()}
            for item in episode_stats
        ]
    )
    (dataset_root / "meta" / "stats.json").write_text(
        json.dumps(serialize_dict(aggregate), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    episode_index = 0
    global_index = 0
    for source_root, _, _, episode_meta in sources:
        for item in episode_meta:
            source_data = (
                source_root
                / "data"
                / f"chunk-{int(item['data/chunk_index']):03d}"
                / f"file-{int(item['data/file_index']):03d}.parquet"
            )
            destination_data = (
                dataset_root
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"file-{episode_index:03d}.parquet"
            )
            task_name = episode_records[episode_index]["tasks"][0]
            _rewrite_episode_table(
                source_data,
                destination_data,
                episode_index=episode_index,
                global_index=global_index,
                task_index=task_indices[task_name],
            )

            for video_key in first_video_keys:
                source_video = (
                    source_root
                    / "videos"
                    / video_key
                    / f"chunk-{int(item[f'videos/{video_key}/chunk_index']):03d}"
                    / f"file-{int(item[f'videos/{video_key}/file_index']):03d}.mp4"
                )
                destination_video = (
                    dataset_root
                    / "videos"
                    / video_key
                    / f"chunk-{episode_index // 1000:03d}"
                    / f"file-{episode_index:03d}.mp4"
                )
                _copy_or_fail(source_video, destination_video)
            length = int(item["length"])
            episode_index += 1
            global_index += length

    print(f"Created: {dataset_root}")
    print(f"Episodes: {total_episodes}")
    print(f"Frames: {output_info['total_frames']}")
    print(f"Cameras: {', '.join(first_video_keys)}")
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        help="Source dataset root; repeat for co-training datasets.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_roots = args.source_roots or [str(DEFAULT_SOURCE_ROOT)]
    build_dataset(
        [Path(path) for path in source_roots],
        Path(args.output_root),
        args.repo_id,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
