#!/usr/bin/env python3
"""Merge one or more compatible local LeRobot datasets into one dataset.

The source repositories are read-only.  Episodes are appended in the order in
which ``--source-root`` is supplied and receive fresh episode/global indices;
task indices are remapped by task text.  All non-index parquet columns are
preserved.  Video files can be hard-linked (the default) or copied.

Example:
  python training/merge_lerobot_datasets.py \
    --source-root /data/run_a --source-root /data/run_b --source-root /data/run_c \
    --output-root outputs/lerobot --repo-id local/stack_cola_real_merged
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_write(path: Path, items: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_tasks(root: Path) -> list[dict[str, Any]]:
    tasks = jsonl_read(root / "meta" / "tasks.jsonl")
    if tasks:
        return tasks
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        return pq.read_table(parquet).to_pylist()
    raise FileNotFoundError(f"missing tasks metadata under {root / 'meta'}")


def load_episodes(root: Path) -> list[dict[str, Any]]:
    episodes = jsonl_read(root / "meta" / "episodes.jsonl")
    if not episodes:
        paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        for path in paths:
            episodes.extend(pq.read_table(path).to_pylist())
    if not episodes:
        raise FileNotFoundError(f"missing episode metadata under {root / 'meta'}")
    return sorted(episodes, key=lambda item: int(item["episode_index"]))


def data_path(root: Path, episode: dict[str, Any]) -> Path:
    if "data/chunk_index" in episode and "data/file_index" in episode:
        return root / "data" / f"chunk-{int(episode['data/chunk_index']):03d}" / f"file-{int(episode['data/file_index']):03d}.parquet"
    index = int(episode["episode_index"])
    return root / "data" / f"chunk-{index // 1000:03d}" / f"file-{index:03d}.parquet"


def video_path(root: Path, episode: dict[str, Any], key: str) -> Path:
    prefix = f"videos/{key}/"
    if f"{prefix}chunk_index" in episode and f"{prefix}file_index" in episode:
        return root / "videos" / key / f"chunk-{int(episode[f'{prefix}chunk_index']):03d}" / f"file-{int(episode[f'{prefix}file_index']):03d}.mp4"
    index = int(episode["episode_index"])
    return root / "videos" / key / f"chunk-{index // 1000:03d}" / f"file-{index:03d}.mp4"


def video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if isinstance(feature, dict) and feature.get("dtype") == "video"]


def replace_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    try:
        index = table.schema.get_field_index(name)
    except AttributeError:
        index = table.column_names.index(name) if name in table.column_names else -1
    if index < 0:
        raise ValueError(f"source parquet is missing required column {name!r}")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing source video: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def task_names(episode: dict[str, Any], source_tasks: dict[int, str]) -> list[str]:
    values = episode.get("tasks", [])
    result = []
    for value in values:
        name = source_tasks.get(int(value), str(value)) if isinstance(value, (int, np.integer)) else str(value)
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError(f"episode {episode.get('episode_index')} has no tasks")
    return result


def merged_stats(roots: list[Path]) -> dict[str, Any] | None:
    """Combine LeRobot min/max/mean/std/count statistics exactly."""
    paths = [root / "meta" / "stats.json" for root in roots]
    if not any(path.is_file() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise ValueError("either every source must have meta/stats.json or none may have it")
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(set(value) != set(inputs[0]) for value in inputs[1:]):
        raise ValueError("stats feature keys differ between source datasets")
    output: dict[str, Any] = {}
    for feature in inputs[0]:
        records = [value[feature] for value in inputs]
        if any(set(record) != {"min", "max", "mean", "std", "count"} for record in records):
            raise ValueError(f"unsupported stats structure for {feature!r}")
        arrays = {name: [np.asarray(record[name], dtype=np.float64) for record in records] for name in records[0]}
        shape = arrays["mean"][0].shape
        if any(value.shape != shape for values in arrays.values() for value in values):
            raise ValueError(f"stats shapes differ for {feature!r}")
        counts = np.stack(arrays["count"])
        if np.any(counts <= 0):
            raise ValueError(f"non-positive stats count for {feature!r}")
        means = np.stack(arrays["mean"])
        total = np.sum(counts, axis=0)
        mean = np.sum(means * counts, axis=0) / total
        variance = np.sum(counts * (np.stack(arrays["std"]) ** 2 + (means - mean) ** 2), axis=0) / total
        output[feature] = {
            "min": np.min(np.stack(arrays["min"]), axis=0).tolist(),
            "max": np.max(np.stack(arrays["max"]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 0)).tolist(),
            "count": total.tolist(),
        }
    return output


def merge(args: argparse.Namespace) -> Path:
    roots = [Path(value).expanduser().resolve() for value in args.source_roots]
    repo_id = Path(args.repo_id)
    if repo_id.is_absolute() or ".." in repo_id.parts:
        raise ValueError("--repo-id must be a relative path without '..'")
    output = (Path(args.output_root).expanduser().resolve() / repo_id).resolve()
    if output in roots:
        raise ValueError("output dataset must not be one of the source datasets")
    infos = []
    for root in roots:
        path = root / "meta" / "info.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset metadata: {path}")
        infos.append(json.loads(path.read_text(encoding="utf-8")))
    first = infos[0]
    keys = video_keys(first)
    for root, info in zip(roots[1:], infos[1:], strict=True):
        if info.get("features") != first.get("features"):
            raise ValueError(f"feature schemas differ: {roots[0]} vs {root}")
        if info.get("fps") != first.get("fps"):
            raise ValueError(f"FPS differs: {roots[0]} vs {root}")
        if video_keys(info) != keys:
            raise ValueError(f"video features differ: {roots[0]} vs {root}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)
    (output / "videos").mkdir(parents=True)

    global_index, output_episode = 0, 0
    output_episodes: list[dict[str, Any]] = []
    output_tasks: dict[str, int] = {}
    provenance: list[dict[str, Any]] = []
    parquet_schema: pa.Schema | None = None
    for root, info in zip(roots, infos, strict=True):
        source_tasks = {int(row["task_index"]): str(row["task"]) for row in load_tasks(root)}
        for episode in load_episodes(root):
            source_path = data_path(root, episode)
            table = pq.read_table(source_path)
            if not len(table):
                raise ValueError(f"empty episode parquet: {source_path}")
            if parquet_schema is None:
                parquet_schema = table.schema.remove_metadata()
            elif table.schema.remove_metadata() != parquet_schema:
                raise ValueError(f"parquet schemas differ: {source_path}")
            names = task_names(episode, source_tasks)
            for name in names:
                output_tasks.setdefault(name, len(output_tasks))
            source_ids = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
            remap = {source_id: output_tasks[name] for source_id, name in source_tasks.items()}
            unknown = set(map(int, source_ids)).difference(remap)
            if unknown:
                raise ValueError(f"{source_path} references unknown task indices: {sorted(unknown)}")
            length = len(table)
            table = replace_column(table, "episode_index", np.full(length, output_episode, dtype=np.int64))
            table = replace_column(table, "index", np.arange(global_index, global_index + length, dtype=np.int64))
            table = replace_column(table, "frame_index", np.arange(length, dtype=np.int64))
            table = replace_column(table, "task_index", np.asarray([remap[int(value)] for value in source_ids], dtype=np.int64))
            destination = output / "data" / f"chunk-{output_episode // 1000:03d}" / f"file-{output_episode:03d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.replace_schema_metadata(None), destination, compression="zstd")
            for key in keys:
                link_or_copy(video_path(root, episode, key), output / "videos" / key / f"chunk-{output_episode // 1000:03d}" / f"file-{output_episode:03d}.mp4", args.video_mode)
            output_episodes.append({"episode_index": output_episode, "tasks": [output_tasks[name] for name in names], "length": length})
            provenance.append({"output_episode_index": output_episode, "source_dataset": str(root), "source_episode_index": int(episode["episode_index"]), "frames": length})
            global_index += length
            output_episode += 1

    info = dict(first)
    info.update({"total_episodes": output_episode, "total_frames": global_index, "total_tasks": len(output_tasks), "total_videos": len(keys), "total_chunks": (output_episode + 999) // 1000, "chunks_size": 1000, "splits": {args.split: f"0:{output_episode}"}})
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    jsonl_write(output / "meta" / "tasks.jsonl", ({"task_index": index, "task": name} for name, index in output_tasks.items()))
    jsonl_write(output / "meta" / "episodes.jsonl", output_episodes)
    jsonl_write(output / "meta" / "episodes_stats.jsonl", [])
    stats = merged_stats(roots)
    if stats is not None:
        (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "merge_provenance.json").write_text(json.dumps({"schema": "tate.merged_lerobot_dataset", "schema_version": 1, "sources": [str(root) for root in roots], "video_mode": args.video_mode, "episodes": provenance}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(output), "episodes": output_episode, "frames": global_index, "tasks": len(output_tasks)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", dest="source_roots", required=True, help="Input LeRobot root; repeat for every dataset, in append order.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "lerobot")
    parser.add_argument("--repo-id", required=True, help="Output directory relative to --output-root, e.g. local/my_merged_dataset.")
    parser.add_argument("--split", default="train", help="Name of the single output split (default: train).")
    parser.add_argument("--video-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
