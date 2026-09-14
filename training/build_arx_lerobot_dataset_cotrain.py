#!/usr/bin/env python3
"""Concatenate ARX EEF LeRobot datasets into one OpenPI-ready cotrain dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from training.build_arx_lerobot_dataset_from_joint import (
        EEF_DIM,
        PreparedEpisode,
        _task_map,
        _video_keys,
        as_abs,
        episode_tasks,
        load_dataset_metadata,
        source_video_path,
        write_lerobot_dataset,
    )
except ImportError:
    from build_arx_lerobot_dataset_from_joint import (
        EEF_DIM,
        PreparedEpisode,
        _task_map,
        _video_keys,
        as_abs,
        episode_tasks,
        load_dataset_metadata,
        source_video_path,
        write_lerobot_dataset,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EGO_ROOT = REPO_ROOT / "outputs" / "lerobot" / "local" / "arx_eef_stack_cube_ego"
DEFAULT_REAL_ROOT = REPO_ROOT / "outputs" / "lerobot" / "local" / "arx_eef_stack_cube_arx"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "lerobot"
DEFAULT_REPO_ID = "local/arx_eef_stack_cube_cotrain"


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


def _json_arrays_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_arrays_to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _load_episode_stats(root: Path) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    path = root / "meta" / "episodes_stats.jsonl"
    if not path.is_file():
        return {}
    stats = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stats[int(record["episode_index"])] = _json_arrays_to_numpy(record["stats"])
    return stats


def _load_prepared_episodes(root: Path) -> tuple[dict[str, Any], list[PreparedEpisode]]:
    info, tasks, episode_meta = load_dataset_metadata(root)
    task_by_index = _task_map(tasks)
    video_keys = _video_keys(info)
    if not video_keys:
        raise ValueError(f"{root} has no video features")
    episode_stats = _load_episode_stats(root)

    episodes = []
    for item in episode_meta:
        data_path = _source_data_path(root, item)
        table = pq.read_table(data_path)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.shape != action.shape or state.shape[-1] != EEF_DIM:
            raise ValueError(f"{data_path} is not a {EEF_DIM}D EEF dataset: {state.shape}, {action.shape}")
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
                source_videos={key: source_video_path(root, item, key) for key in video_keys},
                source_stats=episode_stats.get(int(item["episode_index"]), {}),
            )
        )
    return info, episodes


def build_dataset(source_roots: list[Path], output_root: Path, repo_id: str, overwrite: bool) -> Path:
    if len(source_roots) < 2:
        raise ValueError("provide at least two --source-root datasets for cotrain")

    loaded = [_load_prepared_episodes(root.expanduser().resolve()) for root in source_roots]
    first_info = loaded[0][0]
    first_video_keys = _video_keys(first_info)
    for root, (info, _) in zip(source_roots, loaded, strict=True):
        if _video_keys(info) != first_video_keys:
            raise ValueError(f"video features differ for {root}")
        if int(info.get("fps", 30)) != int(first_info.get("fps", 30)):
            raise ValueError(f"FPS differs for {root}")
    episodes = [episode for _, source_episodes in loaded for episode in source_episodes]
    convention = dict(first_info.get("arx_eef") or {})
    convention["cotrain_sources"] = [str(root) for root in source_roots]
    return write_lerobot_dataset(
        source_info=first_info,
        episodes=episodes,
        output_root=output_root,
        repo_id=repo_id,
        overwrite=overwrite,
        eef_convention=convention,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        default=None,
        help="Converted 16D EEF LeRobot dataset root; repeat for ego and real datasets.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_roots = [as_abs(path) for path in (args.source_roots or [str(DEFAULT_EGO_ROOT), str(DEFAULT_REAL_ROOT)])]
    build_dataset(source_roots, as_abs(args.output_root), args.repo_id, args.overwrite)


if __name__ == "__main__":
    main()
