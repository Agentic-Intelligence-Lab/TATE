#!/usr/bin/env python3
"""Restore LeRobot v3 episode metadata for a contiguous extracted dataset."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _shift_stat(row: dict, key: str, offset: float) -> None:
    for suffix in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        field = f"stats/{key}/{suffix}"
        if field in row:
            row[field] = [float(value) + offset for value in row[field]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    source_meta = sorted((args.source / "meta" / "episodes").rglob("*.parquet"))
    source_rows = [
        row for path in source_meta for row in pq.read_table(path).to_pylist()
    ]
    by_index = {int(row["episode_index"]): row for row in source_rows}
    selected = [by_index[args.start_episode + i] for i in range(args.count)]

    destination = args.destination
    with tempfile.TemporaryDirectory(prefix=".episodes-rebuild-", dir=destination / "meta") as tmp:
        staging = Path(tmp) / "episodes" / "chunk-000"
        global_index = 0
        for derived_index, source_row in enumerate(selected):
            data_path = destination / "data" / "chunk-000" / f"file-{derived_index:03d}.parquet"
            table = pq.read_table(data_path)
            length = len(table)
            if length != int(source_row["length"]):
                raise ValueError(
                    f"episode {derived_index}: target has {length} frames, source has "
                    f"{source_row['length']}"
                )
            expected = table.select(["episode_index", "frame_index", "index"]).to_pylist()
            if (
                expected[0] != {"episode_index": derived_index, "frame_index": 0, "index": global_index}
                or expected[-1]
                != {
                    "episode_index": derived_index,
                    "frame_index": length - 1,
                    "index": global_index + length - 1,
                }
            ):
                raise ValueError(f"episode {derived_index}: target frame indexing is invalid")

            row = dict(source_row)
            source_index = int(row["episode_index"])
            row["episode_index"] = derived_index
            row["data/chunk_index"] = 0
            row["data/file_index"] = derived_index
            row["dataset_from_index"] = global_index
            row["dataset_to_index"] = global_index + length
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = derived_index
            for video_key in ("observation.images.head", "observation.images.left", "observation.images.right"):
                row[f"videos/{video_key}/chunk_index"] = 0
                row[f"videos/{video_key}/file_index"] = derived_index
            _shift_stat(row, "episode_index", derived_index - source_index)
            _shift_stat(row, "index", global_index - int(source_row["dataset_from_index"]))
            output = staging / f"file-{derived_index:03d}.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([row]), output)
            global_index += length

        if (destination / "meta" / "episodes").exists():
            raise FileExistsError(destination / "meta" / "episodes")
        shutil.move(str(Path(tmp) / "episodes"), destination / "meta" / "episodes")
    print(f"wrote {args.count} episode metadata files; total_frames={global_index}")


if __name__ == "__main__":
    main()
