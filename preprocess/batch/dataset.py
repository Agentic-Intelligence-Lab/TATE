"""LeRobot v3 episode discovery without assuming file index equals episode ID."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import pyarrow.parquet as pq


@dataclass(frozen=True)
class EpisodeSource:
    episode_index: int
    length: int
    data_path: Path
    metadata_path: Path
    metadata_row: dict[str, Any]
    video_key: str
    video_path: Path
    video_from_timestamp: float
    video_to_timestamp: float
    fps: float
    crop_start_frames: int = 0
    crop_end_frames: int = 0

    @property
    def start_frame(self) -> int:
        return max(0, int(round(self.video_from_timestamp * self.fps)))

    @property
    def processing_start_frame(self) -> int:
        return self.start_frame + self.crop_start_frames

    @property
    def effective_length(self) -> int:
        return self.length - self.crop_start_frames - self.crop_end_frames

    @property
    def source_stop_frame(self) -> int:
        return self.length - self.crop_end_frames

    @property
    def has_trim(self) -> bool:
        return bool(self.crop_start_frames or self.crop_end_frames)

    def with_trim_seconds(self, start_seconds: float, end_seconds: float) -> "EpisodeSource":
        crop_start = int(round(float(start_seconds) * self.fps))
        crop_end = int(round(float(end_seconds) * self.fps))
        if crop_start < 0 or crop_end < 0:
            raise ValueError("episode trim must be non-negative")
        if crop_start + crop_end >= self.length:
            raise ValueError(
                f"episode {self.episode_index} has {self.length} frames but trim removes "
                f"{crop_start}+{crop_end} frames"
            )
        return replace(
            self,
            crop_start_frames=crop_start,
            crop_end_frames=crop_end,
        )

    def temporal_window(self) -> dict[str, Any]:
        return {
            "source_length": self.length,
            "crop_start_frames": self.crop_start_frames,
            "crop_end_frames": self.crop_end_frames,
            "effective_length": self.effective_length,
            "crop_start_seconds_actual": self.crop_start_frames / self.fps,
            "crop_end_seconds_actual": self.crop_end_frames / self.fps,
            "source_frame_range_half_open": [
                self.crop_start_frames,
                self.source_stop_frame,
            ],
        }

    def identity(self) -> dict[str, Any]:
        data_stat = self.data_path.stat()
        video_stat = self.video_path.stat()
        metadata_stat = self.metadata_path.stat()
        return {
            "episode_index": self.episode_index,
            "length": self.length,
            "temporal_window": self.temporal_window(),
            "data": {
                "path": str(self.data_path),
                "size": data_stat.st_size,
                "mtime_ns": data_stat.st_mtime_ns,
            },
            "metadata": {
                "path": str(self.metadata_path),
                "size": metadata_stat.st_size,
                "mtime_ns": metadata_stat.st_mtime_ns,
            },
            "video": {
                "path": str(self.video_path),
                "size": video_stat.st_size,
                "mtime_ns": video_stat.st_mtime_ns,
                "from_timestamp": self.video_from_timestamp,
                "to_timestamp": self.video_to_timestamp,
            },
        }


def video_frame_count(path: str | Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    return count


def materialize_video_segment(
    source: str | Path,
    destination: str | Path,
    *,
    start_frame: int,
    frame_count: int,
) -> Path:
    """Encode an exact frame interval, atomically, and reuse a valid cached clip."""

    source = Path(source)
    destination = Path(destination)
    if start_frame < 0 or frame_count <= 0:
        raise ValueError(
            f"invalid video segment start/count: {start_frame}/{frame_count}"
        )
    if destination.is_file():
        try:
            if video_frame_count(destination) == frame_count:
                return destination
        except RuntimeError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp-{os.getpid()}.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        actual = video_frame_count(temporary)
        if actual != frame_count:
            raise RuntimeError(
                f"trimmed video has {actual} frames, expected {frame_count}: {source}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_info(dataset_root: str | Path) -> dict[str, Any]:
    path = Path(dataset_root) / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_path(template: str, *, chunk_index: int, file_index: int, video_key: str) -> str:
    return template.format(
        chunk_index=int(chunk_index), file_index=int(file_index), video_key=video_key
    )


def discover_episodes(
    dataset_root: str | Path, video_key: str = "observation.images.head"
) -> list[EpisodeSource]:
    root = Path(dataset_root).resolve()
    info = load_info(root)
    fps = float(info.get("fps") or 0.0)
    if fps <= 0.0:
        raise ValueError(f"dataset FPS must be positive: {root}")
    data_template = str(
        info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    )
    video_template = str(
        info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
    )
    prefix = f"videos/{video_key}/"
    required_video_fields = {
        name: f"{prefix}{name}"
        for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    }

    episodes: dict[int, EpisodeSource] = {}
    metadata_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_files:
        raise FileNotFoundError(f"no episode metadata parquet found under {root}")
    for metadata_path in metadata_files:
        table = pq.read_table(metadata_path)
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index in episodes:
                raise ValueError(f"duplicate episode_index {episode_index} in {root}")
            missing = [field for field in required_video_fields.values() if row.get(field) is None]
            if missing:
                raise ValueError(
                    f"episode {episode_index} metadata lacks video key {video_key!r}: {missing}"
                )
            data_rel = _format_path(
                data_template,
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
                video_key=video_key,
            )
            video_rel = _format_path(
                video_template,
                chunk_index=int(row[required_video_fields["chunk_index"]]),
                file_index=int(row[required_video_fields["file_index"]]),
                video_key=video_key,
            )
            data_path = root / data_rel
            video_path = root / video_rel
            if not data_path.is_file():
                raise FileNotFoundError(f"episode {episode_index} data not found: {data_path}")
            if not video_path.is_file():
                raise FileNotFoundError(f"episode {episode_index} video not found: {video_path}")
            episodes[episode_index] = EpisodeSource(
                episode_index=episode_index,
                length=int(row["length"]),
                data_path=data_path,
                metadata_path=metadata_path,
                metadata_row=row,
                video_key=video_key,
                video_path=video_path,
                video_from_timestamp=float(row[required_video_fields["from_timestamp"]]),
                video_to_timestamp=float(row[required_video_fields["to_timestamp"]]),
                fps=fps,
            )
    return [episodes[index] for index in sorted(episodes)]


def dataset_fingerprint(
    dataset_root: str | Path, episodes: list[EpisodeSource]
) -> str:
    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()
    digest.update((root / "meta" / "info.json").read_bytes())
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        stat = tasks_path.stat()
        digest.update(f"tasks:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for episode in episodes:
        digest.update(
            json.dumps(episode.identity(), sort_keys=True, separators=(",", ":")).encode()
        )
    return digest.hexdigest()
