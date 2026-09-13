"""Materialize resumable WiLoR/EEF diagnostic videos for batch episodes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from preprocess.batch.config import REPO_ROOT, file_sha256, fingerprint
from preprocess.batch.manifest import utc_now


def _output_path(runner: Any, h2g_id: str, episode: Any) -> Path:
    return (
        runner.experiment_dir
        / "artifacts"
        / "visualizations"
        / h2g_id
        / f"episode_{episode.episode_index:06d}"
        / "wilor_eef_vis.mp4"
    )


def visualize_hand2gripper_variants(
    runner: Any,
    *,
    axis_length: float = 0.06,
    ratio_plot_max: float = 1.5,
    max_frames: int | None = None,
) -> None:
    """Render one cache visualization per selected hand2gripper and episode."""

    settings = {
        "axis_length": float(axis_length),
        "ratio_plot_max": float(ratio_plot_max),
        "max_frames": None if max_frames is None else int(max_frames),
    }
    if settings["axis_length"] <= 0.0:
        raise ValueError("visualization axis_length must be positive")
    if settings["ratio_plot_max"] <= 0.0:
        raise ValueError("visualization ratio_plot_max must be positive")
    if settings["max_frames"] is not None and settings["max_frames"] <= 0:
        raise ValueError("visualization max_frames must be positive")

    implementation = REPO_ROOT / "preprocess/visualize_wilor_cache.py"
    h2g_ids = sorted({run["hand2gripper"] for run in runner.runs})
    print("Batch WiLoR visualization")
    for episode in runner.episodes:
        cache = runner.cache_path(episode)
        for h2g_id in h2g_ids:
            h2g = runner.config["hand2gripper_variants"][h2g_id]
            eef = runner.raw_eef_path(h2g_id, episode)
            output = _output_path(runner, h2g_id, episode)
            signature = fingerprint(
                {
                    "stage": "visualize",
                    "version": 1,
                    "episode": episode.identity(),
                    "cache_sha256": file_sha256(cache) if cache.is_file() else None,
                    "eef_sha256": file_sha256(eef) if eef.is_file() else None,
                    "hand2gripper": h2g,
                    "settings": settings,
                    "implementation_sha256": file_sha256(implementation),
                }
            )
            record = runner.manifest.episode_stage(
                "visualizations", h2g_id, episode.episode_index
            )
            if (
                "visualize" not in runner.force_stages
                and record
                and record.get("status") == "complete"
                and record.get("signature") == signature
                and output.is_file()
            ):
                print(
                    f"    visualize[{h2g_id}] episode_"
                    f"{episode.episode_index:06d}: skip"
                )
                continue

            group_meta = {
                "id": h2g_id,
                "hand2gripper_config": h2g,
                "settings": settings,
            }
            temporary: Path | None = None
            try:
                if not cache.is_file():
                    raise FileNotFoundError(f"WiLoR cache is missing: {cache}")
                if not eef.is_file():
                    raise FileNotFoundError(f"raw EEF is missing: {eef}")
                video, _ = runner.processing_video_path(episode)
                runner.manifest.update_episode(
                    "visualizations",
                    h2g_id,
                    episode.episode_index,
                    {
                        "status": "running",
                        "signature": signature,
                        "updated_at": utc_now(),
                    },
                    group_meta=group_meta,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(
                    f".{output.stem}.tmp-{os.getpid()}.mp4"
                )
                command = [
                    sys.executable,
                    "-m",
                    "preprocess.visualize_wilor_cache",
                    "--video",
                    str(video),
                    "--hands",
                    str(cache),
                    "--eef",
                    str(eef),
                    "--eef-config",
                    str(h2g["eef_config"]),
                    "--out",
                    str(temporary),
                    "--hand2gripper-mode",
                    str(h2g["mode"]),
                    "--grasp-mode",
                    str(h2g["grasp_mode"]),
                    "--axis-length",
                    str(settings["axis_length"]),
                    "--ratio-plot-max",
                    str(settings["ratio_plot_max"]),
                ]
                for key in (
                    "grasp_close_ratio",
                    "grasp_open_ratio",
                    "grasp_min_frames",
                ):
                    if h2g.get(key) is not None:
                        command.extend([f"--{key.replace('_', '-')}", str(h2g[key])])
                if settings["max_frames"] is not None:
                    command.extend(["--max-frames", str(settings["max_frames"])])
                log_path = (
                    runner.logs_dir
                    / "visualize"
                    / h2g_id
                    / f"episode_{episode.episode_index:06d}.log"
                )
                runner.run_logged(command, log_path)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("visualizer did not produce a non-empty MP4")
                os.replace(temporary, output)
                runner.manifest.update_episode(
                    "visualizations",
                    h2g_id,
                    episode.episode_index,
                    {
                        "status": "complete",
                        "signature": signature,
                        "updated_at": utc_now(),
                        "source_video": str(video),
                        "source_wilor_cache": str(cache),
                        "source_eef": str(eef),
                        "output": str(output),
                        "output_fingerprint": file_sha256(output),
                        "log": str(log_path),
                    },
                    group_meta=group_meta,
                )
                print(
                    f"    visualize[{h2g_id}] episode_"
                    f"{episode.episode_index:06d}: complete"
                )
            except Exception as error:  # noqa: BLE001
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                runner._record_failure(
                    "visualizations",
                    h2g_id,
                    episode,
                    signature,
                    error,
                    group_meta=group_meta,
                )

