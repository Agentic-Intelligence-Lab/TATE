"""Materialize final EEF variants as self-contained LeRobot v3 datasets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from preprocess.batch.config import file_sha256, fingerprint
from preprocess.batch.dataset import EpisodeSource, load_info, materialize_video_segment
from preprocess.batch.manifest import atomic_write_json, utc_now


EEF_FEATURES = {
    "tate.eef_raw.left.pose": ("float32", [7], ["x", "y", "z", "qx", "qy", "qz", "qw"]),
    "tate.eef_raw.right.pose": ("float32", [7], ["x", "y", "z", "qx", "qy", "qz", "qw"]),
    "tate.eef.left.pose": ("float32", [7], ["x", "y", "z", "qx", "qy", "qz", "qw"]),
    "tate.eef.right.pose": ("float32", [7], ["x", "y", "z", "qx", "qy", "qz", "qw"]),
    "tate.eef.left.gripper": ("int64", [1], ["closed"]),
    "tate.eef.right.gripper": ("int64", [1], ["closed"]),
    "tate.eef.left.grasp_ratio": ("float32", [1], ["ratio"]),
    "tate.eef.right.grasp_ratio": ("float32", [1], ["ratio"]),
    "tate.eef.left.valid": ("bool", [1], ["valid"]),
    "tate.eef.right.valid": ("bool", [1], ["valid"]),
    "tate.source_episode_index": ("int64", [1], ["source_episode_index"]),
    "tate.source_frame_index": ("int64", [1], ["source_frame_index"]),
    "tate.source_timestamp": ("float64", [1], ["source_timestamp"]),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fixed_list(values: np.ndarray, value_type: pa.DataType) -> pa.Array:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"fixed-list values must be 2D, got {values.shape}")
    flat = pa.array(values.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _replace_or_append(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.append_column(name, array) if index < 0 else table.set_column(index, name, array)


def _set_scalar(table: pa.Table, name: str, values: Any, data_type: pa.DataType) -> pa.Table:
    if np.isscalar(values):
        values = [values] * len(table)
    return _replace_or_append(table, name, pa.array(values, type=data_type))


def _pose_xyzw(matrix: Any) -> np.ndarray:
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"invalid EEF pose shape/value: {pose.shape}")
    return np.concatenate([pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()])


def load_eef_columns(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tate.dual_arm_eef" or payload.get("schema_version") != 2:
        raise ValueError(f"unsupported EEF schema: {path}")
    frames = payload.get("frames") or []
    n = len(frames)
    output: dict[str, np.ndarray] = {}
    identity_pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for side, hand_key in (("left", "hand_l"), ("right", "hand_r")):
        poses = np.repeat(identity_pose[None, :], n, axis=0)
        valid = np.zeros(n, dtype=bool)
        gripper = np.zeros(n, dtype=np.int64)
        ratios = np.zeros(n, dtype=np.float32)
        last_gripper = 0
        for index, frame in enumerate(frames):
            hand = frame.get(hand_key)
            if hand is None:
                gripper[index] = last_gripper
                continue
            poses[index] = _pose_xyzw(hand["tcp_pose_eef_frame"]).astype(np.float32)
            valid[index] = True
            last_gripper = int(hand["grasp_state"])
            gripper[index] = last_gripper
            if hand.get("grasp_ratio") is not None:
                ratios[index] = float(hand["grasp_ratio"])
        output[f"{side}.pose"] = poses
        output[f"{side}.valid"] = valid
        output[f"{side}.gripper"] = gripper
        output[f"{side}.grasp_ratio"] = ratios
    metadata = {
        "path": str(path),
        "sha256": file_sha256(path),
        "total_frames": n,
        "coordinate_convention": payload.get("eef_coordinate_convention"),
        "hand2gripper": payload.get("hand2gripper"),
    }
    return output, metadata


def _filter_source_table(episode: EpisodeSource) -> pa.Table:
    table = pq.read_table(episode.data_path)
    if "episode_index" in table.column_names:
        mask = pc.equal(table["episode_index"], pa.scalar(episode.episode_index))
        filtered = table.filter(mask)
        if len(filtered):
            table = filtered
    if len(table) != episode.length:
        raise ValueError(
            f"source data length for episode {episode.episode_index} is {len(table)}, "
            f"metadata says {episode.length}"
        )
    return table.slice(episode.crop_start_frames, episode.effective_length)


def _policy_state_action(
    ik_path: Path,
    final_eef: dict[str, np.ndarray],
    *,
    open_raw: float,
    closed_raw: float,
    action_alignment: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ik_path, allow_pickle=False) as ik:
        left = np.asarray(ik["left_arm_qpos"], dtype=np.float32)
        right = np.asarray(ik["right_arm_qpos"], dtype=np.float32)
    n = len(final_eef["left.valid"])
    if left.shape != (n, 6) or right.shape != (n, 6):
        raise ValueError(
            f"IK arm arrays must be ({n}, 6), got left={left.shape}, right={right.shape}"
        )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"IK contains NaN/Inf: {ik_path}")
    left_gripper = np.where(final_eef["left.gripper"] > 0, closed_raw, open_raw).astype(np.float32)
    right_gripper = np.where(final_eef["right.gripper"] > 0, closed_raw, open_raw).astype(np.float32)
    state = np.column_stack(
        [left, left_gripper[:, None], right, right_gripper[:, None]]
    ).astype(np.float32)
    if action_alignment == "same_frame":
        action = state.copy()
    elif action_alignment == "next_frame":
        action = np.concatenate([state[1:], state[-1:]], axis=0)
    else:
        raise ValueError(f"unknown action alignment: {action_alignment}")
    return state, action


def build_episode_table(
    episode: EpisodeSource,
    *,
    derived_episode_index: int,
    dataset_start_index: int,
    raw_eef_path: Path,
    final_eef_path: Path,
    ik_path: Path | None,
    lerobot_config: dict[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    table = _filter_source_table(episode)
    source_frame_index = (
        np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        if "frame_index" in table.column_names
        else np.arange(
            episode.crop_start_frames,
            episode.source_stop_frame,
            dtype=np.int64,
        )
    )
    source_timestamp = (
        np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        if "timestamp" in table.column_names
        else source_frame_index.astype(np.float64) / episode.fps
    )
    raw, raw_meta = load_eef_columns(raw_eef_path)
    final, final_meta = load_eef_columns(final_eef_path)
    if raw_meta["total_frames"] != len(table) or final_meta["total_frames"] != len(table):
        raise ValueError(
            f"source/EEF length mismatch for episode {episode.episode_index}: "
            f"data={len(table)} raw={raw_meta['total_frames']} final={final_meta['total_frames']}"
        )

    for side in ("left", "right"):
        table = _replace_or_append(
            table,
            f"tate.eef_raw.{side}.pose",
            _fixed_list(raw[f"{side}.pose"], pa.float32()),
        )
        table = _replace_or_append(
            table,
            f"tate.eef.{side}.pose",
            _fixed_list(final[f"{side}.pose"], pa.float32()),
        )
        table = _set_scalar(
            table, f"tate.eef.{side}.gripper", final[f"{side}.gripper"], pa.int64()
        )
        table = _set_scalar(
            table,
            f"tate.eef.{side}.grasp_ratio",
            final[f"{side}.grasp_ratio"],
            pa.float32(),
        )
        table = _set_scalar(
            table, f"tate.eef.{side}.valid", final[f"{side}.valid"], pa.bool_()
        )

    if lerobot_config["replace_state_action"]:
        if ik_path is None or not ik_path.is_file():
            raise FileNotFoundError(
                f"replace_state_action requires a completed IK artifact: {ik_path}"
            )
        state, action = _policy_state_action(
            ik_path,
            final,
            open_raw=lerobot_config["gripper_open_raw"],
            closed_raw=lerobot_config["gripper_closed_raw"],
            action_alignment=lerobot_config["action_alignment"],
        )
        table = _replace_or_append(
            table, "observation.state", _fixed_list(state, pa.float32())
        )
        table = _replace_or_append(table, "action", _fixed_list(action, pa.float32()))

    table = _set_scalar(table, "episode_index", derived_episode_index, pa.int64())
    table = _set_scalar(table, "frame_index", np.arange(len(table)), pa.int64())
    table = _set_scalar(
        table, "index", np.arange(dataset_start_index, dataset_start_index + len(table)), pa.int64()
    )
    table = _set_scalar(
        table, "tate.source_episode_index", episode.episode_index, pa.int64()
    )
    table = _set_scalar(
        table, "tate.source_frame_index", source_frame_index, pa.int64()
    )
    table = _set_scalar(
        table, "tate.source_timestamp", source_timestamp, pa.float64()
    )
    table = _set_scalar(
        table,
        "timestamp",
        source_timestamp - source_timestamp[0],
        pa.float32(),
    )
    if lerobot_config.get("task") is not None:
        table = _set_scalar(table, "task_index", 0, pa.int64())
    table = table.replace_schema_metadata(_huggingface_metadata(table))
    return table, {"raw": raw_meta, "final": final_meta}


def _arrow_dtype_name(data_type: pa.DataType) -> str:
    if pa.types.is_boolean(data_type):
        return "bool"
    if pa.types.is_int64(data_type):
        return "int64"
    if pa.types.is_int32(data_type):
        return "int32"
    if pa.types.is_float32(data_type):
        return "float32"
    if pa.types.is_float64(data_type):
        return "float64"
    raise TypeError(f"unsupported LeRobot Arrow type: {data_type}")


def _huggingface_metadata(table: pa.Table) -> dict[bytes, bytes]:
    features = {}
    for field in table.schema:
        data_type = field.type
        if pa.types.is_fixed_size_list(data_type):
            features[field.name] = {
                "feature": {"dtype": _arrow_dtype_name(data_type.value_type), "_type": "Value"},
                "length": data_type.list_size,
                "_type": "List",
            }
        elif pa.types.is_boolean(data_type) or pa.types.is_integer(data_type) or pa.types.is_floating(data_type):
            features[field.name] = {"dtype": _arrow_dtype_name(data_type), "_type": "Value"}
    value = {
        "info": {"features": features},
        "fingerprint": fingerprint({"schema": str(table.schema.remove_metadata())})[:16],
    }
    return {b"huggingface": json.dumps(value, separators=(",", ":")).encode("utf-8")}


def _column_matrix(column: pa.ChunkedArray) -> np.ndarray | None:
    data_type = column.type
    if pa.types.is_fixed_size_list(data_type) or pa.types.is_list(data_type):
        values = np.asarray(column.to_pylist())
        return values.astype(np.float64) if np.issubdtype(values.dtype, np.number) else None
    if pa.types.is_integer(data_type) or pa.types.is_floating(data_type) or pa.types.is_boolean(data_type):
        return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)[:, None]
    return None


def numeric_stats(table: pa.Table) -> dict[str, dict[str, list[float | int]]]:
    output: dict[str, dict[str, list[float | int]]] = {}
    for name in table.column_names:
        values = _column_matrix(table[name])
        if values is None or values.size == 0:
            continue
        row_mask = np.all(np.isfinite(values), axis=1)
        if name.startswith("tate.eef_raw.left") or name.startswith("tate.eef.left"):
            row_mask &= np.asarray(
                table["tate.eef.left.valid"].to_numpy(zero_copy_only=False), dtype=bool
            )
        elif name.startswith("tate.eef_raw.right") or name.startswith("tate.eef.right"):
            row_mask &= np.asarray(
                table["tate.eef.right.valid"].to_numpy(zero_copy_only=False), dtype=bool
            )
        selected = values[row_mask]
        if not len(selected):
            continue
        quantiles = np.quantile(selected, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
        output[name] = {
            "min": selected.min(axis=0).tolist(),
            "max": selected.max(axis=0).tolist(),
            "mean": selected.mean(axis=0).tolist(),
            "std": selected.std(axis=0).tolist(),
            "count": [int(len(selected))],
            "q01": quantiles[0].tolist(),
            "q10": quantiles[1].tolist(),
            "q50": quantiles[2].tolist(),
            "q90": quantiles[3].tolist(),
            "q99": quantiles[4].tolist(),
        }
    return output


def _feature_info(source_info: dict[str, Any]) -> dict[str, Any]:
    features = dict(source_info.get("features") or {})
    for name, (dtype, shape, names) in EEF_FEATURES.items():
        features[name] = {"dtype": dtype, "shape": shape, "names": names}
    return features


def _materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _video_features(info: dict[str, Any]) -> list[str]:
    return sorted(
        name
        for name, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    )


def _source_video_path(
    source_root: Path, info: dict[str, Any], row: dict[str, Any], video_key: str
) -> Path | None:
    prefix = f"videos/{video_key}/"
    chunk = row.get(prefix + "chunk_index")
    file_index = row.get(prefix + "file_index")
    if chunk is None or file_index is None:
        return None
    template = str(
        info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
    )
    return source_root / template.format(
        video_key=video_key, chunk_index=int(chunk), file_index=int(file_index)
    )


def _update_episode_metadata(
    source_row: dict[str, Any],
    *,
    derived_episode_index: int,
    dataset_start_index: int,
    table: pa.Table,
    video_keys: list[str],
    task_override: str | None,
    episode: EpisodeSource,
    video_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row = _jsonable(source_row)
    row["episode_index"] = derived_episode_index
    row["length"] = len(table)
    row["data/chunk_index"] = 0
    row["data/file_index"] = derived_episode_index
    row["dataset_from_index"] = dataset_start_index
    row["dataset_to_index"] = dataset_start_index + len(table)
    row["meta/episodes/chunk_index"] = 0
    row["meta/episodes/file_index"] = derived_episode_index
    if task_override is not None:
        row["tasks"] = [task_override]
    for video_key in video_keys:
        row[f"videos/{video_key}/chunk_index"] = 0
        row[f"videos/{video_key}/file_index"] = derived_episode_index
        if episode.has_trim:
            row[f"videos/{video_key}/from_timestamp"] = 0.0
            row[f"videos/{video_key}/to_timestamp"] = (
                episode.effective_length / episode.fps
            )
    for key in list(row):
        is_trimmed_video_stat = episode.has_trim and any(
            key.startswith(f"stats/{video_key}/") for video_key in video_keys
        )
        if key.startswith("stats/") and (
            is_trimmed_video_stat
            or not any(key.startswith(f"stats/{video_key}/") for video_key in video_keys)
        ):
            del row[key]
    for feature, stats in numeric_stats(table).items():
        for metric, value in stats.items():
            row[f"stats/{feature}/{metric}"] = value
    for feature, stats in (video_stats or {}).items():
        for metric, value in stats.items():
            row[f"stats/{feature}/{metric}"] = value
    return row


def _empty_video_accumulator() -> dict[str, Any]:
    return {
        "hist": np.zeros((3, 256), dtype=np.int64),
        "pixel_count": 0,
        "frame_count": 0,
    }


def _accumulate_video(path: Path, accumulator: dict[str, Any]) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to decode video for statistics: {path}")
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for channel in range(3):
            accumulator["hist"][channel] += np.bincount(
                frame_rgb[:, :, channel].reshape(-1), minlength=256
            )
        accumulator["pixel_count"] += int(frame_rgb.shape[0] * frame_rgb.shape[1])
        accumulator["frame_count"] += 1
    capture.release()


def _finalize_video_stats(accumulator: dict[str, Any]) -> dict[str, Any]:
    hist = accumulator["hist"]
    pixel_count = int(accumulator["pixel_count"])
    if pixel_count <= 0:
        raise ValueError("cannot compute statistics for an empty video")
    bins = np.arange(256, dtype=np.float64)
    means = (hist @ bins) / pixel_count
    variances = (hist @ (bins * bins)) / pixel_count - means * means
    stds = np.sqrt(np.maximum(variances, 0.0))

    def quantile(channel_hist: np.ndarray, probability: float) -> float:
        target = probability * max(pixel_count - 1, 0)
        return float(np.searchsorted(np.cumsum(channel_hist), target, side="right")) / 255.0

    nonzero = [np.flatnonzero(hist[channel]) for channel in range(3)]
    values = {
        "min": [float(indices[0]) / 255.0 for indices in nonzero],
        "max": [float(indices[-1]) / 255.0 for indices in nonzero],
        "mean": (means / 255.0).tolist(),
        "std": (stds / 255.0).tolist(),
    }
    for label, probability in (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)):
        values[label] = [quantile(hist[channel], probability) for channel in range(3)]
    return {
        key: [[[item]] for item in value]
        for key, value in values.items()
    } | {"count": [int(accumulator["frame_count"])]}


def _write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _package_one_variant(runner, run: dict[str, str], *, force: bool) -> None:
    run_id = run["id"]
    lerobot_cfg = runner.config["lerobot"]
    records = (
        runner.manifest.data.get("variants", {}).get(run_id, {}).get("episodes", {})
    )
    ready: list[tuple[EpisodeSource, dict[str, Any]]] = []
    missing = []
    for episode in runner.episodes:
        record = records.get(str(episode.episode_index))
        required = bool(record and record.get("final_eef"))
        if lerobot_cfg["replace_state_action"]:
            required = required and bool(record.get("ik"))
        if not required:
            missing.append(episode.episode_index)
        else:
            ready.append((episode, record))
    if missing and lerobot_cfg["require_all_episodes"]:
        raise RuntimeError(
            f"variant {run_id} cannot be packaged; incomplete episodes: {missing}"
        )
    if not ready:
        raise RuntimeError(f"variant {run_id} has no complete episodes to package")

    signature = fingerprint(
        {
            "stage": "package",
            "version": 2,
            "source_dataset": runner.source_dataset_fingerprint,
            "lerobot": lerobot_cfg,
            "trajectory": runner.config["trajectory"],
            "episodes": [
                {
                    "source_episode_index": episode.episode_index,
                    "raw_eef": file_sha256(record["raw_eef"]),
                    "final_eef": file_sha256(record["final_eef"]),
                    "ik": file_sha256(record["ik"])
                    if lerobot_cfg["replace_state_action"]
                    else None,
                }
                for episode, record in ready
            ],
        }
    )
    destination = runner.experiment_dir / "datasets" / run_id
    provenance_path = destination / "meta" / "tate_preprocess.json"
    if provenance_path.is_file() and not force:
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("package_signature") == signature:
            print(f"package[{run_id}]: skip")
            runner.manifest.update_variant(
                run_id,
                {
                    "derived_lerobot_dataset": str(destination),
                    "package": {"status": "complete", "signature": signature},
                },
            )
            return
        raise RuntimeError(
            f"derived dataset exists with a different signature: {destination}; "
            "use --force-stage package or a new experiment_id"
        )

    runner.manifest.update_variant(
        run_id,
        {
            "package": {
                "status": "running",
                "signature": signature,
                "updated_at": utc_now(),
            }
        },
    )

    source_root = Path(runner.config["source"]["ego_dataset"])
    source_info = load_info(source_root)
    source_stats_path = source_root / "meta" / "stats.json"
    source_stats = (
        json.loads(source_stats_path.read_text(encoding="utf-8"))
        if source_stats_path.is_file()
        else {}
    )
    video_keys = _video_features(source_info)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-tmp-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "dataset"
        tables: list[pa.Table] = []
        episode_map = []
        dataset_index = 0
        coordinate_convention = None
        dataset_video_accumulators = {
            key: _empty_video_accumulator() for key in video_keys
        }
        for derived_index, (episode, record) in enumerate(ready):
            raw_path = Path(record["raw_eef"])
            final_path = Path(record["final_eef"])
            ik_path = Path(record["ik"]) if record.get("ik") else None
            table, eef_meta = build_episode_table(
                episode,
                derived_episode_index=derived_index,
                dataset_start_index=dataset_index,
                raw_eef_path=raw_path,
                final_eef_path=final_path,
                ik_path=ik_path,
                lerobot_config=lerobot_cfg,
            )
            coordinate_convention = coordinate_convention or eef_meta["final"][
                "coordinate_convention"
            ]
            data_path = staging / "data" / "chunk-000" / f"file-{derived_index:03d}.parquet"
            _write_parquet(table, data_path)
            episode_video_stats: dict[str, dict[str, Any]] = {}
            for video_key in video_keys:
                source_video = _source_video_path(
                    source_root, source_info, episode.metadata_row, video_key
                )
                if source_video is None:
                    continue
                if not source_video.is_file():
                    raise FileNotFoundError(source_video)
                materialized_video = source_video
                if episode.has_trim:
                    prefix = f"videos/{video_key}/"
                    from_timestamp = float(
                        episode.metadata_row.get(prefix + "from_timestamp", 0.0)
                    )
                    source_start_frame = (
                        int(round(from_timestamp * episode.fps))
                        + episode.crop_start_frames
                    )
                    materialized_video = runner.video_segment_path(episode, video_key)
                    materialize_video_segment(
                        source_video,
                        materialized_video,
                        start_frame=source_start_frame,
                        frame_count=episode.effective_length,
                    )
                target_video = (
                    staging
                    / "videos"
                    / video_key
                    / "chunk-000"
                    / f"file-{derived_index:03d}.mp4"
                )
                _materialize_file(
                    materialized_video, target_video, lerobot_cfg["video_mode"]
                )
                if episode.has_trim:
                    episode_accumulator = _empty_video_accumulator()
                    _accumulate_video(materialized_video, episode_accumulator)
                    if episode_accumulator["frame_count"] != episode.effective_length:
                        raise ValueError(
                            f"video {video_key} has {episode_accumulator['frame_count']} "
                            f"frames, expected {episode.effective_length}"
                        )
                    episode_video_stats[video_key] = _finalize_video_stats(
                        episode_accumulator
                    )
                    dataset_video_accumulators[video_key]["hist"] += episode_accumulator[
                        "hist"
                    ]
                    dataset_video_accumulators[video_key]["pixel_count"] += (
                        episode_accumulator["pixel_count"]
                    )
                    dataset_video_accumulators[video_key]["frame_count"] += (
                        episode_accumulator["frame_count"]
                    )
            meta_row = _update_episode_metadata(
                episode.metadata_row,
                derived_episode_index=derived_index,
                dataset_start_index=dataset_index,
                table=table,
                video_keys=video_keys,
                task_override=lerobot_cfg.get("task"),
                episode=episode,
                video_stats=episode_video_stats,
            )
            meta_path = (
                staging / "meta" / "episodes" / "chunk-000" / f"file-{derived_index:03d}.parquet"
            )
            _write_parquet(pa.Table.from_pylist([meta_row]), meta_path)
            tables.append(table)
            episode_map.append(
                {
                    "derived_episode_index": derived_index,
                    "source_episode_index": episode.episode_index,
                    "length": len(table),
                    "temporal_window": episode.temporal_window(),
                    "raw_eef": str(raw_path),
                    "raw_eef_fingerprint": eef_meta["raw"]["sha256"],
                    "final_eef": str(final_path),
                    "final_eef_fingerprint": eef_meta["final"]["sha256"],
                    "ik": None if ik_path is None else str(ik_path),
                    "ik_fingerprint": (
                        None if ik_path is None else file_sha256(ik_path)
                    ),
                }
            )
            dataset_index += len(table)

        tasks_path = source_root / "meta" / "tasks.parquet"
        if lerobot_cfg.get("task") is not None:
            _write_parquet(
                pa.Table.from_pylist(
                    [{"task_index": 0, "task": lerobot_cfg["task"]}]
                ),
                staging / "meta" / "tasks.parquet",
            )
        elif tasks_path.is_file():
            _materialize_file(tasks_path, staging / "meta" / "tasks.parquet", "copy")

        info = dict(source_info)
        features = _feature_info(source_info)
        per_side_frames = (coordinate_convention or {}).get("per_side_frames") or {}
        for side in ("left", "right"):
            if per_side_frames.get(side):
                features[f"tate.eef_raw.{side}.pose"]["frame"] = per_side_frames[side]
                features[f"tate.eef.{side}.pose"]["frame"] = per_side_frames[side]
        info.update(
            {
                "robot_type": "arx5_2025_bimanual",
                "total_episodes": len(tables),
                "total_frames": dataset_index,
                "total_tasks": 1
                if lerobot_cfg.get("task") is not None
                else source_info.get("total_tasks", 1),
                "splits": {"train": f"0:{len(tables)}"},
                "features": features,
                "eef_coordinate_convention": coordinate_convention,
                "tate": {
                    "arm_mode": runner.config["trajectory"]["arm_mode"],
                    "active_sides": runner.config["trajectory"]["active_sides"],
                    "temporal_trim": runner.config["trajectory"]["trim"],
                    "eef_frames": per_side_frames,
                },
            }
        )
        data_size = sum(path.stat().st_size for path in (staging / "data").rglob("*.parquet"))
        video_size = sum(path.stat().st_size for path in (staging / "videos").rglob("*.mp4"))
        info["data_files_size_in_mb"] = round(data_size / 1_000_000, 3)
        info["video_files_size_in_mb"] = round(video_size / 1_000_000, 3)
        atomic_write_json(staging / "meta" / "info.json", info)

        combined = pa.concat_tables(tables, promote_options="default")
        any_trimmed = any(episode.has_trim for episode, _ in ready)
        stats = (
            {
                key: _finalize_video_stats(dataset_video_accumulators[key])
                for key in video_keys
            }
            if any_trimmed
            else {
                key: value
                for key, value in source_stats.items()
                if key in video_keys
            }
        )
        stats.update(numeric_stats(combined))
        atomic_write_json(staging / "meta" / "stats.json", stats)

        provenance = {
            "schema": "tate.lerobot_preprocess_provenance",
            "schema_version": 1,
            "created_at": utc_now(),
            "experiment_id": runner.config["experiment_id"],
            "variant_id": run_id,
            "experiment_manifest": str(runner.manifest.path),
            "experiment_config_fingerprint": runner.config["config_fingerprint"],
            "package_signature": signature,
            "source_ego_dataset": str(source_root),
            "source_ego_dataset_fingerprint": runner.source_dataset_fingerprint,
            "arm_mode": runner.config["trajectory"]["arm_mode"],
            "active_sides": runner.config["trajectory"]["active_sides"],
            "temporal_trim": runner.config["trajectory"]["trim"],
            "hand2gripper_id": run["hand2gripper"],
            "correction_id": run["correction"],
            "eef_coordinate_convention": coordinate_convention,
            "policy_fields": {
                "observation_state": "retargeted ARX joint target"
                if lerobot_cfg["replace_state_action"]
                else "preserved from source dataset",
                "action_alignment": lerobot_cfg["action_alignment"],
                "gripper_open_raw": lerobot_cfg["gripper_open_raw"],
                "gripper_closed_raw": lerobot_cfg["gripper_closed_raw"],
                "task_override": lerobot_cfg.get("task"),
            },
            "image_stats": {
                "policy": (
                    "recomputed from exact decoded trimmed videos"
                    if any_trimmed
                    else "inherited from source dataset; videos are byte-identical"
                ),
                "source": str(source_stats_path),
            },
            "episodes": episode_map,
            "excluded_incomplete_source_episodes": missing,
        }
        atomic_write_json(staging / "meta" / "tate_preprocess.json", provenance)

        if destination.exists():
            if not force:
                raise FileExistsError(destination)
            shutil.rmtree(destination)
        os.replace(staging, destination)

    runner.manifest.update_variant(
        run_id,
        {
            "derived_lerobot_dataset": str(destination),
            "package": {
                "status": "complete",
                "signature": signature,
                "updated_at": utc_now(),
                "episode_count": len(ready),
                "frame_count": dataset_index,
            },
        },
    )
    print(f"package[{run_id}]: complete episodes={len(ready)} frames={dataset_index}")


def package_variants(runner, *, force: bool = False) -> None:
    for run in runner.runs:
        try:
            _package_one_variant(runner, run, force=force)
        except Exception as error:  # noqa: BLE001
            runner.manifest.update_variant(
                run["id"],
                {
                    "package": {
                        "status": "failed",
                        "updated_at": utc_now(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                },
            )
            print(f"package[{run['id']}] FAILED: {type(error).__name__}: {error}")
            if runner.fail_fast:
                raise
