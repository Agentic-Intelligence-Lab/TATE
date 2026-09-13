"""Load ARX FK trajectory JSON files and their authoritative manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import SchemaError, matrix_to_pose_xyzw, read_json, require_schema, resolve_path, sha256_file, validate_trajectory
from ..types import ArmTrajectory, DualArmTrajectory, SIDES


def load_real_fk_json(path: str | Path, episode_id: int | str | None = None) -> DualArmTrajectory:
    path = Path(path).resolve()
    data = read_json(path)
    require_schema(data, "tate.arx_real_flange_trajectory", 1, path)
    convention = data.get("gripper_convention") or {}
    if convention.get("binary_open", 0) != 0 or convention.get("binary_closed", 1) != 1:
        raise SchemaError(f"{path}: real gripper convention must be 0=open, 1=closed")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SchemaError(f"{path}: frames must be a non-empty list")
    timestamps = np.asarray([frame["timestamp_s"] for frame in frames], dtype=np.float64)
    sides: dict[str, ArmTrajectory] = {}
    for side in SIDES:
        poses = np.full((len(frames), 7), np.nan, dtype=np.float64)
        valid = np.zeros(len(frames), dtype=bool)
        binary = np.zeros(len(frames), dtype=np.int8)
        continuous = np.full(len(frames), np.nan, dtype=np.float64)
        names: set[str] = set()
        for index, frame in enumerate(frames):
            arm = (frame.get("arms") or {}).get(side)
            if arm is None:
                continue
            poses[index] = matrix_to_pose_xyzw(
                arm.get("T_tcp_in_output_frame"), f"{path}:arms.{side}[{index}]"
            )
            gripper = arm.get("gripper") or {}
            state = int(gripper.get("binary"))
            if state not in (0, 1):
                raise SchemaError(f"{path}: invalid {side} binary gripper at frame {index}")
            binary[index] = state
            if gripper.get("continuous") is not None:
                continuous[index] = float(gripper["continuous"])
            valid[index] = bool(arm.get("valid", True))
            names.add(str(arm.get("output_frame") or ""))
        names.discard("")
        if len(names) != 1:
            raise SchemaError(f"{path}: expected exactly one {side} output frame, got {sorted(names)}")
        sides[side] = ArmTrajectory(
            pose_xyzw=poses,
            valid=valid,
            gripper_binary=binary,
            gripper_continuous=continuous,
            frame_name=next(iter(names)),
        )
    source_episode = (data.get("source") or {}).get("episode_index")
    return validate_trajectory(
        DualArmTrajectory(
            episode_id=episode_id if episode_id is not None else source_episode,
            timestamps_s=timestamps,
            sides=sides,
            source={"kind": "real_fk_json", "path": str(path), "fingerprint": sha256_file(path)},
        )
    )


def load_real_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    data = read_json(path)
    require_schema(data, "tate.arx_real_flange_dataset_manifest", 1, path)
    outputs: dict[str, Path] = {}
    for record in data.get("episodes") or []:
        episode_id = str(record.get("episode_index"))
        if episode_id in outputs:
            raise SchemaError(f"{path}: duplicate real episode {episode_id}")
        output = resolve_path(record["output"], path.parent)
        if not output.is_file():
            raise FileNotFoundError(f"real manifest episode {episode_id} is missing: {output}")
        outputs[episode_id] = output
    if not outputs:
        raise SchemaError(f"{path}: real manifest has no episodes")
    data["_path"] = str(path)
    data["_fingerprint"] = sha256_file(path)
    return data, outputs
