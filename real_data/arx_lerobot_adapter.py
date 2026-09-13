#!/usr/bin/env python3
"""Convert ARX LeRobot joint episodes into left/right zero-flange frames.

The source LeRobot parquet stores the two six-joint arms plus one scalar
gripper value per side.  This adapter runs ARX forward kinematics with TATE's
MuJoCo scene and exports both the moving flange pose and TCP pose in the
corresponding ``left_flange_zero`` / ``right_flange_zero`` coordinate frame.

The source dataset is never modified.  Gripper output preserves the original
raw value and adds:

* ``continuous``: clipped to [0, 1], where fully open is 0 and fully closed is 1;
* ``binary``: 0=open and 1=closed, using a configurable raw-value threshold.

The ARX raw value increases as the gripper closes: the current recordings use
approximately -3.4 for fully open and 0.1 for fully closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
DEFAULT_CALIBRATION = REPO_ROOT / "cfg" / "preprocess" / "base" / "RealSenseD405.yaml"

SCHEMA_NAME = "tate.arx_real_flange_trajectory"
SCHEMA_VERSION = 1

DEFAULT_GRIPPER_OPEN_RAW = -3.4
DEFAULT_GRIPPER_CLOSED_RAW = 0.1
DEFAULT_GRIPPER_BINARY_THRESHOLD_RAW = -2.6

SIDE_LAYOUT = {
    "left": {
        "state_joint_names": tuple(f"left_joint_{index}" for index in range(1, 7)),
        "state_gripper_name": "left_gripper",
        "model_joint_names": tuple(f"left_joint{index}" for index in range(1, 7)),
        "flange_site": "left_flange",
        "tcp_site": "left_tcp",
        "output_frame": "left_flange_zero",
    },
    "right": {
        "state_joint_names": tuple(f"right_joint_{index}" for index in range(1, 7)),
        "state_gripper_name": "right_gripper",
        "model_joint_names": tuple(f"right_joint{index}" for index in range(11, 17)),
        "flange_site": "right_flange",
        "tcp_site": "right_tcp",
        "output_frame": "right_flange_zero",
    },
}

DEFAULT_STATE_NAMES = [
    *SIDE_LAYOUT["left"]["state_joint_names"],
    SIDE_LAYOUT["left"]["state_gripper_name"],
    *SIDE_LAYOUT["right"]["state_joint_names"],
    SIDE_LAYOUT["right"]["state_gripper_name"],
]


def as_abs(path: str | Path) -> Path:
    result = Path(path).expanduser()
    return result.resolve() if result.is_absolute() else (REPO_ROOT / result).resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def rigid_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or Inf")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def find_dataset_root(path: Path) -> Path | None:
    candidates = [path] if path.is_dir() else list(path.parents)
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    return None


def load_dataset_info(dataset_root: Path | None) -> dict[str, Any]:
    if dataset_root is None:
        return {}
    path = dataset_root / "meta" / "info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def state_names(info: dict[str, Any], source_column: str) -> list[str]:
    feature = (info.get("features") or {}).get(source_column) or {}
    names = feature.get("names")
    if names is None:
        return list(DEFAULT_STATE_NAMES)
    names = [str(name) for name in names]
    missing = [name for name in DEFAULT_STATE_NAMES if name not in names]
    if missing:
        raise ValueError(f"{source_column} metadata is missing ARX fields: {missing}")
    return names


def discover_parquets(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".parquet":
            raise ValueError(f"input file must be parquet: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    data_dir = input_path / "data"
    search_root = data_dir if data_dir.is_dir() else input_path
    paths = sorted(search_root.rglob("*.parquet"))
    paths = [path for path in paths if "meta" not in path.relative_to(search_root).parts]
    if not paths:
        raise FileNotFoundError(f"no parquet data files found under {input_path}")
    return paths


def gripper_continuous(raw: float, open_raw: float, closed_raw: float) -> float:
    """Map the raw ARX scalar to [0, 1]: open=0, closed=1."""
    denominator = closed_raw - open_raw
    if denominator <= 0.0:
        raise ValueError("gripper_closed_raw must be greater than gripper_open_raw")
    return float(np.clip((float(raw) - open_raw) / denominator, 0.0, 1.0))


def gripper_values(
    raw: float,
    action_raw: float | None,
    *,
    open_raw: float,
    closed_raw: float,
    binary_threshold_raw: float,
) -> dict[str, Any]:
    output = {
        "raw": float(raw),
        "continuous": gripper_continuous(raw, open_raw, closed_raw),
        "binary": int(float(raw) >= binary_threshold_raw),
    }
    if action_raw is not None:
        output.update(
            {
                "action_raw": float(action_raw),
                "action_continuous": gripper_continuous(action_raw, open_raw, closed_raw),
                "action_binary": int(float(action_raw) >= binary_threshold_raw),
            }
        )
    return output


def gripper_events(frames: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    """Return the first frame in each new binary state."""
    events = []
    previous = None
    for frame in frames:
        gripper = frame["arms"][side]["gripper"]
        current = int(gripper["binary"])
        if previous is not None and current != previous:
            events.append(
                {
                    "event_index": len(events),
                    "frame_index": int(frame["frame_index"]),
                    "timestamp_s": float(frame["timestamp_s"]),
                    "from_binary": int(previous),
                    "to_binary": current,
                    "transition": "close" if current == 1 else "open",
                    "raw": float(gripper["raw"]),
                    "continuous": float(gripper["continuous"]),
                }
            )
        previous = current
    return events


class ArxForwardKinematics:
    """MuJoCo FK expressed in TATE's per-side zero-flange frames."""

    def __init__(self, scene_path: Path, calibration_path: Path) -> None:
        import mujoco

        self.mujoco = mujoco
        self.scene_path = scene_path
        self.calibration_path = calibration_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8")) or {}
        arm_calibration = calibration.get("arm_extrinsics") or {}
        self.joint_qpos_addresses: dict[str, np.ndarray] = {}
        self.site_ids: dict[str, dict[str, int]] = {}
        self.T_output_in_scene: dict[str, np.ndarray] = {}

        for side, layout in SIDE_LAYOUT.items():
            addresses = []
            for joint_name in layout["model_joint_names"]:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id < 0:
                    raise ValueError(f"MuJoCo scene is missing joint {joint_name!r}")
                addresses.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_qpos_addresses[side] = np.asarray(addresses, dtype=np.int32)

            ids = {}
            for kind in ("flange", "tcp"):
                site_name = layout[f"{kind}_site"]
                site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
                )
                if site_id < 0:
                    raise ValueError(f"MuJoCo scene is missing site {site_name!r}")
                ids[kind] = int(site_id)
            self.site_ids[side] = ids

            side_cfg = arm_calibration.get(side) or {}
            self.T_output_in_scene[side] = rigid_matrix(
                side_cfg.get("T_eef_frame_in_scene"),
                f"arm_extrinsics.{side}.T_eef_frame_in_scene",
            )

        self._validate_zero_flange_frames()

    def _site_pose_in_scene(self, site_id: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[site_id]
        return transform

    def _validate_zero_flange_frames(self) -> None:
        self.data.qpos[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        for side in SIDE_LAYOUT:
            scene_zero = self._site_pose_in_scene(self.site_ids[side]["flange"])
            configured_zero = self.T_output_in_scene[side]
            if not np.allclose(scene_zero, configured_zero, atol=1e-6):
                delta_mm = 1000.0 * np.linalg.norm(
                    scene_zero[:3, 3] - configured_zero[:3, 3]
                )
                raise ValueError(
                    f"{side} zero-flange frame in {self.calibration_path} does not "
                    f"match {self.scene_path} (translation delta {delta_mm:.3f} mm)"
                )

    def forward(self, joints_by_side: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        self.data.qpos[:] = 0.0
        for side in SIDE_LAYOUT:
            joints = np.asarray(joints_by_side[side], dtype=np.float64)
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                raise ValueError(f"{side} joints must be six finite values, got {joints}")
            self.data.qpos[self.joint_qpos_addresses[side]] = joints
        self.mujoco.mj_forward(self.model, self.data)

        output = {}
        for side in SIDE_LAYOUT:
            T_scene_in_output = np.linalg.inv(self.T_output_in_scene[side])
            output[side] = {
                "flange": T_scene_in_output
                @ self._site_pose_in_scene(self.site_ids[side]["flange"]),
                "tcp": T_scene_in_output
                @ self._site_pose_in_scene(self.site_ids[side]["tcp"]),
            }
        return output


def episode_groups(frame: Any) -> list[tuple[int, Any]]:
    if "episode_index" not in frame.columns:
        return [(0, frame.reset_index(drop=True))]
    groups = []
    for episode_index, group in frame.groupby("episode_index", sort=True):
        if "frame_index" in group.columns:
            group = group.sort_values("frame_index", kind="stable")
        groups.append((int(episode_index), group.reset_index(drop=True)))
    return groups


def convert_episode(
    frame: Any,
    *,
    episode_index: int,
    source_path: Path,
    source_column: str,
    names: list[str],
    fps: float,
    fk: ArxForwardKinematics,
    open_raw: float,
    closed_raw: float,
    binary_threshold_raw: float,
) -> dict[str, Any]:
    if source_column not in frame.columns:
        raise ValueError(f"{source_path} has no {source_column!r} column")
    states = np.stack(frame[source_column].to_numpy()).astype(np.float64)
    if states.ndim != 2 or states.shape[1] != len(names):
        raise ValueError(
            f"{source_column} shape {states.shape} does not match {len(names)} feature names"
        )
    actions = None
    if "action" in frame.columns:
        candidate = np.stack(frame["action"].to_numpy()).astype(np.float64)
        if candidate.shape == states.shape:
            actions = candidate

    name_to_index = {name: index for index, name in enumerate(names)}
    side_indices = {}
    for side, layout in SIDE_LAYOUT.items():
        side_indices[side] = {
            "joints": [name_to_index[name] for name in layout["state_joint_names"]],
            "gripper": name_to_index[layout["state_gripper_name"]],
        }

    if "timestamp" in frame.columns:
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        timestamps = timestamps - timestamps[0]
    else:
        timestamps = np.arange(len(frame), dtype=np.float64) / fps
    if "frame_index" in frame.columns:
        frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    else:
        frame_indices = np.arange(len(frame), dtype=np.int64)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"timestamps are not monotonic in episode {episode_index}")

    output_frames = []
    for row_index, (timestamp, frame_index) in enumerate(zip(timestamps, frame_indices)):
        joints_by_side = {
            side: states[row_index, side_indices[side]["joints"]]
            for side in SIDE_LAYOUT
        }
        poses = fk.forward(joints_by_side)
        arms = {}
        for side, layout in SIDE_LAYOUT.items():
            gripper_index = side_indices[side]["gripper"]
            raw = float(states[row_index, gripper_index])
            action_raw = (
                None if actions is None else float(actions[row_index, gripper_index])
            )
            arms[side] = {
                "output_frame": layout["output_frame"],
                "joint_position_rad": joints_by_side[side].tolist(),
                "T_flange_in_output_frame": poses[side]["flange"].tolist(),
                "T_tcp_in_output_frame": poses[side]["tcp"].tolist(),
                "gripper": gripper_values(
                    raw,
                    action_raw,
                    open_raw=open_raw,
                    closed_raw=closed_raw,
                    binary_threshold_raw=binary_threshold_raw,
                ),
            }
        output_frames.append(
            {
                "idx": row_index,
                "frame_index": int(frame_index),
                "timestamp_s": float(timestamp),
                "arms": arms,
            }
        )

    events = {side: gripper_events(output_frames, side) for side in SIDE_LAYOUT}
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "parquet": str(source_path.resolve()),
            "source_column": source_column,
            "episode_index": int(episode_index),
        },
        "robot": "arx5_2025_bimanual",
        "fps": float(fps),
        "total_frames": len(output_frames),
        "coordinate_convention": {
            "axes": "right-handed, z-up, +x forward, +y left",
            "output_frames": {
                side: SIDE_LAYOUT[side]["output_frame"] for side in SIDE_LAYOUT
            },
            "flange_pose_field": "T_flange_in_output_frame",
            "tcp_pose_field": "T_tcp_in_output_frame",
            "T_output_frame_in_scene": {
                side: fk.T_output_in_scene[side].tolist() for side in SIDE_LAYOUT
            },
            "tcp_sites": {
                side: SIDE_LAYOUT[side]["tcp_site"] for side in SIDE_LAYOUT
            },
        },
        "gripper_convention": {
            "raw_open": float(open_raw),
            "raw_closed": float(closed_raw),
            "continuous_open": 0.0,
            "continuous_closed": 1.0,
            "continuous_formula": "clip((raw - raw_open) / (raw_closed - raw_open), 0, 1)",
            "binary_open": 0,
            "binary_closed": 1,
            "binary_threshold_raw": float(binary_threshold_raw),
            "binary_formula": "1 if raw >= binary_threshold_raw else 0",
        },
        "gripper_events": events,
        "frames": output_frames,
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_path = as_abs(args.input)
    output_dir = as_abs(args.out)
    scene_path = as_abs(args.scene)
    calibration_path = as_abs(args.calibration)
    if not (args.gripper_open_raw < args.gripper_binary_threshold_raw < args.gripper_closed_raw):
        raise ValueError(
            "--gripper-binary-threshold-raw must lie strictly between "
            "--gripper-open-raw and --gripper-closed-raw"
        )

    parquets = discover_parquets(input_path)
    dataset_root = find_dataset_root(input_path)
    info = load_dataset_info(dataset_root)
    names = state_names(info, args.source_column)
    fps = float(args.fps if args.fps is not None else info.get("fps", 30.0))
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    import pandas as pd

    fk = ArxForwardKinematics(scene_path, calibration_path)
    outputs = []
    used_names: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"manifest exists; pass --overwrite: {manifest_path}")
    for parquet_path in parquets:
        columns = [args.source_column]
        for optional in ("action", "timestamp", "frame_index", "episode_index"):
            if optional not in columns:
                columns.append(optional)
        frame = pd.read_parquet(parquet_path)
        available = set(frame.columns)
        selected = [column for column in columns if column in available]
        frame = frame[selected]
        for episode_index, episode_frame in episode_groups(frame):
            base_name = f"episode_{episode_index:06d}.json"
            if base_name in used_names:
                base_name = f"{parquet_path.stem}_episode_{episode_index:06d}.json"
            used_names.add(base_name)
            output_path = output_dir / base_name
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
            payload = convert_episode(
                episode_frame,
                episode_index=episode_index,
                source_path=parquet_path,
                source_column=args.source_column,
                names=names,
                fps=fps,
                fk=fk,
                open_raw=args.gripper_open_raw,
                closed_raw=args.gripper_closed_raw,
                binary_threshold_raw=args.gripper_binary_threshold_raw,
            )
            output_path.write_text(
                json.dumps(jsonable(payload), indent=2), encoding="utf-8"
            )
            outputs.append(
                {
                    "episode_index": episode_index,
                    "output": str(output_path.resolve()),
                    "source": str(parquet_path.resolve()),
                    "total_frames": payload["total_frames"],
                    "event_counts": {
                        side: len(payload["gripper_events"][side]) for side in SIDE_LAYOUT
                    },
                }
            )
            print(
                f"Wrote {output_path}  frames={payload['total_frames']}  "
                f"events(left/right)="
                f"{len(payload['gripper_events']['left'])}/"
                f"{len(payload['gripper_events']['right'])}"
            )

    manifest = {
        "schema": "tate.arx_real_flange_dataset_manifest",
        "schema_version": SCHEMA_VERSION,
        "input": str(input_path),
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "scene": str(scene_path),
        "calibration": str(calibration_path),
        "source_column": args.source_column,
        "fps": fps,
        "gripper": {
            "raw_open": args.gripper_open_raw,
            "raw_closed": args.gripper_closed_raw,
            "binary_threshold_raw": args.gripper_binary_threshold_raw,
        },
        "episodes": outputs,
    }
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2), encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="One LeRobot parquet file or a dataset root containing data/**/*.parquet",
    )
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument(
        "--source-column",
        choices=("observation.state", "action"),
        default="observation.state",
        help="Joint state used for FK; observation.state is the measured-state default",
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="ARX MuJoCo scene")
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION),
        help="YAML containing T_eef_frame_in_scene for both arms",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override dataset FPS")
    parser.add_argument(
        "--gripper-open-raw", type=float, default=DEFAULT_GRIPPER_OPEN_RAW
    )
    parser.add_argument(
        "--gripper-closed-raw", type=float, default=DEFAULT_GRIPPER_CLOSED_RAW
    )
    parser.add_argument(
        "--gripper-binary-threshold-raw",
        type=float,
        default=DEFAULT_GRIPPER_BINARY_THRESHOLD_RAW,
        help="Raw values >= threshold are closed (1); default -2.6 is near the open end",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
