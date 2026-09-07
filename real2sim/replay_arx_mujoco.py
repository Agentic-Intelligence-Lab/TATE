#!/usr/bin/env python3
"""Replay ARX trajectories in MuJoCo.

Modes:
  eef   - replay left/right EEF trajectories as markers; robot stays still.
  joint - replay left/right joint qpos from ARX parquet or IK NPZ data.

If --data is omitted, the script only shows/renders the static ARX scene.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "real2sim"

FPS = 30.0
WIDTH = 1280
HEIGHT = 720
RIGHT_ARM_JOINTS = tuple(f"right_joint{i}" for i in range(11, 17))
RIGHT_GRIPPER_JOINTS = ("right_joint17", "right_joint18")
LEFT_ARM_JOINTS = tuple(f"left_joint{i}" for i in range(1, 7))
LEFT_GRIPPER_JOINTS = ("left_joint7", "left_joint8")
ARM_JOINTS_BY_SIDE = {"right": RIGHT_ARM_JOINTS, "left": LEFT_ARM_JOINTS}
GRIPPER_JOINTS_BY_SIDE = {"right": RIGHT_GRIPPER_JOINTS, "left": LEFT_GRIPPER_JOINTS}
ARX_LEFT_SLICE = slice(0, 7)
ARX_RIGHT_SLICE = slice(7, 14)
SIM_GRIPPER_MAX_M = 0.088
REAL_GRIPPER_CLOSED = -3.4
REAL_GRIPPER_OPEN = 0.1
EEF_STYLES = {
    "right": {
        "label": "right",
        "rgba": (0.05, 0.75, 1.0, 0.78),
        "sphere_rgba": (0.05, 0.75, 1.0, 1.0),
    },
    "left": {
        "label": "left",
        "rgba": (1.0, 0.20, 0.90, 0.78),
        "sphere_rgba": (1.0, 0.20, 0.90, 1.0),
    },
}

cv2 = None
mujoco = None


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_runtime(viewer: bool) -> None:
    global cv2, mujoco
    os.environ.setdefault("MUJOCO_GL", "glfw" if viewer else "egl")

    import cv2 as cv2_module
    import mujoco as mujoco_module

    if not hasattr(mujoco_module, "Renderer"):
        from mujoco.rendering.classic.renderer import Renderer

        mujoco_module.Renderer = Renderer
    if viewer:
        import mujoco.viewer  # noqa: F401

    cv2 = cv2_module
    mujoco = mujoco_module


def require_runtime():
    if cv2 is None or mujoco is None:
        raise RuntimeError("call load_runtime() first")
    return cv2, mujoco


def quat_xyzw_to_wxyz(q: np.ndarray) -> list[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def pose_to_pos_quat_xyzw(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(pose, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"EEF pose must be 4x4, got {mat.shape}")
    pos = mat[:3, 3].copy()
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat


def _nan_pose_sequence(n: int) -> np.ndarray:
    return np.full((n, 4, 4), np.nan, dtype=np.float64)


def _arm_display_transforms(data: dict[str, Any]) -> dict[str, np.ndarray]:
    identity = np.eye(4, dtype=np.float64)
    transforms = {"right": identity, "left": identity}
    arm_c2w = data.get("arm_camera_c2w") or {}
    if not isinstance(arm_c2w, dict) or "right" not in arm_c2w or "left" not in arm_c2w:
        return transforms

    right_c2w = np.asarray(arm_c2w["right"], dtype=np.float64)
    left_c2w = np.asarray(arm_c2w["left"], dtype=np.float64)
    if right_c2w.shape != (4, 4) or left_c2w.shape != (4, 4):
        return transforms

    right_to_left = left_c2w @ np.linalg.inv(right_c2w)
    transforms["left"] = np.linalg.inv(right_to_left)
    return transforms


def _eef_scene_transforms(data: dict[str, Any]) -> dict[str, np.ndarray] | None:
    raw = data.get("eef_frame_in_scene") or {}
    if not isinstance(raw, dict) or "right" not in raw or "left" not in raw:
        return None
    transforms = {}
    for side in ("right", "left"):
        mat = np.asarray(raw[side], dtype=np.float64)
        if mat.shape != (4, 4):
            return None
        transforms[side] = mat
    return transforms


def load_eef_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if "frames" in data:
        frames = data["frames"]
        n = len(frames)
        times = np.zeros(n, dtype=np.float64)
        hands = {
            "right": {"pose": _nan_pose_sequence(n), "valid": np.zeros(n, dtype=bool), "grasp": np.zeros(n, dtype=np.int32)},
            "left": {"pose": _nan_pose_sequence(n), "valid": np.zeros(n, dtype=bool), "grasp": np.zeros(n, dtype=np.int32)},
        }
        scene_tf = _eef_scene_transforms(data)
        display_tf = scene_tf if scene_tf is not None else _arm_display_transforms(data)
        display_frame = "scene" if scene_tf is not None else "right_arm_base"

        for i, frame in enumerate(frames):
            stamp = frame.get("ts")
            times[i] = float(stamp) * 1e-9 if stamp is not None else i / FPS
            for side, key in (("right", "hand_r"), ("left", "hand_l")):
                hand = frame.get(key)
                if not hand or hand.get("eef_pose_world") is None:
                    continue
                pose = np.asarray(hand["eef_pose_world"], dtype=np.float64)
                if pose.shape != (4, 4):
                    continue
                hands[side]["pose"][i] = display_tf[side] @ pose
                hands[side]["valid"][i] = True
                hands[side]["grasp"][i] = int(hand.get("grasp_state", 0))
    elif "records" in data:
        times, poses, grasp = [], [], []
        for i, rec in enumerate(data["records"]):
            if not rec.get("valid", True):
                continue
            ee = rec.get("T_ee_in_base")
            if not ee:
                continue
            stamp = rec.get("ts")
            times.append(float(stamp) * 1e-9 if stamp is not None else i / FPS)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R.from_quat(ee["quat_xyzw"]).as_matrix()
            pose[:3, 3] = np.asarray(ee["translation_m"], dtype=np.float64)
            poses.append(pose)
            grasp.append(int(rec.get("grasp", 0)))
        n = len(poses)
        hands = {
            "right": {"pose": np.asarray(poses, dtype=np.float64), "valid": np.ones(n, dtype=bool), "grasp": np.asarray(grasp, dtype=np.int32)},
            "left": {"pose": _nan_pose_sequence(n), "valid": np.zeros(n, dtype=bool), "grasp": np.zeros(n, dtype=np.int32)},
        }
        times = np.asarray(times, dtype=np.float64)
    else:
        raise RuntimeError(f"Unsupported EEF JSON format: {path}")

    if not (hands["right"]["valid"].any() or hands["left"]["valid"].any()):
        raise RuntimeError(f"No left/right EEF data found in {path}")

    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    for side in ("right", "left"):
        pose = hands[side]["pose"]
        hands[side]["pos"] = pose[:, :3, 3]
        quat = np.full((len(t), 4), np.nan, dtype=np.float64)
        for i, valid in enumerate(hands[side]["valid"]):
            if valid:
                quat[i] = R.from_matrix(pose[i, :3, :3]).as_quat()
        hands[side]["quat_xyzw"] = quat

    return {"time_s": t, "hands": hands, "display_frame": locals().get("display_frame", "right_arm_base")}


def _first_npz_key(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float64)
    return None


def gripper_scalar_to_qpos(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.nanmin(values) < -0.1 or np.nanmax(values) > SIM_GRIPPER_MAX_M:
        width = (values - REAL_GRIPPER_CLOSED) / (REAL_GRIPPER_OPEN - REAL_GRIPPER_CLOSED)
        width = np.clip(width, 0.0, 1.0) * SIM_GRIPPER_MAX_M
    else:
        width = np.clip(values, 0.0, SIM_GRIPPER_MAX_M)
    return np.column_stack([0.5 * width, 0.5 * width])


def _joint_qpos_to_arm(side: str, q: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < len(ARM_JOINTS_BY_SIDE[side]):
        raise RuntimeError(f"{side} qpos must be Nx6 or wider, got {q.shape}")

    n = len(q)
    arm = {
        "arm_qpos": q[:, : len(ARM_JOINTS_BY_SIDE[side])],
        "gripper_qpos": np.full((n, 2), 0.044, dtype=np.float64),
    }
    if q.shape[1] >= len(ARM_JOINTS_BY_SIDE[side]) + 2:
        arm["gripper_qpos"] = q[:, len(ARM_JOINTS_BY_SIDE[side]) : len(ARM_JOINTS_BY_SIDE[side]) + 2]
    elif q.shape[1] >= len(ARM_JOINTS_BY_SIDE[side]) + 1:
        arm["gripper_qpos"] = gripper_scalar_to_qpos(q[:, len(ARM_JOINTS_BY_SIDE[side])])
    elif f"{side}_gripper_width_m" in data:
        arm["gripper_qpos"] = gripper_scalar_to_qpos(data[f"{side}_gripper_width_m"])
    elif f"{side}_grasp" in data:
        grasp = np.asarray(data[f"{side}_grasp"], dtype=np.int32).reshape(-1)
        width = np.where(grasp > 0, 0.0, SIM_GRIPPER_MAX_M)
        arm["gripper_qpos"] = np.column_stack([0.5 * width, 0.5 * width])
    return arm


def load_joint_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files}
    right_q = _first_npz_key(data, ("right_joint_qpos", "right_arm_qpos", "joint_qpos"))
    if right_q is None:
        raise RuntimeError(f"Joint replay requires right-arm qpos data in {path}")
    right_q = np.asarray(right_q, dtype=np.float64)
    left_q = _first_npz_key(data, ("left_joint_qpos", "left_arm_qpos"))

    n = len(right_q)
    out = {
        "time_s": np.asarray(data.get("time_s", np.arange(n, dtype=np.float64) / FPS), dtype=np.float64),
        "arms": {"right": _joint_qpos_to_arm("right", right_q, data)},
        "target_pos": _first_npz_key(data, ("right_target_pos_m", "target_pos_m")),
        "target_quat_xyzw": _first_npz_key(data, ("right_target_quat_xyzw", "target_quat_xyzw")),
        "pos_err_m": np.asarray(data.get("right_pos_err_m", data.get("pos_err_m", np.full(n, np.nan))), dtype=np.float64),
        "ang_err_deg": np.asarray(data.get("right_ang_err_deg", data.get("ang_err_deg", np.full(n, np.nan))), dtype=np.float64),
    }
    if left_q is not None:
        left_q = np.asarray(left_q, dtype=np.float64)
        if len(left_q) != n:
            raise RuntimeError(f"left qpos length {len(left_q)} does not match right qpos length {n}")
        out["arms"]["left"] = _joint_qpos_to_arm("left", left_q, data)
    out["arms"]["right"]["target_pos"] = out["target_pos"]
    out["arms"]["right"]["target_quat_xyzw"] = out["target_quat_xyzw"]
    if "left" in out["arms"]:
        out["arms"]["left"]["target_pos"] = _first_npz_key(data, ("left_target_pos_m",))
        out["arms"]["left"]["target_quat_xyzw"] = _first_npz_key(data, ("left_target_quat_xyzw",))

    if len(out["time_s"]) != n:
        raise RuntimeError(f"time_s length {len(out['time_s'])} does not match qpos length {n}")
    out["arm_qpos"] = out["arms"]["right"]["arm_qpos"]
    out["gripper_qpos"] = out["arms"]["right"]["gripper_qpos"]
    return out


def load_joint_parquet(path: Path) -> dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(path)
    source_col = "observation.state" if "observation.state" in df.columns else "action"
    if source_col not in df.columns:
        raise RuntimeError(f"Parquet joint replay requires observation.state or action in {path}")

    state = np.stack(df[source_col].to_numpy()).astype(np.float64)
    if state.ndim != 2 or state.shape[1] < ARX_RIGHT_SLICE.stop:
        raise RuntimeError(f"{source_col} must contain at least 14 values with right-arm data, got {state.shape}")

    left = state[:, ARX_LEFT_SLICE]
    right = state[:, ARX_RIGHT_SLICE]
    n = len(right)
    if "timestamp" in df.columns:
        t = df["timestamp"].to_numpy(dtype=np.float64)
        t -= t[0]
    else:
        t = np.arange(n, dtype=np.float64) / FPS

    return {
        "time_s": t,
        "arms": {
            "right": {
                "arm_qpos": right[:, : len(RIGHT_ARM_JOINTS)],
                "gripper_qpos": gripper_scalar_to_qpos(right[:, len(RIGHT_ARM_JOINTS)]),
            },
            "left": {
                "arm_qpos": left[:, : len(LEFT_ARM_JOINTS)],
                "gripper_qpos": gripper_scalar_to_qpos(left[:, len(LEFT_ARM_JOINTS)]),
            },
        },
        "arm_qpos": right[:, : len(RIGHT_ARM_JOINTS)],
        "gripper_qpos": gripper_scalar_to_qpos(right[:, len(RIGHT_ARM_JOINTS)]),
        "target_pos": None,
        "target_quat_xyzw": None,
        "pos_err_m": np.full(n, np.nan, dtype=np.float64),
        "ang_err_deg": np.full(n, np.nan, dtype=np.float64),
    }


def load_joint_data(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        return load_joint_npz(path)
    if path.suffix.lower() == ".parquet":
        return load_joint_parquet(path)
    raise RuntimeError(f"Unsupported joint data format: {path.suffix}; expected .parquet or .npz")


def qpos_addrs(model, names: tuple[str, ...]) -> np.ndarray:
    _, mj = require_runtime()
    addrs = []
    for name in names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Scene missing joint: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int32)


def actuator_ids(model, names: tuple[str, ...]) -> dict[str, int]:
    _, mj = require_runtime()
    out = {}
    for name in names:
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if aid >= 0:
            out[name] = int(aid)
    return out


def marker_mocap_id(model) -> int | None:
    _, mj = require_runtime()
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if bid < 0:
        return None
    mid = int(model.body_mocapid[bid])
    return mid if mid >= 0 else None


def make_camera():
    _, mj = require_runtime()
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.24, -0.02, 0.22]
    cam.distance = 1.15
    cam.azimuth = -35.0
    cam.elevation = -28.0
    return cam


def setup_model():
    _, mj = require_runtime()
    model = mj.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mj.MjData(model)
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "open")
    if key_id >= 0:
        mj.mj_resetDataKeyframe(model, data, key_id)
    else:
        mj.mj_resetData(model, data)
    return model, data


def set_marker(data, marker_mid: int | None, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    if marker_mid is None:
        return
    data.mocap_pos[marker_mid] = np.asarray(pos, dtype=np.float64)
    data.mocap_quat[marker_mid] = quat_xyzw_to_wxyz(np.asarray(quat_xyzw, dtype=np.float64))


def hide_marker(data, marker_mid: int | None) -> None:
    if marker_mid is None:
        return
    data.mocap_pos[marker_mid] = np.asarray([0.0, 0.0, -10.0], dtype=np.float64)
    data.mocap_quat[marker_mid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def apply_joint_frame(data, replay: dict[str, Any], i: int, arm_addrs, gripper_addrs, acts) -> None:
    data.time = float(replay["time_s"][i])
    for side, arm in replay.get("arms", {}).items():
        arm_qpos = np.asarray(arm["arm_qpos"][i], dtype=np.float64)
        gripper_qpos = np.asarray(arm["gripper_qpos"][i], dtype=np.float64)
        data.qpos[arm_addrs[side]] = arm_qpos
        data.qpos[gripper_addrs[side]] = gripper_qpos
        for name, value in zip(ARM_JOINTS_BY_SIDE[side], arm_qpos):
            if name in acts:
                data.ctrl[acts[name]] = float(value)
        for name, value in zip(GRIPPER_JOINTS_BY_SIDE[side], gripper_qpos):
            if name in acts:
                data.ctrl[acts[name]] = float(value)


def _add_capsule(scn, mj, p0, p1, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(
        geom,
        mj.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mj.mjtCatBit.mjCAT_DECOR
    mj.mjv_connector(
        geom,
        mj.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _add_sphere(scn, mj, pos, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(
        geom,
        mj.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mj.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def draw_path(scn, points, valid=None, rgba=(0.05, 0.75, 1.0, 0.70), stride: int = 2) -> None:
    _, mj = require_runtime()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return
    if valid is None:
        valid = np.isfinite(pts).all(axis=1)
    pts = pts[np.asarray(valid, dtype=bool)]
    if len(pts) < 2:
        return
    sampled = pts[:: max(stride, 1)]
    if not np.allclose(sampled[-1], pts[-1]):
        sampled = np.vstack([sampled, pts[-1]])
    for p0, p1 in zip(sampled[:-1], sampled[1:]):
        _add_capsule(scn, mj, p0, p1, rgba, 0.004)


def draw_eef_marker(scn, pose, rgba, radius: float = 0.018) -> None:
    _, mj = require_runtime()
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return
    origin = pose[:3, 3]
    rot = pose[:3, :3]
    _add_sphere(scn, mj, origin, rgba, radius)
    axis_rgba = (
        (1.0, 0.05, 0.05, 1.0),
        (0.05, 0.8, 0.1, 1.0),
        (0.1, 0.3, 1.0, 1.0),
    )
    for axis_i, axis_color in enumerate(axis_rgba):
        _add_capsule(scn, mj, origin, origin + rot[:, axis_i] * 0.075, axis_color, 0.004)


def draw_eef_replay(scn, replay: dict[str, Any], i: int) -> None:
    for side in ("right", "left"):
        hand = replay["hands"][side]
        style = EEF_STYLES[side]
        draw_path(scn, hand["pos"], hand["valid"], rgba=style["rgba"])
        if hand["valid"][i]:
            draw_eef_marker(scn, hand["pose"][i], style["sphere_rgba"])


def draw_joint_targets(scn, replay: dict[str, Any], i: int) -> None:
    for side, arm in replay.get("arms", {}).items():
        target_pos = arm.get("target_pos")
        if target_pos is None:
            continue
        pos = np.asarray(target_pos[i], dtype=np.float64)
        if pos.shape == (3,) and np.isfinite(pos).all():
            _add_sphere(scn, require_runtime()[1], pos, EEF_STYLES[side]["sphere_rgba"], 0.014)


def eef_info(replay: dict[str, Any], i: int) -> str:
    chunks = []
    for side in ("right", "left"):
        hand = replay["hands"][side]
        if hand["valid"][i]:
            pos = hand["pos"][i]
            chunks.append(f"{side[0].upper()} {pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}")
        else:
            chunks.append(f"{side[0].upper()} n/a")
    return f"cyan=right magenta=left frame={replay.get('display_frame', 'scene')}  " + "  ".join(chunks)


def joint_info(replay: dict[str, Any], i: int) -> str:
    chunks = []
    for side in ("right", "left"):
        arm = replay.get("arms", {}).get(side)
        if arm is None:
            chunks.append(f"{side[0].upper()} n/a")
            continue
        q = arm["arm_qpos"][i]
        chunks.append(f"{side[0].upper()} q0={q[0]:+.2f} q1={q[1]:+.2f} q2={q[2]:+.2f}")
    return "joint arms: " + "  ".join(chunks)


def draw_hud(frame_rgb: np.ndarray, mode: str, i: int, n: int, info: str) -> np.ndarray:
    cv, _ = require_runtime()
    frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
    lines = [f"ARX MuJoCo {mode} replay  frame {i + 1}/{n}", info]
    x, y = 18, 28
    for line in lines:
        cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv.LINE_AA)
        y += 26
    return frame


def launch_viewer(args, replay: dict[str, np.ndarray] | None) -> None:
    _, mj = require_runtime()
    model, data = setup_model()
    marker_mid = marker_mocap_id(model)
    cam = make_camera()

    import mujoco.viewer

    print("Launching MuJoCo viewer. Close the viewer window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = cam.type
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth
        viewer.cam.elevation = cam.elevation

        if replay is None:
            while viewer.is_running():
                with viewer.lock():
                    mj.mj_forward(model, data)
                viewer.set_texts((None, None, "ARX static scene", "no replay data"))
                viewer.sync()
                time.sleep(1.0 / FPS)
            return

        arm_addrs = gripper_addrs = acts = None
        if args.mode == "joint":
            arm_addrs = {side: qpos_addrs(model, joints) for side, joints in ARM_JOINTS_BY_SIDE.items()}
            gripper_addrs = {side: qpos_addrs(model, joints) for side, joints in GRIPPER_JOINTS_BY_SIDE.items()}
            all_joints = LEFT_ARM_JOINTS + LEFT_GRIPPER_JOINTS + RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS
            acts = actuator_ids(model, all_joints)

        i = 0
        while viewer.is_running():
            with viewer.lock():
                if args.mode == "eef":
                    data.time = float(replay["time_s"][i])
                    hide_marker(data, marker_mid)
                    viewer.user_scn.ngeom = 0
                    draw_eef_replay(viewer.user_scn, replay, i)
                else:
                    apply_joint_frame(data, replay, i, arm_addrs, gripper_addrs, acts)
                    viewer.user_scn.ngeom = 0
                    draw_joint_targets(viewer.user_scn, replay, i)
                    if any(arm.get("target_pos") is not None for arm in replay.get("arms", {}).values()):
                        hide_marker(data, marker_mid)
                    elif replay["target_pos"] is not None and replay["target_quat_xyzw"] is not None:
                        set_marker(data, marker_mid, replay["target_pos"][i], replay["target_quat_xyzw"][i])
                mj.mj_forward(model, data)
            detail = eef_info(replay, i) if args.mode == "eef" else joint_info(replay, i)
            viewer.set_texts((None, None, f"ARX {args.mode} replay\n{i + 1}/{len(replay['time_s'])}", detail))
            viewer.sync()
            i = 0 if i == len(replay["time_s"]) - 1 else i + 1
            time.sleep(1.0 / FPS)


def render_mp4(args, replay: dict[str, np.ndarray] | None) -> None:
    cv, mj = require_runtime()
    model, data = setup_model()
    marker_mid = marker_mocap_id(model)
    renderer = mj.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = make_camera()
    out = as_abs(args.out) if args.out else DEFAULT_OUT_DIR / (f"{args.mode}_replay.mp4" if replay else "static.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(str(out), cv.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out}")

    arm_addrs = gripper_addrs = acts = None
    if replay is not None and args.mode == "joint":
        arm_addrs = {side: qpos_addrs(model, joints) for side, joints in ARM_JOINTS_BY_SIDE.items()}
        gripper_addrs = {side: qpos_addrs(model, joints) for side, joints in GRIPPER_JOINTS_BY_SIDE.items()}
        all_joints = LEFT_ARM_JOINTS + LEFT_GRIPPER_JOINTS + RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS
        acts = actuator_ids(model, all_joints)

    try:
        n = len(replay["time_s"]) if replay is not None else int(FPS * 2)
        for i in range(n):
            if replay is None:
                info = "static scene; no replay data"
            elif args.mode == "eef":
                data.time = float(replay["time_s"][i])
                hide_marker(data, marker_mid)
                info = eef_info(replay, i)
            else:
                apply_joint_frame(data, replay, i, arm_addrs, gripper_addrs, acts)
                if any(arm.get("target_pos") is not None for arm in replay.get("arms", {}).values()):
                    hide_marker(data, marker_mid)
                elif replay["target_pos"] is not None and replay["target_quat_xyzw"] is not None:
                    set_marker(data, marker_mid, replay["target_pos"][i], replay["target_quat_xyzw"][i])
                info = joint_info(replay, i)
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if replay is not None and args.mode == "eef" and hasattr(renderer, "scene"):
                draw_eef_replay(renderer.scene, replay, i)
            if replay is not None and args.mode == "joint" and hasattr(renderer, "scene"):
                draw_joint_targets(renderer.scene, replay, i)
            writer.write(draw_hud(renderer.render(), args.mode, i, n, info))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def load_replay(mode: str, data_path: str | None) -> dict[str, np.ndarray] | None:
    if not data_path:
        return None
    path = as_abs(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"data file does not exist: {path}")
    if mode == "eef":
        return load_eef_json(path)
    if mode == "joint":
        return load_joint_data(path)
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ARX EEF or joint trajectories in MuJoCo")
    parser.add_argument("--mode", choices=("eef", "joint"), default="eef")
    parser.add_argument("--data", default=None, help="EEF JSON for eef mode, or ARX parquet for joint mode")
    parser.add_argument("--out", default=None, help="Output mp4 path when --viewer is not set")
    parser.add_argument("--viewer", action="store_true", help="Open interactive viewer instead of rendering mp4")
    args = parser.parse_args()

    if not DEFAULT_SCENE.is_file():
        raise FileNotFoundError(f"scene file does not exist: {DEFAULT_SCENE}")

    load_runtime(args.viewer)
    replay = load_replay(args.mode, args.data)
    if args.viewer:
        launch_viewer(args, replay)
    else:
        render_mp4(args, replay)


if __name__ == "__main__":
    main()
