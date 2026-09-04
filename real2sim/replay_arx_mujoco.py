#!/usr/bin/env python3
"""Replay ARX trajectories in MuJoCo.

Modes:
  eef   - replay right-hand EEF as a mocap marker; robot stays still.
  joint - replay right-arm joint qpos; robot moves.

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


def load_eef_json(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    times, pos, quat, grasp = [], [], [], []

    if "frames" in data:
        for i, frame in enumerate(data["frames"]):
            hand = frame.get("hand_r")
            if not hand or hand.get("eef_pose_world") is None:
                continue
            p, q = pose_to_pos_quat_xyzw(hand["eef_pose_world"])
            stamp = frame.get("ts")
            times.append(float(stamp) * 1e-9 if stamp is not None else i / FPS)
            pos.append(p)
            quat.append(q)
            grasp.append(int(hand.get("grasp_state", 0)))
    elif "records" in data:
        for i, rec in enumerate(data["records"]):
            if not rec.get("valid", True):
                continue
            ee = rec.get("T_ee_in_base")
            if not ee:
                continue
            stamp = rec.get("ts")
            times.append(float(stamp) * 1e-9 if stamp is not None else i / FPS)
            pos.append(np.asarray(ee["translation_m"], dtype=np.float64))
            quat.append(np.asarray(ee["quat_xyzw"], dtype=np.float64))
            grasp.append(int(rec.get("grasp", 0)))
    else:
        raise RuntimeError(f"Unsupported EEF JSON format: {path}")

    if not pos:
        raise RuntimeError(f"No right-hand EEF data found in {path}")

    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return {
        "time_s": t,
        "pos": np.asarray(pos, dtype=np.float64),
        "quat_xyzw": np.asarray(quat, dtype=np.float64),
        "grasp": np.asarray(grasp, dtype=np.int32),
    }


def _first_npz_key(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float64)
    return None


def load_joint_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files}
    q = _first_npz_key(data, ("right_joint_qpos", "right_arm_qpos", "joint_qpos"))
    if q is None:
        raise RuntimeError(f"Joint replay requires right-arm qpos data in {path}")
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < len(RIGHT_ARM_JOINTS):
        raise RuntimeError(f"Right-arm qpos must be Nx6 or wider, got {q.shape}")

    n = len(q)
    out = {
        "time_s": np.asarray(data.get("time_s", np.arange(n, dtype=np.float64) / FPS), dtype=np.float64),
        "arm_qpos": q[:, : len(RIGHT_ARM_JOINTS)],
        "gripper_qpos": np.full((n, 2), 0.044, dtype=np.float64),
        "target_pos": _first_npz_key(data, ("right_target_pos_m", "target_pos_m")),
        "target_quat_xyzw": _first_npz_key(data, ("right_target_quat_xyzw", "target_quat_xyzw")),
        "pos_err_m": np.asarray(data.get("pos_err_m", np.full(n, np.nan)), dtype=np.float64),
        "ang_err_deg": np.asarray(data.get("ang_err_deg", np.full(n, np.nan)), dtype=np.float64),
    }

    if q.shape[1] >= len(RIGHT_ARM_JOINTS) + 2:
        out["gripper_qpos"] = q[:, len(RIGHT_ARM_JOINTS) : len(RIGHT_ARM_JOINTS) + 2]
    elif q.shape[1] >= len(RIGHT_ARM_JOINTS) + 1:
        width = np.clip(q[:, len(RIGHT_ARM_JOINTS)], 0.0, 0.088)
        out["gripper_qpos"] = np.column_stack([0.5 * width, 0.5 * width])
    elif "gripper_width_m" in data:
        width = np.clip(np.asarray(data["gripper_width_m"], dtype=np.float64).reshape(-1), 0.0, 0.088)
        out["gripper_qpos"] = np.column_stack([0.5 * width, 0.5 * width])
    elif "grasp" in data:
        grasp = np.asarray(data["grasp"], dtype=np.int32).reshape(-1)
        width = np.where(grasp > 0, 0.0, 0.088)
        out["gripper_qpos"] = np.column_stack([0.5 * width, 0.5 * width])

    if len(out["time_s"]) != n:
        raise RuntimeError(f"time_s length {len(out['time_s'])} does not match qpos length {n}")
    return out


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
    cam.azimuth = 145.0
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


def apply_joint_frame(data, arm_addrs, gripper_addrs, acts, frame: dict[str, Any]) -> None:
    data.time = float(frame["time_s"])
    arm_qpos = np.asarray(frame["arm_qpos"], dtype=np.float64)
    gripper_qpos = np.asarray(frame["gripper_qpos"], dtype=np.float64)
    data.qpos[arm_addrs] = arm_qpos
    data.qpos[gripper_addrs] = gripper_qpos
    for name, value in zip(RIGHT_ARM_JOINTS, arm_qpos):
        if name in acts:
            data.ctrl[acts[name]] = float(value)
    for name, value in zip(RIGHT_GRIPPER_JOINTS, gripper_qpos):
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


def draw_path(scn, points, stride: int = 2) -> None:
    _, mj = require_runtime()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return
    sampled = pts[:: max(stride, 1)]
    if not np.allclose(sampled[-1], pts[-1]):
        sampled = np.vstack([sampled, pts[-1]])
    for p0, p1 in zip(sampled[:-1], sampled[1:]):
        _add_capsule(scn, mj, p0, p1, (0.05, 0.75, 1.0, 0.70), 0.004)


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
            arm_addrs = qpos_addrs(model, RIGHT_ARM_JOINTS)
            gripper_addrs = qpos_addrs(model, RIGHT_GRIPPER_JOINTS)
            acts = actuator_ids(model, RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS)

        i = 0
        while viewer.is_running():
            with viewer.lock():
                if args.mode == "eef":
                    data.time = float(replay["time_s"][i])
                    set_marker(data, marker_mid, replay["pos"][i], replay["quat_xyzw"][i])
                    viewer.user_scn.ngeom = 0
                    draw_path(viewer.user_scn, replay["pos"])
                else:
                    frame = {k: v[i] for k, v in replay.items() if isinstance(v, np.ndarray) and len(v) == len(replay["time_s"])}
                    apply_joint_frame(data, arm_addrs, gripper_addrs, acts, frame)
                    if replay["target_pos"] is not None and replay["target_quat_xyzw"] is not None:
                        set_marker(data, marker_mid, replay["target_pos"][i], replay["target_quat_xyzw"][i])
                mj.mj_forward(model, data)
            viewer.set_texts((None, None, f"ARX {args.mode} replay\n{i + 1}/{len(replay['time_s'])}", "right arm data"))
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
        arm_addrs = qpos_addrs(model, RIGHT_ARM_JOINTS)
        gripper_addrs = qpos_addrs(model, RIGHT_GRIPPER_JOINTS)
        acts = actuator_ids(model, RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS)

    try:
        n = len(replay["time_s"]) if replay is not None else int(FPS * 2)
        for i in range(n):
            if replay is None:
                info = "static scene; no replay data"
            elif args.mode == "eef":
                data.time = float(replay["time_s"][i])
                set_marker(data, marker_mid, replay["pos"][i], replay["quat_xyzw"][i])
                info = f"EEF target in robot base: {replay['pos'][i]}"
            else:
                frame = {k: v[i] for k, v in replay.items() if isinstance(v, np.ndarray) and len(v) == len(replay["time_s"])}
                apply_joint_frame(data, arm_addrs, gripper_addrs, acts, frame)
                if replay["target_pos"] is not None and replay["target_quat_xyzw"] is not None:
                    set_marker(data, marker_mid, replay["target_pos"][i], replay["target_quat_xyzw"][i])
                info = f"right arm qpos: {replay['arm_qpos'][i]}"
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if replay is not None and args.mode == "eef" and hasattr(renderer, "scene"):
                draw_path(renderer.scene, replay["pos"])
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
        return load_joint_npz(path)
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ARX EEF or joint trajectories in MuJoCo")
    parser.add_argument("--mode", choices=("eef", "joint"), default="eef")
    parser.add_argument("--data", default=None, help="EEF JSON for eef mode, or joint NPZ for joint mode")
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
