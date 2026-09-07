#!/usr/bin/env python3
"""Retarget HumanEgo EEF targets to both ARX MuJoCo arms with mink."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL2SIM_DIR = Path(__file__).resolve().parent
if str(REAL2SIM_DIR) not in sys.path:
    sys.path.insert(0, str(REAL2SIM_DIR))

from replay_arx_mujoco import FPS, load_eef_json  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
DEFAULT_EEF = REPO_ROOT / "outputs" / "test" / "preprocess" / "eef.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "test" / "ik" / "dual_arm_ik_mink.npz"

SIDE_SPECS = {
    "left": {
        "arm_joints": tuple(f"left_joint{i}" for i in range(1, 7)),
        "gripper_joints": ("left_joint7", "left_joint8"),
        "site": "left_tcp",
    },
    "right": {
        "arm_joints": tuple(f"right_joint{i}" for i in range(11, 17)),
        "gripper_joints": ("right_joint17", "right_joint18"),
        "site": "right_tcp",
    },
}
GRIPPER_OPEN_WIDTH_M = 0.088

mujoco = None


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_runtime():
    global mujoco
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mujoco_module

    mujoco = mujoco_module
    return mujoco_module


def import_mink():
    try:
        import mink
    except ImportError as exc:
        raise ImportError("mink is not installed in this Python environment.") from exc
    return mink


def object_id(model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Scene is missing {name!r}")
    return int(idx)


def joint_qpos_addrs(model, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names],
        dtype=np.int32,
    )


def joint_dof_addrs(model, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names],
        dtype=np.int32,
    )


def joint_bounds(model, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for name in names:
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo.append(model.jnt_range[jid, 0])
        hi.append(model.jnt_range[jid, 1])
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def actuator_ids(model, names: tuple[str, ...]) -> dict[str, int]:
    ids = {}
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            ids[name] = int(aid)
    return ids


def frame_pose(data, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.site_xpos[frame_id].copy(), data.site_xmat[frame_id].reshape(3, 3).copy()


def orientation_error_deg(actual_rot: np.ndarray, target_rot: np.ndarray) -> float:
    err = R.from_matrix(actual_rot.T @ target_rot).as_rotvec()
    return float(np.linalg.norm(err) * 180.0 / np.pi)


def se3_from_matrix(mink, pose: np.ndarray):
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(pose[:3, :3]),
        translation=pose[:3, 3],
    )


def set_gripper(data, spec: dict[str, Any], gripper_addrs: np.ndarray, act_ids: dict[str, int], grasp: int) -> np.ndarray:
    width = 0.0 if int(grasp) > 0 else GRIPPER_OPEN_WIDTH_M
    qpos = np.asarray([0.5 * width, 0.5 * width], dtype=np.float64)
    data.qpos[gripper_addrs] = qpos
    for name, value in zip((f"{j}_pos" for j in spec["gripper_joints"]), qpos):
        if name in act_ids:
            data.ctrl[act_ids[name]] = float(value)
    return qpos


def set_arm(data, spec: dict[str, Any], arm_addrs: np.ndarray, act_ids: dict[str, int], q: np.ndarray) -> None:
    data.qpos[arm_addrs] = q
    for name, value in zip((f"{j}_pos" for j in spec["arm_joints"]), q):
        if name in act_ids:
            data.ctrl[act_ids[name]] = float(value)


def posture_target(q0: np.ndarray, arm_addrs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    q = np.asarray(q0, dtype=np.float64).copy()
    q[arm_addrs] = q_ref
    return q


def posture_task(mink, model, q_target: np.ndarray, active_dofs: np.ndarray, active_cost: float):
    cost = np.full(model.nv, 80.0, dtype=np.float64)
    cost[active_dofs] = float(active_cost)
    task = mink.PostureTask(model, cost=cost)
    task.set_target(q_target)
    return task


def load_targets(eef_path: Path) -> dict[str, Any]:
    replay = load_eef_json(eef_path)
    targets = {"time_s": np.asarray(replay["time_s"], dtype=np.float64), "sides": {}}
    for side in ("right", "left"):
        hand = replay["hands"][side]
        valid = np.asarray(hand["valid"], dtype=bool)
        poses = np.asarray(hand["pose"], dtype=np.float64)
        targets["sides"][side] = {
            "valid": valid,
            "pose": poses,
            "pos_m": poses[:, :3, 3],
            "quat_xyzw": np.asarray(hand["quat_xyzw"], dtype=np.float64),
            "grasp": np.asarray(hand["grasp"], dtype=np.int32),
        }
    if not targets["sides"]["right"]["valid"].any():
        raise RuntimeError(f"No right-hand EEF targets found in {eef_path}")
    if not targets["sides"]["left"]["valid"].any():
        raise RuntimeError(f"No left-hand EEF targets found in {eef_path}")
    return targets


def solve_frame(
    mink,
    model,
    data,
    *,
    spec: dict[str, Any],
    q_base: np.ndarray,
    q_seed: np.ndarray,
    q_ref: np.ndarray,
    target_pose: np.ndarray,
    arm_addrs: np.ndarray,
    gripper_addrs: np.ndarray,
    active_dofs: np.ndarray,
    act_ids: dict[str, int],
    target_site_id: int,
    grasp: int,
    pos_weight: float,
    ori_weight: float,
    smooth_weight: float,
    n_iter: int,
    dt: float,
    solver: str,
    damping: float,
    lm_damping: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data.qpos[:] = q_base
    set_arm(data, spec, arm_addrs, act_ids, q_seed)
    set_gripper(data, spec, gripper_addrs, act_ids, grasp)
    mujoco.mj_forward(model, data)

    q0 = data.qpos.copy()
    configuration = mink.Configuration(model, q0)
    frame_task = mink.FrameTask(
        frame_name=spec["site"],
        frame_type="site",
        position_cost=float(pos_weight),
        orientation_cost=float(ori_weight),
        lm_damping=float(lm_damping),
    )
    frame_task.set_target(se3_from_matrix(mink, target_pose))
    posture = posture_task(mink, model, posture_target(q0, arm_addrs, q_ref), active_dofs, smooth_weight)
    limits = [mink.ConfigurationLimit(model)]

    success = True
    status = solver
    nfev = 0
    for _ in range(int(n_iter)):
        try:
            vel = mink.solve_ik(
                configuration,
                [frame_task, posture],
                float(dt),
                solver,
                damping=float(damping),
                limits=limits,
            )
        except Exception as exc:  # noqa: BLE001
            success = False
            status = f"failed:{type(exc).__name__}"
            break
        configuration.integrate_inplace(vel, float(dt))
        nfev += 1

    q_sol = np.asarray(configuration.q[arm_addrs], dtype=np.float64)
    data.qpos[:] = q_base
    set_arm(data, spec, arm_addrs, act_ids, q_sol)
    gripper_qpos = set_gripper(data, spec, gripper_addrs, act_ids, grasp)
    mujoco.mj_forward(model, data)
    actual_pos, actual_rot = frame_pose(data, target_site_id)
    target_pos = target_pose[:3, 3]
    target_rot = target_pose[:3, :3]
    metrics = {
        "success": bool(success),
        "status": status,
        "nfev": int(nfev),
        "pos_err_m": float(np.linalg.norm(actual_pos - target_pos)),
        "ang_err_deg": orientation_error_deg(actual_rot, target_rot),
        "achieved_pos_m": actual_pos,
        "achieved_quat_xyzw": R.from_matrix(actual_rot).as_quat(),
    }
    return q_sol.copy(), gripper_qpos.copy(), metrics


def solve_side(
    side: str,
    *,
    mink,
    model,
    data,
    q_base: np.ndarray,
    act_ids: dict[str, int],
    targets: dict[str, Any],
    initial_q: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = SIDE_SPECS[side]
    arm_addrs = joint_qpos_addrs(model, spec["arm_joints"])
    gripper_addrs = joint_qpos_addrs(model, spec["gripper_joints"])
    active_dofs = joint_dof_addrs(model, spec["arm_joints"])
    bounds = joint_bounds(model, spec["arm_joints"])
    target_site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, spec["site"])
    q_warm = np.clip(initial_q, bounds[0], bounds[1])

    side_targets = targets["sides"][side]
    valid = side_targets["valid"]
    n = len(targets["time_s"])
    joint_qpos = np.zeros((n, len(spec["arm_joints"]) + len(spec["gripper_joints"])), dtype=np.float64)
    arm_qpos = np.zeros((n, len(spec["arm_joints"])), dtype=np.float64)
    gripper_qpos = np.zeros((n, len(spec["gripper_joints"])), dtype=np.float64)
    achieved_pos_m = np.full((n, 3), np.nan, dtype=np.float64)
    achieved_quat_xyzw = np.full((n, 4), np.nan, dtype=np.float64)
    pos_err_m = np.full(n, np.nan, dtype=np.float64)
    ang_err_deg = np.full(n, np.nan, dtype=np.float64)
    nfev = np.zeros(n, dtype=np.int32)
    success = np.zeros(n, dtype=bool)
    status_labels = np.full(n, "missing_target", dtype=object)

    t0 = time.time()
    last_grasp = 0
    for i in range(n):
        if valid[i]:
            last_grasp = int(side_targets["grasp"][i])
            q_sol, grip, metrics = solve_frame(
                mink,
                model,
                data,
                spec=spec,
                q_base=q_base,
                q_seed=q_warm,
                q_ref=q_warm,
                target_pose=side_targets["pose"][i],
                arm_addrs=arm_addrs,
                gripper_addrs=gripper_addrs,
                active_dofs=active_dofs,
                act_ids=act_ids,
                target_site_id=target_site_id,
                grasp=last_grasp,
                pos_weight=args.pos_weight,
                ori_weight=args.ori_weight,
                smooth_weight=args.smooth_weight,
                n_iter=args.n_iter,
                dt=args.dt,
                solver=args.solver,
                damping=args.damping,
                lm_damping=args.lm_damping,
            )
            q_warm = q_sol.copy()
            success[i] = metrics["success"]
            achieved_pos_m[i] = metrics["achieved_pos_m"]
            achieved_quat_xyzw[i] = metrics["achieved_quat_xyzw"]
            pos_err_m[i] = metrics["pos_err_m"]
            ang_err_deg[i] = metrics["ang_err_deg"]
            nfev[i] = metrics["nfev"]
            status_labels[i] = str(metrics["status"])
        else:
            grip = set_gripper(data, spec, gripper_addrs, act_ids, last_grasp)

        arm_qpos[i] = q_warm
        gripper_qpos[i] = grip
        joint_qpos[i, : len(spec["arm_joints"])] = q_warm
        joint_qpos[i, len(spec["arm_joints"]) :] = grip

        if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or i == n - 1):
            valid_count = int(valid[: i + 1].sum())
            mean_mm = float(np.nanmean(pos_err_m[: i + 1]) * 1000.0) if valid_count else float("nan")
            print(
                f"{side:>5s} frame {i:5d} ({i + 1}/{n})  "
                f"valid={valid_count:3d}  mean_pos={mean_mm:6.1f}mm  {time.time() - t0:.1f}s"
            )

    return {
        "joint_qpos": joint_qpos,
        "arm_qpos": arm_qpos,
        "gripper_qpos": gripper_qpos,
        "achieved_pos_m": achieved_pos_m,
        "achieved_quat_xyzw": achieved_quat_xyzw,
        "pos_err_m": pos_err_m,
        "ang_err_deg": ang_err_deg,
        "nfev": nfev,
        "success": success,
        "valid": valid,
        "status_labels": status_labels.astype(str),
    }


def solve_trajectory(args: argparse.Namespace) -> Path:
    load_runtime()
    mink = import_mink()

    scene_path = as_abs(args.scene)
    eef_path = as_abs(args.eef)
    out_path = as_abs(args.out)
    targets = load_targets(eef_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "open")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    q_base = data.qpos.copy()
    all_actuators = []
    for spec in SIDE_SPECS.values():
        all_actuators.extend(f"{j}_pos" for j in spec["arm_joints"] + spec["gripper_joints"])
    act_ids = actuator_ids(model, tuple(all_actuators))
    initial_q = np.asarray(args.initial_q, dtype=np.float64)
    if initial_q.shape != (6,):
        raise RuntimeError("--initial-q must contain 6 values")

    t0 = time.time()
    results = {}
    for side in ("right", "left"):
        results[side] = solve_side(
            side,
            mink=mink,
            model=model,
            data=data,
            q_base=q_base,
            act_ids=act_ids,
            targets=targets,
            initial_q=initial_q,
            args=args,
        )

    n = len(targets["time_s"])
    full_qpos = np.zeros((n, model.nq), dtype=np.float64)
    ctrl = np.zeros((n, model.nu), dtype=np.float64)
    side_addrs = {
        side: (
            joint_qpos_addrs(model, spec["arm_joints"]),
            joint_qpos_addrs(model, spec["gripper_joints"]),
        )
        for side, spec in SIDE_SPECS.items()
    }
    for i in range(n):
        data.qpos[:] = q_base
        data.ctrl[:] = 0.0
        for side, spec in SIDE_SPECS.items():
            arm_addrs, gripper_addrs = side_addrs[side]
            set_arm(data, spec, arm_addrs, act_ids, results[side]["arm_qpos"][i])
            data.qpos[gripper_addrs] = results[side]["gripper_qpos"][i]
            for name, value in zip((f"{j}_pos" for j in spec["gripper_joints"]), results[side]["gripper_qpos"][i]):
                if name in act_ids:
                    data.ctrl[act_ids[name]] = float(value)
        mujoco.mj_forward(model, data)
        full_qpos[i] = data.qpos.copy()
        ctrl[i] = data.ctrl.copy()

    right = results["right"]
    left = results["left"]
    combined_success = right["success"] & (~left["valid"] | left["success"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        time_s=targets["time_s"],
        frame_idx=np.arange(n, dtype=np.int32),
        right_joint_qpos=right["joint_qpos"],
        right_arm_qpos=right["arm_qpos"],
        left_joint_qpos=left["joint_qpos"],
        left_arm_qpos=left["arm_qpos"],
        joint_qpos=right["joint_qpos"],
        full_qpos=full_qpos,
        ctrl=ctrl,
        right_target_pos_m=targets["sides"]["right"]["pos_m"],
        right_target_quat_xyzw=targets["sides"]["right"]["quat_xyzw"],
        left_target_pos_m=targets["sides"]["left"]["pos_m"],
        left_target_quat_xyzw=targets["sides"]["left"]["quat_xyzw"],
        target_pos_m=targets["sides"]["right"]["pos_m"],
        target_quat_xyzw=targets["sides"]["right"]["quat_xyzw"],
        right_achieved_pos_m=right["achieved_pos_m"],
        right_achieved_quat_xyzw=right["achieved_quat_xyzw"],
        left_achieved_pos_m=left["achieved_pos_m"],
        left_achieved_quat_xyzw=left["achieved_quat_xyzw"],
        right_pos_err_m=right["pos_err_m"],
        right_ang_err_deg=right["ang_err_deg"],
        left_pos_err_m=left["pos_err_m"],
        left_ang_err_deg=left["ang_err_deg"],
        pos_err_m=right["pos_err_m"],
        ang_err_deg=right["ang_err_deg"],
        right_nfev=right["nfev"],
        left_nfev=left["nfev"],
        success=combined_success,
        right_success=right["success"],
        left_success=left["success"],
        right_valid=right["valid"],
        left_valid=left["valid"],
        right_grasp=targets["sides"]["right"]["grasp"],
        left_grasp=targets["sides"]["left"]["grasp"],
        solver=np.asarray("mink"),
        right_status_labels=right["status_labels"],
        left_status_labels=left["status_labels"],
        scene_path=np.asarray(str(scene_path)),
        eef_path=np.asarray(str(eef_path)),
        elapsed_s=np.asarray(time.time() - t0, dtype=np.float64),
    )
    print(
        f"Wrote {out_path}  frames={n}  "
        f"right={int(right['success'].sum())}/{int(right['valid'].sum())}  "
        f"left={int(left['success'].sum())}/{int(left['valid'].sum())}  "
        f"mean_pos_r={float(np.nanmean(right['pos_err_m'])) * 1000:.1f}mm  "
        f"mean_pos_l={float(np.nanmean(left['pos_err_m'])) * 1000:.1f}mm"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eef", default=str(DEFAULT_EEF), help="HumanEgo EEF JSON path")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output IK NPZ path")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="MuJoCo scene XML path")
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--lm-damping", type=float, default=1e-2)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--ori-weight", type=float, default=0.2)
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--initial-q", type=float, nargs=6, default=[0.0, 1.2, 1.5, 0.0, 0.0, 0.0])
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    solve_trajectory(args)


if __name__ == "__main__":
    main()
