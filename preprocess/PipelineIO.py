"""Serialization and coordinate conversion for the split WiLoR pipeline.

The WiLoR cache is deliberately expressed in the camera frame.  Camera-to-arm
extrinsics are only consumed when exporting EEF targets, so changing an
extrinsic calibration never requires another expensive WiLoR inference pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from preprocess.Hand2Gripper import FingerCenter, MODES, debounce_grasp, make_hand2gripper


WILOR_SCHEMA_VERSION = 1
EEF_SCHEMA_VERSION = 2


def safe_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def select_intrinsics(
    calibration: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """Select an exact-resolution intrinsic profile.

    Intrinsics are not silently rescaled.  In particular, changing from 16:9
    to 4:3 can include sensor cropping and cannot be inferred from image size.
    """

    intrinsics = calibration.get("intrinsics") or {}
    profile_name = f"{int(width)}x{int(height)}"
    profiles = intrinsics.get("profiles") or {}
    selected = profiles.get(profile_name)

    resolution = calibration.get("resolution") or {}
    top_level_matches = (
        int(resolution.get("width", -1)) == int(width)
        and int(resolution.get("height", -1)) == int(height)
    )
    if selected is None and top_level_matches:
        selected = intrinsics
        profile_name = "top_level"
    if selected is None:
        available = sorted(str(key) for key in profiles)
        if top_level_matches:
            available.append("top_level")
        raise ValueError(
            f"No exact camera intrinsic profile for {width}x{height}. "
            f"Available profiles: {available or ['none']}. Recalibrate or add "
            "an exact profile; automatic aspect-ratio-changing scaling is unsafe."
        )

    K = _matrix(selected.get("K", selected.get("k")), (3, 3), f"intrinsics {profile_name}.K")
    d = np.asarray(selected.get("d", selected.get("D", [])), dtype=np.float64).reshape(-1)
    if d.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"intrinsics {profile_name}.d must contain OpenCV distortion coefficients, "
            f"got length {d.size}"
        )
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"intrinsics {profile_name}.K must have positive focal lengths")
    return K, d, profile_name


def load_arm_camera_transforms(
    calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return camera poses in each arm's configured output coordinate frame."""

    arm_cfg = calibration.get("arm_extrinsics") or {}
    output: dict[str, np.ndarray] = {}
    root_camera: dict[str, np.ndarray] = {}
    eef_in_arm: dict[str, np.ndarray] = {}
    eef_in_scene: dict[str, np.ndarray | None] = {}
    frame_names: dict[str, str] = {}
    for side in ("right", "left"):
        side_cfg = arm_cfg.get(side) or {}
        key = f"T_cam_in_{side}_arm_base"
        if side_cfg.get(key) is None:
            raise ValueError(f"Camera calibration is missing arm_extrinsics.{side}.{key}")
        T_cam_in_arm = _matrix(side_cfg[key], (4, 4), key)
        T_eef_in_arm = _matrix(
            side_cfg.get("T_eef_frame_in_arm_base", np.eye(4)),
            (4, 4),
            f"arm_extrinsics.{side}.T_eef_frame_in_arm_base",
        )
        if not np.allclose(T_cam_in_arm[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"{key} has an invalid homogeneous last row")
        if not np.allclose(T_eef_in_arm[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"{side} T_eef_frame_in_arm_base has an invalid last row")

        root_camera[side] = T_cam_in_arm
        eef_in_arm[side] = T_eef_in_arm
        scene_value = side_cfg.get("T_eef_frame_in_scene")
        eef_in_scene[side] = (
            None
            if scene_value is None
            else _matrix(
                scene_value,
                (4, 4),
                f"arm_extrinsics.{side}.T_eef_frame_in_scene",
            )
        )
        output[side] = np.linalg.inv(T_eef_in_arm) @ T_cam_in_arm
        frame_names[side] = str(side_cfg.get("eef_frame", f"{side}_arm_base"))

    metadata = {
        "frame_names": frame_names,
        "T_cam_in_arm_base": {side: safe_list(value) for side, value in root_camera.items()},
        "T_eef_frame_in_arm_base": {side: safe_list(value) for side, value in eef_in_arm.items()},
        "T_eef_frame_in_scene": {side: safe_list(value) for side, value in eef_in_scene.items()},
        "T_cam_in_output_frame": {side: safe_list(value) for side, value in output.items()},
    }
    return output, metadata


def _pose_to_camera(pose: Any, processing_c2w: np.ndarray) -> Any:
    if pose is None:
        return None
    return np.linalg.inv(processing_c2w) @ np.asarray(pose, dtype=np.float64)


def _point_to_camera(point: Any, processing_c2w: np.ndarray) -> Any:
    if point is None:
        return None
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return (np.linalg.inv(processing_c2w) @ point_h)[:3]


def _vector_to_camera(vector: Any, processing_c2w: np.ndarray) -> Any:
    if vector is None:
        return None
    return processing_c2w[:3, :3].T @ np.asarray(vector, dtype=np.float64).reshape(3)


def _pack_hand_camera(hand: Any, processing_c2w: np.ndarray) -> dict[str, Any] | None:
    if hand is None:
        return None
    return {
        "is_right": bool(hand.is_right),
        "confidence": safe_list(hand.confidence),
        "grasp_state": int(hand.grasp_state),
        "depth_source": getattr(hand, "depth_source", None),
        "principal_point_corrected": getattr(hand, "principal_point_corrected", None),
        "keypoints_3d_cam": safe_list(hand.hand_keypoints_3d),
        "keypoints_2d": safe_list(hand.hand_keypoints_2d),
        "midpoint_pose_cam": safe_list(_pose_to_camera(hand.midpoint_pose_opt_world, processing_c2w)),
        "wrist_pose_cam": safe_list(_pose_to_camera(hand.wrist_pose_opt_world, processing_c2w)),
        "midpoint_translation_cam": safe_list(
            _point_to_camera(hand.midpoint_translation_opt_world, processing_c2w)
        ),
        "midpoint_linear_velocity_cam": safe_list(
            _vector_to_camera(hand.midpoint_lin_vel_opt_world, processing_c2w)
        ),
        "midpoint_angular_velocity_cam": safe_list(
            _vector_to_camera(hand.midpoint_ang_vel_opt_world, processing_c2w)
        ),
        "thumb_translation_cam": safe_list(
            _point_to_camera(hand.thumb_translation_opt_world, processing_c2w)
        ),
        "index_translation_cam": safe_list(
            _point_to_camera(hand.index_translation_opt_world, processing_c2w)
        ),
        "thumb_base_cam": safe_list(_point_to_camera(hand.thumb_base_opt_world, processing_c2w)),
        "index_base_cam": safe_list(_point_to_camera(hand.index_base_opt_world, processing_c2w)),
        "distance_midpoint_to_wrist_m": safe_list(hand.distance_midpoint2wrist_opt_world),
    }


def build_wilor_cache(
    aria_cam: Any,
    aria_hands: Any,
    source_video: str | None,
    calibration_meta: dict,
    wilor_meta: dict | None = None,
    linear_speed_limit: float = 0.0,
    angular_speed_limit: float = 0.0,
) -> dict:
    frames = []
    for cam_data, hands_data in zip(aria_cam.cam, aria_hands.hands):
        frames.append(
            {
                "idx": int(cam_data.idx),
                "ts": int(cam_data.ts),
                "hand_r": _pack_hand_camera(hands_data.hand_r, cam_data.c2w),
                "hand_l": _pack_hand_camera(hands_data.hand_l, cam_data.c2w),
            }
        )
    payload = {
        "schema": "tate.wilor_hands",
        "schema_version": WILOR_SCHEMA_VERSION,
        "source_video": source_video,
        "total_frames": len(frames),
        "fps": float(aria_cam.fps),
        "width": int(aria_cam.w),
        "height": int(aria_cam.h),
        "coordinate_frame": "opencv_camera_x_right_y_down_z_forward",
        "keypoint_order": "aria_humanego_21",
        "K": safe_list(aria_cam.k),
        "d": safe_list(aria_cam.d),
        "camera_calibration": calibration_meta,
        "wilor": wilor_meta or {},
        "frames": frames,
    }
    enforce_cache_kinematic_limits(payload, linear_speed_limit, angular_speed_limit)
    return payload


def enforce_cache_kinematic_limits(
    payload: dict, linear_speed_limit: float, angular_speed_limit: float
) -> None:
    """Enforce frame-to-frame limits on cached midpoint EEF targets in-place."""
    fps = float(payload.get("fps") or 0.0)
    if fps <= 0.0:
        raise ValueError("WiLoR cache must contain a positive fps")
    dt = 1.0 / fps
    max_distance = float(linear_speed_limit) * dt
    max_angle = float(angular_speed_limit) * dt

    for hand_key in ("hand_r", "hand_l"):
        previous_pose = None
        for frame in payload.get("frames", []):
            hand = frame.get(hand_key)
            if hand is None or hand.get("midpoint_pose_cam") is None:
                previous_pose = None
                continue
            pose = _matrix(hand["midpoint_pose_cam"], (4, 4), "midpoint_pose_cam").copy()
            original_position = pose[:3, 3].copy()
            if previous_pose is not None:
                delta = original_position - previous_pose[:3, 3]
                distance = float(np.linalg.norm(delta))
                if linear_speed_limit > 0.0 and distance > max_distance:
                    pose[:3, 3] = previous_pose[:3, 3] + delta * (max_distance / distance)

                rotvec = _rotation_vector(previous_pose[:3, :3].T @ pose[:3, :3])
                angle = float(np.linalg.norm(rotvec))
                if angular_speed_limit > 0.0 and angle > max_angle:
                    pose[:3, :3] = previous_pose[:3, :3] @ _rotation_matrix(
                        rotvec * (max_angle / angle)
                    )

            correction = pose[:3, 3] - original_position
            hand["midpoint_pose_cam"] = safe_list(pose)
            hand["midpoint_translation_cam"] = safe_list(pose[:3, 3])
            for point_key in (
                "thumb_translation_cam",
                "index_translation_cam",
                "thumb_base_cam",
                "index_base_cam",
            ):
                if hand.get(point_key) is not None:
                    hand[point_key] = safe_list(
                        np.asarray(hand[point_key], dtype=np.float64) + correction
                    )
            hand["midpoint_linear_velocity_cam"] = safe_list(
                np.zeros(3)
                if previous_pose is None
                else (pose[:3, 3] - previous_pose[:3, 3]) / dt
            )
            hand["midpoint_angular_velocity_cam"] = safe_list(
                np.zeros(3)
                if previous_pose is None
                else _rotation_vector(previous_pose[:3, :3].T @ pose[:3, :3]) / dt
            )
            previous_pose = pose

    payload["kinematic_limits"] = {
        "linear_speed_m_s": float(linear_speed_limit),
        "angular_speed_rad_s": float(angular_speed_limit),
        "applied_to": "camera-frame HumanEgo midpoint pose",
    }


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Stable SO(3) logarithm implemented without a scipy dependency."""
    trace = float(np.trace(rotation))
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        # This branch is only a safety fallback; normal cached trajectories are
        # already temporally sign-consistent and remain far from pi per frame.
        from scipy.spatial.transform import Rotation as SciPyRotation

        return SciPyRotation.from_matrix(rotation).as_rotvec()
    axis = np.asarray(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def _rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        return np.eye(3)
    axis = rotvec / angle
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def write_json(payload: dict, path: str | os.PathLike[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return output


def _transform_pose(pose: Any, T_cam_in_output: np.ndarray) -> Any:
    if pose is None:
        return None
    return T_cam_in_output @ _matrix(pose, (4, 4), "cached hand pose")


def _transform_point(point: Any, T_cam_in_output: np.ndarray) -> Any:
    if point is None:
        return None
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return (T_cam_in_output @ point_h)[:3]


def _transform_vector(vector: Any, T_cam_in_output: np.ndarray) -> Any:
    if vector is None:
        return None
    return T_cam_in_output[:3, :3] @ np.asarray(vector, dtype=np.float64).reshape(3)


def _validate_rigid_transform(value: Any, name: str) -> np.ndarray:
    transform = _matrix(value, (4, 4), name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform


def _load_tcp_local_transforms(
    calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    extra = calibration.get("extra_transforms") or {}
    T_hand_to_ee = _validate_rigid_transform(
        extra.get("T_hand_to_ee"), "extra_transforms.T_hand_to_ee"
    )
    T_ee_axis_correct = _validate_rigid_transform(
        extra.get("T_ee_axis_correct"), "extra_transforms.T_ee_axis_correct"
    )
    legacy_transform = T_hand_to_ee @ T_ee_axis_correct
    configured_by_side = extra.get("T_tcp_in_hand") or {}
    transforms: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    for side in ("right", "left"):
        configured = configured_by_side.get(side)
        if configured is None:
            transforms[side] = legacy_transform.copy()
            sources[side] = "legacy T_hand_to_ee @ T_ee_axis_correct"
        else:
            transforms[side] = _validate_rigid_transform(
                configured, f"extra_transforms.T_tcp_in_hand.{side}"
            )
            sources[side] = f"extra_transforms.T_tcp_in_hand.{side}"
    return transforms, {
        "T_hand_to_ee": safe_list(T_hand_to_ee),
        "T_ee_axis_correct": safe_list(T_ee_axis_correct),
        "T_tcp_in_hand": {
            side: safe_list(transform) for side, transform in transforms.items()
        },
        "source_by_side": sources,
        "composition": "T_tcp_in_cam = T_hand2gripper_hand_in_cam @ T_tcp_in_hand[side]",
    }


def _target_to_eef_record(
    target: Any,
    side: str,
    T_cam_in_output: np.ndarray,
    T_tcp_in_hand: np.ndarray,
    frame_name: str,
) -> dict[str, Any] | None:
    if target is None:
        return None
    T_hand_in_cam = np.asarray(target.T_hand_in_cam, dtype=np.float64)
    T_tcp_in_cam = T_hand_in_cam @ T_tcp_in_hand
    T_tcp_in_output = T_cam_in_output @ T_tcp_in_cam
    return {
        "is_right": side == "right",
        "eef_frame": frame_name,
        "pose_semantics": "arx_tcp",
        "confidence": safe_list(target.confidence),
        "grasp_state": int(target.grasp_state),
        "grasp_ratio": safe_list(target.grasp_ratio),
        "hand2gripper_mode": target.mode,
        "hand2gripper_hand_pose_cam": safe_list(T_hand_in_cam),
        # Legacy field name retained for downstream consumers that predate mode selection.
        "finger_center_hand_pose_cam": safe_list(T_hand_in_cam),
        "tcp_pose_cam": safe_list(T_tcp_in_cam),
        "tcp_pose_eef_frame": safe_list(T_tcp_in_output),
        # Compatibility aliases consumed by replay and IK.
        "eef_pose_cam": safe_list(T_tcp_in_cam),
        "eef_pose_world": safe_list(T_tcp_in_output),
        "eef_translation_world": safe_list(T_tcp_in_output[:3, 3]),
        "eef_linear_velocity_world": None,
        "eef_angular_velocity_world": None,
    }


def _limit_eef_records(
    frames: list[dict[str, Any]],
    arm_transforms: dict[str, np.ndarray],
    fps: float,
    linear_speed_limit: float,
    angular_speed_limit: float,
) -> None:
    """Limit TCP motion in camera space, then regenerate arm-frame poses."""
    if fps <= 0.0:
        raise ValueError("WiLoR cache must contain a positive fps")
    dt = 1.0 / fps
    max_distance = linear_speed_limit * dt
    max_angle = angular_speed_limit * dt
    for side, hand_key in (("right", "hand_r"), ("left", "hand_l")):
        previous = None
        for frame in frames:
            record = frame.get(hand_key)
            if record is None:
                previous = None
                continue
            pose = _matrix(record["tcp_pose_cam"], (4, 4), "tcp_pose_cam").copy()
            if previous is not None:
                delta = pose[:3, 3] - previous[:3, 3]
                distance = float(np.linalg.norm(delta))
                if linear_speed_limit > 0.0 and distance > max_distance:
                    pose[:3, 3] = previous[:3, 3] + delta * (max_distance / distance)
                rotvec = _rotation_vector(previous[:3, :3].T @ pose[:3, :3])
                angle = float(np.linalg.norm(rotvec))
                if angular_speed_limit > 0.0 and angle > max_angle:
                    pose[:3, :3] = previous[:3, :3] @ _rotation_matrix(
                        rotvec * (max_angle / angle)
                    )
            output_pose = arm_transforms[side] @ pose
            record["tcp_pose_cam"] = safe_list(pose)
            record["tcp_pose_eef_frame"] = safe_list(output_pose)
            record["eef_pose_cam"] = safe_list(pose)
            record["eef_pose_world"] = safe_list(output_pose)
            record["eef_translation_world"] = safe_list(output_pose[:3, 3])
            record["eef_linear_velocity_world"] = safe_list(
                np.zeros(3)
                if previous is None
                else arm_transforms[side][:3, :3]
                @ ((pose[:3, 3] - previous[:3, 3]) / dt)
            )
            record["eef_angular_velocity_world"] = safe_list(
                np.zeros(3)
                if previous is None
                else arm_transforms[side][:3, :3]
                @ (_rotation_vector(previous[:3, :3].T @ pose[:3, :3]) / dt)
            )
            previous = pose


def export_eef_from_cache(
    cache_path: str | os.PathLike[str],
    calibration_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    eef_config_path: str | os.PathLike[str] | None = None,
    grasp_close_ratio: float | None = None,
    grasp_open_ratio: float | None = None,
    grasp_min_frames: int | None = None,
    hand2gripper_mode: str | None = None,
    grasp_mode: str | None = None,
    arm_mode: str | None = None,
    active_sides: list[str] | tuple[str, ...] | None = None,
) -> dict:
    with open(cache_path, "r", encoding="utf-8") as stream:
        cache = json.load(stream)
    if (
        cache.get("schema") != "tate.wilor_hands"
        or cache.get("schema_version") != WILOR_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported WiLoR cache schema: {cache.get('schema')!r} v{cache.get('schema_version')!r}"
        )
    if cache.get("keypoint_order") != "aria_humanego_21":
        raise ValueError(
            "Independent EEF export requires keypoint_order='aria_humanego_21'; "
            f"got {cache.get('keypoint_order')!r}"
        )

    calibration = load_yaml(calibration_path)
    eef_config = load_yaml(eef_config_path) if eef_config_path else {}
    hand_cfg = eef_config.get("hand2gripper") or {}
    mode = str(hand2gripper_mode or hand_cfg.get("mode", FingerCenter.MODE_NAME))
    grasp_mode = str(grasp_mode or hand_cfg.get("grasp_mode", mode))
    if mode not in MODES or grasp_mode not in MODES:
        raise ValueError(
            f"mode/grasp_mode must be chosen from {sorted(MODES)}; "
            f"got {mode!r}/{grasp_mode!r}"
        )
    active_sides = list(active_sides or ("left", "right"))
    if (
        not active_sides
        or any(side not in {"left", "right"} for side in active_sides)
        or len(active_sides) != len(set(active_sides))
    ):
        raise ValueError("active_sides must contain unique values chosen from left/right")
    arm_mode = str(
        arm_mode or ("single_arm" if len(active_sides) == 1 else "bimanual")
    )
    if arm_mode == "single_arm" and len(active_sides) != 1:
        raise ValueError("single_arm EEF export requires exactly one active side")
    if arm_mode == "bimanual" and set(active_sides) != {"left", "right"}:
        raise ValueError("bimanual EEF export requires left and right active sides")
    close_ratio = (
        grasp_close_ratio
        if grasp_close_ratio is not None
        else hand_cfg.get("grasp_close_ratio")
    )
    open_ratio = (
        grasp_open_ratio
        if grasp_open_ratio is not None
        else hand_cfg.get("grasp_open_ratio")
    )
    min_frames = (
        grasp_min_frames
        if grasp_min_frames is not None
        else hand_cfg.get("grasp_min_frames")
    )
    pose_converters = {
        side: make_hand2gripper(
            mode,
            grasp_close_ratio=close_ratio,
            grasp_open_ratio=open_ratio,
            grasp_min_frames=min_frames,
        )
        for side in ("right", "left")
    }
    grasp_converters = (
        pose_converters
        if grasp_mode == mode
        else {
            side: make_hand2gripper(
                grasp_mode,
                grasp_close_ratio=close_ratio,
                grasp_open_ratio=open_ratio,
                grasp_min_frames=min_frames,
            )
            for side in ("right", "left")
        }
    )
    tcp_local_transforms, tcp_transform_meta = _load_tcp_local_transforms(calibration)
    arm_transforms, transform_meta = load_arm_camera_transforms(calibration)
    frame_names = transform_meta["frame_names"]
    frames = []
    for frame in cache.get("frames", []):
        output_frame = {"idx": int(frame["idx"]), "ts": int(frame["ts"])}
        for side, hand_key in (("right", "hand_r"), ("left", "hand_l")):
            source_hand = frame.get(hand_key)
            target = (
                None
                if side not in active_sides or source_hand is None
                else pose_converters[side].from_hand_record(
                    source_hand, is_right=side == "right"
                )
            )
            if target is not None and grasp_converters is not pose_converters:
                grasp_target = grasp_converters[side].from_hand_record(
                    source_hand, is_right=side == "right"
                )
                if grasp_target is not None:
                    target.grasp_state = grasp_target.grasp_state
                    target.grasp_ratio = grasp_target.grasp_ratio
            output_frame[hand_key] = _target_to_eef_record(
                target,
                side,
                arm_transforms[side],
                tcp_local_transforms[side],
                frame_names[side],
            )
        frames.append(output_frame)

    debounced_by_side = {}
    for side, hand_key in (("right", "hand_r"), ("left", "hand_l")):
        states = [
            None if frame.get(hand_key) is None else frame[hand_key]["grasp_state"]
            for frame in frames
        ]
        fixed, changed = debounce_grasp(
            states, grasp_converters[side].grasp_min_frames
        )
        for frame, state in zip(frames, fixed):
            if frame.get(hand_key) is not None:
                frame[hand_key]["grasp_state"] = int(state)
        debounced_by_side[side] = int(changed)

    limits = eef_config.get("kinematic_limits") or {}
    linear_limit = float(limits.get("linear_speed_m_s", 0.0))
    angular_limit = float(limits.get("angular_speed_rad_s", 0.0))
    _limit_eef_records(
        frames,
        arm_transforms,
        float(cache.get("fps") or 0.0),
        linear_limit,
        angular_limit,
    )

    payload = {
        "schema": "tate.dual_arm_eef",
        "schema_version": EEF_SCHEMA_VERSION,
        "source_video": cache.get("source_video"),
        "source_wilor_cache": str(Path(cache_path)),
        "source_video_original": cache.get("source_video_original", cache.get("source_video")),
        "temporal_window": cache.get("temporal_window"),
        "arm_mode": arm_mode,
        "active_sides": active_sides,
        "total_frames": len(frames),
        "fps": cache.get("fps"),
        "width": cache.get("width"),
        "height": cache.get("height"),
        "k": cache.get("K"),
        "d": cache.get("d"),
        "eef_coordinate_convention": {
            "pose_semantics": "arx_tcp",
            "tcp_orientation_applied": True,
            "per_side_frames": frame_names,
            "axes": "right-handed, z-up, +x forward, +y left",
            "compatibility_note": "eef_pose_world is an alias of tcp_pose_eef_frame",
        },
        "hand2gripper": {
            "mode": mode,
            "pose_mode": mode,
            "grasp_mode": grasp_mode,
            "available_modes": sorted(MODES),
            "input_keypoint_order": cache.get("keypoint_order"),
            "grasp_hysteresis": {
                "close_ratio": grasp_converters["right"].grasp_close_ratio,
                "open_ratio": grasp_converters["right"].grasp_open_ratio,
                "min_frames": grasp_converters["right"].grasp_min_frames,
                "debounced_frames": debounced_by_side,
                "signal": (
                    "thumb-virtual-fingertip distance / wrist-middle-MCP palm size"
                    if grasp_mode == "qwen"
                    else "thumb-index tip distance / wrist-middle-MCP palm size"
                ),
            },
        },
        "tcp_local_transform": tcp_transform_meta,
        "kinematic_limits": {
            "linear_speed_m_s": linear_limit,
            "angular_speed_rad_s": angular_limit,
            "applied_to": "exported ARX TCP target",
        },
        "arm_transforms": transform_meta,
        "eef_frame_in_scene": transform_meta["T_eef_frame_in_scene"],
        "camera_calibration": {
            "path": str(Path(calibration_path)),
            "camera_model": calibration.get("camera_model"),
        },
        "frames": frames,
    }
    write_json(payload, output_path)
    return payload
