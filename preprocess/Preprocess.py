# -*- coding: utf-8 -*-
# @FileName: Preprocess.py

"""
WiLoR-only preprocessing entry point.

Outputs exactly the requested sequence-level artifacts:
  1. preprocess/hand_keypoints_eef_vis.mp4
  2. preprocess/eef.json

No per-frame RGB images or per-frame JSON files are written by this entry point.
"""

import argparse
import json
import os
import re
import traceback
from typing import Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from preprocess.AriaCamTypes import AriaCam, AriaCamData
from preprocess.WiLoRHands import WiLoRHandsGenerator
import preprocess.WiLoRHands as WiLoRHandsModule
from utils.utils_io import load_cfg
from utils.utils_math import time_it


def get_task_list(parent_path, run_range=None):
    pattern = re.compile(r"^mps_.*_(\d+)_vrs$")
    task_folders = []
    if not os.path.exists(parent_path):
        print(f"[Error] Parent path not found: {parent_path}")
        return []

    for item in os.listdir(parent_path):
        full_path = os.path.join(parent_path, item)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(item)
        if match:
            task_folders.append({"index": int(match.group(1)), "path": full_path})

    task_folders.sort(key=lambda x: x["index"])
    if run_range:
        start, end = run_range
        task_folders = [f for f in task_folders if start <= f["index"] <= end]
        print(f"[Range] Filtering indices from {start} to {end}. Found {len(task_folders)} task(s).")

    return [f["path"] for f in task_folders]


def _load_master_cfg(cfg_path: str) -> dict:
    if not cfg_path or not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_cfg_path(master_cfg: dict, key: str, fallback: str) -> str:
    path = master_cfg.get(key, fallback)
    if os.path.exists(path):
        return path
    candidate = os.path.normpath(os.path.join(os.path.dirname(fallback), os.path.basename(path)))
    return candidate if os.path.exists(candidate) else path


def _safe_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _cfg_get(cfg: dict, *keys):
    value = cfg
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_cfg_value(cfg: dict, paths):
    for path in paths:
        value = _cfg_get(cfg, *path)
        if value is not None and value != "":
            return value
    return None


def _cfg_matrix(value, shape, name: str):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.shape != shape:
        if arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)
        else:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _cfg_vector(value, name: str):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"{name} must contain OpenCV distortion coefficients, got length {arr.size}")
    return arr


class Preprocess:
    """Runs WiLoR hand reconstruction and exports HumanEgo-style hand products."""

    def __init__(
        self,
        mps_path,
        cfg_path,
        task=None,
        export_video=True,
        export_gif=False,
        video_path: Optional[str] = None,
    ):
        self.mps_path = mps_path
        self.cfg_path = cfg_path
        self.task = task
        self.export_video = export_video
        self.export_gif = export_gif
        self.video_path = video_path

        self.master_cfg = _load_master_cfg(cfg_path)
        self.aria_hands_cfg_path = _resolve_cfg_path(
            self.master_cfg,
            "AriaHands_path",
            "./cfg/preprocess/base/AriaHands.yaml",
        )
        self.aria_cam_cfg_path = _resolve_cfg_path(
            self.master_cfg,
            "AriaCam_path",
            "./cfg/preprocess/base/AriaCam.yaml",
        )
        self.aria_slam_cfg_path = _resolve_cfg_path(
            self.master_cfg,
            "AriaSlam_path",
            "./cfg/preprocess/base/AriaSlam.yaml",
        )
        self.aria_phases_cfg_path = _resolve_cfg_path(
            self.master_cfg,
            "AriaPhases_path",
            "./cfg/preprocess/base/AriaPhases.yaml",
        )
        self.camera_calibration_cfg_path = _resolve_cfg_path(
            self.master_cfg,
            "CameraCalibration_path",
            "./cfg/preprocess/base/RealSenseD405.yaml",
        )
        self.aria_cam_cfg = load_cfg(self.aria_cam_cfg_path)
        self.camera_calibration_cfg = _load_master_cfg(self.camera_calibration_cfg_path)
        self.camera_calibration_meta = {
            "path": self.camera_calibration_cfg_path,
            "camera_model": self.camera_calibration_cfg.get("camera_model"),
            "used_fields": [],
            "fallback_fields": ["K", "d", "c2w"],
        }
        self.arm_root_camera_c2w = {"right": None, "left": None}
        self.arm_camera_c2w = {"right": None, "left": None}
        self.arm_eef_frame_in_arm_base = {"right": np.eye(4, dtype=np.float64), "left": np.eye(4, dtype=np.float64)}
        self.arm_eef_frame_in_scene = {"right": None, "left": None}
        self.arm_frame_names = {"right": "right_flange_zero", "left": "left_flange_zero"}
        self.has_vrs_input = False

    def _camera_params_from_video_defaults(self, w: int, h: int):
        focal = float(self.master_cfg.get("video_focal_px", 500.0))
        k = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        d = np.zeros(8, dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        return k, d, c2w

    def _load_video_camera_params(self, w: int, h: int):
        k, d, c2w = self._camera_params_from_video_defaults(w, h)
        cfg = self.camera_calibration_cfg or {}
        used_fields = []

        if not cfg:
            self.camera_calibration_meta["used_fields"] = used_fields
            print("[Camera] No calibration YAML loaded; using video default intrinsics/extrinsics.")
            return k, d, c2w

        calib_w = _cfg_get(cfg, "resolution", "width")
        calib_h = _cfg_get(cfg, "resolution", "height")
        if calib_w and calib_h and (int(calib_w) != w or int(calib_h) != h):
            print(f"[Warn] Calibration resolution {calib_w}x{calib_h} does not match video {w}x{h}.")

        k_cfg = _cfg_matrix(
            _first_cfg_value(cfg, [("intrinsics", "K"), ("intrinsics", "k"), ("K",), ("k",)]),
            (3, 3),
            "camera intrinsics K",
        )
        d_cfg = _cfg_vector(
            _first_cfg_value(cfg, [("intrinsics", "d"), ("intrinsics", "D"), ("d",), ("D",)]),
            "camera distortion d",
        )
        c2w_cfg = _cfg_matrix(
            _first_cfg_value(cfg, [("extrinsics", "c2w"), ("c2w",)]),
            (4, 4),
            "camera extrinsics c2w",
        )

        if k_cfg is not None:
            k = k_cfg
            used_fields.append("K")
        if d_cfg is not None:
            d = d_cfg
            used_fields.append("d")
        if c2w_cfg is not None:
            c2w = c2w_cfg
            used_fields.append("c2w")

        self._load_arm_camera_extrinsics(c2w)
        self.camera_calibration_meta["used_fields"] = used_fields
        self.camera_calibration_meta["fallback_fields"] = [name for name in ("K", "d", "c2w") if name not in used_fields]
        if used_fields:
            print(f"[Camera] Using calibration fields from {self.camera_calibration_cfg_path}: {', '.join(used_fields)}")
        else:
            print(f"[Camera] Calibration YAML has no numeric K/d/c2w yet; using video defaults.")
        return k, d, c2w

    def _load_arm_camera_extrinsics(self, default_c2w: np.ndarray) -> None:
        cfg = self.camera_calibration_cfg or {}
        arm_cfg = cfg.get("arm_extrinsics") if isinstance(cfg, dict) else {}
        used = []

        right_c2w = _cfg_matrix(
            _first_cfg_value(
                arm_cfg or {},
                [("right", "T_cam_in_right_arm_base"), ("right", "c2w")],
            ),
            (4, 4),
            "right arm camera extrinsics",
        )
        left_c2w = _cfg_matrix(
            _first_cfg_value(
                arm_cfg or {},
                [("left", "T_cam_in_left_arm_base"), ("left", "c2w")],
            ),
            (4, 4),
            "left arm camera extrinsics",
        )
        right_eef_in_arm = _cfg_matrix(
            _first_cfg_value(
                arm_cfg or {},
                [("right", "T_eef_frame_in_arm_base")],
            ),
            (4, 4),
            "right EEF frame in arm base",
        )
        left_eef_in_arm = _cfg_matrix(
            _first_cfg_value(
                arm_cfg or {},
                [("left", "T_eef_frame_in_arm_base")],
            ),
            (4, 4),
            "left EEF frame in arm base",
        )
        right_eef_in_scene = _cfg_matrix(
            _first_cfg_value(
                arm_cfg or {},
                [("right", "T_eef_frame_in_scene")],
            ),
            (4, 4),
            "right EEF frame in scene",
        )
        left_eef_in_scene = _cfg_matrix(
            _first_cfg_value(
                arm_cfg or {},
                [("left", "T_eef_frame_in_scene")],
            ),
            (4, 4),
            "left EEF frame in scene",
        )

        # Remember which per-arm camera poses were actually configured before
        # applying fallbacks.  A measured T_cam_in_*_arm_base must take
        # precedence over the legacy global camera-to-scene pose.
        has_measured_c2w = {"right": right_c2w is not None, "left": left_c2w is not None}
        if right_c2w is None:
            right_c2w = default_c2w
        else:
            used.append("right")
        if left_c2w is None:
            left_c2w = default_c2w
        else:
            used.append("left")
        if right_eef_in_arm is None:
            right_eef_in_arm = np.eye(4, dtype=np.float64)
        if left_eef_in_arm is None:
            left_eef_in_arm = np.eye(4, dtype=np.float64)

        self.arm_root_camera_c2w = {"right": right_c2w, "left": left_c2w}
        self.arm_eef_frame_in_arm_base = {"right": right_eef_in_arm, "left": left_eef_in_arm}
        self.arm_eef_frame_in_scene = {"right": right_eef_in_scene, "left": left_eef_in_scene}
        self.arm_camera_c2w = {}
        for side, arm_base_c2w, eef_in_arm, eef_in_scene in (
            ("right", right_c2w, right_eef_in_arm, right_eef_in_scene),
            ("left", left_c2w, left_eef_in_arm, left_eef_in_scene),
        ):
            if has_measured_c2w[side] or eef_in_scene is None:
                self.arm_camera_c2w[side] = np.linalg.inv(eef_in_arm) @ arm_base_c2w
            else:
                self.arm_camera_c2w[side] = np.linalg.inv(eef_in_scene) @ default_c2w
        for side in ("right", "left"):
            frame_name = _cfg_get(arm_cfg or {}, side, "eef_frame")
            if frame_name:
                self.arm_frame_names[side] = str(frame_name)
        self.camera_calibration_meta["arm_extrinsics_used"] = used

    def _build_aria_cam_from_video(self, video_path: str) -> AriaCam:
        self.has_vrs_input = False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or float(getattr(self.aria_cam_cfg, "fps", 30))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        k, d, c2w = self._load_video_camera_params(w, h)

        aria_cam = AriaCam(mps_path=self.mps_path)
        aria_cam.fps = fps
        aria_cam.h = h
        aria_cam.w = w
        aria_cam.k = k
        aria_cam.d = d
        aria_cam.c2d = np.eye(4, dtype=np.float64)

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = int(round(idx * 1e9 / fps))
            aria_cam.tss.append(ts)
            aria_cam.cam.append(
                AriaCamData(
                    idx=idx,
                    ts=ts,
                    img=frame,
                    fov=0.0,
                    h=h,
                    w=w,
                    k=k,
                    d=d,
                    c2w=c2w,
                    c2d=np.eye(4, dtype=np.float64),
                    d2w=c2w,
                )
            )
            idx += 1
        cap.release()

        if not aria_cam.cam:
            raise RuntimeError(f"No frames read from video: {video_path}")
        return aria_cam

    def _build_aria_cam_from_vrs(self) -> AriaCam:
        from projectaria_tools.core import data_provider, mps
        from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
        from projectaria_tools.core.sensor_data import TimeDomain
        from preprocess.AriaCam import AriaCamGenerator

        self.has_vrs_input = True
        vrs_path = os.path.join(self.mps_path, "sample.vrs")
        hand_tracking_path = os.path.join(self.mps_path, "hand_tracking", "hand_tracking_results.csv")
        vrs_provider = data_provider.create_vrs_data_provider(vrs_path)
        mps_provider = MpsDataProvider(MpsDataPathsProvider(self.mps_path).get_data_paths())

        aria_hands_mps = mps.hand_tracking.read_hand_tracking_results(hand_tracking_path)
        rgb_tss = vrs_provider.get_timestamps_ns(vrs_provider.get_stream_id_from_label("camera-rgb"), TimeDomain.DEVICE_TIME)
        start_idx = len(rgb_tss) - len(aria_hands_mps)
        end_idx = start_idx + len(aria_hands_mps) - 1
        print(f"[Aria] start_idx={start_idx}, end_idx={end_idx}")

        aria_cam_rgb_generator = AriaCamGenerator(
            self.mps_path,
            self.aria_cam_cfg_path,
            vrs_provider,
            mps_provider,
            label="rgb",
        )
        return aria_cam_rgb_generator.get_aria_cam(start_idx, end_idx)

    def _build_aria_cam(self) -> AriaCam:
        if self.video_path:
            return self._build_aria_cam_from_video(self.video_path)
        if os.path.isfile(self.mps_path) and self.mps_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            self.video_path = self.mps_path
            self.mps_path = os.path.splitext(self.mps_path)[0] + "_preprocess"
            return self._build_aria_cam_from_video(self.video_path)
        if os.path.exists(os.path.join(self.mps_path, "sample.vrs")):
            return self._build_aria_cam_from_vrs()
        raise RuntimeError("Provide --video_path for mp4 input, or set --mps_path to a folder containing sample.vrs.")

    def _arm_c2w_for_hand(self, is_right: bool, default_c2w: np.ndarray) -> np.ndarray:
        side = "right" if is_right else "left"
        return self.arm_camera_c2w.get(side) if self.arm_camera_c2w.get(side) is not None else default_c2w

    @staticmethod
    def _pose_from_processing_world(pose, processing_c2w, output_c2w):
        if pose is None:
            return None
        pose_cam = np.linalg.inv(processing_c2w) @ pose
        return output_c2w @ pose_cam

    @staticmethod
    def _point_from_processing_world(point, processing_c2w, output_c2w):
        if point is None:
            return None
        p = np.asarray(point, dtype=np.float64).reshape(3)
        p_h = np.ones(4, dtype=np.float64)
        p_h[:3] = p
        return (output_c2w @ (np.linalg.inv(processing_c2w) @ p_h))[:3]

    @staticmethod
    def _vec_from_processing_world(vec, processing_c2w, output_c2w):
        if vec is None:
            return None
        v = np.asarray(vec, dtype=np.float64).reshape(3)
        return output_c2w[:3, :3] @ (processing_c2w[:3, :3].T @ v)

    def _hand_to_eef_json(self, hand, processing_c2w, output_c2w):
        if hand is None or hand.midpoint_pose_opt_world is None:
            return None
        is_right = bool(hand.is_right)
        eef_pose_cam = np.linalg.inv(processing_c2w) @ hand.midpoint_pose_opt_world
        eef_pose_world = output_c2w @ eef_pose_cam
        return {
            "is_right": is_right,
            "eef_frame": self.arm_frame_names["right" if is_right else "left"],
            "confidence": _safe_list(hand.confidence),
            "grasp_state": int(hand.grasp_state),
            "eef_pose_world": _safe_list(eef_pose_world),
            "eef_pose_cam": _safe_list(eef_pose_cam),
            "eef_translation_world": _safe_list(self._point_from_processing_world(hand.midpoint_translation_opt_world, processing_c2w, output_c2w)),
            "eef_linear_velocity_world": _safe_list(self._vec_from_processing_world(hand.midpoint_lin_vel_opt_world, processing_c2w, output_c2w)),
            "eef_angular_velocity_world": _safe_list(self._vec_from_processing_world(hand.midpoint_ang_vel_opt_world, processing_c2w, output_c2w)),
            "wrist_pose_world": _safe_list(self._pose_from_processing_world(hand.wrist_pose_opt_world, processing_c2w, output_c2w)),
            "thumb_translation_world": _safe_list(self._point_from_processing_world(hand.thumb_translation_opt_world, processing_c2w, output_c2w)),
            "index_translation_world": _safe_list(self._point_from_processing_world(hand.index_translation_opt_world, processing_c2w, output_c2w)),
            "thumb_base_world": _safe_list(self._point_from_processing_world(hand.thumb_base_opt_world, processing_c2w, output_c2w)),
            "index_base_world": _safe_list(self._point_from_processing_world(hand.index_base_opt_world, processing_c2w, output_c2w)),
            "distance_midpoint2wrist": _safe_list(hand.distance_midpoint2wrist_opt_world),
        }

    def _export_eef_json(self, aria_cam: AriaCam, aria_hands, save_path: str) -> None:
        frames = []
        for idx, cam_d in enumerate(aria_cam.cam):
            data = aria_hands.hands[idx]
            right_c2w = self._arm_c2w_for_hand(True, cam_d.c2w)
            left_c2w = self._arm_c2w_for_hand(False, cam_d.c2w)
            frames.append(
                {
                    "idx": int(cam_d.idx),
                    "ts": int(cam_d.ts),
                    "hand_r": self._hand_to_eef_json(data.hand_r, cam_d.c2w, right_c2w),
                    "hand_l": self._hand_to_eef_json(data.hand_l, cam_d.c2w, left_c2w),
                }
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        payload = {
            "source_video": self.video_path,
            "mps_path": self.mps_path,
            "total_frames": len(frames),
            "fps": float(aria_cam.fps),
            "height": int(aria_cam.h),
            "width": int(aria_cam.w),
            "k": _safe_list(aria_cam.k),
            "d": _safe_list(aria_cam.d),
            "c2w": _safe_list(aria_cam.cam[0].c2w) if aria_cam.cam else None,
            "eef_coordinate_convention": {
                "origin": "flange position at zero joint position",
                "axes": "right-handed, z-up, +x forward from flange, +y left",
                "per_side_frames": self.arm_frame_names,
            },
            "arm_root_camera_c2w": {side: _safe_list(value) for side, value in self.arm_root_camera_c2w.items()},
            "eef_frame_in_arm_base": {side: _safe_list(value) for side, value in self.arm_eef_frame_in_arm_base.items()},
            "eef_frame_in_scene": {side: _safe_list(value) for side, value in self.arm_eef_frame_in_scene.items()},
            "arm_camera_c2w": {side: _safe_list(value) for side, value in self.arm_camera_c2w.items()},
            "camera_calibration": self.camera_calibration_meta,
            "frames": frames,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Output] EEF JSON saved to: {save_path}")

    def _build_optional_aria_motion(self, aria_cam: AriaCam, aria_hands):
        if not self.has_vrs_input:
            return None, None, None, None

        try:
            import preprocess.AriaSlam as AriaSlamModule
            from preprocess.AriaSlam import AriaSlamGenerator
            from preprocess.AriaPhases import AriaPhasesGenerator

            AriaSlamModule.AriaSlamOps.save_professional_plot = staticmethod(lambda *args, **kwargs: None)

            aria_slam_generator = AriaSlamGenerator(self.mps_path, self.aria_slam_cfg_path, aria_cam)
            aria_slam = aria_slam_generator.get_aria_slam()
            aria_phases_generator = AriaPhasesGenerator(
                self.mps_path,
                self.aria_phases_cfg_path,
                aria_cam,
                aria_slam,
                aria_hands,
            )
            aria_phases = aria_phases_generator.get_aria_phases()
            return aria_slam_generator, aria_slam, aria_phases_generator, aria_phases
        except Exception as exc:
            print(f"[Warn] Failed to build SLAM/phase overlays: {exc}")
            traceback.print_exc()
            return None, None, None, None

    def _export_hand_keypoints_eef_video(
        self,
        aria_cam: AriaCam,
        aria_hands,
        generator: WiLoRHandsGenerator,
        save_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        first = aria_cam.cam[0].img
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, float(aria_cam.fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open output video writers")

        for idx, cam_d in enumerate(tqdm(aria_cam.cam, desc="Export Hand Keypoints/EEF Video")):
            data = aria_hands.hands[idx]
            img = cam_d.img.copy()
            img = WiLoRHandsModule.AriaHandsOps.draw_aria_hands_skeleton(
                img,
                data,
                cam_d.k,
                cam_d.d,
                cam_d.c2w,
                getattr(generator.cfg, "grasp_threshold", 0.105),
                full_skeleton=True,
            )
            img = generator.draw_aria_hands_panel(img, idx, data)
            writer.write(img)

        writer.release()
        print(f"[Output] Hand keypoints/EEF video saved to: {save_path}")

    @time_it
    def run(self) -> None:
        aria_cam = self._build_aria_cam()
        print(f"[Input] Loaded {len(aria_cam.cam)} frames ({aria_cam.w}x{aria_cam.h}, {float(aria_cam.fps):.3f} FPS)")

        # The original WiLoR generator also writes analysis PNGs. Disable that
        # side effect here so this command writes only the requested artifacts.
        WiLoRHandsModule.AriaHandsOps.save_hands_analysis_plots_two = staticmethod(lambda *args, **kwargs: None)

        generator = WiLoRHandsGenerator(self.mps_path, self.aria_hands_cfg_path, aria_cam)
        aria_hands = generator.get_aria_hands()

        output_dir = os.path.join(self.mps_path, "preprocess")
        if self.export_video:
            self._export_hand_keypoints_eef_video(
                aria_cam,
                aria_hands,
                generator,
                os.path.join(output_dir, "hand_keypoints_eef_vis.mp4"),
            )
        self._export_eef_json(aria_cam, aria_hands, os.path.join(output_dir, "eef.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiLoR HumanEgo hand preprocessing")
    parser.add_argument("--mps_path", type=str, required=True, help="Output/session folder, or a Project Aria MPS folder with sample.vrs")
    parser.add_argument("--video_path", type=str, default=None, help="Optional mp4 input path")
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/base/Preprocess.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--no-video", action="store_false", dest="export_video", help="Disable MP4 video export")
    parser.add_argument("--no-gif", action="store_false", dest="export_gif", help="Ignored; kept for CLI compatibility")
    parser.set_defaults(export_video=True, export_gif=False)
    args = parser.parse_args()

    if args.video_path or (os.path.isfile(args.mps_path) and args.mps_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))):
        final_tasks = [args.mps_path]
    elif os.path.exists(os.path.join(args.mps_path, "sample.vrs")):
        final_tasks = [args.mps_path]
    else:
        final_tasks = get_task_list(args.mps_path, args.range)

    if not final_tasks:
        print("[Error] No valid task found.")
        raise SystemExit(1)

    for path in final_tasks:
        try:
            Preprocess(
                mps_path=path,
                cfg_path=args.cfg_path,
                task=args.task,
                export_video=args.export_video,
                export_gif=args.export_gif,
                video_path=args.video_path,
            ).run()
        except Exception:
            print(f"[Critical Error] Task failed: {path}")
            print(traceback.format_exc())
